"""Run model-written SQL safely.

Defence is layered: only a single SELECT statement is accepted by the caller, SQLite's own
authorizer allows reads from an explicit table allow-list only (so `jobs`, `count_lines`,
`sqlite_master` and pragmas are unreachable), the connection is read-only, runaway queries
are interrupted after a time limit, and the number of returned rows is capped.
"""
import sqlite3
import time

from .db import connect

ALLOWED_TABLES = {"customer_traffic", "videos", "tracks", "crossings"}
DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
MAX_ROWS = 50
TIMEOUT_SEC = 5.0


class QueryRejected(Exception):
    pass


class QueryTimeout(Exception):
    pass


def _make_authorizer(real_tables):
    """SQLite reports a CTE or subquery alias as a "table read" too, so a pure allow-list would
    block legitimate WITH clauses. Names of real tables must be on the allow-list; internal
    (`sqlite_*`) and pragma table-functions are always denied; anything else is an alias."""

    def authorizer(action, arg1, arg2, _db, _source):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:  # arg1 = table, arg2 = column
            name = (arg1 or "").lower()
            if arg1 in ALLOWED_TABLES:
                return sqlite3.SQLITE_OK
            if arg1 in real_tables or name.startswith(("sqlite_", "pragma_")):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:  # arg2 = function name
            return sqlite3.SQLITE_DENY if (arg2 or "").lower() in DENIED_FUNCTIONS else sqlite3.SQLITE_OK
        if action == getattr(sqlite3, "SQLITE_RECURSIVE", -1):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY  # inserts, updates, DDL, pragmas, attach, transactions...

    return authorizer


def execute_readonly(sql, timeout=TIMEOUT_SEC, max_rows=MAX_ROWS):
    """Return (rows, truncated). Raises QueryRejected or QueryTimeout."""
    conn = connect()
    try:
        conn.execute("PRAGMA query_only = ON")
        real = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        conn.set_authorizer(_make_authorizer(real))
        deadline = time.monotonic() + timeout
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        try:
            cur = conn.execute(sql)
            fetched = cur.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as e:
            msg = str(e).lower()
            if "interrupted" in msg:
                raise QueryTimeout(f"Query stopped after {timeout:.0f} s.") from e
            if "not authorized" in msg or "prohibited" in msg:
                raise QueryRejected("That query touches data the analyst may not read.") from e
            raise
        return [dict(r) for r in fetched[:max_rows]], len(fetched) > max_rows
    finally:
        conn.close()
