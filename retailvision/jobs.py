"""Persistent single-worker job queue backed by SQLite.

Jobs survive a server restart: queued and interrupted ("running") jobs are re-queued by
`resume()` as long as their uploaded file is still on disk. One job runs at a time because
the CPU/GPU is the bottleneck. Analysis writes its results in a single final commit, so
re-running an interrupted job cannot leave partial rows behind.
"""
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from .db import connect

MAX_QUEUED = int(os.getenv("MAX_QUEUED_JOBS", "5"))
KEEP_FINISHED = 100

_pool = ThreadPoolExecutor(max_workers=1)
_handlers = {}


class QueueFull(Exception):
    pass


def register(kind, fn):
    """fn(**params) -> JSON-serialisable result."""
    _handlers[kind] = fn


def _update(conn, job_id, **fields):
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))
    conn.commit()


def submit(kind, params, upload_path=None):
    """Queue a job; raises QueueFull if too many are waiting. Returns the job id."""
    conn = connect()
    try:
        active = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
        if active >= MAX_QUEUED:
            raise QueueFull(f"{active} analyses already queued; try again when one finishes.")
        job_id = uuid.uuid4().hex[:12]
        conn.execute("INSERT INTO jobs (id, kind, params, status, created, upload_path) VALUES (?,?,?,?,?,?)",
                     (job_id, kind, json.dumps(params), "queued", time.time(), upload_path))
        conn.commit()
    finally:
        conn.close()
    _pool.submit(_run, job_id)
    return job_id


def _run(job_id):
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["status"] in ("done", "error"):
            return
        _update(conn, job_id, status="running", started=time.time())
        try:
            result = _handlers[row["kind"]](**json.loads(row["params"]))
            _update(conn, job_id, status="done", finished=time.time(), result=json.dumps(result))
        except Exception as e:  # surfaced to the client, not swallowed
            _update(conn, job_id, status="error", finished=time.time(), error=f"{type(e).__name__}: {e}")
        finally:
            path = row["upload_path"]
            if path and os.path.exists(path):
                os.unlink(path)
            _prune(conn)
    finally:
        conn.close()


def _prune(conn):
    conn.execute("""DELETE FROM jobs WHERE status IN ('done','error') AND id NOT IN
                    (SELECT id FROM jobs WHERE status IN ('done','error') ORDER BY created DESC LIMIT ?)""",
                 (KEEP_FINISHED,))
    conn.commit()


def resume(upload_dir=None):
    """Re-queue unfinished jobs after a restart; fail the ones whose upload is gone; remove
    orphaned uploads. Returns the number of jobs re-queued."""
    conn = connect()
    requeued, keep = 0, set()
    try:
        rows = conn.execute("SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY created").fetchall()
        for r in rows:
            if r["upload_path"] and not os.path.exists(r["upload_path"]):
                _update(conn, r["id"], status="error", finished=time.time(),
                        error="Upload file was lost before the job could run; please upload again.")
                continue
            _update(conn, r["id"], status="queued", started=None)
            if r["upload_path"]:
                keep.add(os.path.abspath(r["upload_path"]))
            _pool.submit(_run, r["id"])
            requeued += 1
    finally:
        conn.close()
    if upload_dir and os.path.isdir(upload_dir):
        for name in os.listdir(upload_dir):
            p = os.path.abspath(os.path.join(upload_dir, name))
            if p not in keep and os.path.isfile(p):
                os.unlink(p)
    return requeued


def _present(row):
    return {"id": row["id"], "status": row["status"], "kind": row["kind"],
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": row["error"], "created": row["created"],
            "started": row["started"], "finished": row["finished"]}


def get(job_id):
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _present(row) if row else None
    finally:
        conn.close()


def recent(limit=20):
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [_present(r) for r in rows]
    finally:
        conn.close()
