"""Local LLM analyst (no external API).

A small open-source instruction model (runs on your own machine) turns the
question into a SQL query, we run it read-only, then the same model explains
the rows in plain English. Forecast questions are routed to our own LSTM.
"""
import os
import re
import sqlite3
import threading

from .sqlguard import QueryRejected, QueryTimeout, execute_readonly
from .forecast import forecast

MODEL_NAME = os.getenv("LOCAL_LLM", "Qwen/Qwen2.5-1.5B-Instruct")
FORECAST_WORDS = ("forecast", "predict", "next hour", "will be busy", "expect")

SCHEMA = """SQLite tables:
customer_traffic(timestamp TEXT, people_count INTEGER)
videos(id, filename, fps, duration_sec, created)
tracks(video_id, track_id, first_sec, last_sec, dwell_sec, path_length, avg_speed,
       stationary_ratio, zone TEXT, role TEXT, carrying TEXT, segment TEXT)
crossings(video_id, line_name, direction[in|out], sec)  -- one row per person crossing a counting line
role is customer|employee; carrying is basket|bag|nothing;
segment is passer-by|browser|engaged-shopper; zone is r{row}c{col} on a 3x3 grid."""

SQL_SYSTEM = """You write exactly one SQLite SELECT query that answers the question. Output only SQL.
Rules:
- One row of `tracks` = one person. For counts or stats about people, shoppers, visitors, segments,
  zones, roles or dwell time use `tracks`. Use `customer_traffic` only for people counts over time.
- passer-by, browser and engaged-shopper are values of the `segment` column. `role` only holds
  customer, employee or unknown.
- Questions about people crossing, entering, leaving, "going in" or "going out" use `crossings`
  (one row per person per crossing): COUNT(*) with a WHERE on direction.
- Use `videos` only for questions about videos. Do not JOIN tables unless the question needs it.
- Add a WHERE filter only if the question explicitly names it. Never invent filters.
- "per X" / "each X" means GROUP BY X and select X together with the aggregate.
- Use only the columns listed in the schema. There is no `customer`, `person`, `visitors` or
  `traffic` column; `zone` exists only in `tracks`.
- `role` is "unknown" for almost every row (staff/customer labelling is switched off), so never
  filter on role unless the question explicitly asks about employees or staff.
- Customers, shoppers, visitors, people seen or tracked all mean rows of `tracks`: count them
  with COUNT(*) FROM tracks. `people_count` is how many people were visible at one moment, so
  use MAX or AVG on it and never SUM it to count people.
- "Traffic records", "readings", "entries" or "measurements" count rows of `customer_traffic`
  (one row per second of video), not people.
- "Busiest zone", "most visited area" or "zone traffic" mean the zone with the most rows in
  `tracks`: GROUP BY zone ORDER BY COUNT(*) DESC.

""" + SCHEMA + """

Examples:
Q: Which area of the store is visited the most?
SQL: SELECT zone FROM tracks GROUP BY zone ORDER BY COUNT(*) DESC LIMIT 1
Q: How many visitors did we detect overall?
SQL: SELECT COUNT(*) FROM tracks
Q: What was the highest number of people present at once?
SQL: SELECT MAX(people_count) FROM customer_traffic
Q: How many shoppers are engaged compared with passers-by?
SQL: SELECT segment, COUNT(*) FROM tracks WHERE segment IN ('engaged-shopper','passer-by') GROUP BY segment
Q: How many occupancy readings are stored?
SQL: SELECT COUNT(*) FROM customer_traffic
Q: How many of the people are browsers?
SQL: SELECT COUNT(*) FROM tracks WHERE segment='browser'
Q: How many tracked people were classified as employees?
SQL: SELECT COUNT(*) FROM tracks WHERE role='employee'
Q: What is the average speed of all people?
SQL: SELECT AVG(avg_speed) FROM tracks
Q: Which role has the most people?
SQL: SELECT role FROM tracks GROUP BY role ORDER BY COUNT(*) DESC LIMIT 1
Q: How many people are there in each carrying category?
SQL: SELECT carrying, COUNT(*) FROM tracks GROUP BY carrying
Q: What is the minimum people count recorded?
SQL: SELECT MIN(people_count) FROM customer_traffic
Q: How many people stayed longer than 60 seconds in zone r0c0?
SQL: SELECT COUNT(*) FROM tracks WHERE dwell_sec > 60 AND zone='r0c0'
Q: How many videos are longer than 100 seconds?
SQL: SELECT COUNT(*) FROM videos WHERE duration_sec > 100
Q: At what time did the people count reach its lowest?
SQL: SELECT timestamp FROM customer_traffic ORDER BY people_count ASC LIMIT 1
Q: How many people left through the line?
SQL: SELECT COUNT(*) FROM crossings WHERE direction='out'
Q: How many crossings were there in each direction?
SQL: SELECT direction, COUNT(*) FROM crossings GROUP BY direction
Q: What is the average path length per zone?
SQL: SELECT zone, AVG(path_length) FROM tracks GROUP BY zone"""

from .device import get_device

_llm = {}
_llm_lock = threading.Lock()


def _load_models():
    """Load tokenizer and model. Pure function: nothing is published until both exist."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = get_device()
    dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()
    return {"tok": tok, "model": model, "dev": dev}


def is_ready():
    return bool(_llm.get("ready"))


def load_llm():
    """Thread-safe, load-once. The startup preload and the first question can arrive together:
    the old version published the tokenizer before the model existed, so a question arriving
    mid-load skipped loading and crashed with KeyError('model'). Now later callers wait on the
    lock, and `ready` is set only after everything is in place."""
    if _llm.get("ready"):
        return
    with _llm_lock:
        if _llm.get("ready"):
            return
        loaded = _load_models()
        _llm.update(loaded)
        _llm["ready"] = True


def _generate_local(system, user, max_new_tokens=200):
    import torch
    load_llm()
    tok, model, dev = _llm["tok"], _llm["model"], _llm["dev"]
    text = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _generate(system, user, max_new_tokens=200):
    return _generate_local(system, user, max_new_tokens)



def _safe_select(sql):
    sql = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.M).strip().rstrip(";")
    if not sql.lower().startswith("select") or ";" in sql:
        return None
    if re.search(r"\b(insert|update|delete|drop|alter|attach|pragma|create)\b", sql, re.I):
        return None
    return sql


def _column_hint(error_text):
    """For 'no such column: x' say which allowed table(s) actually contain x. The model often
    uses the right idea on the wrong table, and repeats itself unless told where the column is."""
    m = re.search(r"no such column: (?:\w+\.)?(\w+)", error_text)
    if not m:
        return ""
    wanted, owners = m.group(1).lower(), []
    from .db import connect
    from .sqlguard import ALLOWED_TABLES
    conn = connect()
    try:
        for t in sorted(ALLOWED_TABLES):
            if wanted in {r["name"].lower() for r in conn.execute(f"PRAGMA table_info({t})")}:
                owners.append(t)
    finally:
        conn.close()
    if owners:
        return f" The column `{wanted}` does not exist in that table; it exists only in: {', '.join(owners)}."
    return f" There is no column named `{wanted}` anywhere; remove it or use a listed column."


def drop_unusable_role_filter(sql):
    """Remove `role = 'customer'|'employee'` predicates when attribute labelling is off.

    With ENABLE_CLIP_ATTRIBUTES unset every row has role 'unknown', so such a filter can only
    return zero rows. Dropping it is correct by construction, not a guess about intent.
    """
    from .classifier import attributes_enabled
    if attributes_enabled():
        return sql
    pred = r"role\s*=\s*'(?:customer|employee)'"
    out = re.sub(rf"\s+WHERE\s+{pred}\s+AND\s+", " WHERE ", sql, flags=re.I)
    out = re.sub(rf"\s+AND\s+{pred}", "", out, flags=re.I)
    out = re.sub(rf"\s+WHERE\s+{pred}", "", out, flags=re.I)
    return out


# The only values these columns can ever hold. A predicate with any other literal cannot match
# a row, so it is always an error to keep it.
ENUM_DOMAINS = {
    "segment": {"passer-by", "browser", "engaged-shopper"},
    "role": {"customer", "employee", "unknown"},
    "carrying": {"basket", "bag", "nothing", "unknown"},
    "direction": {"in", "out"},
}


def drop_impossible_enum_filters(sql):
    """Drop `segment = 'shoppers'`-style predicates whose literal is not a possible value."""
    out = sql
    for col, domain in ENUM_DOMAINS.items():
        for m in list(re.finditer(rf"\b{col}\s*=\s*'([^']*)'", out, flags=re.I)):
            if m.group(1).lower() in domain:
                continue
            pred = re.escape(m.group(0))
            out = re.sub(rf"\s+WHERE\s+{pred}\s+AND\s+", " WHERE ", out, flags=re.I)
            out = re.sub(rf"\s+AND\s+{pred}", "", out, flags=re.I)
            out = re.sub(rf"\s+WHERE\s+{pred}", "", out, flags=re.I)
    return out


def drop_hallucinated_zone_filter(sql, question):
    """Remove `zone = 'r0c0'` / `zone IN (...)` predicates whose zone code the question never
    mentions. The small model sometimes attaches a made-up zone literal to an unrelated question
    ("average path length of passers-by" -> WHERE zone = 'r0c0'), silently shrinking the answer."""
    q = question.lower()

    def invented(codes):
        return not any(c.lower() in q for c in codes)

    pat_eq = re.compile(r"zone\s*=\s*'(r\d+c\d+)'", re.I)
    pat_in = re.compile(r"zone\s+IN\s*\(([^)]*)\)", re.I)
    out = sql
    for pat, get in ((pat_eq, lambda m: [m.group(1)]),
                     (pat_in, lambda m: re.findall(r"'(r\d+c\d+)'", m.group(1)))):
        m = pat.search(out)
        if not m:
            continue
        codes = get(m)
        if codes and invented(codes):
            pred = re.escape(m.group(0))
            out = re.sub(rf"\s+WHERE\s+{pred}\s+AND\s+", " WHERE ", out, flags=re.I)
            out = re.sub(rf"\s+AND\s+{pred}", "", out, flags=re.I)
            out = re.sub(rf"\s+WHERE\s+{pred}", "", out, flags=re.I)
    return out


_SEGMENT_WORDS = (("passer-by", r"\bpassers?[- ]?by\b"), ("browser", r"\bbrowsers?\b"),
                  ("engaged-shopper", r"\bengaged(?:[- ]shoppers?)?\b"))


def ensure_segment_filter(sql, question):
    """If the question names exactly one behaviour group ("passer-by", "browser", "engaged
    shopper") and a plain single-table query over `tracks` ignores it, add the filter.
    Skipped when two groups are compared, when the query already mentions `segment`, or when it
    groups or joins (then the model deliberately did something else)."""
    named = [value for value, pat in _SEGMENT_WORDS if re.search(pat, question, re.I)]
    low = sql.lower()
    if len(named) != 1 or "segment" in low or re.search(r"\b(join|group\s+by)\b", low):
        return sql
    if not re.search(r"\bfrom\s+tracks\b", low):
        return sql
    pred = f"segment = '{named[0]}'"
    if re.search(r"\bwhere\b", sql, re.I):
        return re.sub(r"\bwhere\b", f"WHERE {pred} AND", sql, count=1, flags=re.I)
    m = re.search(r"\b(order\s+by|limit)\b", sql, re.I)
    return f"{sql[:m.start()]}WHERE {pred} {sql[m.start():]}" if m else f"{sql} WHERE {pred}"


def swap_to_owning_table(sql, error_text):
    """Single-table query that names a column living in exactly one other allowed table:
    point it at that table (e.g. `direction` is in `crossings`, not `tracks`)."""
    m = re.search(r"no such column: (?:\w+\.)?(\w+)", error_text)
    if not m or re.search(r"\bjoin\b", sql, re.I):
        return None
    tables = re.findall(r"\bFROM\s+(\w+)", sql, flags=re.I)
    if len(tables) != 1:
        return None
    from .db import connect
    from .sqlguard import ALLOWED_TABLES
    wanted, owners = m.group(1).lower(), []
    conn = connect()
    try:
        for t in sorted(ALLOWED_TABLES):
            if wanted in {r["name"].lower() for r in conn.execute(f"PRAGMA table_info({t})")}:
                owners.append(t)
    finally:
        conn.close()
    if len(owners) != 1 or owners[0].lower() == tables[0].lower():
        return None
    return re.sub(rf"\bFROM\s+{tables[0]}\b", f"FROM {owners[0]}", sql, count=1, flags=re.I)


def sql_with_repair(question, max_attempts=2):
    """Generate SQL for `question`, run it, and on a SQL error show the model the error and let
    it correct the query once (small models often invent a column on the first try).

    Returns {"sql", "rows", "truncated", "executed", "attempts", "error"}; `executed` is True
    only if a query actually ran and returned.
    """
    sql = _safe_select(_generate(SQL_SYSTEM, "Q: " + question + "\nSQL:"))
    if sql is not None:
        sql = drop_impossible_enum_filters(sql)
        sql = ensure_segment_filter(drop_hallucinated_zone_filter(drop_unusable_role_filter(sql), question), question)
    out = {"sql": sql, "rows": None, "truncated": False, "executed": False, "attempts": 0, "error": None}
    if sql is None:
        out["error"] = "The model did not produce a single SELECT statement."
        return out
    for attempt in range(1, max_attempts + 2):  # +1: a deterministic swap does not use a model call
        out["attempts"], out["sql"] = attempt, sql
        try:
            out["rows"], out["truncated"] = execute_readonly(sql)
            out["executed"], out["error"] = True, None
            return out
        except QueryRejected as e:  # policy decision, not something to "repair" around
            out["error"] = str(e)
            return out
        except QueryTimeout as e:
            out["error"] = f"{e} Try a narrower question."
            return out
        except (sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
            out["error"] = f"The query failed: {e}"
            swapped = swap_to_owning_table(sql, str(e))
            if swapped is not None and swapped != sql:  # deterministic fix beats asking again
                sql = swapped
                continue
            if attempt >= max_attempts + 1:
                return out
            fixed = _safe_select(_generate(
                SQL_SYSTEM,
                f"Q: {question}\nSQL: {sql}\nThat query failed with: {e}.{_column_hint(str(e))}\n"
                "Write a corrected single SELECT. Use only the tables and columns in the schema.\nSQL:"))
            if fixed is None or fixed == sql:
                return out
            sql = fixed
    return out


def ask(question):
    if any(w in question.lower() for w in FORECAST_WORDS):
        return {"answer": "Footfall forecast from our LSTM model.", "data": forecast(), "executed": False}

    r = sql_with_repair(question)
    if not r["executed"]:
        msg = r["error"] or "I could not turn that into a query."
        return {"answer": f"I could not answer that from the data. {msg} Try rephrasing the question.",
                "sql": r["sql"], "executed": False, "attempts": r["attempts"], "error": r["error"]}
    answer = _generate("You are a retail analytics assistant. Using only the data given, "
                       "answer the question in 1-3 simple sentences.",
                       f"Question: {question}\nSQL: {r['sql']}\nRows: {r['rows']}"
                       + ("\n(Only the first rows are shown; there are more.)" if r["truncated"] else ""))
    return {"answer": answer, "sql": r["sql"], "rows": r["rows"], "truncated": r["truncated"],
            "executed": True, "attempts": r["attempts"]}
