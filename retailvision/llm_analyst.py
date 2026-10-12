"""Local LLM analyst (no external API).

A small open-source instruction model (runs on your own machine) turns the
question into a SQL query, we run it read-only, then the same model explains
the rows in plain English. Forecast questions are routed to our own LSTM.
"""
import os
import re

from .db import connect
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
- Use only the columns listed in the schema.

""" + SCHEMA + """

Examples:
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


def load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not _llm:
        dev = get_device()
        dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
        _llm["tok"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _llm["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(dev).eval()
        _llm["dev"] = dev


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


def ask(question):
    if any(w in question.lower() for w in FORECAST_WORDS):
        return {"answer": "Footfall forecast from our LSTM model.", "data": forecast()}

    sql = _safe_select(_generate(SQL_SYSTEM, "Q: " + question + "\nSQL:"))
    if sql is None:
        return {"answer": "I could not turn that into a safe query. Try rephrasing."}
    conn = connect()
    try:
        conn.execute("PRAGMA query_only = ON")
        rows = [dict(r) for r in conn.execute(sql).fetchmany(50)]
    except Exception as e:
        return {"answer": f"The generated query failed: {e}", "sql": sql}
    finally:
        conn.close()
    answer = _generate("You are a retail analytics assistant. Using only the data given, "
                       "answer the question in 1-3 simple sentences.",
                       f"Question: {question}\nSQL: {sql}\nRows: {rows}")
    return {"answer": answer, "sql": sql, "rows": rows}
