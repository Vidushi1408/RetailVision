"""Text-to-SQL accuracy of the local LLM analyst against gold queries.

Usage: python evaluation/llm_eval.py [path/to/test.db]
A question passes if the model's SQL returns the same rows as the gold SQL.
"""
import sys
import time

import retailvision.db as db

db.DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/eval.db"
from retailvision import llm_analyst as la  # noqa: E402

QA = [
    ("How many people were tracked in total?", "SELECT COUNT(*) FROM tracks"),
    ("How many videos have been analyzed?", "SELECT COUNT(*) FROM videos"),
    ("What is the average dwell time in seconds?", "SELECT AVG(dwell_sec) FROM tracks"),
    ("What is the longest dwell time?", "SELECT MAX(dwell_sec) FROM tracks"),
    ("What is the shortest dwell time?", "SELECT MIN(dwell_sec) FROM tracks"),
    ("How many engaged shoppers are there?", "SELECT COUNT(*) FROM tracks WHERE segment='engaged-shopper'"),
    ("How many passer-by are there?", "SELECT COUNT(*) FROM tracks WHERE segment='passer-by'"),
    ("How many browsers are there?", "SELECT COUNT(*) FROM tracks WHERE segment='browser'"),
    ("Which zone has the most people?", "SELECT zone FROM tracks GROUP BY zone ORDER BY COUNT(*) DESC LIMIT 1"),
    ("Which zone has the fewest people?", "SELECT zone FROM tracks GROUP BY zone ORDER BY COUNT(*) ASC LIMIT 1"),
    ("How many people were in zone r1c1?", "SELECT COUNT(*) FROM tracks WHERE zone='r1c1'"),
    ("What is the average dwell time in zone r1c1?", "SELECT AVG(dwell_sec) FROM tracks WHERE zone='r1c1'"),
    ("Which zone has the longest average dwell time?", "SELECT zone FROM tracks GROUP BY zone ORDER BY AVG(dwell_sec) DESC LIMIT 1"),
    ("What is the average speed of engaged shoppers?", "SELECT AVG(avg_speed) FROM tracks WHERE segment='engaged-shopper'"),
    ("How many people stayed longer than 30 seconds?", "SELECT COUNT(*) FROM tracks WHERE dwell_sec > 30"),
    ("How many people stayed less than 5 seconds?", "SELECT COUNT(*) FROM tracks WHERE dwell_sec < 5"),
    ("What is the maximum number of people counted at one time?", "SELECT MAX(people_count) FROM customer_traffic"),
    ("What is the average people count?", "SELECT AVG(people_count) FROM customer_traffic"),
    ("How many traffic records are there?", "SELECT COUNT(*) FROM customer_traffic"),
    ("When was the busiest moment?", "SELECT timestamp FROM customer_traffic ORDER BY people_count DESC, timestamp ASC LIMIT 1"),
    ("What is the total duration of all videos in seconds?", "SELECT SUM(duration_sec) FROM videos"),
    ("What is the frame rate of the first video?", "SELECT fps FROM videos ORDER BY id ASC LIMIT 1"),
    ("What is the average path length of passer-by?", "SELECT AVG(path_length) FROM tracks WHERE segment='passer-by'"),
    ("How many segments of each type are there?", "SELECT segment, COUNT(*) FROM tracks GROUP BY segment"),
    ("How many people are in each zone?", "SELECT zone, COUNT(*) FROM tracks GROUP BY zone"),
    ("What is the average dwell time per segment?", "SELECT segment, AVG(dwell_sec) FROM tracks GROUP BY segment"),
    ("How many people have a stationary ratio above 0.5?", "SELECT COUNT(*) FROM tracks WHERE stationary_ratio > 0.5"),
    ("What is the average stationary ratio?", "SELECT AVG(stationary_ratio) FROM tracks"),
    ("Which segment has the highest average dwell time?", "SELECT segment FROM tracks GROUP BY segment ORDER BY AVG(dwell_sec) DESC LIMIT 1"),
    ("How many people were first seen after 100 seconds?", "SELECT COUNT(*) FROM tracks WHERE first_sec > 100"),
    # added after a user report: the dashboard's own suggestion chips produced failing SQL
    ("How many unique customers were engaged shoppers vs browsers?",
     "SELECT segment, COUNT(*) FROM tracks WHERE segment IN ('engaged-shopper','browser') GROUP BY segment"),
    ("Which store zone has the highest customer traffic?",
     "SELECT zone FROM tracks GROUP BY zone ORDER BY COUNT(*) DESC LIMIT 1"),
    ("How many people were visible at the busiest time?", "SELECT MAX(people_count) FROM customer_traffic"),
    ("How many customers were seen in total?", "SELECT COUNT(*) FROM tracks"),
    ("How many people were passers-by versus browsers?",
     "SELECT segment, COUNT(*) FROM tracks WHERE segment IN ('passer-by','browser') GROUP BY segment"),
    ("How many people entered the store?", "SELECT COUNT(*) FROM crossings WHERE direction = 'in'"),
]


def norm(rows):
    out = []
    for r in rows:
        out.append(tuple(round(v, 2) if isinstance(v, float) else v for v in r))
    return sorted(out, key=str)


def run(sql):
    conn = db.connect()
    try:
        conn.execute("PRAGMA query_only = ON")
        return norm([tuple(r) for r in conn.execute(sql).fetchall()])
    finally:
        conn.close()


def project(rows, width):
    return sorted((r[:width] for r in rows), key=str)


passed, lenient, repaired, results = 0, 0, 0, []
for q, gold in QA:
    t = time.time()
    r = la.sql_with_repair(q)
    sql = r["sql"]
    try:
        want = run(gold)
        got = norm([tuple(row.values()) for row in r["rows"]]) if r["executed"] else None
        ok = got is not None and got == want
        okl = got is not None and bool(want) and project(got, len(want[0])) == want
        lenient += okl
        err = "" if r["executed"] else (r["error"] or "not executed")
    except Exception as e:
        ok, err = False, str(e)[:60]
    passed += ok
    repaired += bool(okl and r["attempts"] > 1)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {time.time() - t:4.1f}s  {q}" + ("" if ok else f"\n      got: {sql}  {err}\n      gold: {gold}"))
print(f"\nstrict:  {passed}/{len(QA)} ({100 * passed / len(QA):.0f}%)")
print(f"repaired by retry: {repaired}")
print(f"lenient: {lenient}/{len(QA)} ({100 * lenient / len(QA):.0f}%)  (extra columns allowed)")
