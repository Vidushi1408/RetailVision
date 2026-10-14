"""D4/D5: persistent jobs, auth, rate limits and the SQL guard."""
import os
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np
from fastapi.testclient import TestClient

import retailvision.db as db

_tmp = tempfile.mkdtemp()
db.DB_PATH = os.path.join(_tmp, "sec.db")
os.environ["UPLOAD_DIR"] = os.path.join(_tmp, "uploads")
import main  # noqa: E402
from retailvision import auth, jobs, sqlguard  # noqa: E402

client = TestClient(main.app)


def blank_video(path, frames=10):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 240))
    for _ in range(frames):
        w.write(np.full((240, 320, 3), 80, np.uint8))
    w.release()


def wait_for(job_id, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        j = jobs.get(job_id)
        if j and j["status"] in ("done", "error"):
            return j
        time.sleep(0.3)
    raise AssertionError("job did not finish")


class TestSqlGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        conn = db.connect()
        conn.executemany("INSERT INTO customer_traffic (timestamp, people_count) VALUES (?, ?)",
                         [(f"2026-01-01 00:00:{i % 60:02d}", i) for i in range(200)])
        conn.commit()
        conn.close()

    def test_plain_select_works_and_caps_rows(self):
        rows, truncated = sqlguard.execute_readonly("SELECT * FROM customer_traffic")
        self.assertEqual(len(rows), sqlguard.MAX_ROWS)
        self.assertTrue(truncated)
        rows, truncated = sqlguard.execute_readonly("SELECT COUNT(*) AS n FROM customer_traffic")
        self.assertGreaterEqual(rows[0]["n"], 200)  # other test modules may add rows
        self.assertFalse(truncated)

    def test_ctes_are_allowed_but_cannot_reach_other_tables(self):
        rows, _ = sqlguard.execute_readonly(
            "WITH t AS (SELECT people_count FROM customer_traffic) SELECT COUNT(*) AS n FROM t")
        self.assertGreaterEqual(rows[0]["n"], 200)
        with self.assertRaises(sqlguard.QueryRejected):
            sqlguard.execute_readonly("WITH t AS (SELECT * FROM jobs) SELECT * FROM t")

    def test_tables_outside_allow_list_are_rejected(self):
        for sql in ("SELECT * FROM jobs", "SELECT * FROM count_lines",
                    "SELECT name FROM sqlite_master", "SELECT * FROM pragma_table_info('tracks')"):
            with self.assertRaises(sqlguard.QueryRejected, msg=sql):
                sqlguard.execute_readonly(sql)

    def test_writes_and_pragmas_are_rejected(self):
        before, _ = sqlguard.execute_readonly("SELECT COUNT(*) AS n FROM customer_traffic")
        for sql in ("INSERT INTO customer_traffic VALUES (1, 'x', 1)", "DELETE FROM tracks",
                    "DROP TABLE tracks", "PRAGMA table_info(tracks)", "ATTACH DATABASE 'x.db' AS x"):
            with self.assertRaises(Exception, msg=sql):
                sqlguard.execute_readonly(sql)
        after, _ = sqlguard.execute_readonly("SELECT COUNT(*) AS n FROM customer_traffic")
        self.assertEqual(after[0]["n"], before[0]["n"])  # nothing was modified

    def test_runaway_query_is_interrupted(self):
        runaway = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
                   "SELECT COUNT(*) FROM c")
        start = time.monotonic()
        with self.assertRaises(sqlguard.QueryTimeout):
            sqlguard.execute_readonly(runaway, timeout=0.5)
        self.assertLess(time.monotonic() - start, 5)


class TestAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_open_when_no_key_configured(self):
        with mock.patch.dict(os.environ, {"RETAILVISION_API_KEY": ""}):
            self.assertEqual(client.get("/overview/").status_code, 200)
            self.assertFalse(client.get("/health").json()["auth_required"])

    def test_key_required_when_configured(self):
        with mock.patch.dict(os.environ, {"RETAILVISION_API_KEY": "s3cret"}):
            c = TestClient(main.app)
            self.assertEqual(c.get("/overview/").status_code, 401)
            self.assertEqual(c.get("/lines/").status_code, 401)
            self.assertEqual(c.put("/lines/", json={"name": "a", "x1": 0, "y1": .5, "x2": 1, "y2": .5}).status_code, 401)
            self.assertEqual(c.post("/analyze_video/", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code, 401)
            self.assertEqual(c.get("/ask/", params={"q": "forecast"}).status_code, 401)
            # public routes stay reachable so the page can ask for the key
            self.assertEqual(c.get("/").status_code, 200)
            self.assertEqual(c.get("/health").status_code, 200)
            self.assertEqual(c.get("/static/dashboard.js").status_code, 200)
            # header
            self.assertEqual(c.get("/overview/", headers={"X-API-Key": "s3cret"}).status_code, 200)
            self.assertEqual(c.get("/overview/", headers={"X-API-Key": "wrong"}).status_code, 401)

    def test_login_sets_cookie(self):
        with mock.patch.dict(os.environ, {"RETAILVISION_API_KEY": "s3cret"}):
            c = TestClient(main.app)
            self.assertEqual(c.post("/login", json={"key": "nope"}).status_code, 401)
            self.assertEqual(c.get("/overview/").status_code, 401)
            ok = c.post("/login", json={"key": "s3cret"})
            self.assertEqual(ok.status_code, 200)
            self.assertIn("httponly", ok.headers["set-cookie"].lower())
            self.assertIn("samesite=strict", ok.headers["set-cookie"].lower())
            self.assertEqual(c.get("/overview/").status_code, 200)

    def test_question_length_and_rate_limit(self):
        with mock.patch.dict(os.environ, {"RETAILVISION_API_KEY": ""}):
            self.assertEqual(client.get("/ask/", params={"q": "x" * 501}).status_code, 422)
            with mock.patch.object(main, "ask_limiter", auth.RateLimiter(2)):
                # "forecast" questions bypass the LLM, so this stays fast
                codes = [client.get("/ask/", params={"q": "forecast"}).status_code for _ in range(3)]
            self.assertEqual(codes, [200, 200, 429])

    def test_rate_limiter_window(self):
        rl = auth.RateLimiter(2, window=0.2)
        self.assertTrue(rl.allow("a") and rl.allow("a"))
        self.assertFalse(rl.allow("a"))
        self.assertTrue(rl.allow("b"))  # separate clients
        time.sleep(0.25)
        self.assertTrue(rl.allow("a"))


class TestPersistentJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        cls.upload_dir = os.environ["UPLOAD_DIR"]
        os.makedirs(cls.upload_dir, exist_ok=True)

    def _queue_row(self, status, upload_path, job_id):
        import json
        conn = db.connect()
        conn.execute("INSERT INTO jobs (id, kind, params, status, created, upload_path) VALUES (?,?,?,?,?,?)",
                     (job_id, "analyze", json.dumps({"path": upload_path, "filename": "blank.mp4", "use_clip": False}),
                      status, time.time(), upload_path))
        conn.commit()
        conn.close()

    def test_interrupted_and_queued_jobs_resume_after_restart(self):
        for status, jid in (("queued", "jobq00000001"), ("running", "jobr00000001")):
            path = os.path.join(self.upload_dir, f"{jid}.mp4")
            blank_video(path)
            self._queue_row(status, path, jid)
        self.assertGreaterEqual(jobs.resume(self.upload_dir), 2)  # what a restart does
        for jid in ("jobq00000001", "jobr00000001"):
            j = wait_for(jid)
            self.assertEqual(j["status"], "done", j)
            self.assertIn("unique_people", j["result"])
            self.assertFalse(os.path.exists(os.path.join(self.upload_dir, f"{jid}.mp4")))

    def test_job_with_lost_upload_fails_cleanly(self):
        self._queue_row("queued", os.path.join(self.upload_dir, "gone.mp4"), "jobgone00001")
        jobs.resume(self.upload_dir)
        j = jobs.get("jobgone00001")
        self.assertEqual(j["status"], "error")
        self.assertIn("lost", j["error"])

    def test_orphaned_uploads_are_removed_on_resume(self):
        stray = os.path.join(self.upload_dir, "stray.mp4")
        open(stray, "wb").write(b"junk")
        jobs.resume(self.upload_dir)
        self.assertFalse(os.path.exists(stray))

    def test_queue_limit_returns_429_and_cleans_up(self):
        vid = os.path.join(_tmp, "q.mp4")
        blank_video(vid)
        before = set(os.listdir(self.upload_dir))
        with mock.patch.object(jobs, "MAX_QUEUED", 0), open(vid, "rb") as f:
            r = client.post("/analyze_video/", params={"background": True, "use_clip": False},
                            files={"file": ("q.mp4", f, "video/mp4")})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(set(os.listdir(self.upload_dir)), before)  # rejected upload not kept

    def test_jobs_listing(self):
        r = client.get("/jobs/")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json()["jobs"], list)


if __name__ == "__main__":
    unittest.main()
