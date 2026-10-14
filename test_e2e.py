"""End-to-end: upload -> background job -> tracking -> crossings -> DB -> API -> analyst answer.

Only the two heavy models are replaced, so the test needs no weights or downloads and is
deterministic:
  * the YOLO detector is scripted: five people on a 320x240 frame - two walk DOWN across the
    middle line (-> "in"), one walks UP (-> "out"), one stands still far from the line, and
    one hovers ON the line with detection jitter (must NOT be counted). The asymmetry
    (2 in, 1 out) means an inverted direction convention cannot go unnoticed;
  * the local LLM is scripted to return one SQL query and one sentence.
Everything else (BYTETracker association, crossing logic, merge, KMeans, SQLite, the job
queue, FastAPI routes, the SQL guard and answer assembly) is the real code.
"""
import os
import tempfile
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import retailvision.db as db

_tmp = tempfile.mkdtemp()
db.DB_PATH = os.path.join(_tmp, "e2e.db")
os.environ["UPLOAD_DIR"] = os.path.join(_tmp, "uploads")
import main  # noqa: E402
from retailvision import llm_analyst, tracker  # noqa: E402

from scripted_detector import FRAMES, FPS, ScriptedDetector, write_video  # noqa: E402


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Own database and upload dir: exact counts below must not depend on what other test
        # modules (which share this process) have written.
        cls._old_db, cls._old_uploads = db.DB_PATH, main.UPLOAD_DIR
        db.DB_PATH = os.path.join(tempfile.mkdtemp(), "e2e_isolated.db")
        main.UPLOAD_DIR = os.path.join(tempfile.mkdtemp(), "uploads")
        db.init_db()
        cls.client = TestClient(main.app)
        cls.video = os.path.join(_tmp, "scripted.mp4")
        write_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        db.DB_PATH, main.UPLOAD_DIR = cls._old_db, cls._old_uploads

    def _upload_and_wait(self):
        detector = ScriptedDetector()
        with mock.patch.object(tracker, "get_model", lambda *a, **k: detector), open(self.video, "rb") as f:
            r = self.client.post("/analyze_video/", params={"background": True, "use_clip": False},
                                 files={"file": ("scripted.mp4", f, "video/mp4")})
            self.assertEqual(r.status_code, 200, r.text)
            job_id = r.json()["job_id"]
            for _ in range(240):  # up to ~60 s
                job = self.client.get(f"/jobs/{job_id}").json()
                if job["status"] in ("done", "error"):
                    break
                time.sleep(0.25)
        self.assertEqual(job["status"], "done", job)
        return job["result"]

    def test_full_pipeline(self):
        summary = self._upload_and_wait()

        # 1. summary produced by the real pipeline
        self.assertEqual(summary["unique_people"], 5)
        self.assertEqual(summary["crossings"]["entrance"], {"in": 2, "out": 1})
        self.assertEqual(summary["occupancy"], {"mean": 5.0, "peak": 5})
        self.assertAlmostEqual(summary["duration_sec"], FRAMES / FPS, delta=0.3)
        self.assertEqual(sum(summary["segments"].values()), 5)

        # 2. persisted and served by the API
        vid = summary["video_id"]
        self.assertEqual(self.client.get(f"/crossings/?video_id={vid}").json(),
                         {str(vid): {"entrance": {"in": 2, "out": 1}}})
        overview = self.client.get("/overview/").json()
        self.assertEqual(overview["total_tracks"], 5)
        self.assertEqual(overview["crossings_in"], 2)
        self.assertEqual(overview["crossings_out"], 1)
        self.assertEqual(overview["peak_occupancy"], 5)
        self.assertEqual(sum(self.client.get("/segments/").json()[i]["n"]
                             for i in range(len(self.client.get("/segments/").json()))), 5)

        # 3. the upload was cleaned up after the job
        self.assertEqual(os.listdir(main.UPLOAD_DIR), [])

        # 4. analyst: scripted LLM -> real SQL guard -> real DB -> assembled answer
        replies = iter(["```sql\nSELECT COUNT(*) AS n FROM crossings WHERE direction = 'in'\n```",
                        "Two people crossed the line going in."])
        with mock.patch.object(llm_analyst, "_generate", lambda *a, **k: next(replies)):
            r = self.client.get("/ask/", params={"q": "How many people went in?"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["rows"], [{"n": 2}])
        self.assertEqual(body["answer"], "Two people crossed the line going in.")
        self.assertIn("crossings", body["sql"])

    def test_analyst_cannot_be_steered_to_other_tables(self):
        replies = iter(["SELECT upload_path FROM jobs"])
        with mock.patch.object(llm_analyst, "_generate", lambda *a, **k: next(replies)):
            body = self.client.get("/ask/", params={"q": "show the jobs table"}).json()
        self.assertIn("may not read", body["answer"])
        self.assertNotIn("rows", body)


if __name__ == "__main__":
    unittest.main()
