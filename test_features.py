"""D2 cameras, D3 stream sources, D6 privacy/retention and the DB migration."""
import datetime
import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import retailvision.db as db

_tmp = tempfile.mkdtemp()
db.DB_PATH = os.path.join(_tmp, "features.db")
os.environ["UPLOAD_DIR"] = os.path.join(_tmp, "uploads")
import main  # noqa: E402
from retailvision import privacy, tracker  # noqa: E402
from scripted_detector import EXPECTED, fresh_detector_factory, write_video  # noqa: E402


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_db, cls._old_uploads = db.DB_PATH, main.UPLOAD_DIR
        db.DB_PATH = os.path.join(tempfile.mkdtemp(), "f.db")
        main.UPLOAD_DIR = os.path.join(tempfile.mkdtemp(), "uploads")
        db.init_db()
        cls.client = TestClient(main.app)
        cls.video = os.path.join(_tmp, "scene.mp4")
        write_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        db.DB_PATH, main.UPLOAD_DIR = cls._old_db, cls._old_uploads

    def wait_job(self, job_id, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            j = self.client.get(f"/jobs/{job_id}").json()
            if j["status"] in ("done", "error"):
                return j
            time.sleep(0.2)
        self.fail("job timed out")

    def upload(self, **params):
        with mock.patch.object(tracker, "get_model", fresh_detector_factory()), open(self.video, "rb") as f:
            r = self.client.post("/analyze_video/", params=params,
                                 files={"file": ("scene.mp4", f, "video/mp4")})
        return r


class TestMigration(unittest.TestCase):
    def test_old_database_gains_camera_columns_without_losing_data(self):
        path = os.path.join(tempfile.mkdtemp(), "old.db")
        old = sqlite3.connect(path)
        old.executescript("""
            CREATE TABLE count_lines (name TEXT PRIMARY KEY, x1 REAL, y1 REAL, x2 REAL, y2 REAL, created TEXT);
            CREATE TABLE videos (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, fps REAL,
                                 duration_sec REAL, created TEXT);
            INSERT INTO count_lines VALUES ('door', 0, 0.5, 1, 0.5, '2026-01-01');
            INSERT INTO videos (filename, fps, duration_sec, created) VALUES ('a.mp4', 5, 3, '2026-01-01');
        """)
        old.commit()
        old.close()
        saved, db.DB_PATH = db.DB_PATH, path
        try:
            db.init_db()
            db.init_db()  # idempotent
            conn = db.connect()
            line = conn.execute("SELECT name, camera FROM count_lines").fetchone()
            vid = conn.execute("SELECT filename, camera FROM videos").fetchone()
            conn.close()
        finally:
            db.DB_PATH = saved
        self.assertEqual((line["name"], line["camera"]), ("door", "default"))
        self.assertEqual((vid["filename"], vid["camera"]), ("a.mp4", "default"))


class TestCameras(Base):
    def test_lines_are_per_camera_and_used_by_matching_analysis(self):
        self.client.put("/lines/", json={"name": "door-line", "camera": "door",
                                         "x1": 0.0, "y1": 0.3, "x2": 1.0, "y2": 0.3})
        listed = self.client.get("/lines/").json()["lines"]
        self.assertEqual([(l["name"], l["camera"]) for l in listed], [("door-line", "door")])

        r = self.upload(camera="door", use_clip=False)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["camera"], "door")
        self.assertEqual(list(body["crossings"]), ["door-line"])

        # a different camera does not see the door's lines: falls back to the default line
        other = self.upload(camera="till", use_clip=False).json()
        self.assertEqual(list(other["crossings"]), ["entrance"])
        self.assertEqual(other["crossings"]["entrance"], {"in": EXPECTED["in"], "out": EXPECTED["out"]})

        conn = db.connect()
        cams = [r["camera"] for r in conn.execute("SELECT camera FROM videos ORDER BY id")]
        conn.close()
        self.assertEqual(cams, ["door", "till"])

    def test_bad_camera_name_is_rejected(self):
        self.assertEqual(self.upload(camera="bad;name", use_clip=False).status_code, 422)
        r = self.client.put("/lines/", json={"name": "x", "camera": "../etc", "x1": 0, "y1": .5, "x2": 1, "y2": .5})
        self.assertEqual(r.status_code, 422)


class TestStreams(Base):
    def test_only_names_are_exposed_and_unknown_streams_rejected(self):
        secret = "rtsp://admin:hunter2@10.0.0.9/stream"
        with mock.patch.dict(os.environ, {"STREAM_SOURCES": f'{{"door": "{secret}"}}'}):
            r = self.client.get("/streams/")
            self.assertEqual(r.json()["streams"], ["door"])
            self.assertNotIn("hunter2", r.text)
            self.assertEqual(self.client.post("/analyze_stream/", params={"name": "nope"}).status_code, 404)
            self.assertEqual(self.client.post("/analyze_stream/", params={"name": "door", "seconds": 0}).status_code, 422)
            self.assertEqual(self.client.post("/analyze_stream/", params={"name": "door", "seconds": 9999}).status_code, 422)

    def test_stream_is_recorded_then_analysed(self):
        with mock.patch.dict(os.environ, {"STREAM_SOURCES": f'{{"entrance-cam": "{self.video}"}}'}), \
                mock.patch.object(tracker, "get_model", fresh_detector_factory()):
            r = self.client.post("/analyze_stream/", params={"name": "entrance-cam", "seconds": 10,
                                                             "camera": "front"})
            self.assertEqual(r.status_code, 200, r.text)
            job = self.wait_job(r.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        res = job["result"]
        self.assertEqual(res["unique_people"], EXPECTED["unique_people"])
        self.assertEqual(res["camera"], "front")
        self.assertEqual(res["crossings"]["entrance"], {"in": 2, "out": 1})

    def test_unreachable_stream_fails_cleanly(self):
        with mock.patch.dict(os.environ, {"STREAM_SOURCES": '{"dead": "/no/such/file.mp4"}'}):
            r = self.client.post("/analyze_stream/", params={"name": "dead", "seconds": 5})
            job = self.wait_job(r.json()["job_id"])
        self.assertEqual(job["status"], "error")
        self.assertIn("open stream", job["error"])


class TestPrivacy(Base):
    def _seed(self):
        conn = db.connect()
        now = datetime.datetime.now()
        old = now - datetime.timedelta(days=60)
        for label, when in (("old", old), ("new", now)):
            vid = conn.execute("INSERT INTO videos (filename, fps, duration_sec, created, camera) VALUES (?,?,?,?,?)",
                               (label, 5, 1, when.isoformat(), "default")).lastrowid
            conn.execute("INSERT INTO tracks (video_id, track_id, dwell_sec) VALUES (?,?,?)", (vid, 1, 2.0))
            conn.execute("INSERT INTO crossings (video_id, line_name, direction, sec) VALUES (?,?,?,?)",
                         (vid, "entrance", "in", 1.0))
            conn.execute("INSERT INTO customer_traffic (timestamp, people_count) VALUES (?,?)",
                         (when.strftime("%Y-%m-%d %H:%M:%S"), 3))
        conn.commit()
        conn.close()

    def _counts(self):
        conn = db.connect()
        out = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
               for t in ("videos", "tracks", "crossings", "customer_traffic")}
        conn.close()
        return out

    def test_purge_removes_only_data_older_than_cutoff(self):
        privacy.wipe_all()
        self._seed()
        self.assertEqual(self._counts(), {"videos": 2, "tracks": 2, "crossings": 2, "customer_traffic": 2})
        deleted = privacy.purge_older_than(30)
        self.assertEqual((deleted["videos"], deleted["tracks"], deleted["crossings"], deleted["customer_traffic"]),
                         (1, 1, 1, 1))
        self.assertEqual(self._counts(), {"videos": 1, "tracks": 1, "crossings": 1, "customer_traffic": 1})
        conn = db.connect()
        self.assertEqual(conn.execute("SELECT filename FROM videos").fetchone()[0], "new")
        conn.close()

    def test_wipe_needs_confirmation_and_keeps_line_settings(self):
        privacy.wipe_all()
        self.client.put("/lines/", json={"name": "keep-me", "x1": 0, "y1": .5, "x2": 1, "y2": .5})
        self._seed()
        self.assertEqual(self.client.delete("/privacy/data").status_code, 400)
        self.assertEqual(self._counts()["videos"], 2)
        r = self.client.delete("/privacy/data", params={"confirm": "yes"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sum(self._counts().values()), 0)
        self.assertIn("keep-me", [l["name"] for l in self.client.get("/lines/").json()["lines"]])

    def test_privacy_endpoints_and_auth(self):
        self.assertIn("does_not_store", self.client.get("/privacy/").json())
        self.assertEqual(self.client.post("/privacy/purge", params={"days": -1}).status_code, 422)
        with mock.patch.dict(os.environ, {"RETAILVISION_API_KEY": "k"}):
            c = TestClient(main.app)
            self.assertEqual(c.delete("/privacy/data", params={"confirm": "yes"}).status_code, 401)
            self.assertEqual(c.post("/privacy/purge", params={"days": 1}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
