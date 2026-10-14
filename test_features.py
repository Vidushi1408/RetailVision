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
from retailvision import monitor, privacy, tracker  # noqa: E402
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

        # a different camera does not see the door's lines, and there is no arbitrary default:
        # it reports no crossings but still reports people and occupancy
        other = self.upload(camera="till", use_clip=False).json()
        self.assertEqual(other["crossings"], {})
        self.assertEqual(other["unique_people"], EXPECTED["unique_people"])
        self.assertEqual(other["occupancy"]["peak"], EXPECTED["occupancy_peak"])

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
        self.client.put("/lines/", json={"name": "entrance", "camera": "front",
                                         "x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5})
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


class TestMonitor(Base):
    def _wait(self, pred, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            time.sleep(0.2)
        return False

    def test_rolling_windows_accumulate_then_stop(self):
        env = {"STREAM_SOURCES": f'{{"loop": "{self.video}"}}'}
        with mock.patch.dict(os.environ, env), mock.patch.object(tracker, "get_model", fresh_detector_factory()):
            r = self.client.post("/monitor/start", params={"name": "loop", "window": 5, "camera": "lobby"})
            self.assertEqual(r.status_code, 200, r.text)
            # second start while running is refused
            self.assertEqual(self.client.post("/monitor/start", params={"name": "loop"}).status_code, 409)
            ok = self._wait(lambda: self.client.get("/monitor/").json()["monitors"]["loop"]["windows_done"] >= 2)
            self.assertTrue(ok, self.client.get("/monitor/").text)
            stopped = self.client.post("/monitor/stop", params={"name": "loop"}).json()
        self.assertFalse(stopped["running"])
        self.assertGreaterEqual(stopped["windows_done"], 2)
        self.assertEqual(stopped["last_result"]["unique_people"], EXPECTED["unique_people"])
        conn = db.connect()
        rows = conn.execute("SELECT COUNT(*) FROM videos WHERE camera = 'lobby' AND filename = 'stream:loop'").fetchone()[0]
        conn.close()
        self.assertGreaterEqual(rows, 2)
        self.assertIsNotNone(self.client.get("/monitor/").json()["server_rss_mb"])

    def test_unknown_stream_and_unknown_stop(self):
        with mock.patch.dict(os.environ, {"STREAM_SOURCES": ""}):
            self.assertEqual(self.client.post("/monitor/start", params={"name": "nope"}).status_code, 404)
        self.assertEqual(self.client.post("/monitor/stop", params={"name": "never-started"}).status_code, 404)

    def test_dead_source_gives_up_after_repeated_errors(self):
        env = {"STREAM_SOURCES": '{"dead": "/no/such/camera.mp4"}'}
        with mock.patch.dict(os.environ, env), mock.patch.object(monitor, "MAX_CONSECUTIVE_ERRORS", 2), \
                mock.patch.object(monitor, "ERROR_BACKOFF_SEC", 0.05):
            self.assertEqual(self.client.post("/monitor/start", params={"name": "dead", "window": 5}).status_code, 200)
            ok = self._wait(lambda: not self.client.get("/monitor/").json()["monitors"]["dead"]["running"])
            st = self.client.get("/monitor/").json()["monitors"]["dead"]
        self.assertTrue(ok)
        self.assertGreaterEqual(st["errors"], 2)
        self.assertEqual(st["windows_done"], 0)
        self.assertIn("open stream", st["last_error"])


class TestForecastMethods(Base):
    """SYNTHETIC data only (clearly fabricated for testing the selection logic, not evidence
    that any real store benefits)."""

    def _seed(self, hours, fn, skip=lambda h: False):
        conn = db.connect()
        conn.execute("DELETE FROM customer_traffic")
        start = datetime.datetime(2026, 1, 1)
        for h in range(hours):
            if skip(h):
                continue
            ts = (start + datetime.timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("INSERT INTO customer_traffic (timestamp, people_count) VALUES (?,?)", (ts, fn(h)))
        conn.commit()
        conn.close()

    def test_seasonal_wins_on_a_strict_daily_cycle(self):
        import math
        from retailvision.forecast import forecast
        # exact 24 h cycle with a sharp shape the last value cannot track, no noise
        self._seed(24 * 6, lambda h: round(30 + 25 * math.sin((h % 24) / 24 * 2 * math.pi)))
        r = forecast(steps=6)
        self.assertIn("seasonal", r["backtest_mae"])
        self.assertEqual(min(r["backtest_mae"], key=r["backtest_mae"].get) in ("seasonal", "lstm"), True)
        self.assertLessEqual(r["backtest_mae"]["seasonal"], r["backtest_mae"]["last_value"])
        self.assertEqual(len(r["forecast_next_hours"]), 6)

    def test_gaps_do_not_break_the_seasonal_lookup(self):
        from retailvision.forecast import forecast
        self._seed(24 * 5, lambda h: 10 + (h % 24), skip=lambda h: h % 7 == 0)  # a missing hour every 7
        r = forecast(steps=4)
        self.assertEqual(len(r["forecast_next_hours"]), 4)
        self.assertTrue(all(v >= 0 for v in r["forecast_next_hours"]))

    def test_too_little_data_has_no_seasonal_option(self):
        from retailvision.forecast import forecast
        self._seed(14, lambda h: 5 + h % 3)
        r = forecast(steps=3)
        self.assertNotIn("seasonal", r.get("backtest_mae", {}))


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
