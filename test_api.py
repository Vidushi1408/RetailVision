"""API behaviour: upload validation, size limit, background jobs, tracker state reset."""
import os
import tempfile
import time
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

import retailvision.db as db

_tmp = tempfile.mkdtemp()
db.DB_PATH = os.path.join(_tmp, "test.db")  # must be set before main touches the DB
import main  # noqa: E402

client = TestClient(main.app)


def make_video(path, frames=20, fps=5):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for _ in range(frames):
        w.write(np.full((240, 320, 3), 90, np.uint8))
    w.release()


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()
        cls.video = os.path.join(_tmp, "blank.mp4")
        make_video(cls.video)

    def _post(self, **params):
        with open(self.video, "rb") as f:
            return client.post("/analyze_video/", params=params,
                               files={"file": ("blank.mp4", f, "video/mp4")})

    def test_rejects_unsupported_extension(self):
        r = client.post("/analyze_video/", files={"file": ("notes.txt", b"hello", "text/plain")})
        self.assertEqual(r.status_code, 400)

    def test_rejects_oversized_upload(self):
        old, main.MAX_UPLOAD_MB = main.MAX_UPLOAD_MB, 0  # 0 MB limit -> anything is too big
        try:
            self.assertEqual(self._post().status_code, 413)
        finally:
            main.MAX_UPLOAD_MB = old

    def test_sync_analysis_returns_summary(self):
        r = self._post(use_clip=False)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["unique_people"], 0)  # blank video: nobody there

    def test_background_job_completes(self):
        r = self._post(use_clip=False, background=True)
        self.assertEqual(r.status_code, 200)
        job_id = r.json()["job_id"]
        for _ in range(120):
            j = client.get(f"/jobs/{job_id}").json()
            if j["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        self.assertEqual(j["status"], "done", j)
        self.assertIn("unique_people", j["result"])

    def test_unknown_job_is_404(self):
        self.assertEqual(client.get("/jobs/doesnotexist").status_code, 404)

    def test_tracker_state_resets_between_videos(self):
        from retailvision.tracker import track_video
        clip = "data/mall_clip.mp4"
        if not os.path.exists(clip):
            self.skipTest("mall clip not available")
        short = os.path.join(_tmp, "short.mp4")
        cap, vw = cv2.VideoCapture(clip), None
        for _ in range(30):
            ok, fr = cap.read()
            if vw is None:
                vw = cv2.VideoWriter(short, cv2.VideoWriter_fourcc(*"mp4v"), 2.0, fr.shape[1::-1])
            vw.write(fr)
        vw.release()
        a = track_video(short)[0]["raw_ids"]
        b = track_video(short)[0]["raw_ids"]
        self.assertEqual(a, b, "cached model leaked tracker state into the second run")


if __name__ == "__main__":
    unittest.main()
