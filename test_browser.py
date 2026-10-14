"""E3: real-browser test of the dashboard (Chromium via Playwright).

The server runs in-process on a free port with the scripted detector, so results are known:
5 people, 2 in / 1 out, occupancy 5. Skipped automatically if Playwright or Chromium is missing.
Install: pip install playwright && python -m playwright install chromium
"""
import os
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

import requests

import retailvision.db as db

_tmp = tempfile.mkdtemp()
db.DB_PATH = os.path.join(_tmp, "browser.db")
os.environ["UPLOAD_DIR"] = os.path.join(_tmp, "uploads")
os.environ["PRELOAD_MODELS"] = "none"
import main  # noqa: E402
from retailvision import tracker  # noqa: E402
from scripted_detector import EXPECTED, fresh_detector_factory, write_video  # noqa: E402

try:
    from playwright.sync_api import sync_playwright, expect
    HAVE_PLAYWRIGHT = True
except ImportError:  # pragma: no cover
    HAVE_PLAYWRIGHT = False


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_PLAYWRIGHT, "playwright is not installed")
class TestDashboardInBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uvicorn
        cls._old_db, cls._old_up = db.DB_PATH, main.UPLOAD_DIR
        db.DB_PATH = os.path.join(tempfile.mkdtemp(), "b.db")
        main.UPLOAD_DIR = os.path.join(tempfile.mkdtemp(), "uploads")
        db.init_db()
        cls.patch = mock.patch.object(tracker, "get_model", fresh_detector_factory())
        cls.patch.start()
        cls.video = os.path.join(_tmp, "scene.mp4")
        write_video(cls.video)

        cls.port = free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=cls.port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            try:
                if requests.get(cls.base + "/health", timeout=0.5).ok:
                    break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not start")
        cls.pw = sync_playwright().start()
        try:
            cls.browser = cls.pw.chromium.launch()
        except Exception as e:  # chromium not installed
            cls.pw.stop()
            cls.server.should_exit = True
            cls.patch.stop()
            raise unittest.SkipTest(f"Chromium unavailable: {e}")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls.patch.stop()
        db.DB_PATH, main.UPLOAD_DIR = cls._old_db, cls._old_up

    def setUp(self):
        self.page = self.browser.new_page()
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))

    def tearDown(self):
        self.page.close()

    # ------------------------------------------------------------------
    def test_1_page_loads_without_script_errors(self):
        self.page.goto(self.base)
        expect(self.page.locator("#val-total-shoppers")).to_be_visible()
        expect(self.page.locator("#lines-card")).to_be_visible()
        expect(self.page.locator("#privacy-card")).to_be_visible()
        self.assertEqual(self.errors, [])

    def test_2_upload_shows_known_results(self):
        self.page.goto(self.base)
        expect(self.page.locator("#toggle-use-clip")).not_to_be_checked()  # CLIP off by default
        self.page.set_input_files("#video-file-input", self.video)
        self.page.click("#start-analysis-btn")
        expect(self.page.locator("#progress-status-text")).to_have_text("Complete!", timeout=60000)
        expect(self.page.locator("#summary-unique-people")).to_have_text(str(EXPECTED["unique_people"]))
        expect(self.page.locator("#summary-crossings")).to_have_text(f"{EXPECTED['in']} / {EXPECTED['out']}")
        expect(self.page.locator("#summary-occupancy")).to_have_text("5 / 5")
        expect(self.page.locator("#val-crossings")).to_have_text(f"{EXPECTED['in']} / {EXPECTED['out']}")
        self.assertEqual(self.errors, [])

    def test_3_draw_and_save_a_counting_line(self):
        self.page.goto(self.base)
        self.page.set_input_files("#lines-frame-input", self.video)  # mp4v: needs the server fallback
        expect(self.page.locator("#lines-status")).to_contain_text("Frame loaded", timeout=20000)
        canvas = self.page.locator("#lines-canvas")
        canvas.scroll_into_view_if_needed()
        box = canvas.bounding_box()
        # position= is relative to the element, and Playwright scrolls it into view first
        canvas.click(position={"x": box["width"] * 0.1, "y": box["height"] * 0.4})
        canvas.click(position={"x": box["width"] * 0.9, "y": box["height"] * 0.4})
        expect(self.page.locator("#lines-status")).to_contain_text("Line drawn")
        self.page.fill("#line-camera", "e3cam")
        self.page.fill("#line-name", "browser-line")
        self.page.click("#line-save")
        expect(self.page.locator("#saved-lines")).to_contain_text("browser-line")
        expect(self.page.locator("#saved-lines")).to_contain_text("e3cam")
        saved = requests.get(self.base + "/lines/").json()["lines"]
        mine = [l for l in saved if l["name"] == "browser-line"][0]
        self.assertEqual(mine["camera"], "e3cam")
        self.assertAlmostEqual(mine["y1"], 0.4, delta=0.03)
        self.assertAlmostEqual(mine["x1"], 0.1, delta=0.03)

    def test_4_privacy_wipe_asks_for_confirmation_then_deletes(self):
        requests.post(self.base + "/analyze_video/", params={"use_clip": False, "camera": "default"},
                      files={"file": ("scene.mp4", open(self.video, "rb"), "video/mp4")}).raise_for_status()
        self.assertGreater(requests.get(self.base + "/overview/").json()["total_tracks"], 0)
        self.page.goto(self.base)
        dialogs = []
        self.page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
        self.page.click("#wipe-run")
        expect(self.page.locator("#privacy-status")).to_contain_text("All analytics deleted")
        self.assertEqual(len(dialogs), 1)
        self.assertIn("cannot be undone", dialogs[0])
        self.assertEqual(requests.get(self.base + "/overview/").json()["total_tracks"], 0)

    def test_6_analyst_labels_executed_and_failed_queries_honestly(self):
        from retailvision import llm_analyst
        self.page.goto(self.base)
        box, send = self.page.locator("#analyst-query-input"), self.page.locator("#analyst-submit-btn")
        chat = self.page.locator("#analyst-chat-display")

        replies = iter(["SELECT COUNT(*) AS n FROM tracks", "There are some tracks."])
        with mock.patch.object(llm_analyst, "_generate", lambda *a, **k: next(replies)):
            box.fill("how many people?")
            send.click()
            expect(chat).to_contain_text("Executed SQL", timeout=15000)
        expect(chat).to_contain_text("Result (1 row)")
        expect(chat).not_to_contain_text("Not executed")

        replies = iter(["SELECT nonsense FROM tracks", "SELECT nonsense FROM tracks"])
        with mock.patch.object(llm_analyst, "_generate", lambda *a, **k: next(replies)):
            box.fill("something unanswerable")
            send.click()
            expect(chat).to_contain_text("Not executed", timeout=15000)
        # the failed query must never be presented as executed
        last = chat.locator(".sql-failed").last
        expect(last).to_contain_text("SELECT nonsense FROM tracks")

    def test_5_dashboard_asks_for_key_once_when_server_requires_one(self):
        with mock.patch.dict(os.environ, {"RETAILVISION_API_KEY": "browser-key"}):
            prompts = []
            self.page.on("dialog", lambda d: (prompts.append(d.type), d.accept("browser-key")))
            self.page.goto(self.base)
            expect(self.page.locator("#privacy-summary")).to_contain_text("Stored:", timeout=15000)
            self.assertEqual(prompts, ["prompt"])  # one prompt even though many calls got a 401
            # and the page's own API calls work afterwards (cookie set)
            status = self.page.evaluate("fetch('/overview/').then(r => r.status)")
            self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
