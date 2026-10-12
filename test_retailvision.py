"""End-to-end regression and verification test suite for RetailVision."""
import unittest

from retailvision.db import connect, init_db
from retailvision.device import get_device
from retailvision.forecast import forecast


class TestRetailVision(unittest.TestCase):
    def test_01_device_detection(self):
        dev = get_device()
        self.assertIn(dev, ("mps", "cuda", "cpu"))
        print(f"\n[PASS] Verified hardware acceleration device: {dev}")

    def test_02_database_init(self):
        init_db()
        conn = connect()
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        self.assertIn("customer_traffic", tables)
        self.assertIn("videos", tables)
        self.assertIn("tracks", tables)
        print("[PASS] Verified database schema and tables.")

    def test_03_forecast_caching_and_output(self):
        res1 = forecast(steps=6)
        self.assertIn("forecast_next_hours", res1)
        self.assertEqual(len(res1["forecast_next_hours"]), 6)
        self.assertIn("future_labels", res1)
        self.assertEqual(len(res1["future_labels"]), 6)
        
        # Second call tests fast cached inference
        res2 = forecast(steps=6)
        self.assertEqual(res1["forecast_next_hours"], res2["forecast_next_hours"])
        print(f"[PASS] Verified forecast engine (Status: {res1.get('status')}).")

    def test_04_direct_fastapi_endpoints(self):
        from main import app, get_overview, get_forecast, customer_traffic, segments, ask
        
        # Test overview
        ov = get_overview()
        self.assertIn("total_tracks", ov)
        self.assertIn("video_count", ov)
        self.assertIn("zones", ov)

        # Test customer traffic
        ct = customer_traffic()
        self.assertIn("customer_traffic", ct)

        # Test segments
        segs = segments()
        self.assertIsInstance(segs, list)

        # Test ask forecast routing
        ask_res = ask("What is the footfall forecast for the next hour?")
        self.assertIn("answer", ask_res)
        self.assertIn("Footfall forecast", ask_res["answer"])
        print("[PASS] Verified FastAPI controller endpoints.")


if __name__ == "__main__":
    unittest.main()
