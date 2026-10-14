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

    def _with_hourly_points(self, n):
        """Run forecast() against a throwaway database holding n hourly occupancy points."""
        import datetime
        import os
        import tempfile
        from unittest import mock
        import retailvision.db as db
        path = os.path.join(tempfile.mkdtemp(), "fc.db")
        with mock.patch.object(db, "DB_PATH", path):
            db.init_db()
            conn = db.connect()
            start = datetime.datetime(2026, 1, 1)
            for h in range(n):
                ts = (start + datetime.timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("INSERT INTO customer_traffic (timestamp, people_count) VALUES (?, ?)", (ts, 10 + h % 4))
            conn.commit()
            conn.close()
            return forecast(steps=6)

    def test_03_forecast_has_no_fabricated_numbers(self):
        # no data at all: no forecast, not an invented default
        res = self._with_hourly_points(0)
        self.assertEqual(res["status"], "no_data")
        self.assertEqual(res["forecast_next_hours"], [])
        self.assertEqual(res["future_labels"], [])

        # too little history for the LSTM: the flat average observed so far, no made-up wiggle
        res = self._with_hourly_points(5)
        self.assertEqual(res["status"], "baseline_extrapolation")
        vals = res["forecast_next_hours"]
        self.assertEqual(len(vals), 6)
        self.assertEqual(len(set(vals)), 1, "baseline must be flat")
        self.assertAlmostEqual(vals[0], round(sum(10 + h % 4 for h in range(5)) / 5, 1))
        self.assertEqual(len(res["future_labels"]), 6)

        # enough history: a real method is chosen and its backtest errors are reported
        res = self._with_hourly_points(40)
        self.assertIn(res["status"], ("lstm_trained", "last_value_baseline", "seasonal_baseline"))
        self.assertEqual(len(res["forecast_next_hours"]), 6)
        self.assertIn("backtest_mae", res)
        print(f"[PASS] Verified forecast engine (Status: {res['status']}).")

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
