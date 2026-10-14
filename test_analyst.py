"""Analyst robustness: repair of failing SQL, deterministic fixes, honest `executed` flag.
The LLM is scripted, so these run without weights."""
import os
import tempfile
import unittest
from unittest import mock

import retailvision.db as db

db.DB_PATH = os.path.join(tempfile.mkdtemp(), "analyst.db")
from retailvision import llm_analyst as la  # noqa: E402


def scripted(*replies):
    it = iter(replies)
    return lambda *a, **k: next(it)


class TestAnalyst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.DB_PATH = os.path.join(tempfile.mkdtemp(), "analyst.db")
        db.init_db()
        conn = db.connect()
        conn.executemany("INSERT INTO tracks (video_id, track_id, dwell_sec, zone, role, segment) VALUES (1,?,?,?,?,?)",
                         [(i, 10.0 + i, "r0c0" if i < 3 else "r1c1", "unknown", "browser") for i in range(5)])
        conn.executemany("INSERT INTO crossings (video_id, line_name, direction, sec) VALUES (1,'door',?,1.0)",
                         [("in",), ("in",), ("out",)])
        conn.commit()
        conn.close()

    def test_failed_query_is_repaired_by_a_second_model_call(self):
        gen = scripted("SELECT COUNT(DISTINCT customer) FROM tracks", "SELECT COUNT(*) FROM tracks")
        with mock.patch.object(la, "_generate", gen):
            r = la.sql_with_repair("how many customers?")
        self.assertTrue(r["executed"])
        self.assertEqual(r["attempts"], 2)
        self.assertEqual(r["rows"], [{"COUNT(*)": 5}])

    def test_column_on_the_wrong_table_is_swapped_without_asking_the_model_again(self):
        calls = []
        def gen(*a, **k):
            calls.append(1)
            return "SELECT COUNT(*) FROM tracks WHERE direction = 'in'"
        with mock.patch.object(la, "_generate", gen):
            r = la.sql_with_repair("how many people entered?")
        self.assertTrue(r["executed"], r)
        self.assertEqual(r["rows"], [{"COUNT(*)": 2}])
        self.assertIn("FROM crossings", r["sql"])
        self.assertEqual(len(calls), 1)  # only the initial generation

    def test_role_filter_is_dropped_when_attributes_are_off(self):
        with mock.patch.dict(os.environ, {"ENABLE_CLIP_ATTRIBUTES": "0"}):
            self.assertEqual(la.drop_unusable_role_filter("SELECT COUNT(*) FROM tracks WHERE role = 'customer'"),
                             "SELECT COUNT(*) FROM tracks")
            self.assertEqual(la.drop_unusable_role_filter(
                "SELECT COUNT(*) FROM tracks WHERE role = 'customer' AND zone = 'r0c0'"),
                "SELECT COUNT(*) FROM tracks WHERE zone = 'r0c0'")
            self.assertEqual(la.drop_unusable_role_filter(
                "SELECT COUNT(*) FROM tracks WHERE zone = 'r0c0' AND role = 'employee'"),
                "SELECT COUNT(*) FROM tracks WHERE zone = 'r0c0'")
        with mock.patch.dict(os.environ, {"ENABLE_CLIP_ATTRIBUTES": "1"}):  # labels are real: keep the filter
            q = "SELECT COUNT(*) FROM tracks WHERE role = 'customer'"
            self.assertEqual(la.drop_unusable_role_filter(q), q)

    def test_role_filter_does_not_zero_out_the_answer(self):
        with mock.patch.dict(os.environ, {"ENABLE_CLIP_ATTRIBUTES": "0"}), \
                mock.patch.object(la, "_generate", scripted("SELECT COUNT(*) FROM tracks WHERE role = 'customer'")):
            r = la.sql_with_repair("how many customers?")
        self.assertEqual(r["rows"], [{"COUNT(*)": 5}])

    def test_unfixable_query_reports_not_executed(self):
        gen = scripted("SELECT nonsense FROM tracks", "SELECT nonsense FROM tracks")
        with mock.patch.object(la, "_generate", gen):
            r = la.sql_with_repair("?")
        self.assertFalse(r["executed"])
        self.assertIn("nonsense", r["error"])

    def test_ask_never_claims_execution_when_it_failed(self):
        gen = scripted("SELECT nonsense FROM tracks", "SELECT nonsense FROM tracks")
        with mock.patch.object(la, "_generate", gen):
            body = la.ask("anything")
        self.assertFalse(body["executed"])
        self.assertNotIn("rows", body)
        self.assertIn("could not answer", body["answer"])

    def test_policy_rejection_is_not_retried(self):
        calls = []
        def gen(*a, **k):
            calls.append(1)
            return "SELECT * FROM jobs"
        with mock.patch.object(la, "_generate", gen):
            r = la.sql_with_repair("show jobs")
        self.assertFalse(r["executed"])
        self.assertEqual(len(calls), 1)
        self.assertIn("may not read", r["error"])

    def test_successful_ask_reports_executed_with_rows(self):
        gen = scripted("SELECT zone FROM tracks GROUP BY zone ORDER BY COUNT(*) DESC LIMIT 1",
                       "The busiest zone is r1c1.")  # scripted text; rows below are the real data
        with mock.patch.object(la, "_generate", gen):
            body = la.ask("busiest zone?")
        self.assertTrue(body["executed"])
        self.assertEqual(body["rows"], [{"zone": "r0c0"}])  # 3 tracks in r0c0 vs 2 in r1c1
        self.assertEqual(body["answer"], "The busiest zone is r1c1.")  # text is whatever the model says


class TestModelLoading(unittest.TestCase):
    def setUp(self):
        self._saved = dict(la._llm)
        la._llm.clear()

    def tearDown(self):
        la._llm.clear()
        la._llm.update(self._saved)

    def test_concurrent_first_use_loads_once_and_never_sees_a_half_loaded_model(self):
        import threading, time
        calls, errors = [], []

        def slow_loader():
            calls.append(1)
            time.sleep(0.4)  # a question arrives while this is running
            return {"tok": object(), "model": object(), "dev": "cpu"}

        def use():
            try:
                la.load_llm()
                # what _generate_local does next; used to raise KeyError('model') mid-load
                _ = la._llm["tok"], la._llm["model"], la._llm["dev"]
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        with mock.patch.object(la, "_load_models", slow_loader):
            threads = [threading.Thread(target=use) for _ in range(6)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1, "model must be loaded exactly once")
        self.assertTrue(la.is_ready())

    def test_not_ready_before_load(self):
        self.assertFalse(la.is_ready())


if __name__ == "__main__":
    unittest.main()
