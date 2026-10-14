"""Rolling camera monitor: analyse a configured stream in back-to-back windows until stopped.

Each window is recorded for `window_sec` and analysed through the normal job queue (so it is
persistent, serialised with uploads and visible in /jobs/). Results land in the same tables as
uploads, tagged with the camera, so hourly occupancy accumulates for the forecaster.
This is near-continuous (gaps while a window is analysed), not frame-by-frame real time.
"""
import os
import subprocess
import threading
import time

from . import jobs, streams

MAX_CONSECUTIVE_ERRORS = 5
ERROR_BACKOFF_SEC = 5.0
_monitors = {}
_lock = threading.Lock()


def rss_mb():
    """Resident memory of this process in MB (ps works on macOS and Linux)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
        return round(int(out.stdout.strip()) / 1024, 1)
    except Exception:  # noqa: BLE001
        return None


class _Monitor:
    def __init__(self, name, window_sec, camera):
        self.name, self.window_sec, self.camera = name, window_sec, camera
        self.stop_event = threading.Event()
        self.state = {"running": True, "windows_done": 0, "errors": 0, "consecutive_errors": 0,
                      "last_result": None, "last_error": None, "started": time.time(),
                      "rss_mb_start": rss_mb(), "rss_mb": rss_mb(), "rss_mb_peak": rss_mb()}
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"monitor-{name}")

    def _run(self):
        s = self.state
        while not self.stop_event.is_set():
            try:
                job_id = jobs.submit("stream", {"name": self.name, "seconds": self.window_sec, "camera": self.camera})
            except jobs.QueueFull:
                self.stop_event.wait(ERROR_BACKOFF_SEC)
                continue
            job = None
            while not self.stop_event.is_set():
                job = jobs.get(job_id)
                if job and job["status"] in ("done", "error"):
                    break
                self.stop_event.wait(0.5)
            if job is None or job["status"] not in ("done", "error"):
                break
            if job["status"] == "done":
                s["windows_done"] += 1
                s["consecutive_errors"] = 0
                r = job["result"]
                s["last_result"] = {k: r.get(k) for k in ("unique_people", "occupancy", "crossings", "duration_sec")}
            else:
                s["errors"] += 1
                s["consecutive_errors"] += 1
                s["last_error"] = job["error"]
                if s["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
                    break
                self.stop_event.wait(ERROR_BACKOFF_SEC)
            cur = rss_mb()
            s["rss_mb"] = cur
            if cur is not None and (s["rss_mb_peak"] is None or cur > s["rss_mb_peak"]):
                s["rss_mb_peak"] = cur
        s["running"] = False


def start(name, window_sec=30, camera="default"):
    if name not in streams.configured():
        raise KeyError(name)
    window_sec = max(5, min(int(window_sec), streams.MAX_SECONDS))
    with _lock:
        cur = _monitors.get(name)
        if cur and cur.state["running"]:
            raise RuntimeError(f"'{name}' is already being monitored.")
        m = _Monitor(name, window_sec, camera)
        _monitors[name] = m
        m.thread.start()
    return m.state


def stop(name):
    with _lock:
        m = _monitors.get(name)
    if m is None:
        raise KeyError(name)
    m.stop_event.set()
    m.thread.join(timeout=5)
    return m.state


def status():
    with _lock:
        items = list(_monitors.items())
    return {n: {**m.state, "window_sec": m.window_sec, "camera": m.camera,
                "uptime_sec": round(time.time() - m.state["started"], 1)} for n, m in items}
