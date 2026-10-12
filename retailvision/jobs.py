"""Tiny in-process job queue so long video analyses don't block the HTTP request."""
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

_pool = ThreadPoolExecutor(max_workers=1)  # one video at a time: the GPU/CPU is the bottleneck
_jobs, _lock = {}, threading.Lock()
MAX_JOBS_KEPT = 50


def submit(fn, *args, cleanup_path=None, **kwargs):
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"id": job_id, "status": "queued", "created": time.time(),
                         "result": None, "error": None}
        for old in sorted(_jobs, key=lambda k: _jobs[k]["created"])[:-MAX_JOBS_KEPT]:
            _jobs.pop(old, None)

    def run():
        _jobs[job_id]["status"] = "running"
        try:
            _jobs[job_id]["result"] = fn(*args, **kwargs)
            _jobs[job_id]["status"] = "done"
        except Exception as e:  # surfaced to the client, not swallowed
            _jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
            _jobs[job_id]["status"] = "error"
        finally:
            if cleanup_path and os.path.exists(cleanup_path):
                os.unlink(cleanup_path)

    _pool.submit(run)
    return job_id


def get(job_id):
    return _jobs.get(job_id)
