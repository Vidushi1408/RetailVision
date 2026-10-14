"""Configured camera sources (RTSP/HTTP URLs or file paths) that can be analysed by name.

STREAM_SOURCES is JSON like {"door": "rtsp://user:pw@192.168.1.20/stream1"}. Only names are ever
sent to the browser, never the URL (it may hold credentials), and callers cannot supply their
own URL, so the server cannot be pointed at arbitrary hosts or local files.

Analysis records up to N seconds from the source into a temporary clip and then runs the normal
pipeline on it: near-live rather than frame-by-frame real time.
"""
import json
import os
import tempfile
import time

import cv2

MAX_SECONDS = 300


def configured():
    raw = os.getenv("STREAM_SOURCES", "")
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def record(name, seconds):
    """Capture up to `seconds` of video from a configured source. Returns a temp .mp4 path."""
    sources = configured()
    if name not in sources:
        raise KeyError(f"Unknown stream '{name}'.")
    seconds = max(1, min(int(seconds), MAX_SECONDS))
    cap = cv2.VideoCapture(sources[name])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open stream '{name}'.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 120 else 5.0
    max_frames = int(seconds * fps)
    deadline = time.monotonic() + seconds + 15
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer, frames = None, 0
    try:
        while frames < max_frames and time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        os.unlink(path)
        raise RuntimeError(f"No frames received from stream '{name}'.")
    return path
