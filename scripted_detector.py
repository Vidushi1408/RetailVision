"""Test helper: a scripted stand-in for the YOLO detector plus a matching blank video.

Scene (320x240, 14 frames at 5 fps): five people - two walk DOWN across the middle line
(2 "in"), one walks UP (1 "out"), one stands still far from the line, one hovers ON the line
with +-0.012 jitter (must not be counted). Used by test_e2e, test_features and test_browser.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes

W, H, FRAMES, FPS = 320, 240, 14, 5
EXPECTED = {"unique_people": 5, "in": 2, "out": 1, "occupancy_mean": 5.0, "occupancy_peak": 5}


def person_box(cx, cy, w=0.125, h=0.34):
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H, 0.9, 0]


def scripted_frame(i):
    t = i / (FRAMES - 1)
    return [person_box(0.15, 0.15 + 0.70 * t),            # down  -> in
            person_box(0.35, 0.15 + 0.70 * t),            # down  -> in
            person_box(0.55, 0.25),                        # still
            person_box(0.85, 0.85 - 0.70 * t),            # up    -> out
            person_box(0.68, 0.5 + (0.012 if i % 2 else -0.012))]  # jitter on the line


class ScriptedDetector:
    """Returns the next scripted frame on every predict() call (one video per instance)."""

    def __init__(self):
        self.calls = 0

    def predict(self, frame, **kwargs):
        boxes = Boxes(torch.tensor(scripted_frame(self.calls), dtype=torch.float32), (H, W))
        self.calls += 1
        return [SimpleNamespace(boxes=boxes)]


def write_video(path):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(FRAMES):
        vw.write(np.full((H, W, 3), 100, np.uint8))
    vw.release()


def fresh_detector_factory():
    """A get_model replacement that hands every video its own scripted detector."""
    return lambda *a, **k: ScriptedDetector()
