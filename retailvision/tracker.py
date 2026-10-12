"""Video -> per-person tracks using YOLOv8 + ByteTrack."""
import math
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO

from .device import get_device

PERSON = 0
ZONE_GRID = 3  # frame is split into a 3x3 grid of named zones


def zone_name(cx, cy, w, h):
    col = min(int(cx / w * ZONE_GRID), ZONE_GRID - 1)
    row = min(int(cy / h * ZONE_GRID), ZONE_GRID - 1)
    return f"r{row}c{col}"


TRACKER_CFG = os.path.join(os.path.dirname(__file__), "bytetrack_retail.yaml")


def merge_fragments(tracks, max_gap=4.0, max_dist=0.12):
    """Stitch broken IDs: if track B starts shortly after track A ends (never overlapping),
    and close to where A ended, treat them as the same person. Greedy, nearest end wins.
    Returns (merged_tracks, parent) where parent maps every original id to its root id."""
    order = sorted(tracks, key=lambda i: tracks[i]["points"][0][0])
    parent, ends = {}, {}  # ends: root id -> (end_sec, x, y) of its latest point
    for i in order:
        s0, x0, y0 = tracks[i]["points"][0]
        best, best_d = None, max_dist
        for root, (e, ex, ey) in ends.items():
            d = math.hypot(x0 - ex, y0 - ey)
            if 0 < s0 - e <= max_gap and d < best_d:
                best, best_d = root, d
        root = i if best is None else best
        parent[i] = root
        ends[root] = tracks[i]["points"][-1]
    merged = {}
    for i in order:
        m = merged.setdefault(parent[i], {"points": [], "zones": defaultdict(int)})
        m["points"].extend(tracks[i]["points"])
        for z, c in tracks[i]["zones"].items():
            m["zones"][z] += c
    return merged, parent


_models = {}


def get_model(model_path="yolov8s.pt"):
    """Load each YOLO model once per process (loading dominates short videos)."""
    if model_path not in _models:
        _models[model_path] = YOLO(model_path)
    return _models[model_path]


def track_video(path, model_path="yolov8s.pt", stride=None, imgsz=1280, conf=0.15):
    """Returns (meta, tracks, crops).

    `stride` = process every Nth frame; default None picks ~5 processed frames per
    second of video (1 for low-fps CCTV). imgsz/conf defaults come from the Phase 1
    evaluation on the Mall dataset (MAE 4.1 people vs 13.2 for 640/0.25).

    tracks[id] = dict(points=[(sec,cx,cy)], zones=Counter-like dict)
    crops[id]  = best (largest) person crop, used for CLIP classification.
    """
    dev = get_device()
    model = get_model(model_path)
    if getattr(model, "predictor", None) is not None:  # cached model: forget the previous video
        for t in getattr(model.predictor, "trackers", []):
            t.reset()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if stride is None:
        stride = max(1, round(fps / 5))
    tracks = defaultdict(lambda: {"points": [], "zones": defaultdict(int)})
    crops, crop_area = {}, defaultdict(int)
    frame_idx, frames_seen, per_second = 0, 0, defaultdict(set)
    w = h = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride:
            continue
        frames_seen += 1
        h, w = frame.shape[:2]
        sec = frame_idx / fps
        res = model.track(frame, persist=True, classes=[PERSON],
                          tracker=TRACKER_CFG, imgsz=imgsz, conf=conf,
                          device=dev, verbose=False)[0]

        if res.boxes.id is None:
            continue
        for box, tid in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist()):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h  # normalised
            t = tracks[tid]
            t["points"].append((sec, cx, cy))
            t["zones"][zone_name(cx * w, cy * h, w, h)] += 1
            per_second[int(sec)].add(tid)
            area = (x2 - x1) * (y2 - y1)
            if area > crop_area[tid]:
                crop_area[tid] = area
                crops[tid] = frame[max(y1, 0):y2, max(x1, 0):x2].copy()
    cap.release()
    raw_ids = len(tracks)
    tracks, parent = merge_fragments(dict(tracks))
    best_crops = {}
    for i, c in crops.items():  # keep the largest crop among a merged person's fragments
        r = parent[i]
        if r not in best_crops or c.size > best_crops[r].size:
            best_crops[r] = c
    crops = best_crops
    meta = {"fps": fps, "duration_sec": frame_idx / fps,
            "stride": stride, "raw_ids": raw_ids,
            "per_second_counts": {s: len(ids) for s, ids in per_second.items()}}
    return meta, tracks, crops


def track_features(t):
    """Trajectory -> numeric feature vector for behaviour segmentation."""
    pts = t["points"]
    first, last = pts[0][0], pts[-1][0]
    path = speed_sum = still = 0.0
    for (s0, x0, y0), (s1, x1, y1) in zip(pts, pts[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        dt = max(s1 - s0, 1e-6)
        path += d
        speed_sum += d / dt
        still += d / dt < 0.01
    n = max(len(pts) - 1, 1)
    zone = max(t["zones"], key=t["zones"].get)
    return {"first_sec": first, "last_sec": last, "dwell_sec": last - first,
            "path_length": path, "avg_speed": speed_sum / n,
            "stationary_ratio": still / n, "zone": zone}
