"""Video -> per-person tracks using YOLOv8 + ByteTrack."""
import math
import os
from collections import defaultdict

import cv2
import numpy as np
import yaml
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from .device import get_device

PERSON = 0
ZONE_GRID = 3  # frame is split into a 3x3 grid of named zones


def zone_name(cx, cy, w, h):
    col = min(int(cx / w * ZONE_GRID), ZONE_GRID - 1)
    row = min(int(cy / h * ZONE_GRID), ZONE_GRID - 1)
    return f"r{row}c{col}"


TRACKER_CFG = os.path.join(os.path.dirname(__file__), "bytetrack_retail.yaml")


def appearance_feature(crop_bgr, bins=(8, 8, 4)):
    """Colour signature of the torso and legs (Hellinger-normalised HSV histograms).

    Phase 1 check on hand-identified fragments of one person: AUC 0.86 vs 0.46 for CLIP image
    embeddings, which mostly encode the background. Weak, so only used to gate nearby candidates.
    """
    h, w = crop_bgr.shape[:2]
    body = crop_bgr[int(.15 * h):int(.9 * h), int(.2 * w):int(.8 * w)]
    if body.size == 0:
        return None
    hsv, parts = cv2.cvtColor(body, cv2.COLOR_BGR2HSV), []
    for p in (hsv[:hsv.shape[0] // 2], hsv[hsv.shape[0] // 2:]):
        hist = cv2.calcHist([p], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).ravel()
        parts.append(hist / max(hist.sum(), 1e-9))
    v = np.sqrt(np.concatenate(parts))
    return v / max(np.linalg.norm(v), 1e-9)


def merge_fragments(tracks, max_gap=4.0, max_dist=0.12, feats=None,
                    appearance_thr=0.8, long_gap=20.0, long_dist=0.25,
                    dt=None, min_coverage=0.5):
    """Stitch broken IDs into one person.

    Rule 1 (always): B starts within `max_gap` s and `max_dist` (fraction of the frame) of where
    A ended, with no overlap in time.
    Rule 2 (only if `feats` given): B starts within `long_gap` s / `long_dist` AND their colour
    signatures match (cosine >= appearance_thr), which recovers stationary people who were
    occluded for longer. Greedy; the nearest end wins.
    Coverage guard (needs `dt` = seconds per processed frame): a merge is rejected if the merged
    track would be observed in less than `min_coverage` of its time span; that pattern means a
    chain of different people was glued together and would inflate dwell time.
    Returns (merged_tracks, parent) where parent maps every original id to its root id.
    """
    order = sorted(tracks, key=lambda i: tracks[i]["points"][0][0])
    parent, ends, root_feats = {}, {}, {}
    root_start, root_n = {}, {}
    for i in order:
        s0, x0, y0 = tracks[i]["points"][0]
        fi = feats.get(i) if feats else None
        best, best_d = None, float("inf")
        for root, (e, ex, ey) in ends.items():
            gap, d = s0 - e, math.hypot(x0 - ex, y0 - ey)
            if gap <= 0:
                continue
            ok = gap <= max_gap and d < max_dist
            if not ok and fi is not None and gap <= long_gap and d < long_dist:
                ok = any(float(fi @ f) >= appearance_thr for f in root_feats.get(root, []))
            if ok and dt is not None:
                span = tracks[i]["points"][-1][0] - root_start[root]
                n = root_n[root] + len(tracks[i]["points"])
                ok = span <= 0 or n * dt / span >= min_coverage
            if ok and d < best_d:
                best, best_d = root, d
        root = i if best is None else best
        parent[i] = root
        root_start.setdefault(root, tracks[i]["points"][0][0])
        root_n[root] = root_n.get(root, 0) + len(tracks[i]["points"])
        ends[root] = tracks[i]["points"][-1]
        if fi is not None:
            root_feats.setdefault(root, []).append(fi)
    merged = {}
    for i in order:
        m = merged.setdefault(parent[i], {"points": [], "zones": defaultdict(int)})
        m["points"].extend(tracks[i]["points"])
        for z, c in tracks[i]["zones"].items():
            m["zones"][z] += c
    return merged, parent


REVERSAL_WINDOW = float(os.getenv("REVERSAL_WINDOW", "3.0"))  # seconds; 0 disables


def line_crossings(points, line, margin=0.02, reversal_window=0.0):
    """Directed crossings of one segment by one track.

    line = (ax, ay, bx, by) in normalised frame coordinates. "in" means moving from the left-hand
    side of A->B to the right-hand side (in image coordinates, y down); "out" is the reverse.
    A crossing is counted only when the track moves from beyond `margin` on one side to beyond
    `margin` on the other, so jitter around the line is not counted, and only when the crossing
    point lies within the segment (walking around its end does not count).
    With `reversal_window` > 0, a crossing that is undone by the opposite crossing within that many
    seconds is dropped together with its partner: on real doorway footage 18 of 87 events were a
    track flipping back and forth (box jitter, merged group boxes, ID switches), and nobody
    genuinely crosses a line twice in three seconds.
    Returns [(sec, "in"|"out"), ...].
    """
    ax, ay, bx, by = line
    ex, ey = bx - ax, by - ay
    length = math.hypot(ex, ey)
    events, state = [], None
    for sec, x, y in points:
        dist = (ex * (y - ay) - ey * (x - ax)) / length  # signed distance, + = right-hand side
        if dist > margin:
            new = "pos"
        elif dist < -margin:
            new = "neg"
        else:
            continue
        if state is not None and new != state:
            u = ((x - ax) * ex + (y - ay) * ey) / (length ** 2)
            if -0.05 <= u <= 1.05:
                events.append((sec, "in" if new == "pos" else "out"))
        state = new
    if reversal_window > 0:
        kept = []
        for sec, d in events:
            if kept and kept[-1][1] != d and sec - kept[-1][0] <= reversal_window:
                kept.pop()
            else:
                kept.append((sec, d))
        events = kept
    return events


_models = {}


def get_model(model_path="yolov8s.pt"):
    """Load each YOLO model once per process (loading dominates short videos)."""
    if model_path not in _models:
        _models[model_path] = YOLO(model_path)
    return _models[model_path]


def track_video(path, model_path="yolov8s.pt", stride=None, imgsz=1280, conf=0.15,
                use_appearance=False, lines=None):
    """Returns (meta, tracks, crops).

    `stride` = process every Nth frame; default None picks ~5 processed frames per
    second of video (1 for low-fps CCTV). imgsz/conf defaults come from the Phase 1
    evaluation on the Mall dataset (MAE 4.1 people vs 13.2 for 640/0.25).

    tracks[id] = dict(points=[(sec,cx,cy)], zones=Counter-like dict)
    crops[id]  = best (largest) person crop, used for CLIP classification.
    """
    dev = get_device()
    model = get_model(model_path)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if stride is None:
        stride = max(1, round(fps / 5))
    # Detect and associate separately: model.track() only returns *confirmed* tracks, which at
    # low fps hides most people (Phase 1: 13.7 vs 23.9 detections per frame on the Mall clip).
    # A fresh BYTETracker per video also means no state leaks between videos.
    with open(TRACKER_CFG) as f:
        byte = BYTETracker(args=SimpleNamespace(**yaml.safe_load(f)))
    tracks = defaultdict(lambda: {"points": [], "zones": defaultdict(int)})
    crops, crop_area = {}, defaultdict(int)
    frame_idx, frames_seen, per_second = 0, 0, defaultdict(set)
    occupancy = defaultdict(list)  # second -> detections per processed frame (incl. id-less)
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
        det = model.predict(frame, classes=[PERSON], imgsz=imgsz, conf=conf,
                            device=dev, verbose=False)[0].boxes.cpu().numpy()
        occupancy[int(sec)].append(len(det))
        out = byte.update(det, frame)
        if len(out) == 0:
            continue
        for row in out:
            box, tid = row[:4], int(row[4])
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
    # Count crossings on RAW tracks: merged tracks may chain different people.
    crossings = {name: sorted(e for t in tracks.values() for e in line_crossings(t["points"], ln, reversal_window=REVERSAL_WINDOW))
                 for name, ln in (lines or {}).items()}
    feats = None
    if use_appearance:
        feats = {i: f for i, c in crops.items() if (f := appearance_feature(c)) is not None}
    tracks, parent = merge_fragments(dict(tracks), feats=feats, dt=stride / fps)
    best_crops = {}
    for i, c in crops.items():  # keep the largest crop among a merged person's fragments
        r = parent[i]
        if r not in best_crops or c.size > best_crops[r].size:
            best_crops[r] = c
    crops = best_crops
    meta = {"fps": fps, "duration_sec": frame_idx / fps,
            "stride": stride, "raw_ids": raw_ids, "parent": parent, "crossings": crossings,
            "occupancy_per_second": {sec: float(np.mean(v)) for sec, v in occupancy.items()},
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
