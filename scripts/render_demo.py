"""Render an annotated demo video: person boxes + track ids, counting line, live in/out
counters and occupancy. Uses the same detector, tracker and crossing logic as the pipeline.

Usage: PYTHONPATH=. python scripts/render_demo.py input.mp4 output.mp4 [line=0,0.5,1,0.5]
"""
import sys
from types import SimpleNamespace

import cv2
import yaml
from ultralytics.trackers.byte_tracker import BYTETracker

from retailvision.device import get_device
from retailvision.tracker import TRACKER_CFG, get_model, line_crossings


def main(src, dst, line=(0.0, 0.5, 1.0, 0.5), imgsz=1280, conf=0.15):
    model, dev = get_model(), get_device()
    with open(TRACKER_CFG) as f:
        tracker = BYTETracker(args=SimpleNamespace(**yaml.safe_load(f)))
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 2.0
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    history, counted = {}, {"in": 0, "out": 0}
    p1, p2 = (int(line[0] * w), int(line[1] * h)), (int(line[2] * w), int(line[3] * h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        det = model.predict(frame, classes=[0], imgsz=imgsz, conf=conf, device=dev,
                            verbose=False)[0].boxes.cpu().numpy()
        for x1, y1, x2, y2 in det.xyxy.astype(int):  # every detection (thin white)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        for row in tracker.update(det, frame):  # confirmed tracks (thick green + id)
            x1, y1, x2, y2, tid = map(int, row[:5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, str(tid), (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            pts = history.setdefault(tid, [])
            pts.append((n / fps, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
            ev = line_crossings(pts, line)
            seen = len(pts) > 1 and len(line_crossings(pts[:-1], line)) or 0
            for _, d in ev[seen:]:
                counted[d] += 1
        cv2.line(frame, p1, p2, (0, 215, 255), 2)
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(frame, f"occupancy {len(det)}   crossed IN {counted['in']}   OUT {counted['out']}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
    cap.release()
    out.release()
    print(f"wrote {dst}: {n} frames, in={counted['in']} out={counted['out']}")


if __name__ == "__main__":
    line = tuple(float(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0.0, 0.5, 1.0, 0.5)
    main(sys.argv[1], sys.argv[2], line)
