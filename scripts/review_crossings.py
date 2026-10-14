"""Validation helper: save a labelled crop for every counted line crossing so a person can
check them by eye (precision, direction, and the same person counted twice).

Usage: PYTHONPATH=. python scripts/review_crossings.py video.mp4 out_dir [x1,y1,x2,y2]
Writes out_dir/sheet_N.jpg contact sheets and prints the event list.
"""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from ultralytics.trackers.byte_tracker import BYTETracker

from retailvision.device import get_device
from retailvision.tracker import REVERSAL_WINDOW, TRACKER_CFG, get_model, line_crossings

CELL_W, CELL_H, COLS, ROWS = 150, 190, 6, 5


def main(src, out_dir, line=(0.05, 0.55, 0.95, 0.55), imgsz=1280, conf=0.15, stride=None):
    os.makedirs(out_dir, exist_ok=True)
    model, dev = get_model(), get_device()
    with open(TRACKER_CFG) as f:
        tracker = BYTETracker(args=SimpleNamespace(**yaml.safe_load(f)))
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    stride = stride or max(1, round(fps / 5))
    w, h = int(cap.get(3)), int(cap.get(4))
    history, seen_events, events, n = {}, {}, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % stride:
            continue
        det = model.predict(frame, classes=[0], imgsz=imgsz, conf=conf, device=dev, verbose=False)[0].boxes.cpu().numpy()
        for row in tracker.update(det, frame):
            x1, y1, x2, y2, tid = map(int, row[:5])
            pts = history.setdefault(tid, [])
            pts.append((n / fps, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
            ev = line_crossings(pts, line)  # raw candidates; cancellations are applied at the end
            for sec, d in ev[seen_events.get(tid, 0):]:
                pad = int(0.15 * max(x2 - x1, y2 - y1))
                crop = frame[max(y1 - pad, 0):min(y2 + pad, h), max(x1 - pad, 0):min(x2 + pad, w)].copy()
                events.append({"sec": sec, "dir": d, "tid": tid, "crop": crop})
            seen_events[tid] = len(ev)
    cap.release()
    # keep only the events that survive reversal cancellation on each track's full history
    final = {(tid, round(s, 3), d) for tid, pts in history.items()
             for s, d in line_crossings(pts, line, reversal_window=REVERSAL_WINDOW)}
    candidates = len(events)
    events = [e for e in events if (e["tid"], round(e["sec"], 3), e["dir"]) in final]
    events.sort(key=lambda e: e["sec"])
    print(f"{candidates} raw candidates -> {len(events)} after cancelling reversals within {REVERSAL_WINDOW:g}s")
    per_sheet = COLS * ROWS
    for s in range(0, len(events), per_sheet):
        sheet = np.full((ROWS * CELL_H, COLS * CELL_W, 3), 255, np.uint8)
        for k, e in enumerate(events[s:s + per_sheet]):
            r, c = divmod(k, COLS)
            im = e["crop"]
            sc = min((CELL_W - 6) / im.shape[1], (CELL_H - 26) / im.shape[0])
            im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            y0, x0 = r * CELL_H + 22, c * CELL_W + 3
            sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            col = (0, 140, 0) if e["dir"] == "in" else (0, 0, 200)
            cv2.putText(sheet, f"#{s + k + 1} {e['dir']} {e['sec']:.0f}s id{e['tid']}", (x0, r * CELL_H + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.imwrite(os.path.join(out_dir, f"sheet_{s // per_sheet + 1}.jpg"), sheet)
    print(f"{len(events)} crossings: in={sum(e['dir'] == 'in' for e in events)} out={sum(e['dir'] == 'out' for e in events)}")
    for i, e in enumerate(events, 1):
        print(f"#{i} t={e['sec']:.1f}s {e['dir']} id={e['tid']}")


if __name__ == "__main__":
    ln = tuple(float(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0.05, 0.55, 0.95, 0.55)
    main(sys.argv[1], sys.argv[2], ln)
