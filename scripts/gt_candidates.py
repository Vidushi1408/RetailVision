"""Build a candidate pool of line crossings for hand annotation (precision AND recall).

Two detector passes are merged: the production settings, and a looser pass (lower confidence,
no jitter margin, no reversal cancelling) that surfaces crossings the production run may have
missed. Every candidate is saved as a numbered crop on a contact sheet plus metadata JSON; a
person then marks each as a real, distinct crossing or not, and `score_crossings.py` compares
the production counter against that ground truth.

Usage: PYTHONPATH=. python scripts/gt_candidates.py video.mp4 out_dir x1,y1,x2,y2
"""
import json
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from ultralytics.trackers.byte_tracker import BYTETracker

from retailvision.device import get_device
from retailvision.tracker import TRACKER_CFG, get_model, line_crossings

CELL_W, CELL_H, COLS, ROWS = 180, 230, 5, 4


def run_pass(src, line, conf, margin, window):
    model, dev = get_model(), get_device()
    with open(TRACKER_CFG) as f:
        trk = BYTETracker(args=SimpleNamespace(**yaml.safe_load(f)))
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    stride = max(1, round(fps / 5))
    w, h = int(cap.get(3)), int(cap.get(4))
    hist, seen, ev = {}, {}, []
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if n % stride:
            continue
        det = model.predict(fr, classes=[0], imgsz=1280, conf=conf, device=dev, verbose=False)[0].boxes.cpu().numpy()
        for row in trk.update(det, fr):
            x1, y1, x2, y2, tid = map(int, row[:5])
            p = hist.setdefault(tid, [])
            p.append((n / fps, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
            e = line_crossings(p, line, margin)
            for s, d in e[seen.get(tid, 0):]:
                pad = int(0.15 * max(x2 - x1, y2 - y1))
                crop = fr[max(y1 - pad, 0):min(y2 + pad, h), max(x1 - pad, 0):min(x2 + pad, w)].copy()
                ev.append({"sec": s, "dir": d, "tid": tid, "x": (x1 + x2) / 2 / w, "crop": crop})
            seen[tid] = len(e)
    cap.release()
    final = {(t, round(s, 3), d) for t, p in hist.items() for s, d in line_crossings(p, line, margin, window)}
    for e in ev:
        e["kept"] = (e["tid"], round(e["sec"], 3), e["dir"]) in final
    return ev


def main(src, out, line):
    os.makedirs(out, exist_ok=True)
    prod = run_pass(src, line, conf=0.15, margin=0.02, window=3.0)
    loose = run_pass(src, line, conf=0.08, margin=0.0, window=0.0)
    for e in prod:
        e["source"] = "production"
    pool = list(prod)
    extra = 0
    for e in loose:  # add looser-pass events that no production candidate explains
        if not any(abs(e["sec"] - p["sec"]) <= 1.5 and e["dir"] == p["dir"] and abs(e["x"] - p["x"]) <= 0.2 for p in prod):
            e["source"], e["kept"] = "loose-only", False
            pool.append(e)
            extra += 1
    pool.sort(key=lambda e: e["sec"])
    meta = []
    per = COLS * ROWS
    for s in range(0, len(pool), per):
        sheet = np.full((ROWS * CELL_H, COLS * CELL_W, 3), 255, np.uint8)
        for k, e in enumerate(pool[s:s + per]):
            i = s + k + 1
            r, c = divmod(k, COLS)
            im = e["crop"]
            sc = min((CELL_W - 6) / im.shape[1], (CELL_H - 26) / im.shape[0])
            im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            y0, x0 = r * CELL_H + 24, c * CELL_W + 3
            sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            col = (0, 140, 0) if e["dir"] == "in" else (0, 0, 200)
            tag = "" if e["source"] == "production" else " *"
            cv2.putText(sheet, f"#{i} {e['dir']} {e['sec']:.1f}s{tag}", (x0, r * CELL_H + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.imwrite(os.path.join(out, f"cand_{s // per + 1}.jpg"), sheet)
    for i, e in enumerate(pool, 1):
        meta.append({"n": i, "sec": round(e["sec"], 2), "dir": e["dir"], "tid": e["tid"],
                     "x": round(e["x"], 3), "source": e["source"], "counted": e["kept"]})
    json.dump(meta, open(os.path.join(out, "candidates.json"), "w"), indent=1)
    print(f"{len(prod)} production candidates ({sum(e['kept'] for e in prod)} counted after cancelling), "
          f"{extra} extra from the loose pass, {len(pool)} to annotate on {-(-len(pool) // per)} sheets")


if __name__ == "__main__":
    ln = tuple(float(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0.05, 0.55, 0.95, 0.55)
    main(sys.argv[1], sys.argv[2], ln)
