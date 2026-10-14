"""Best (largest) crop per tracked person, as a numbered contact sheet, for hand labelling
attributes such as "carrying a bag".

Usage: PYTHONPATH=. python scripts/attribute_crops.py video.mp4 out_dir [n=60] [min_height_px=120]
Writes out_dir/crops/<n>.jpg, out_dir/sheet_K.jpg and out_dir/index.json.
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
from retailvision.tracker import TRACKER_CFG, get_model


def main(src, out, n=60, min_h=120):
    os.makedirs(os.path.join(out, "crops"), exist_ok=True)
    model, dev = get_model(), get_device()
    with open(TRACKER_CFG) as f:
        trk = BYTETracker(args=SimpleNamespace(**yaml.safe_load(f)))
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    stride = max(1, round(fps / 5))
    best, k = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        k += 1
        if k % stride:
            continue
        det = model.predict(fr, classes=[0], imgsz=1280, conf=0.25, device=dev, verbose=False)[0].boxes.cpu().numpy()
        H, W = fr.shape[:2]
        for row in trk.update(det, fr):
            x1, y1, x2, y2, tid = map(int, row[:5])
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            area = (x2 - x1) * (y2 - y1)
            full_body = y2 < H - 4 and y1 > 2          # not cut by the frame edge
            if full_body and (tid not in best or area > best[tid][0]):
                best[tid] = (area, fr[y1:y2, x1:x2].copy(), k / fps)
    cap.release()
    chosen = [(t, v) for t, v in sorted(best.items(), key=lambda kv: -kv[1][0]) if v[1].shape[0] >= min_h][:n]
    chosen.sort(key=lambda tv: tv[1][2])
    index, cw, ch, cols = [], 150, 270, 6
    for s in range(0, len(chosen), 30):
        sheet = np.full((5 * ch, cols * cw, 3), 255, np.uint8)
        for j, (tid, (_, crop, sec)) in enumerate(chosen[s:s + 30]):
            i = s + j + 1
            cv2.imwrite(os.path.join(out, "crops", f"{i}.jpg"), crop)
            sc = min((cw - 6) / crop.shape[1], (ch - 26) / crop.shape[0])
            im = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            r, c = divmod(j, cols)
            y0, x0 = r * ch + 24, c * cw + 3
            sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.putText(sheet, f"#{i} {sec:.0f}s", (x0, r * ch + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            index.append({"n": i, "tid": tid, "sec": round(sec, 1), "h": int(crop.shape[0]), "w": int(crop.shape[1])})
        cv2.imwrite(os.path.join(out, f"sheet_{s // 30 + 1}.jpg"), sheet)
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    hs = [r["h"] for r in index]
    print(f"{len(index)} crops; height px min/median/max = {min(hs)}/{int(np.median(hs))}/{max(hs)}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 60, int(a[4]) if len(a) > 4 else 120)
