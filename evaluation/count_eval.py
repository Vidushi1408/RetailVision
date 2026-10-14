"""Per-frame people-count accuracy vs Mall dataset ground truth.

Usage: python evaluation/count_eval.py [model] [imgsz] [conf] [every_nth_frame]
Dataset: data/mall_dataset (CUHK Mall, research use only) + data/mall_counts.npy
"""
import glob
import sys

import numpy as np
from ultralytics import YOLO

from retailvision.device import get_device

model = sys.argv[1] if len(sys.argv) > 1 else "yolov8s.pt"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
step = int(sys.argv[4]) if len(sys.argv) > 4 else 10

gt = np.load("data/mall_counts.npy")
frames = sorted(glob.glob("data/mall_dataset/frames/seq_*.jpg"))
idx = list(range(0, len(frames), step))
m, dev = YOLO(model), get_device()
pred = np.array([len(m.predict(frames[i], classes=[0], imgsz=imgsz, conf=conf,
                               device=dev, verbose=False)[0].boxes) for i in idx])
g = gt[idx]
print(f"{model} imgsz={imgsz} conf={conf} frames={len(idx)}: "
      f"MAE={np.abs(pred - g).mean():.1f} bias={(pred - g).mean():+.1f} "
      f"MAPE={100 * np.mean(np.abs(pred - g) / g):.1f}%")
