"""Does the LSTM beat trivial baselines? Walk-forward test on the Mall ground-truth counts.

Usage: python evaluation/forecast_eval.py [bin_frames] [horizon]
Series = mean people count per `bin_frames` consecutive frames. Train on first 70%,
test on last 30%; each test point is predicted `horizon` steps ahead from the true past.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

from retailvision.device import get_device
from retailvision.forecast import LSTMForecaster, WINDOW

bin_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 10
horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 1
raw = np.load("data/mall_counts.npy").astype(np.float32)
n = len(raw) // bin_frames
s = raw[: n * bin_frames].reshape(n, bin_frames).mean(1)
split = int(0.7 * n)
mu, sd = s[:split].mean(), s[:split].std()
z = (s - mu) / sd
dev = get_device()


def windows(arr, lo, hi):
    X = np.stack([arr[i:i + WINDOW] for i in range(lo, hi - WINDOW)])[..., None]
    return X, arr[lo + WINDOW:hi, None]


X, y = windows(z, 0, split)
torch.manual_seed(0)
model, opt, lossf = LSTMForecaster().to(dev), None, nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
Xt, yt = torch.tensor(X, device=dev), torch.tensor(y, device=dev)
for _ in range(200):
    opt.zero_grad(); lossf(model(Xt), yt).backward(); opt.step()
model.eval()


def lstm_pred(past):
    w = list(past[-WINDOW:])
    with torch.no_grad():
        for _ in range(horizon):
            p = model(torch.tensor(w[-WINDOW:], dtype=torch.float32, device=dev)[None, :, None]).item()
            w.append(p)
    return w[-1]


rows = {"last value": [], "mean of window": [], "train mean": [], "LSTM": []}
truth = []
for t in range(split, n - horizon + 1):
    past = z[:t]
    truth.append(z[t + horizon - 1])
    rows["last value"].append(past[-1])
    rows["mean of window"].append(past[-WINDOW:].mean())
    rows["train mean"].append(0.0)
    rows["LSTM"].append(lstm_pred(past))
truth = np.array(truth) * sd + mu
print(f"series points={n} (bin={bin_frames} frames) train={split} test={len(truth)} horizon={horizon} steps")
for k, v in rows.items():
    p = np.array(v) * sd + mu
    print(f"  {k:15s} MAE={np.abs(p - truth).mean():5.2f}  RMSE={np.sqrt(((p - truth) ** 2).mean()):5.2f}")
