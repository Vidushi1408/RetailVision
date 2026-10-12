"""LSTM footfall forecaster trained on the stored traffic history."""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from .db import connect

import datetime
import os

from .device import get_device

WINDOW = 6
MODEL_CACHE_PATH = "store_lstm.pt"
_cached_forecaster = {"model": None, "data_count": -1, "loss": 0.0}


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def hourly_series():
    conn = connect()
    rows = conn.execute("SELECT timestamp, people_count FROM customer_traffic ORDER BY timestamp ASC").fetchall()
    conn.close()
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["timestamp"][:13]].append(r["people_count"])  # YYYY-MM-DD HH
    sorted_items = sorted(buckets.items())
    series = [float(np.mean(v)) for _, v in sorted_items]
    last_timestamp = sorted_items[-1][0] if sorted_items else None
    return series, last_timestamp, len(rows)


def forecast(steps=6, epochs=200):
    series, last_ts, row_count = hourly_series()
    dev = get_device()

    # Generate future hour labels
    if last_ts:
        try:
            base_dt = datetime.datetime.strptime(last_ts, "%Y-%m-%d %H")
        except Exception:
            base_dt = datetime.datetime.now()
    else:
        base_dt = datetime.datetime.now()

    future_labels = [
        (base_dt + datetime.timedelta(hours=i + 1)).strftime("%H:00")
        for i in range(steps)
    ]

    # Graceful fallback if insufficient hourly points
    if len(series) < WINDOW + 3:
        avg_traffic = float(np.mean(series)) if series else 5.0
        # Light variance based on typical retail curve
        baseline_preds = [float(round(max(avg_traffic * (1 + 0.1 * float(np.sin(i))), 1.0), 1)) for i in range(steps)]
        return {
            "status": "baseline_extrapolation",
            "message": f"Need at least {WINDOW + 3} hourly data points for deep LSTM; showing trend baseline.",
            "history_points": len(series),
            "future_labels": future_labels,
            "forecast_next_hours": baseline_preds
        }


    s = np.array(series, dtype=np.float32)
    mu, sd = float(s.mean()), float(s.std() or 1.0)
    z = (s - mu) / sd

    def train(arr):
        X = np.stack([arr[i:i + WINDOW] for i in range(len(arr) - WINDOW)])[..., None]
        y = arr[WINDOW:, None]
        X, y = torch.tensor(X, device=dev), torch.tensor(y, device=dev)
        torch.manual_seed(0)
        m, loss_fn = LSTMForecaster().to(dev), nn.MSELoss()
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(m(X), y)
            loss.backward()
            opt.step()
        return m.eval(), float(loss.item())

    def step(m, window):
        inp = torch.tensor(window[-WINDOW:], dtype=torch.float32, device=dev)[None, :, None]
        with torch.no_grad():
            return float(m(inp).item())

    # Honest backtest: train on the first 80%, predict the rest one step ahead, and compare the
    # LSTM with "same as last value". Phase 1 showed the LSTM often does NOT beat that baseline.
    split = max(int(0.8 * len(z)), WINDOW + 3)
    bt_model, _ = train(z[:split])
    lstm_err = [abs(step(bt_model, list(z[:t])) - z[t]) for t in range(split, len(z))]
    naive_err = [abs(z[t - 1] - z[t]) for t in range(split, len(z))]
    lstm_mae, naive_mae = float(np.mean(lstm_err)) * sd, float(np.mean(naive_err)) * sd
    use_lstm = lstm_mae < naive_mae

    loss_val = _cached_forecaster["loss"]
    model = None
    if use_lstm:
        if (_cached_forecaster["model"] is not None
                and _cached_forecaster["data_count"] == row_count):
            model = _cached_forecaster["model"]
        else:
            model, loss_val = train(z)
            _cached_forecaster.update(model=model, data_count=row_count, loss=loss_val)

    window, preds = list(z[-WINDOW:]), []
    for _ in range(steps):
        p = step(model, window) if use_lstm else window[-1]
        preds.append(p)
        window.append(p)

    return {
        "status": "lstm_trained" if use_lstm else "last_value_baseline",
        "method": "LSTM" if use_lstm else "last observed value (LSTM did not beat it in backtest)",
        "backtest_mae": {"lstm": round(lstm_mae, 2), "last_value": round(naive_mae, 2)},
        "train_loss": round(loss_val, 4),
        "history_points": len(series),
        "future_labels": future_labels,
        "forecast_next_hours": [round(float(max(p * sd + mu, 0.0)), 1) for p in preds]
    }
