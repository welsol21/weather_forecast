"""Direct 24-hour temperature forecast from segment history.

Instead of predicting segment parameters, predict the next 24 hourly
temperature values directly.

Input:  full sequence of segment feature vectors (with temporal features)
Output: next 24 hourly temperature values (normalized by last known temp)

Training: for each segment i, target = actual hourly temps for the next
24 hours starting at segment i's end time.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_patterns.pattern.segmentation_temperature import (
    TRY_ORDER, TemperatureSegment, TemperatureFit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/temperature_24h.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SEGMENTS_PATH = Path("artifacts/temperature_segments.json")
MODEL_PATH    = Path("artifacts/temperature_24h.pt")

ODE_FEAT_DIM  = 12
TIME_FEAT_DIM = 4
FEAT_DIM      = ODE_FEAT_DIM + TIME_FEAT_DIM   # 16
HORIZON       = 24   # hours to predict

D_MODEL   = 128
N_HEAD    = 4
N_LAYERS  = 3
DIM_FF    = 256
DROPOUT   = 0.1
LR        = 3e-4
EPOCHS    = 500

EQ_IDX = {eq: i for i, eq in enumerate(TRY_ORDER)}


# ── Features ───────────────────────────────────────────────────────────────────

def _global_scales(segments: list) -> dict:
    Ls = [abs(s.fit.L) for s in segments if abs(s.fit.L) > 1e-6]
    As = [abs(s.fit.A) for s in segments if abs(s.fit.A) > 1e-6]
    Bs = [abs(s.fit.B) for s in segments if abs(s.fit.B) > 1e-6]
    cs = [abs(s.fit.c) for s in segments if abs(s.fit.c) > 1e-6]
    return {
        "L": float(np.percentile(Ls, 90)) if Ls else 1.0,
        "A": float(np.percentile(As, 90)) if As else 1.0,
        "B": float(np.percentile(Bs, 90)) if Bs else 1.0,
        "c": float(np.percentile(cs, 90)) if cs else 1.0,
    }

def _time_features(ts: pd.Timestamp) -> np.ndarray:
    doy = ts.day_of_year / 365.0
    hod = ts.hour / 24.0
    return np.array([
        math.sin(2 * math.pi * doy), math.cos(2 * math.pi * doy),
        math.sin(2 * math.pi * hod), math.cos(2 * math.pi * hod),
    ], dtype=np.float32)

def segment_to_vector(seg, scales: dict) -> np.ndarray:
    v = np.zeros(ODE_FEAT_DIM, dtype=np.float32)
    v[EQ_IDX[seg.fit.eq_type]] = 1.0
    v[6]  = seg.fit.L   / scales["L"]
    v[7]  = seg.fit.c   / scales["c"]
    v[8]  = seg.fit.lam
    v[9]  = seg.fit.A   / scales["A"]
    v[10] = seg.fit.B   / scales["B"]
    v[11] = seg.duration_hours / 24.0
    return np.concatenate([v, _time_features(seg.start_time)])


# ── Model ──────────────────────────────────────────────────────────────────────

class Forecast24h(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(FEAT_DIM, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEAD, DIM_FF, DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, HORIZON),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, seq_len, FEAT_DIM)
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        out = self.encoder(self.proj(x), mask=mask)
        return self.head(out)   # (1, seq_len, HORIZON)


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_targets(
    segments: list,
    hourly_temp: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """For each segment, extract the next HORIZON hourly temps as target.

    Returns:
        valid_mask: bool array of len(segments) — True if target available
        targets: (N_valid, HORIZON) array of delta temps (actual - x0)
        x0_values: last known temperature before each target window
    """
    targets, x0_values, valid_mask = [], [], []

    for seg in segments:
        # x0 = last known temperature at end of this segment
        end_time = seg.end_time
        x0 = seg.x0   # start value; use actual end if available
        # Try to get actual end value from hourly data
        if end_time in hourly_temp.index:
            x0 = float(hourly_temp.loc[end_time])

        # Target: next HORIZON hours after segment end
        target_times = [end_time + pd.Timedelta(hours=h+1) for h in range(HORIZON)]
        vals = [hourly_temp.get(t, np.nan) for t in target_times]

        if any(np.isnan(v) for v in vals):
            valid_mask.append(False)
            targets.append(np.zeros(HORIZON, dtype=np.float32))
            x0_values.append(x0)
            continue

        # Normalize: predict delta from x0
        delta = np.array(vals, dtype=np.float32) - x0
        targets.append(delta)
        x0_values.append(x0)
        valid_mask.append(True)

    return np.array(valid_mask), np.array(targets, dtype=np.float32), x0_values


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    model: Forecast24h,
    vectors: np.ndarray,       # (N, FEAT_DIM)
    targets: np.ndarray,       # (N, HORIZON)
    valid_mask: np.ndarray,    # (N,) bool
    device: str,
) -> list[float]:
    model.to(device)

    X = torch.tensor(vectors,  dtype=torch.float32).unsqueeze(0).to(device)
    Y = torch.tensor(targets,  dtype=torch.float32).unsqueeze(0).to(device)
    M = torch.tensor(valid_mask, dtype=torch.bool).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X)                      # (1, N, HORIZON)
        # Only compute loss on positions where target is valid
        pred_valid = pred[0][M]              # (N_valid, HORIZON)
        tgt_valid  = Y[0][M]
        loss = nn.functional.mse_loss(pred_valid, tgt_valid)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % 50 == 0:
            rmse_c = math.sqrt(losses[-1])
            logger.info("epoch=%d/%d  loss=%.6f  rmse=%.3f°C  lr=%.2e",
                        epoch + 1, EPOCHS, losses[-1], rmse_c,
                        scheduler.get_last_lr()[0])
    return losses


# ── Main ───────────────────────────────────────────────────────────────────────

def load_segments() -> list:
    with open(SEGMENTS_PATH, encoding="utf-8") as f:
        records = json.load(f)
    segments = []
    for r in records:
        fit = TemperatureFit(**r["fit"])
        segments.append(TemperatureSegment(
            segment_id=r["segment_id"],
            start_index=r["start_index"],
            end_index=r["end_index"],
            start_time=pd.Timestamp(r["start_time"]),
            end_time=pd.Timestamp(r["end_time"]),
            duration_hours=r["duration_hours"],
            fit=fit,
            x0=r["x0"],
        ))
    return segments


def main():
    Path("artifacts").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    segments = load_segments()
    logger.info("Loaded %d segments", len(segments))

    # Load hourly temperature
    df = pd.read_csv("hly4935_subset.csv", skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    hourly_temp = df.set_index("date")["temp"].dropna()
    logger.info("Hourly data: %d points %s → %s",
                len(hourly_temp), hourly_temp.index[0], hourly_temp.index[-1])

    scales = _global_scales(segments)
    vectors = np.array([segment_to_vector(s, scales) for s in segments])

    valid_mask, targets, x0_values = build_targets(segments, hourly_temp)
    logger.info("Valid training positions: %d / %d", valid_mask.sum(), len(segments))

    model = Forecast24h()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d", n_params)

    logger.info("Training, epochs=%d", EPOCHS)
    t0 = time.perf_counter()
    losses = train(model, vectors, targets, valid_mask, device)
    logger.info("Training complete elapsed_s=%.1f final_loss=%.6f rmse=%.3f°C",
                time.perf_counter() - t0, losses[-1], math.sqrt(losses[-1]))
    torch.save(model.state_dict(), MODEL_PATH)

    # --- Forecast February 1 first 24 hours ---
    logger.info("=== FEBRUARY 1 FORECAST (first 24h) ===")
    model.eval()
    with torch.no_grad():
        X = torch.tensor(vectors, dtype=torch.float32).unsqueeze(0).to(device)
        mask = nn.Transformer.generate_square_subsequent_mask(len(vectors), device=device)
        enc_out = model.encoder(model.proj(X), mask=mask)
        pred_delta = model.head(enc_out)[0, -1].cpu().numpy()   # last segment → 24h

    # x0 = last known temperature (end of last January segment)
    last_seg = segments[-1]
    x0 = float(hourly_temp.get(last_seg.end_time, last_seg.x0))
    pred_values = x0 + pred_delta

    actual = hourly_temp.loc["2026-02-01 00:00":"2026-02-01 23:00"]
    feb1_start = pd.Timestamp("2026-02-01 00:00")
    pred_times = [feb1_start + pd.Timedelta(hours=h) for h in range(HORIZON)]

    errors = []
    logger.info("%-20s  %6s  %6s  %6s", "timestamp", "actual", "pred", "error")
    for i in range(HORIZON):
        ts = pred_times[i]
        act = float(actual.get(ts, np.nan))
        pred = float(pred_values[i])
        err = act - pred
        errors.append(err)
        logger.info("  %s  %6.2f  %6.2f  %+.2f", ts, act, pred, err)

    valid_errors = [e for e in errors if not np.isnan(e)]
    mae  = float(np.mean(np.abs(valid_errors)))
    rmse = float(np.sqrt(np.mean(np.array(valid_errors) ** 2)))
    logger.info("First 24h MAE=%.3f°C  RMSE=%.3f°C", mae, rmse)

    result = {
        "n_segments": len(segments),
        "final_train_loss": round(float(losses[-1]), 6),
        "final_train_rmse_celsius": round(math.sqrt(losses[-1]), 3),
        "february_1_24h_mae_celsius": round(mae, 3),
        "february_1_24h_rmse_celsius": round(rmse, 3),
    }
    Path("artifacts/temperature_24h_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info("Done. MAE=%.3f°C", mae)


if __name__ == "__main__":
    main()
