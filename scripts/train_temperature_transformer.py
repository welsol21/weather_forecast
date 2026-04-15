"""Temperature sequence Transformer — full history, time-aware.

Architecture:
  - Input: all N segments as one sequence (GPT-style causal training)
  - Feature vector: 12 ODE features + 4 temporal (sin/cos of day-of-year,
    sin/cos of hour-of-day) = 16 dims total
  - Transformer encoder with causal mask, d_model=128, 4 heads, 3 layers
  - Output head: predict next segment's 12 ODE features

Training:
  - One pass over the full 1707-segment sequence
  - Predict each position t+1 given 0..t (causal)
  - No fixed window — the model sees the full history

Inference:
  - Feed all 1707 January segments
  - Autoregressively predict until February (672h) is covered
  - Temporal features of predicted segments are computed from their
    predicted start time (x0 from previous segment's end)
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
    _predict_value, build_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/temperature_transformer.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SEGMENTS_PATH = Path("segments/temperature_segments.json")
MODEL_PATH    = Path("artifacts/temperature_transformer.pt")

ODE_FEAT_DIM  = 12    # ODE feature vector
TIME_FEAT_DIM = 4     # sin/cos day-of-year + sin/cos hour-of-day
FEAT_DIM      = ODE_FEAT_DIM + TIME_FEAT_DIM   # 16

D_MODEL       = 128
N_HEAD        = 4
N_LAYERS      = 3
DIM_FF        = 256
DROPOUT       = 0.1

LR            = 3e-4
EPOCHS        = 500
FEB_HOURS     = 672


# ── Feature encoding ───────────────────────────────────────────────────────────

EQ_IDX = {eq: i for i, eq in enumerate(TRY_ORDER)}

def _global_scales(segments: list[TemperatureSegment]) -> dict[str, float]:
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
    """4 temporal features: sin/cos of day-of-year and hour-of-day."""
    doy = ts.day_of_year / 365.0
    hod = ts.hour / 24.0
    return np.array([
        math.sin(2 * math.pi * doy),
        math.cos(2 * math.pi * doy),
        math.sin(2 * math.pi * hod),
        math.cos(2 * math.pi * hod),
    ], dtype=np.float32)

def segment_to_ode_vector(seg: TemperatureSegment, scales: dict) -> np.ndarray:
    v = np.zeros(ODE_FEAT_DIM, dtype=np.float32)
    v[EQ_IDX[seg.fit.eq_type]] = 1.0
    v[6]  = seg.fit.L   / scales["L"]
    v[7]  = seg.fit.c   / scales["c"]
    v[8]  = seg.fit.lam
    v[9]  = seg.fit.A   / scales["A"]
    v[10] = seg.fit.B   / scales["B"]
    v[11] = seg.duration_hours / 24.0
    return v

def segment_to_full_vector(seg: TemperatureSegment, scales: dict) -> np.ndarray:
    ode = segment_to_ode_vector(seg, scales)
    time_feat = _time_features(seg.start_time)
    return np.concatenate([ode, time_feat])

def vector_to_fit(v: np.ndarray, scales: dict) -> tuple[TemperatureFit, int]:
    eq_idx  = int(np.argmax(v[:6]))
    eq_type = TRY_ORDER[eq_idx]
    L   = float(v[6]) * scales["L"]
    c   = float(v[7]) * scales["c"]
    lam = float(max(v[8], 0.0))
    A   = float(v[9]) * scales["A"]
    B   = float(v[10]) * scales["B"]
    dur = max(1, int(round(float(v[11]) * 24)))
    return TemperatureFit(eq_type=eq_type, L=L, c=c, lam=lam, A=A, B=B), dur


# ── Model ──────────────────────────────────────────────────────────────────────

class TemperatureTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(FEAT_DIM, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD,
            dim_feedforward=DIM_FF, dropout=DROPOUT,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, ODE_FEAT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, seq_len, FEAT_DIM)
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        out = self.encoder(self.input_proj(x), mask=mask)
        return self.head(out)   # (1, seq_len, ODE_FEAT_DIM)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    model: TemperatureTransformer,
    full_vectors: np.ndarray,   # (N, FEAT_DIM)
    ode_targets: np.ndarray,    # (N, ODE_FEAT_DIM)  — shifted by 1
    device: str,
) -> list[float]:
    model.to(device)
    # Whole sequence fits in one forward pass on GPU
    X = torch.tensor(full_vectors[:-1], dtype=torch.float32).unsqueeze(0).to(device)
    Y = torch.tensor(ode_targets[1:],   dtype=torch.float32).unsqueeze(0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X)           # (1, N-1, ODE_FEAT_DIM)
        loss = criterion(pred, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % 50 == 0:
            logger.info("epoch=%d/%d loss=%.6f lr=%.2e",
                        epoch + 1, EPOCHS, losses[-1],
                        scheduler.get_last_lr()[0])
    return losses


# ── Inference ──────────────────────────────────────────────────────────────────

def forecast_february(
    model: TemperatureTransformer,
    all_segments: list[TemperatureSegment],
    scales: dict,
    device: str,
) -> list[tuple[TemperatureFit, int, float]]:
    model.eval()

    # Full history as seed
    history = [segment_to_full_vector(s, scales) for s in all_segments]

    # x0 of first predicted segment = last value of last January segment
    last_seg = all_segments[-1]
    prev_end_value = _predict_value(last_seg.fit, float(last_seg.duration_hours - 1))
    prev_end_time  = last_seg.end_time

    predicted: list[tuple[TemperatureFit, int, float]] = []
    covered = 0

    with torch.no_grad():
        while covered < FEB_HOURS:
            X = torch.tensor(np.array(history), dtype=torch.float32) \
                    .unsqueeze(0).to(device)
            pred_ode = model(X)[0, -1].cpu().numpy()   # last position → next segment

            fit, dur = vector_to_fit(pred_ode, scales)
            x0 = prev_end_value

            predicted.append((fit, dur, x0))
            covered += dur

            # Build full vector for predicted segment (temporal from predicted time)
            next_start_time = prev_end_time + pd.Timedelta(hours=1)
            time_feat = _time_features(next_start_time)
            next_vec = np.concatenate([pred_ode, time_feat.astype(np.float32)])
            history.append(next_vec)

            prev_end_value = x0 + _predict_value(fit, float(dur - 1)) - _predict_value(fit, 0.0)
            prev_end_time  = next_start_time + pd.Timedelta(hours=dur - 1)

    return predicted


def decode_to_series(
    predicted: list[tuple[TemperatureFit, int, float]],
    start_time: pd.Timestamp,
) -> pd.Series:
    timestamps, values = [], []
    current_time = start_time
    for fit, dur, x0 in predicted:
        shift = x0 - _predict_value(fit, 0.0)
        for step in range(dur):
            values.append(_predict_value(fit, float(step)) + shift)
            timestamps.append(current_time + pd.Timedelta(hours=step))
        current_time += pd.Timedelta(hours=dur)
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="temperature_predicted")


# ── Main ───────────────────────────────────────────────────────────────────────

def load_segments() -> list[TemperatureSegment]:
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

    scales = _global_scales(segments)
    logger.info("Scales: %s", {k: f"{v:.3f}" for k, v in scales.items()})

    full_vectors = np.array([segment_to_full_vector(s, scales) for s in segments])
    ode_targets  = np.array([segment_to_ode_vector(s, scales)  for s in segments])
    logger.info("Sequence length: %d  feat_dim: %d", len(full_vectors), FEAT_DIM)

    model = TemperatureTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model params: %d", n_params)

    logger.info("Training on full sequence, epochs=%d", EPOCHS)
    t0 = time.perf_counter()
    losses = train(model, full_vectors, ode_targets, device)
    logger.info("Training complete elapsed_s=%.1f final_loss=%.6f",
                time.perf_counter() - t0, losses[-1])
    torch.save(model.state_dict(), MODEL_PATH)

    # --- Forecast February ---
    logger.info("=== FORECASTING FEBRUARY 2026 ===")
    predicted = forecast_february(model, segments, scales, device)
    logger.info("Predicted %d segments covering %d hours",
                len(predicted), sum(d for _, d, _ in predicted))

    pred_series = decode_to_series(predicted, pd.Timestamp("2026-02-01")).loc["2026-02-01":"2026-02-28"]

    df = pd.read_csv("hly4935_subset.csv", skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    actual = df.set_index("date")["temp"].loc["2026-02-01":"2026-02-28"].dropna()

    common = pred_series.index.intersection(actual.index)
    errors = actual.loc[common].values - pred_series.loc[common].values
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    logger.info("Points compared: %d", len(common))
    logger.info("MAE:  %.3f°C", mae)
    logger.info("RMSE: %.3f°C", rmse)
    logger.info("--- First 24 hours ---")
    for i in range(min(24, len(common))):
        ts = common[i]
        logger.info("  %s  actual=%5.2f  pred=%5.2f  err=%+.2f",
                    ts, float(actual.loc[ts]),
                    float(pred_series.loc[ts]), float(errors[i]))

    pred_eq_dist = {}
    for eq in TRY_ORDER:
        count = sum(1 for fit, _, _ in predicted if fit.eq_type == eq)
        if count:
            pred_eq_dist[eq] = count

    result = {
        "architecture": f"Transformer d_model={D_MODEL} heads={N_HEAD} layers={N_LAYERS}",
        "n_segments": len(segments),
        "sequence_length": len(full_vectors),
        "final_loss": round(float(losses[-1]), 6),
        "february_mae_celsius": round(mae, 3),
        "february_rmse_celsius": round(rmse, 3),
        "february_points": len(common),
        "predicted_segment_count": len(predicted),
        "predicted_eq_distribution": pred_eq_dist,
    }
    Path("artifacts/temperature_transformer_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info("Report saved.")
    logger.info("GRU baseline: MAE=3.921°C  |  Transformer: MAE=%.3f°C", mae)


if __name__ == "__main__":
    main()
