"""Temperature segment sequence forecaster.

Architecture:
  - Encoder: full history of ODE segments (causal Transformer)
  - Head: predict next K_MAX segments at once (direct multi-step, no autoregression)

Forecasting for horizon T:
  1. K = round(T / mean_segment_duration(current_month))  — seasonal constant
  2. Model predicts K_MAX segments in one forward pass
  3. Take first K segments, decode to hourly values until T hours covered

K is not predicted by the model — it is computed analytically from the
season and horizon. Mean segment duration per month is derived from training data.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_patterns.pattern.segmentation_temperature import (
    TRY_ORDER, TemperatureSegment, TemperatureFit,
    _predict_value,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/temperature_two_pass.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SEGMENTS_PATH = Path("artifacts/temperature_segments.json")
MODEL_PATH    = Path("artifacts/temperature_two_pass.pt")

ODE_FEAT_DIM  = 12
TIME_FEAT_DIM = 4
FEAT_DIM      = ODE_FEAT_DIM + TIME_FEAT_DIM   # encoder input: 16

K_MAX    = 48    # max segments predicted per forward pass
D_MODEL  = 128
N_HEAD   = 4
N_LAYERS = 3
DIM_FF   = 256
DROPOUT  = 0.1
LR       = 3e-4
EPOCHS   = 600

EQ_IDX = {eq: i for i, eq in enumerate(TRY_ORDER)}


# ── Feature helpers ────────────────────────────────────────────────────────────

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

def segment_to_ode_vector(seg, scales: dict) -> np.ndarray:
    v = np.zeros(ODE_FEAT_DIM, dtype=np.float32)
    v[EQ_IDX[seg.fit.eq_type]] = 1.0
    v[6]  = seg.fit.L   / scales["L"]
    v[7]  = seg.fit.c   / scales["c"]
    v[8]  = seg.fit.lam
    v[9]  = seg.fit.A   / scales["A"]
    v[10] = seg.fit.B   / scales["B"]
    v[11] = seg.duration_hours / 24.0
    return v

def segment_to_full_vector(seg, scales: dict) -> np.ndarray:
    return np.concatenate([segment_to_ode_vector(seg, scales),
                           _time_features(seg.start_time)])

def vector_to_fit(v: np.ndarray, scales: dict) -> tuple:
    eq_type = TRY_ORDER[int(np.argmax(v[:6]))]
    L   = float(v[6]) * scales["L"]
    c   = float(v[7]) * scales["c"]
    lam = float(max(v[8], 0.0))
    A   = float(v[9]) * scales["A"]
    B   = float(v[10]) * scales["B"]
    dur = max(1, int(round(float(v[11]) * 24)))
    return TemperatureFit(eq_type=eq_type, L=L, c=c, lam=lam, A=A, B=B), dur


# ── Historical K lookup ────────────────────────────────────────────────────────

def compute_K_from_history(
    segments: list,
    query_date: pd.Timestamp,
    T_hours: float,
    window_days: int = 7,
) -> int:
    """Count how many segments historically covered T_hours starting from
    the same day-of-year as query_date (across all years in training data).

    For each year, find the segment active at query_date's day-of-year
    (within ±window_days), then count segments forward until T_hours covered.
    Returns round(mean count across years), minimum 1.
    """
    query_doy = query_date.day_of_year
    years = sorted({s.start_time.year for s in segments})
    k_observations = []

    for year in years:
        # Find all segment start indices near query_doy in this year
        candidates = [
            i for i, s in enumerate(segments)
            if s.start_time.year == year
            and abs(s.start_time.day_of_year - query_doy) <= window_days
        ]
        if not candidates:
            continue

        # Take the closest one
        anchor_idx = min(candidates,
                         key=lambda i: abs(segments[i].start_time.day_of_year - query_doy))

        # Count segments from anchor forward until T_hours covered
        cumulative = 0
        k = 0
        for j in range(anchor_idx, len(segments)):
            cumulative += segments[j].duration_hours
            k += 1
            if cumulative >= T_hours or k >= K_MAX:
                break

        if k > 0:
            k_observations.append(k)

    if not k_observations:
        return max(1, round(T_hours / 30.0))   # fallback: 30h mean

    K = max(1, min(K_MAX, round(float(np.mean(k_observations)))))
    return K


# ── Model ──────────────────────────────────────────────────────────────────────

class SegmentForecaster(nn.Module):
    """Causal Transformer: segment history → next K_MAX segment ODE vectors."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(FEAT_DIM, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEAD, DIM_FF, DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2),
            nn.ReLU(),
            nn.Linear(D_MODEL * 2, K_MAX * ODE_FEAT_DIM),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, N, FEAT_DIM) → h: (N, D_MODEL)  causal."""
        N = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)
        return self.encoder(self.input_proj(x), mask=mask)[0]   # (N, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, N, FEAT_DIM) → segs: (N, K_MAX, ODE_FEAT_DIM)"""
        h = self.encode(x)                              # (N, D_MODEL)
        return self.head(h).view(-1, K_MAX, ODE_FEAT_DIM)  # (N, K_MAX, ODE_DIM)


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_dataset(
    ode_vectors: np.ndarray,   # (N, ODE_FEAT_DIM)
) -> tuple[np.ndarray, np.ndarray]:
    """For each position i, target = next K_MAX segment ODE vectors (padded).

    Returns:
        seg_targets (N, K_MAX, ODE_FEAT_DIM)
        seg_masks   (N, K_MAX)  — True where target is valid (not padding)
    """
    N = len(ode_vectors)
    seg_targets = np.zeros((N, K_MAX, ODE_FEAT_DIM), dtype=np.float32)
    seg_masks   = np.zeros((N, K_MAX), dtype=bool)

    for i in range(N - 1):
        available = min(K_MAX, N - 1 - i)
        seg_targets[i, :available] = ode_vectors[i + 1 : i + 1 + available]
        seg_masks[i, :available]   = True

    return seg_targets, seg_masks


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    model: SegmentForecaster,
    full_vectors: np.ndarray,   # (N, FEAT_DIM)
    seg_targets: np.ndarray,    # (N, K_MAX, ODE_FEAT_DIM)
    seg_masks: np.ndarray,      # (N, K_MAX)
    device: str,
) -> list[float]:
    model.to(device)

    X      = torch.tensor(full_vectors, dtype=torch.float32).unsqueeze(0).to(device)
    segs_t = torch.tensor(seg_targets,  dtype=torch.float32).to(device)
    masks_t = torch.tensor(seg_masks,   dtype=torch.bool).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        segs_pred = model(X)   # (N, K_MAX, ODE_FEAT_DIM)

        # Masked MSE: only on valid (non-padding) targets
        mask_exp = masks_t.unsqueeze(-1).expand_as(segs_pred)
        loss = ((segs_pred - segs_t) ** 2)[mask_exp].mean()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(float(loss.item()))
        if (epoch + 1) % 50 == 0:
            logger.info("epoch=%d/%d  loss=%.4f  lr=%.2e",
                        epoch + 1, EPOCHS, losses[-1],
                        scheduler.get_last_lr()[0])

    return losses


# ── Inference ──────────────────────────────────────────────────────────────────

def forecast(
    model: SegmentForecaster,
    full_vectors: np.ndarray,
    T_hours: float,
    query_date: pd.Timestamp,
    segments: list,
    scales: dict,
    device: str,
) -> tuple[list, int]:
    """Returns (list of (TemperatureFit, duration_hours), K_used)."""
    K = compute_K_from_history(segments, query_date, T_hours)
    logger.info("T=%dh  query=%s  K=%d (from history)",
                int(T_hours), query_date.strftime("%m-%d"), K)

    model.eval()
    with torch.no_grad():
        X = torch.tensor(full_vectors, dtype=torch.float32).unsqueeze(0).to(device)
        segs_pred = model(X)          # (N, K_MAX, ODE_FEAT_DIM)
        segs_np = segs_pred[-1, :K].cpu().numpy()   # last position → first K

    result = []
    for v in segs_np:
        fit, dur = vector_to_fit(v, scales)
        result.append((fit, dur))

    return result, K


def decode_to_series(
    predicted: list[tuple],
    x0: float,
    start_time: pd.Timestamp,
    n_hours: int,
) -> pd.Series:
    timestamps, values = [], []
    current_time = start_time
    current_x0 = x0

    for fit, dur in predicted:
        shift = current_x0 - _predict_value(fit, 0.0)
        for step in range(dur):
            if len(timestamps) >= n_hours:
                break
            timestamps.append(current_time + pd.Timedelta(hours=step))
            values.append(_predict_value(fit, float(step)) + shift)
        if len(timestamps) >= n_hours:
            break
        current_x0 = _predict_value(fit, float(dur - 1)) + shift
        current_time += pd.Timedelta(hours=dur)

    return pd.Series(values, index=pd.DatetimeIndex(timestamps))


# ── Segment loader ─────────────────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    Path("artifacts").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    segments = load_segments()
    logger.info("Loaded %d segments", len(segments))

    scales       = _global_scales(segments)
    full_vectors = np.array([segment_to_full_vector(s, scales) for s in segments])
    ode_vectors  = np.array([segment_to_ode_vector(s, scales)  for s in segments])

    # Build training dataset
    seg_targets, seg_masks = build_dataset(ode_vectors)
    logger.info("Dataset: %d positions  K_MAX=%d", len(ode_vectors), K_MAX)

    model = SegmentForecaster()
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    logger.info("Training epochs=%d", EPOCHS)
    t0 = time.perf_counter()
    losses = train(model, full_vectors, seg_targets, seg_masks, device)
    logger.info("Training done  elapsed=%.1fs  final_loss=%.4f",
                time.perf_counter() - t0, losses[-1])
    torch.save(model.state_dict(), MODEL_PATH)

    # ── Forecast: February 1, T=24h ──
    logger.info("=== FORECAST: Feb 1 2026, T=24h ===")
    last_seg = segments[-1]
    x0 = float(last_seg.x0)
    feb1 = pd.Timestamp("2026-02-01 00:00")

    predicted, K = forecast(model, full_vectors, 24, feb1, segments, scales, device)
    for i, (fit, dur) in enumerate(predicted):
        logger.info("  seg %d: eq=%s  dur=%dh  L=%.2f  A=%.2f  B=%.2f",
                    i + 1, fit.eq_type, dur, fit.L, fit.A, fit.B)

    pred_24h = decode_to_series(predicted, x0, feb1, 24)

    df = pd.read_csv("hly4935_subset.csv", skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    actual = df.set_index("date")["temp"]

    errors = []
    logger.info("%-22s  %6s  %6s  %6s", "timestamp", "actual", "pred", "error")
    for ts in pred_24h.index:
        act = float(actual.get(ts, float("nan")))
        pred = float(pred_24h.loc[ts])
        err = act - pred
        errors.append(err)
        logger.info("  %s  %6.2f  %6.2f  %+.2f", ts, act, pred, err)

    valid = [e for e in errors if not math.isnan(e)]
    mae_24  = float(np.mean(np.abs(valid)))
    rmse_24 = float(np.sqrt(np.mean(np.array(valid) ** 2)))
    logger.info("T=24h  MAE=%.3f°C  RMSE=%.3f°C", mae_24, rmse_24)

    # ── Forecast: first week, T=168h ──
    logger.info("=== FORECAST: Feb 1–7 2026, T=168h ===")
    predicted_w, K_w = forecast(model, full_vectors, 168, feb1, segments, scales, device)
    pred_week = decode_to_series(predicted_w, x0, feb1, 168)
    actual_week = actual.loc["2026-02-01":"2026-02-07"].dropna()
    common = pred_week.index.intersection(actual_week.index)
    if len(common):
        errs_w = actual_week.loc[common].values - pred_week.loc[common].values
        mae_w  = float(np.mean(np.abs(errs_w)))
        rmse_w = float(np.sqrt(np.mean(errs_w ** 2)))
        logger.info("T=168h  K=%d  points=%d  MAE=%.3f°C  RMSE=%.3f°C",
                    K_w, len(common), mae_w, rmse_w)

    result = {
        "n_segments": len(segments),
        "final_loss": round(float(losses[-1]), 4),
        "forecast_24h":  {"K": K,   "mae": round(mae_24,  3), "rmse": round(rmse_24, 3)},
    }
    Path("artifacts/temperature_two_pass_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info("Done. MAE(24h)=%.3f°C", mae_24)


if __name__ == "__main__":
    main()
