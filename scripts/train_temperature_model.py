"""Train sequence model on temperature segments and forecast February 2026.

Feature vector per segment (12 floats):
  [0..5]  equation type one-hot: constant, linear, exponential,
                                  harmonic, linear_harmonic, damped_harmonic
  [6]     L  (normalized by global temp std)
  [7]     c  (normalized)
  [8]     lam
  [9]     A  (normalized)
  [10]    B  (normalized)
  [11]    duration_hours / 24

Model: 1-layer GRU, history=14 segments → predict next 1 segment (autoregressive).
Inference: seed with last 14 January segments, predict until February is covered.
Decode: chain predicted equations using x0 = last value of previous segment.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_patterns.pattern.segmentation_temperature import (
    EQ_CONSTANT, EQ_LINEAR, EQ_EXPONENTIAL,
    EQ_HARMONIC, EQ_LINEAR_HARMONIC, EQ_DAMPED_HARMONIC,
    TRY_ORDER, TemperatureSegment, TemperatureFit,
    _predict_value, build_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/temperature_model.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

REPORT_PATH    = Path("artifacts/temperature_segmentation_report.json")
SEGMENTS_PATH  = Path("artifacts/temperature_segments.json")
MODEL_PATH     = Path("artifacts/temperature_gru.pt")

HISTORY_LEN    = 14     # segments of history
FEAT_DIM       = 12     # feature vector size per segment
HIDDEN_DIM     = 64
N_LAYERS       = 1
LR             = 1e-3
EPOCHS         = 300
BATCH_SIZE     = 32
FEB_HOURS      = 672    # February 2026


# ── Feature encoding / decoding ────────────────────────────────────────────────

EQ_IDX = {eq: i for i, eq in enumerate(TRY_ORDER)}   # 6 types

def _global_scales(segments: list[TemperatureSegment]) -> dict[str, float]:
    Ls   = [abs(s.fit.L) for s in segments if abs(s.fit.L) > 1e-6]
    As   = [abs(s.fit.A) for s in segments if abs(s.fit.A) > 1e-6]
    Bs   = [abs(s.fit.B) for s in segments if abs(s.fit.B) > 1e-6]
    cs   = [abs(s.fit.c) for s in segments if abs(s.fit.c) > 1e-6]
    return {
        "L":  float(np.percentile(Ls, 90)) if Ls else 1.0,
        "A":  float(np.percentile(As, 90)) if As else 1.0,
        "B":  float(np.percentile(Bs, 90)) if Bs else 1.0,
        "c":  float(np.percentile(cs, 90)) if cs else 1.0,
    }

def segment_to_vector(seg: TemperatureSegment, scales: dict) -> np.ndarray:
    v = np.zeros(FEAT_DIM, dtype=np.float32)
    v[EQ_IDX[seg.fit.eq_type]] = 1.0
    v[6]  = seg.fit.L   / scales["L"]
    v[7]  = seg.fit.c   / scales["c"]
    v[8]  = seg.fit.lam
    v[9]  = seg.fit.A   / scales["A"]
    v[10] = seg.fit.B   / scales["B"]
    v[11] = seg.duration_hours / 24.0
    return v

def vector_to_fit(v: np.ndarray, scales: dict) -> tuple[TemperatureFit, int]:
    eq_idx  = int(np.argmax(v[:6]))
    eq_type = TRY_ORDER[eq_idx]
    L   = float(v[6])  * scales["L"]
    c   = float(v[7])  * scales["c"]
    lam = float(max(v[8], 0.0))
    A   = float(v[9])  * scales["A"]
    B   = float(v[10]) * scales["B"]
    dur = max(1, int(round(float(v[11]) * 24)))
    fit = TemperatureFit(eq_type=eq_type, L=L, c=c, lam=lam, A=A, B=B)
    return fit, dur


# ── Model ──────────────────────────────────────────────────────────────────────

class TemperatureGRU(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.gru  = nn.GRU(feat_dim, hidden_dim, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])   # predict next segment from last hidden state


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_dataset(
    segments: list[TemperatureSegment],
    scales: dict,
    history_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = np.array([segment_to_vector(s, scales) for s in segments])
    X, Y = [], []
    for i in range(history_len, len(vectors)):
        X.append(vectors[i - history_len : i])
        Y.append(vectors[i])
    return torch.tensor(np.array(X), dtype=torch.float32), \
           torch.tensor(np.array(Y), dtype=torch.float32)


# ── Training ───────────────────────────────────────────────────────────────────

def train(model: TemperatureGRU, X: torch.Tensor, Y: torch.Tensor, device: str) -> list[float]:
    model.to(device)
    X, Y = X.to(device), Y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        avg = epoch_loss / len(X)
        losses.append(avg)
        if (epoch + 1) % 50 == 0:
            logger.info("epoch=%d/%d loss=%.6f", epoch + 1, EPOCHS, avg)
    return losses


# ── Inference ──────────────────────────────────────────────────────────────────

def forecast_february(
    model: TemperatureGRU,
    seed_segments: list[TemperatureSegment],
    scales: dict,
    feb_hours: int,
    device: str,
) -> list[tuple[TemperatureFit, int, float]]:
    """Autoregressively predict segments until feb_hours are covered.

    Returns list of (fit, duration_hours, x0) for each predicted segment.
    """
    model.eval()
    vectors = [segment_to_vector(s, scales) for s in seed_segments[-HISTORY_LEN:]]
    history = list(vectors)

    # x0 of first predicted segment = last actual temperature value
    last_x0 = float(seed_segments[-1].x0)
    last_fit = seed_segments[-1].fit
    last_dur = seed_segments[-1].duration_hours
    prev_x0 = _predict_value(last_fit, float(last_dur - 1))

    predicted: list[tuple[TemperatureFit, int, float]] = []
    covered = 0

    with torch.no_grad():
        while covered < feb_hours:
            x = torch.tensor(np.array(history[-HISTORY_LEN:]), dtype=torch.float32) \
                    .unsqueeze(0).to(device)
            pred_vec = model(x).squeeze(0).cpu().numpy()
            fit, dur = vector_to_fit(pred_vec, scales)

            predicted.append((fit, dur, prev_x0))
            history.append(pred_vec)
            covered += dur
            # x0 of next segment = last predicted value of current segment
            prev_x0 = _predict_value(fit, float(dur - 1))
            # Un-normalize back using prev_x0 as offset anchor
            # (L is relative in normalised space; add actual x0 offset)

    return predicted


def decode_to_series(
    predicted: list[tuple[TemperatureFit, int, float]],
    start_time: pd.Timestamp,
) -> pd.Series:
    timestamps, values = [], []
    current_time = start_time
    for fit, dur, x0 in predicted:
        for step in range(dur):
            t_val = _predict_value(fit, float(step))
            # Shift so that t=0 matches x0
            shift = x0 - _predict_value(fit, 0.0)
            value = t_val + shift
            timestamps.append(current_time + pd.Timedelta(hours=step))
            values.append(value)
        current_time += pd.Timedelta(hours=dur)
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="temperature_predicted")


# ── Main ───────────────────────────────────────────────────────────────────────

def load_segments() -> list[TemperatureSegment]:
    with open(SEGMENTS_PATH, encoding="utf-8") as f:
        records = json.load(f)
    segments = []
    for r in records:
        fit = TemperatureFit(**r["fit"])
        seg = TemperatureSegment(
            segment_id=r["segment_id"],
            start_index=r["start_index"],
            end_index=r["end_index"],
            start_time=pd.Timestamp(r["start_time"]),
            end_time=pd.Timestamp(r["end_time"]),
            duration_hours=r["duration_hours"],
            fit=fit,
            x0=r["x0"],
        )
        segments.append(seg)
    return segments


def load_actual_february(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()
    series = df["temp"].loc["2026-02-01":"2026-02-28"].dropna()
    return series


def main():
    Path("artifacts").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    # --- Load segments ---
    logger.info("Loading segments from %s", SEGMENTS_PATH)
    segments = load_segments()
    logger.info("Loaded %d segments", len(segments))

    scales = _global_scales(segments)
    logger.info("Scales: %s", {k: f"{v:.3f}" for k, v in scales.items()})

    # --- Build dataset ---
    X, Y = build_dataset(segments, scales, HISTORY_LEN)
    logger.info("Dataset: X=%s Y=%s", tuple(X.shape), tuple(Y.shape))

    # --- Train ---
    model = TemperatureGRU(FEAT_DIM, HIDDEN_DIM, N_LAYERS)
    logger.info("Training GRU hidden=%d layers=%d epochs=%d", HIDDEN_DIM, N_LAYERS, EPOCHS)
    t0 = time.perf_counter()
    losses = train(model, X, Y, device)
    logger.info("Training complete elapsed_s=%.1f final_loss=%.6f", time.perf_counter() - t0, losses[-1])
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    # --- Forecast February ---
    logger.info("=== FORECASTING FEBRUARY 2026 ===")
    # Seed: last HISTORY_LEN segments before February (all from January)
    jan_segments = [s for s in segments if s.start_time < pd.Timestamp("2026-02-01")]
    seed = jan_segments[-HISTORY_LEN:]
    logger.info("Seed: last %d January segments, last ends at %s",
                len(seed), seed[-1].end_time)

    predicted = forecast_february(model, seed, scales, FEB_HOURS, device)
    logger.info("Predicted %d segments covering %d hours",
                len(predicted), sum(d for _, d, _ in predicted))

    feb_start = pd.Timestamp("2026-02-01 00:00")
    pred_series = decode_to_series(predicted, feb_start).loc["2026-02-01":"2026-02-28"]

    # --- Load actual ---
    actual = load_actual_february(Path("hly4935_subset.csv"))
    common_idx = pred_series.index.intersection(actual.index)
    pred_aligned = pred_series.loc[common_idx]
    actual_aligned = actual.loc[common_idx]

    errors = actual_aligned.values - pred_aligned.values
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    logger.info("=== FEBRUARY RESULTS ===")
    logger.info("Points compared: %d", len(common_idx))
    logger.info("MAE:  %.3f°C", mae)
    logger.info("RMSE: %.3f°C", rmse)
    logger.info("--- First 24 hours ---")
    for i in range(min(24, len(common_idx))):
        ts = common_idx[i]
        logger.info("  %s  actual=%5.2f  pred=%5.2f  err=%+.2f",
                    ts, float(actual_aligned.iloc[i]),
                    float(pred_aligned.iloc[i]), float(errors[i]))

    # --- Save results ---
    result = {
        "n_segments_train": len(segments),
        "final_loss": float(losses[-1]),
        "february_mae_celsius": round(mae, 3),
        "february_rmse_celsius": round(rmse, 3),
        "february_points": len(common_idx),
        "predicted_segment_count": len(predicted),
        "predicted_eq_distribution": {},
    }
    for eq in TRY_ORDER:
        count = sum(1 for fit, _, _ in predicted if fit.eq_type == eq)
        if count:
            result["predicted_eq_distribution"][eq] = count

    out = Path("artifacts/temperature_model_report.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Report saved to %s", out)


if __name__ == "__main__":
    main()
