"""Joint three-channel (temperature + pressure + wind speed) sequence forecaster.

Input:  segments/joint_segments.json  (built by run_joint_segmentation.py)
Output: artifacts/joint_two_pass.pt

Each pattern vector:
  [temp_onehot(6), temp_L, temp_c, temp_lam, temp_A, temp_B,   # 11
   pres_onehot(3), pres_L, pres_c, pres_lam, pres_A,           # 7
   wind_onehot(3), wind_L, wind_c, wind_lam, wind_A,           # 7
   duration/24,                                                  # 1
   sin/cos day-of-year, sin/cos hour-of-day]                    # 4
  = 30 dims total

K is derived from history: for query date + horizon T, count how many
joint patterns historically covered T hours on this day-of-year (±7 days).
Model predicts K_MAX patterns in one forward pass; first K are used.
end_time = start_time + duration_hours (reconstructed at decode time).
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
    TRY_ORDER as TEMP_EQ_ORDER, _predict_value as temp_predict,
    TemperatureFit,
)
from weather_patterns.pattern.segmentation_pressure import (
    TRY_ORDER as PRES_EQ_ORDER, _predict_value as pres_predict,
    PressureFit,
)
from weather_patterns.pattern.segmentation_windspeed import (
    TRY_ORDER as WIND_EQ_ORDER, _predict_value as wind_predict,
    WindspeedFit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/joint_two_pass.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SEGMENTS_PATH = Path("segments/joint_segments.json")
MODEL_PATH    = Path("artifacts/joint_two_pass.pt")

# Feature dimensions
TEMP_EQ_N  = len(TEMP_EQ_ORDER)   # 6
PRES_EQ_N  = len(PRES_EQ_ORDER)   # 3
WIND_EQ_N  = len(WIND_EQ_ORDER)   # 3
TEMP_PARAM = 5   # L, c, lam, A, B
PRES_PARAM = 4   # L, c, lam, A
WIND_PARAM = 4   # L, c, lam, A
ODE_DIM = TEMP_EQ_N + TEMP_PARAM + PRES_EQ_N + PRES_PARAM + WIND_EQ_N + WIND_PARAM + 1  # +1 dur = 26
TIME_DIM = 4
FEAT_DIM = ODE_DIM + TIME_DIM   # 30

K_MAX    = 48
D_MODEL  = 128
N_HEAD   = 4
N_LAYERS = 3
DIM_FF   = 256
DROPOUT  = 0.1
LR       = 3e-4
EPOCHS   = 600

TEMP_EQ_IDX = {eq: i for i, eq in enumerate(TEMP_EQ_ORDER)}
PRES_EQ_IDX = {eq: i for i, eq in enumerate(PRES_EQ_ORDER)}
WIND_EQ_IDX = {eq: i for i, eq in enumerate(WIND_EQ_ORDER)}


# ── Normalisation scales ───────────────────────────────────────────────────────

def _global_scales(segments: list[dict]) -> dict:
    def p90(vals):
        return float(np.percentile(vals, 90)) if vals else 1.0

    return {
        "temp_L":   p90([abs(s["temp_fit"]["L"])   for s in segments if abs(s["temp_fit"]["L"])   > 1e-6]),
        "temp_c":   p90([abs(s["temp_fit"]["c"])   for s in segments if abs(s["temp_fit"]["c"])   > 1e-6]),
        "temp_lam": p90([abs(s["temp_fit"]["lam"]) for s in segments if abs(s["temp_fit"]["lam"]) > 1e-6]),
        "temp_A":   p90([abs(s["temp_fit"]["A"])   for s in segments if abs(s["temp_fit"]["A"])   > 1e-6]),
        "temp_B":   p90([abs(s["temp_fit"]["B"])   for s in segments if abs(s["temp_fit"]["B"])   > 1e-6]),
        "pres_L":   p90([abs(s["pres_fit"]["L"])   for s in segments if abs(s["pres_fit"]["L"])   > 1e-6]),
        "pres_c":   p90([abs(s["pres_fit"]["c"])   for s in segments if abs(s["pres_fit"]["c"])   > 1e-6]),
        "pres_lam": p90([abs(s["pres_fit"]["lam"]) for s in segments if abs(s["pres_fit"]["lam"]) > 1e-6]),
        "pres_A":   p90([abs(s["pres_fit"]["A"])   for s in segments if abs(s["pres_fit"]["A"])   > 1e-6]),
        "wind_L":   p90([abs(s["wind_fit"]["L"])   for s in segments if abs(s["wind_fit"]["L"])   > 1e-6]),
        "wind_c":   p90([abs(s["wind_fit"]["c"])   for s in segments if abs(s["wind_fit"]["c"])   > 1e-6]),
        "wind_lam": p90([abs(s["wind_fit"]["lam"]) for s in segments if abs(s["wind_fit"]["lam"]) > 1e-6]),
        "wind_A":   p90([abs(s["wind_fit"]["A"])   for s in segments if abs(s["wind_fit"]["A"])   > 1e-6]),
    }


# ── Feature vectors ────────────────────────────────────────────────────────────

def _time_features(ts: pd.Timestamp) -> np.ndarray:
    doy = ts.day_of_year / 365.0
    hod = ts.hour / 24.0
    return np.array([
        math.sin(2 * math.pi * doy), math.cos(2 * math.pi * doy),
        math.sin(2 * math.pi * hod), math.cos(2 * math.pi * hod),
    ], dtype=np.float32)


def segment_to_ode_vector(seg: dict, sc: dict) -> np.ndarray:
    v = np.zeros(ODE_DIM, dtype=np.float32)
    tf = seg["temp_fit"]
    pf = seg["pres_fit"]
    wf = seg["wind_fit"]

    # Temperature [0..10]
    v[TEMP_EQ_IDX[tf["eq_type"]]] = 1.0
    b = TEMP_EQ_N
    v[b+0] = tf["L"]   / sc["temp_L"]
    v[b+1] = tf["c"]   / sc["temp_c"]
    v[b+2] = tf["lam"] / sc["temp_lam"]
    v[b+3] = tf["A"]   / sc["temp_A"]
    v[b+4] = tf["B"]   / sc["temp_B"]

    # Pressure [11..17]
    b = TEMP_EQ_N + TEMP_PARAM
    v[b + PRES_EQ_IDX[pf["eq_type"]]] = 1.0
    b += PRES_EQ_N
    v[b+0] = pf["L"]   / sc["pres_L"]
    v[b+1] = pf["c"]   / sc["pres_c"]
    v[b+2] = pf["lam"] / sc["pres_lam"]
    v[b+3] = pf["A"]   / sc["pres_A"]

    # Wind speed [18..24]
    b = TEMP_EQ_N + TEMP_PARAM + PRES_EQ_N + PRES_PARAM
    v[b + WIND_EQ_IDX[wf["eq_type"]]] = 1.0
    b += WIND_EQ_N
    v[b+0] = wf["L"]   / sc["wind_L"]
    v[b+1] = wf["c"]   / sc["wind_c"]
    v[b+2] = wf["lam"] / sc["wind_lam"]
    v[b+3] = wf["A"]   / sc["wind_A"]

    # Duration [25]
    v[ODE_DIM - 1] = seg["duration_hours"] / 24.0
    return v


def segment_to_full_vector(seg: dict, sc: dict) -> np.ndarray:
    return np.concatenate([
        segment_to_ode_vector(seg, sc),
        _time_features(pd.Timestamp(seg["start_time"])),
    ])


def vector_to_fits(v: np.ndarray, sc: dict) -> tuple[TemperatureFit, PressureFit, WindspeedFit, int]:
    # Temperature
    temp_eq = TEMP_EQ_ORDER[int(np.argmax(v[:TEMP_EQ_N]))]
    b = TEMP_EQ_N
    temp_fit = TemperatureFit(
        eq_type=temp_eq,
        L   = float(v[b+0]) * sc["temp_L"],
        c   = float(v[b+1]) * sc["temp_c"],
        lam = float(max(v[b+2], 0.0)) * sc["temp_lam"],
        A   = float(v[b+3]) * sc["temp_A"],
        B   = float(v[b+4]) * sc["temp_B"],
    )

    # Pressure
    b = TEMP_EQ_N + TEMP_PARAM
    pres_eq = PRES_EQ_ORDER[int(np.argmax(v[b: b + PRES_EQ_N]))]
    b += PRES_EQ_N
    pres_fit = PressureFit(
        eq_type=pres_eq,
        L   = float(v[b+0]) * sc["pres_L"],
        c   = float(v[b+1]) * sc["pres_c"],
        lam = float(max(v[b+2], 0.0)) * sc["pres_lam"],
        A   = float(v[b+3]) * sc["pres_A"],
    )

    # Wind
    b = TEMP_EQ_N + TEMP_PARAM + PRES_EQ_N + PRES_PARAM
    wind_eq = WIND_EQ_ORDER[int(np.argmax(v[b: b + WIND_EQ_N]))]
    b += WIND_EQ_N
    wind_fit = WindspeedFit(
        eq_type=wind_eq,
        L   = float(v[b+0]) * sc["wind_L"],
        c   = float(v[b+1]) * sc["wind_c"],
        lam = float(max(v[b+2], 0.0)) * sc["wind_lam"],
        A   = float(v[b+3]) * sc["wind_A"],
    )

    dur = max(1, int(round(float(v[ODE_DIM - 1]) * 24)))
    return temp_fit, pres_fit, wind_fit, dur


# ── Historical K ───────────────────────────────────────────────────────────────

def compute_K_from_history(
    segments: list[dict],
    query_date: pd.Timestamp,
    T_hours: float,
    window_days: int = 7,
) -> int:
    query_doy = query_date.day_of_year
    years = sorted({pd.Timestamp(s["start_time"]).year for s in segments})
    k_obs = []

    for year in years:
        candidates = [
            i for i, s in enumerate(segments)
            if pd.Timestamp(s["start_time"]).year == year
            and abs(pd.Timestamp(s["start_time"]).day_of_year - query_doy) <= window_days
        ]
        if not candidates:
            continue
        anchor = min(candidates,
                     key=lambda i: abs(pd.Timestamp(segments[i]["start_time"]).day_of_year - query_doy))
        cumulative = 0
        k = 0
        for j in range(anchor, len(segments)):
            cumulative += segments[j]["duration_hours"]
            k += 1
            if cumulative >= T_hours or k >= K_MAX:
                break
        if k > 0:
            k_obs.append(k)

    if not k_obs:
        return max(1, round(T_hours / 8.0))
    return max(1, min(K_MAX, round(float(np.mean(k_obs)))))


# ── Model ──────────────────────────────────────────────────────────────────────

class JointForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(FEAT_DIM, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEAD, DIM_FF, DROPOUT, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2),
            nn.ReLU(),
            nn.Linear(D_MODEL * 2, K_MAX * ODE_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)
        h = self.encoder(self.input_proj(x), mask=mask)[0]
        return self.head(h).view(-1, K_MAX, ODE_DIM)


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_dataset(ode_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = len(ode_vectors)
    targets = np.zeros((N, K_MAX, ODE_DIM), dtype=np.float32)
    masks   = np.zeros((N, K_MAX), dtype=bool)
    for i in range(N - 1):
        available = min(K_MAX, N - 1 - i)
        targets[i, :available] = ode_vectors[i + 1: i + 1 + available]
        masks[i, :available]   = True
    return targets, masks


# ── Training ───────────────────────────────────────────────────────────────────

def train(model, full_vectors, targets, masks, device) -> list[float]:
    model.to(device)
    X         = torch.tensor(full_vectors, dtype=torch.float32).unsqueeze(0).to(device)
    targets_t = torch.tensor(targets,      dtype=torch.float32).to(device)
    masks_t   = torch.tensor(masks,        dtype=torch.bool).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred     = model(X)
        mask_exp = masks_t.unsqueeze(-1).expand_as(pred)
        loss     = ((pred - targets_t) ** 2)[mask_exp].mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % 50 == 0:
            logger.info("epoch=%d/%d  loss=%.4f  lr=%.2e",
                        epoch + 1, EPOCHS, losses[-1], scheduler.get_last_lr()[0])

    return losses


# ── Inference ──────────────────────────────────────────────────────────────────

def forecast(model, full_vectors, T_hours, query_date, segments, sc, device):
    K = compute_K_from_history(segments, query_date, T_hours)
    logger.info("T=%dh  query=%s  K=%d (from history)",
                int(T_hours), query_date.strftime("%m-%d"), K)

    model.eval()
    with torch.no_grad():
        X = torch.tensor(full_vectors, dtype=torch.float32).unsqueeze(0).to(device)
        pred_np = model(X)[-1, :K].cpu().numpy()

    return [vector_to_fits(v, sc) for v in pred_np], K


def decode_to_series(
    predicted: list,
    temp_x0: float, pres_x0: float, wind_x0: float,
    start_time: pd.Timestamp,
    n_hours: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    temp_ts, temp_vals = [], []
    pres_ts, pres_vals = [], []
    wind_ts, wind_vals = [], []
    current_time = start_time
    cur_temp = temp_x0
    cur_pres = pres_x0
    cur_wind = wind_x0

    for tf, pf, wf, dur in predicted:
        temp_shift = cur_temp - temp_predict(tf, 0.0)
        pres_shift = cur_pres - pres_predict(pf, 0.0)
        wind_shift = cur_wind - wind_predict(wf, 0.0)

        for step in range(dur):
            if len(temp_ts) >= n_hours:
                break
            ts = current_time + pd.Timedelta(hours=step)
            temp_ts.append(ts); temp_vals.append(temp_predict(tf, float(step)) + temp_shift)
            pres_ts.append(ts); pres_vals.append(pres_predict(pf, float(step)) + pres_shift)
            wind_ts.append(ts); wind_vals.append(max(0.0, wind_predict(wf, float(step)) + wind_shift))

        if len(temp_ts) >= n_hours:
            break
        cur_temp = temp_predict(tf, float(dur - 1)) + temp_shift
        cur_pres = pres_predict(pf, float(dur - 1)) + pres_shift
        cur_wind = max(0.0, wind_predict(wf, float(dur - 1)) + wind_shift)
        current_time += pd.Timedelta(hours=dur)

    idx = pd.DatetimeIndex(temp_ts)
    return (
        pd.Series(temp_vals, index=idx),
        pd.Series(pres_vals, index=idx),
        pd.Series(wind_vals, index=idx),
    )


# ── Eval helper ────────────────────────────────────────────────────────────────

def _mae_rmse(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    errs = []
    for ts in pred.index:
        act = float(actual.get(ts, float("nan")))
        if math.isfinite(act):
            errs.append(act - float(pred.loc[ts]))
    if not errs:
        return float("nan"), float("nan")
    a = np.array(errs)
    return float(np.mean(np.abs(a))), float(np.sqrt(np.mean(a ** 2)))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    Path("artifacts").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    segments = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d joint segments", len(segments))

    sc = _global_scales(segments)
    logger.info("Scales: %s", {k: f"{v:.4f}" for k, v in sc.items()})

    full_vectors = np.array([segment_to_full_vector(s, sc) for s in segments])
    ode_vectors  = np.array([segment_to_ode_vector(s, sc)  for s in segments])
    logger.info("Feature dim: %d  ODE dim: %d  K_MAX: %d", FEAT_DIM, ODE_DIM, K_MAX)

    targets, masks = build_dataset(ode_vectors)
    logger.info("Dataset: %d positions", len(ode_vectors))

    model = JointForecaster()
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    logger.info("Training epochs=%d …", EPOCHS)
    t0 = time.perf_counter()
    losses = train(model, full_vectors, targets, masks, device)
    logger.info("Training done  elapsed=%.1fs  final_loss=%.4f",
                time.perf_counter() - t0, losses[-1])
    torch.save(model.state_dict(), MODEL_PATH)

    # ── Load actuals ──
    df = pd.read_csv("hly4935_subset.csv", skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()

    last_seg = segments[-1]
    temp_x0  = float(last_seg["temp_x0"])
    pres_x0  = float(last_seg["pres_x0"])
    wind_x0  = float(last_seg["wind_x0"])
    feb1     = pd.Timestamp("2026-02-01 00:00")

    for T, label in [(24, "24h"), (168, "168h")]:
        logger.info("=== FORECAST: Feb 1 2026, T=%s ===", label)
        predicted, K = forecast(model, full_vectors, T, feb1, segments, sc, device)
        pred_temp, pred_pres, pred_wind = decode_to_series(
            predicted, temp_x0, pres_x0, wind_x0, feb1, T
        )

        mae_t, rmse_t = _mae_rmse(df["temp"], pred_temp)
        mae_p, rmse_p = _mae_rmse(df["msl"],  pred_pres)
        mae_w, rmse_w = _mae_rmse(df["wdsp"], pred_wind)

        if T == 24:
            logger.info("%-20s  %6s %6s %6s  |  %7s %7s  |  %6s %6s",
                        "timestamp", "aT", "pT", "eT", "aP", "pP", "aW", "pW")
            for ts in pred_temp.index:
                at = float(df["temp"].get(ts, float("nan")))
                ap = float(df["msl"].get(ts,  float("nan")))
                aw = float(df["wdsp"].get(ts, float("nan")))
                logger.info("  %s  %6.1f %6.1f %+.2f  |  %7.1f %7.1f  |  %6.1f %6.1f",
                            ts, at, float(pred_temp.loc[ts]), at - float(pred_temp.loc[ts]),
                            ap, float(pred_pres.loc[ts]),
                            aw, float(pred_wind.loc[ts]))

        logger.info("T=%s K=%d | temp MAE=%.3f°C RMSE=%.3f°C | pres MAE=%.3f hPa RMSE=%.3f hPa | wind MAE=%.3f kt RMSE=%.3f kt",
                    label, K, mae_t, rmse_t, mae_p, rmse_p, mae_w, rmse_w)

    Path("artifacts/joint_two_pass_report.json").write_text(
        json.dumps({"n_segments": len(segments), "final_loss": round(float(losses[-1]), 4)},
                   indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
