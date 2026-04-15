"""Joint three-channel (temperature + pressure + wind speed) sequence forecaster.

Training approach: symmetric 90-day window centred on the forecast point.
  - Tail (input):  segments in [center - HALF_WINDOW_DAYS, center)
  - Head (target): segments in [center, center + HALF_WINDOW_DAYS)
  - Center shifts by 1 hour across all valid positions in training data.
  - Both tail and head are zero-padded to MAX_SEQ_LEN.

To experiment with different forecast horizons change HALF_WINDOW_DAYS.

Input:  segments/joint_segments.json
Output: artifacts/joint_two_pass.pt
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

# ── Window config (change this to experiment with different horizons) ──────────
HALF_WINDOW_DAYS = 45          # tail = 45 days back, head = 45 days forward
HALF_WINDOW_H    = HALF_WINDOW_DAYS * 24   # 1080 hours
MAX_SEQ_LEN      = 256         # pad tail and head to this length
CENTER_STEP_H    = 1           # shift center by 1 hour per training sample

# ── Feature dimensions ─────────────────────────────────────────────────────────
TEMP_EQ_N  = len(TEMP_EQ_ORDER)   # 6
PRES_EQ_N  = len(PRES_EQ_ORDER)   # 3
WIND_EQ_N  = len(WIND_EQ_ORDER)   # 3
TEMP_PARAM = 5   # L, c, lam, A, B
PRES_PARAM = 4   # L, c, lam, A
WIND_PARAM = 4   # L, c, lam, A
ODE_DIM = TEMP_EQ_N + TEMP_PARAM + PRES_EQ_N + PRES_PARAM + WIND_EQ_N + WIND_PARAM + 1  # 26
TIME_DIM = 4
FEAT_DIM = ODE_DIM + TIME_DIM   # 30

BATCH_SIZE = 32
D_MODEL    = 128
N_HEAD     = 4
N_LAYERS   = 3
DIM_FF     = 256
DROPOUT    = 0.1
LR         = 3e-4
EPOCHS     = 100

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


def segment_to_vector(seg: dict, sc: dict) -> np.ndarray:
    v = np.zeros(FEAT_DIM, dtype=np.float32)
    tf, pf, wf = seg["temp_fit"], seg["pres_fit"], seg["wind_fit"]

    # Temperature one-hot + params [0..10]
    v[TEMP_EQ_IDX[tf["eq_type"]]] = 1.0
    b = TEMP_EQ_N
    v[b+0]=tf["L"]/sc["temp_L"]; v[b+1]=tf["c"]/sc["temp_c"]
    v[b+2]=tf["lam"]/sc["temp_lam"]; v[b+3]=tf["A"]/sc["temp_A"]; v[b+4]=tf["B"]/sc["temp_B"]

    # Pressure one-hot + params [11..17]
    b = TEMP_EQ_N + TEMP_PARAM
    v[b + PRES_EQ_IDX[pf["eq_type"]]] = 1.0
    b += PRES_EQ_N
    v[b+0]=pf["L"]/sc["pres_L"]; v[b+1]=pf["c"]/sc["pres_c"]
    v[b+2]=pf["lam"]/sc["pres_lam"]; v[b+3]=pf["A"]/sc["pres_A"]

    # Wind one-hot + params [18..24]
    b = TEMP_EQ_N + TEMP_PARAM + PRES_EQ_N + PRES_PARAM
    v[b + WIND_EQ_IDX[wf["eq_type"]]] = 1.0
    b += WIND_EQ_N
    v[b+0]=wf["L"]/sc["wind_L"]; v[b+1]=wf["c"]/sc["wind_c"]
    v[b+2]=wf["lam"]/sc["wind_lam"]; v[b+3]=wf["A"]/sc["wind_A"]

    # Duration [25]
    v[ODE_DIM - 1] = seg["duration_hours"] / 24.0

    # Time features [26..29]
    v[ODE_DIM:] = _time_features(pd.Timestamp(seg["start_time"]))
    return v


def vector_to_fits(v: np.ndarray, sc: dict) -> tuple[TemperatureFit, PressureFit, WindspeedFit, int]:
    temp_eq = TEMP_EQ_ORDER[int(np.argmax(v[:TEMP_EQ_N]))]
    b = TEMP_EQ_N
    temp_fit = TemperatureFit(eq_type=temp_eq,
        L=float(v[b+0])*sc["temp_L"], c=float(v[b+1])*sc["temp_c"],
        lam=float(max(v[b+2],0))*sc["temp_lam"],
        A=float(v[b+3])*sc["temp_A"], B=float(v[b+4])*sc["temp_B"])

    b = TEMP_EQ_N + TEMP_PARAM
    pres_eq = PRES_EQ_ORDER[int(np.argmax(v[b:b+PRES_EQ_N]))]
    b += PRES_EQ_N
    pres_fit = PressureFit(eq_type=pres_eq,
        L=float(v[b+0])*sc["pres_L"], c=float(v[b+1])*sc["pres_c"],
        lam=float(max(v[b+2],0))*sc["pres_lam"], A=float(v[b+3])*sc["pres_A"])

    b = TEMP_EQ_N + TEMP_PARAM + PRES_EQ_N + PRES_PARAM
    wind_eq = WIND_EQ_ORDER[int(np.argmax(v[b:b+WIND_EQ_N]))]
    b += WIND_EQ_N
    wind_fit = WindspeedFit(eq_type=wind_eq,
        L=float(v[b+0])*sc["wind_L"], c=float(v[b+1])*sc["wind_c"],
        lam=float(max(v[b+2],0))*sc["wind_lam"], A=float(v[b+3])*sc["wind_A"])

    dur = max(1, int(round(float(v[ODE_DIM - 1]) * 24)))
    return temp_fit, pres_fit, wind_fit, dur


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_dataset(segments: list[dict], sc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each valid center hour build:
      tail_seqs[i]: segments in [center - HALF_WINDOW_H, center), padded to MAX_SEQ_LEN
      head_seqs[i]: segments in [center, center + HALF_WINDOW_H), padded to MAX_SEQ_LEN
      tail_masks[i]: bool mask — True where real segment, False where padding

    Center shifts by CENTER_STEP_H hours across the full training range.
    """
    starts = np.array([pd.Timestamp(s["start_time"]).value for s in segments], dtype=np.int64)
    vectors = np.array([segment_to_vector(s, sc) for s in segments], dtype=np.float32)

    first_ts = pd.Timestamp(segments[0]["start_time"])
    last_ts  = pd.Timestamp(segments[-1]["start_time"])

    center_start = first_ts + pd.Timedelta(hours=HALF_WINDOW_H)
    center_end   = last_ts  - pd.Timedelta(hours=HALF_WINDOW_H)

    centers = pd.date_range(center_start, center_end, freq=f"{CENTER_STEP_H}h")
    n = len(centers)
    logger.info("Building dataset: %d centers  half_window=%dd  max_seq=%d",
                n, HALF_WINDOW_DAYS, MAX_SEQ_LEN)

    tail_seqs  = np.zeros((n, MAX_SEQ_LEN, FEAT_DIM), dtype=np.float32)
    head_seqs  = np.zeros((n, MAX_SEQ_LEN, FEAT_DIM), dtype=np.float32)
    tail_masks = np.zeros((n, MAX_SEQ_LEN), dtype=bool)
    head_masks = np.zeros((n, MAX_SEQ_LEN), dtype=bool)

    ns_per_h = int(pd.Timedelta(hours=1).value)

    for i, center in enumerate(centers):
        cv = center.value
        wv = HALF_WINDOW_H * ns_per_h

        tail_idx = np.where((starts >= cv - wv) & (starts < cv))[0]
        head_idx = np.where((starts >= cv)      & (starts < cv + wv))[0]

        t_len = min(len(tail_idx), MAX_SEQ_LEN)
        h_len = min(len(head_idx), MAX_SEQ_LEN)

        # fill from the END of the tail (most recent segments last)
        tail_seqs[i, MAX_SEQ_LEN - t_len:] = vectors[tail_idx[-t_len:]]
        tail_masks[i, MAX_SEQ_LEN - t_len:] = True

        head_seqs[i, :h_len] = vectors[head_idx[:h_len]]
        head_masks[i, :h_len] = True

        if (i + 1) % 10000 == 0:
            logger.info("  %d / %d centers processed", i + 1, n)

    logger.info("Dataset built: %d samples", n)
    return tail_seqs, head_seqs, tail_masks


# ── Model ──────────────────────────────────────────────────────────────────────

class JointForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(FEAT_DIM, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEAD, DIM_FF, DROPOUT, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        # head predicts the full head sequence: MAX_SEQ_LEN * FEAT_DIM
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 2),
            nn.ReLU(),
            nn.Linear(D_MODEL * 2, MAX_SEQ_LEN * FEAT_DIM),
        )

    def forward(self, tail: torch.Tensor, tail_mask: torch.Tensor) -> torch.Tensor:
        # tail: [B, MAX_SEQ_LEN, FEAT_DIM]
        # tail_mask: [B, MAX_SEQ_LEN] — True = real, False = padding
        # key_padding_mask for Transformer expects True = IGNORE
        pad_mask = ~tail_mask   # [B, MAX_SEQ_LEN]
        h = self.encoder(self.input_proj(tail), src_key_padding_mask=pad_mask)
        # use last real token: last non-padded position
        # since tail is right-aligned (padding on left), last position is always real
        last = h[:, -1, :]                                       # [B, D_MODEL]
        return self.head(last).view(-1, MAX_SEQ_LEN, FEAT_DIM)   # [B, MAX_SEQ_LEN, FEAT_DIM]


# ── Training ───────────────────────────────────────────────────────────────────

def train(model, tail_seqs, head_seqs, tail_masks, head_masks, device) -> list[float]:
    model.to(device)
    n = len(tail_seqs)

    tail_t  = torch.tensor(tail_seqs,  dtype=torch.float32)
    head_t  = torch.tensor(head_seqs,  dtype=torch.float32)
    tmask_t = torch.tensor(tail_masks, dtype=torch.bool)
    hmask_t = torch.tensor(head_masks, dtype=torch.bool)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n, BATCH_SIZE):
            idx = perm[start: start + BATCH_SIZE]
            tb  = tail_t[idx].to(device)
            hb  = head_t[idx].to(device)
            tm  = tmask_t[idx].to(device)
            hm  = hmask_t[idx].to(device)

            optimizer.zero_grad()
            pred = model(tb, tm)                        # [B, MAX_SEQ_LEN, FEAT_DIM]
            # loss only on real (non-padded) head positions
            mask_exp = hm.unsqueeze(-1).expand_as(pred)
            loss = ((pred - hb) ** 2)[mask_exp].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches  += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            logger.info("epoch=%d/%d  loss=%.4f  lr=%.2e",
                        epoch + 1, EPOCHS, avg, scheduler.get_last_lr()[0])

    return losses


# ── Inference ──────────────────────────────────────────────────────────────────

def forecast(model, segments: list[dict], sc: dict, center: pd.Timestamp, device):
    starts  = np.array([pd.Timestamp(s["start_time"]).value for s in segments], dtype=np.int64)
    vectors = np.array([segment_to_vector(s, sc) for s in segments], dtype=np.float32)

    cv  = center.value
    wv  = int(HALF_WINDOW_H * pd.Timedelta(hours=1).value)
    idx = np.where((starts >= cv - wv) & (starts < cv))[0]

    tail = np.zeros((1, MAX_SEQ_LEN, FEAT_DIM), dtype=np.float32)
    mask = np.zeros((1, MAX_SEQ_LEN), dtype=bool)
    t_len = min(len(idx), MAX_SEQ_LEN)
    tail[0, MAX_SEQ_LEN - t_len:] = vectors[idx[-t_len:]]
    mask[0, MAX_SEQ_LEN - t_len:] = True

    model.eval()
    with torch.no_grad():
        tail_t = torch.tensor(tail, dtype=torch.float32).to(device)
        mask_t = torch.tensor(mask, dtype=torch.bool).to(device)
        pred   = model(tail_t, mask_t)[0].cpu().numpy()  # [MAX_SEQ_LEN, FEAT_DIM]

    # decode only real (non-zero duration) predictions
    result = []
    for v in pred:
        tf, pf, wf, dur = vector_to_fits(v, sc)
        if dur > 0:
            result.append((tf, pf, wf, dur))
    return result


def decode_to_series(
    predicted: list,
    temp_x0: float, pres_x0: float, wind_x0: float,
    start_time: pd.Timestamp,
    n_hours: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    temp_ts, temp_vals, pres_vals, wind_vals = [], [], [], []
    current_time = start_time
    cur_temp, cur_pres, cur_wind = temp_x0, pres_x0, wind_x0

    for tf, pf, wf, dur in predicted:
        temp_shift = cur_temp - temp_predict(tf, 0.0)
        pres_shift = cur_pres - pres_predict(pf, 0.0)
        wind_shift = cur_wind - wind_predict(wf, 0.0)
        for step in range(dur):
            if len(temp_ts) >= n_hours:
                break
            ts = current_time + pd.Timedelta(hours=step)
            temp_ts.append(ts)
            temp_vals.append(temp_predict(tf, float(step)) + temp_shift)
            pres_vals.append(pres_predict(pf, float(step)) + pres_shift)
            wind_vals.append(max(0.0, wind_predict(wf, float(step)) + wind_shift))
        if len(temp_ts) >= n_hours:
            break
        cur_temp = temp_predict(tf, float(dur-1)) + temp_shift
        cur_pres = pres_predict(pf, float(dur-1)) + pres_shift
        cur_wind = max(0.0, wind_predict(wf, float(dur-1)) + wind_shift)
        current_time += pd.Timedelta(hours=dur)

    idx = pd.DatetimeIndex(temp_ts)
    return pd.Series(temp_vals, index=idx), pd.Series(pres_vals, index=idx), pd.Series(wind_vals, index=idx)


# ── Eval ───────────────────────────────────────────────────────────────────────

def _mae_rmse(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    errs = []
    for ts in pred.index:
        act = float(actual.get(ts, float("nan")))
        if math.isfinite(act):
            errs.append(act - float(pred.loc[ts]))
    if not errs:
        return float("nan"), float("nan")
    a = np.array(errs)
    return float(np.mean(np.abs(a))), float(np.sqrt(np.mean(a**2)))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    Path("artifacts").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s  half_window=%dd  max_seq=%d", device, HALF_WINDOW_DAYS, MAX_SEQ_LEN)

    segments = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d joint segments", len(segments))

    sc = _global_scales(segments)
    tail_seqs, head_seqs, tail_masks = build_dataset(segments, sc)

    # head_masks: True where head sequence has a real segment
    head_masks = (head_seqs.sum(axis=-1) != 0)

    model = JointForecaster()
    logger.info("Model params: %d  FEAT_DIM=%d  MAX_SEQ_LEN=%d",
                sum(p.numel() for p in model.parameters()), FEAT_DIM, MAX_SEQ_LEN)

    logger.info("Training epochs=%d  batch=%d …", EPOCHS, BATCH_SIZE)
    t0 = time.perf_counter()
    losses = train(model, tail_seqs, head_seqs, tail_masks, head_masks, device)
    logger.info("Training done  elapsed=%.1fs  final_loss=%.4f",
                time.perf_counter() - t0, losses[-1])
    torch.save(model.state_dict(), MODEL_PATH)

    # ── Evaluate: Feb 1 2026 ──
    df = pd.read_csv("hly4935_subset.csv", skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()

    feb1 = pd.Timestamp("2026-02-01 00:00")
    # x0 = last known values before feb1
    last_known = df.loc[:"2026-01-31 23:00"].iloc[-1]
    temp_x0 = float(last_known["temp"])
    pres_x0 = float(last_known["msl"])
    wind_x0 = float(last_known["wdsp"])

    for T, label in [(24, "24h"), (168, "168h"), (HALF_WINDOW_H, f"{HALF_WINDOW_DAYS}d")]:
        predicted = forecast(model, segments, sc, feb1, device)
        pred_temp, pred_pres, pred_wind = decode_to_series(
            predicted, temp_x0, pres_x0, wind_x0, feb1, T
        )
        mae_t, rmse_t = _mae_rmse(df["temp"], pred_temp)
        mae_p, rmse_p = _mae_rmse(df["msl"],  pred_pres)
        mae_w, rmse_w = _mae_rmse(df["wdsp"], pred_wind)
        logger.info("T=%-6s | temp MAE=%.3f°C RMSE=%.3f°C | pres MAE=%.3f hPa RMSE=%.3f hPa | wind MAE=%.3f kt RMSE=%.3f kt",
                    label, mae_t, rmse_t, mae_p, rmse_p, mae_w, rmse_w)

    Path("artifacts/joint_two_pass_report.json").write_text(
        json.dumps({"n_segments": len(segments), "half_window_days": HALF_WINDOW_DAYS,
                    "final_loss": round(float(losses[-1]), 4)}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
