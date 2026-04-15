"""Build joint two-channel segments from temperature + pressure.

Reuses already-computed segments (no re-segmentation of raw data):
  segments/temperature_segments.json
  segments/pressure_segments.json

Algorithm:
  1. Collect all start_time and end_time from both channels.
  2. Union → sort → deduplicate → shared time grid.
  3. For each interval [tᵢ, tᵢ₊₁], slice raw CSV and refit best
     equation independently for temperature and pressure.
  4. Save to segments/joint_segments.json.

end_time is not stored — it equals start_time + duration_hours and is
reconstructed at decode time.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_patterns.pattern.segmentation_temperature import (
    TemperatureFit, _best_fit as temp_best_fit,
)
from weather_patterns.pattern.segmentation_pressure import (
    PressureFit, _best_fit as pres_best_fit,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CSV_PATH       = Path("hly4935_subset.csv")
TEMP_SEGS_PATH = Path("segments/temperature_segments.json")
PRES_SEGS_PATH = Path("segments/pressure_segments.json")
OUT_PATH       = Path("segments/joint_segments.json")

TRAIN_START = "2020-01-01"
TRAIN_END   = "2026-01-31"


# ── Load CSV ───────────────────────────────────────────────────────────────────

def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()
    df = df.loc[TRAIN_START:TRAIN_END]
    logger.info("CSV loaded: %d rows, columns: %s", len(df), list(df.columns))
    return df


# ── Load segment timestamps ────────────────────────────────────────────────────

def load_timestamps(path: Path) -> list[pd.Timestamp]:
    records = json.loads(path.read_text(encoding="utf-8"))
    times = []
    for r in records:
        times.append(pd.Timestamp(r["start_time"]))
        times.append(pd.Timestamp(r["end_time"]))
    return times


# ── Refit helpers ──────────────────────────────────────────────────────────────

def fit_temp(df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> dict:
    slice_ = df["temp"].loc[t_start:t_end].dropna()
    if len(slice_) == 0:
        return {"eq_type": "constant", "L": 0.0, "c": 0.0, "lam": 0.0, "A": 0.0, "B": 0.0, "rms": 0.0, "n_points": 0}
    t = np.arange(len(slice_), dtype=float)
    y = slice_.values.astype(float)
    fit: TemperatureFit = temp_best_fit(t, y)
    return {
        "eq_type": fit.eq_type,
        "L": fit.L, "c": fit.c, "lam": fit.lam, "A": fit.A, "B": fit.B,
        "rms": fit.rms, "n_points": fit.n_points,
    }


def fit_pres(df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> dict:
    slice_ = df["msl"].loc[t_start:t_end].dropna()
    if len(slice_) == 0:
        return {"eq_type": "constant", "L": 0.0, "c": 0.0, "lam": 0.0, "A": 0.0, "rms": 0.0, "n_points": 0}
    t = np.arange(len(slice_), dtype=float)
    y = slice_.values.astype(float)
    fit: PressureFit = pres_best_fit(t, y)
    return {
        "eq_type": fit.eq_type,
        "L": fit.L, "c": fit.c, "lam": fit.lam, "A": fit.A,
        "rms": fit.rms, "n_points": fit.n_points,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    df = load_csv()

    logger.info("Loading segment timestamps …")
    temp_times = load_timestamps(TEMP_SEGS_PATH)
    pres_times = load_timestamps(PRES_SEGS_PATH)

    all_times = sorted(set(temp_times + pres_times))
    logger.info("Union grid: %d unique timestamps (temp=%d pres=%d)",
                len(all_times), len(temp_times) // 2, len(pres_times) // 2)

    records = []
    n = len(all_times)
    for i in range(n - 1):
        t_start = all_times[i]
        t_end   = all_times[i + 1]
        duration_hours = int(round((t_end - t_start).total_seconds() / 3600))
        if duration_hours <= 0:
            continue

        temp_fit = fit_temp(df, t_start, t_end)
        pres_fit = fit_pres(df, t_start, t_end)

        records.append({
            "segment_id":    i,
            "start_time":    str(t_start),
            "duration_hours": duration_hours,
            "temp_x0":       float(df["temp"].loc[t_start:t_start].iloc[0])
                             if not df["temp"].loc[t_start:t_start].empty else 0.0,
            "pres_x0":       float(df["msl"].loc[t_start:t_start].iloc[0])
                             if not df["msl"].loc[t_start:t_start].empty else 0.0,
            "temp_fit":      temp_fit,
            "pres_fit":      pres_fit,
        })

        if (i + 1) % 500 == 0 or i == n - 2:
            logger.info("  %d / %d intervals processed", i + 1, n - 1)

    Path("segments").mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    logger.info("Done: %d joint segments saved to %s  (%.1fs)",
                len(records), OUT_PATH, elapsed)

    # Quick stats
    durs = [r["duration_hours"] for r in records]
    logger.info("Duration: mean=%.1fh  min=%dh  max=%dh",
                float(np.mean(durs)), min(durs), max(durs))
    for ch in ("temp", "pres"):
        from collections import Counter
        eq_dist = Counter(r[f"{ch}_fit"]["eq_type"] for r in records)
        logger.info("%s equations: %s", ch, dict(eq_dist))


if __name__ == "__main__":
    main()
