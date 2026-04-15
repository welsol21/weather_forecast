"""Run pressure segmentation for 2020-01-01 to 2026-01-31.

Usage:
    python scripts/run_pressure_segmentation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_patterns.pattern.segmentation_pressure import (
    build_report, segment_pressure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/pressure_segmentation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

CSV_PATH    = Path("hly4935_subset.csv")
TRAIN_START = "2020-01-01"
TRAIN_END   = "2026-01-31"
FEB_START   = "2026-02-01"
FEB_END     = "2026-02-28"


def load_pressure(path: Path, date_start: str, date_end: str) -> pd.Series:
    df = pd.read_csv(path, skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()
    series = df["msl"].loc[date_start:date_end].dropna()
    logger.info("loaded pressure n=%d start=%s end=%s",
                len(series), series.index[0], series.index[-1])
    return series


def main():
    Path("artifacts").mkdir(exist_ok=True)

    logger.info("=== PRESSURE SEGMENTATION: %s to %s ===", TRAIN_START, TRAIN_END)
    train_series = load_pressure(CSV_PATH, TRAIN_START, TRAIN_END)
    segments, eq_counts = segment_pressure(
        series=train_series.values,
        timestamps=train_series.index,
    )
    report = build_report(segments, eq_counts)

    logger.info("=== REPORT ===")
    logger.info("Total segments: %d", report["total_segments"])
    logger.info("Mean duration:  %.1f hours", report["mean_duration_h"])
    logger.info("Min duration:   %d hours", report["min_duration_h"])
    logger.info("Max duration:   %d hours", report["max_duration_h"])
    logger.info("Overall RMS:    %.3f hPa", report["overall_mean_rms_hpa"])
    logger.info("--- Equation distribution ---")
    for eq, stats in report["by_equation_type"].items():
        logger.info(
            "  %-15s %3d segments (%5.1f%%)  mean_dur=%5.1fh  mean_rms=%.3f hPa",
            eq, stats["count"], stats["pct"],
            stats["mean_duration_h"], stats["mean_rms_hpa"],
        )

    # February inference — extend last segment's equation into February
    logger.info("=== FEBRUARY 2026 INFERENCE (last-segment extrapolation) ===")
    feb_series = load_pressure(CSV_PATH, FEB_START, FEB_END)
    from weather_patterns.pattern.segmentation_pressure import _predict_value
    last_seg = segments[-1]
    fit = last_seg.fit
    hours_offset = [(ts - last_seg.start_time).total_seconds() / 3600.0
                    for ts in feb_series.index]
    predicted = np.array([_predict_value(fit, t) for t in hours_offset])
    errors = feb_series.values - predicted
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    logger.info("eq=%s mae=%.3f hPa rmse=%.3f hPa n=%d",
                fit.eq_type, mae, rmse, len(feb_series))

    # Save segments
    seg_records = []
    for s in segments:
        seg_records.append({
            "segment_id":    s.segment_id,
            "start_index":   s.start_index,
            "end_index":     s.end_index,
            "start_time":    str(s.start_time),
            "end_time":      str(s.end_time),
            "duration_hours": s.duration_hours,
            "x0":            s.x0,
            "fit": {
                "eq_type": s.fit.eq_type,
                "L":       s.fit.L,
                "c":       s.fit.c,
                "A":       s.fit.A,
                "lam":     s.fit.lam,
                "rms":     s.fit.rms,
                "n_points": s.fit.n_points,
            },
        })
    segs_path = Path("segments/pressure_segments.json")
    segs_path.write_text(json.dumps(seg_records, indent=2), encoding="utf-8")
    logger.info("Segments saved to %s (%d segments)", segs_path, len(seg_records))

    out = {"segmentation_report": report,
           "february_extrapolation": {"mae_hpa": round(mae, 3), "rmse_hpa": round(rmse, 3)}}
    Path("artifacts/pressure_segmentation_report.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
