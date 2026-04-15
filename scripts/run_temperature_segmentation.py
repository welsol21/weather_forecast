"""Run temperature segmentation for 2025-01-31 to 2026-01-31.
Check inference on February 2026.

Usage:
    python scripts/run_temperature_segmentation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_patterns.pattern.segmentation_temperature import (
    build_report,
    segment_temperature,
    _predict_value,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("artifacts/temperature_segmentation.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

CSV_PATH = Path("hly4935_subset.csv")
TRAIN_START = "2020-01-01"
TRAIN_END   = "2026-01-31"
FEB_START   = "2026-02-01"
FEB_END     = "2026-02-28"


def load_temperature(path: Path, date_start: str, date_end: str) -> pd.Series:
    df = pd.read_csv(path, skiprows=23)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    temp_col = "temp"
    series = df[temp_col].loc[date_start:date_end].dropna()
    logger.info("loaded temperature n=%d start=%s end=%s", len(series), series.index[0], series.index[-1])
    return series


def run_inference_february(
    last_segment,
    feb_series: pd.Series,
) -> dict:
    """Project last January segment equation forward into February."""
    fit = last_segment.fit
    feb_start_time = feb_series.index[0]
    last_seg_start_time = last_segment.start_time

    # t for each February point relative to the start of the last segment
    hours_offset = [(ts - last_seg_start_time).total_seconds() / 3600.0 for ts in feb_series.index]

    predicted = np.array([_predict_value(fit, t) for t in hours_offset])
    actual    = feb_series.values
    errors    = actual - predicted
    mae       = float(np.mean(np.abs(errors)))
    rmse      = float(np.sqrt(np.mean(errors ** 2)))

    logger.info(
        "february_inference eq=%s mae=%.3f°C rmse=%.3f°C n=%d",
        fit.eq_type, mae, rmse, len(feb_series),
    )

    preview = []
    for i in range(min(10, len(feb_series))):
        preview.append({
            "timestamp": str(feb_series.index[i]),
            "actual": round(float(actual[i]), 2),
            "predicted": round(float(predicted[i]), 2),
            "error": round(float(errors[i]), 2),
        })

    return {
        "last_segment_eq": fit.eq_type,
        "last_segment_start": str(last_segment.start_time),
        "last_segment_duration_h": last_segment.duration_hours,
        "february_n_points": len(feb_series),
        "february_mae_celsius": round(mae, 3),
        "february_rmse_celsius": round(rmse, 3),
        "preview_first_10": preview,
    }


def main():
    Path("artifacts").mkdir(exist_ok=True)

    # --- Training period ---
    logger.info("=== TEMPERATURE SEGMENTATION: %s to %s ===", TRAIN_START, TRAIN_END)
    train_series = load_temperature(CSV_PATH, TRAIN_START, TRAIN_END)
    segments, eq_counts = segment_temperature(
        series=train_series.values,
        timestamps=train_series.index,
    )
    report = build_report(segments, eq_counts)

    logger.info("=== REPORT ===")
    logger.info("Total segments: %d", report["total_segments"])
    logger.info("Mean duration:  %.1f hours", report["mean_duration_h"])
    logger.info("Min duration:   %d hours", report["min_duration_h"])
    logger.info("Max duration:   %d hours", report["max_duration_h"])
    logger.info("Overall RMS:    %.3f°C", report["overall_mean_rms_celsius"])
    logger.info("--- Equation distribution ---")
    for eq, stats in report["by_equation_type"].items():
        logger.info(
            "  %-20s %3d segments (%5.1f%%)  mean_dur=%5.1fh  mean_rms=%.3f°C",
            eq, stats["count"], stats["pct"],
            stats["mean_duration_h"], stats["mean_rms_celsius"],
        )

    # --- February inference ---
    logger.info("=== FEBRUARY 2026 INFERENCE ===")
    feb_series = load_temperature(CSV_PATH, FEB_START, FEB_END)
    last_segment = segments[-1]
    inference = run_inference_february(last_segment, feb_series)

    logger.info("Last January segment: eq=%s duration=%dh start=%s",
        inference["last_segment_eq"],
        inference["last_segment_duration_h"],
        inference["last_segment_start"],
    )
    logger.info("February MAE:  %.3f°C", inference["february_mae_celsius"])
    logger.info("February RMSE: %.3f°C", inference["february_rmse_celsius"])
    logger.info("--- First 10 hours ---")
    for row in inference["preview_first_10"]:
        logger.info("  %s  actual=%.2f  pred=%.2f  err=%+.2f",
            row["timestamp"], row["actual"], row["predicted"], row["error"])

    # --- Save full report ---
    output = {"segmentation_report": report, "february_inference": inference}
    out_path = Path("artifacts/temperature_segmentation_report.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Report saved to %s", out_path)

    # --- Save segments for model training ---
    seg_records = []
    for s in segments:
        seg_records.append({
            "segment_id": s.segment_id,
            "start_index": s.start_index,
            "end_index": s.end_index,
            "start_time": str(s.start_time),
            "end_time": str(s.end_time),
            "duration_hours": s.duration_hours,
            "x0": s.x0,
            "fit": {
                "eq_type": s.fit.eq_type,
                "L": s.fit.L,
                "c": s.fit.c,
                "lam": s.fit.lam,
                "A": s.fit.A,
                "B": s.fit.B,
                "rms": s.fit.rms,
                "n_points": s.fit.n_points,
            },
        })
    segs_path = Path("artifacts/temperature_segments.json")
    segs_path.write_text(json.dumps(seg_records, indent=2), encoding="utf-8")
    logger.info("Segments saved to %s (%d segments)", segs_path, len(seg_records))


if __name__ == "__main__":
    main()
