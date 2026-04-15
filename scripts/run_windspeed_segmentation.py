"""Run wind speed segmentation for 2020-01-01 to 2026-01-31."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_patterns.pattern.segmentation_windspeed import (
    build_report, segment_windspeed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CSV_PATH    = Path("hly4935_subset.csv")
TRAIN_START = "2020-01-01"
TRAIN_END   = "2026-01-31"


def load_windspeed(path: Path, date_start: str, date_end: str) -> pd.Series:
    df = pd.read_csv(path, skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()
    series = df["wdsp"].loc[date_start:date_end].dropna()
    logger.info("loaded wdsp n=%d start=%s end=%s",
                len(series), series.index[0], series.index[-1])
    return series


def main():
    Path("segments").mkdir(exist_ok=True)

    logger.info("=== WINDSPEED SEGMENTATION: %s to %s ===", TRAIN_START, TRAIN_END)
    train_series = load_windspeed(CSV_PATH, TRAIN_START, TRAIN_END)
    segments, eq_counts = segment_windspeed(
        series=train_series.values.astype(float),
        timestamps=train_series.index,
    )
    report = build_report(segments, eq_counts)

    logger.info("=== REPORT ===")
    logger.info("Total segments: %d", report["total_segments"])
    logger.info("Mean duration:  %.1f hours", report["mean_duration_h"])
    logger.info("Min / Max:      %d / %d hours", report["min_duration_h"], report["max_duration_h"])
    logger.info("Overall RMS:    %.3f kt", report["overall_mean_rms_kt"])
    for eq, stats in report["by_equation_type"].items():
        logger.info("  %-15s %3d segments (%5.1f%%)  mean_dur=%5.1fh  mean_rms=%.3f kt",
                    eq, stats["count"], stats["pct"],
                    stats["mean_duration_h"], stats["mean_rms_kt"])

    seg_records = []
    for s in segments:
        seg_records.append({
            "segment_id":     s.segment_id,
            "start_index":    s.start_index,
            "end_index":      s.end_index,
            "start_time":     str(s.start_time),
            "end_time":       str(s.end_time),
            "duration_hours": s.duration_hours,
            "x0":             s.x0,
            "fit": {
                "eq_type":  s.fit.eq_type,
                "L":        s.fit.L,
                "c":        s.fit.c,
                "lam":      s.fit.lam,
                "A":        s.fit.A,
                "rms":      s.fit.rms,
                "n_points": s.fit.n_points,
            },
        })

    out_path = Path("segments/windspeed_segments.json")
    out_path.write_text(json.dumps(seg_records, indent=2), encoding="utf-8")
    logger.info("Segments saved to %s (%d segments)", out_path, len(seg_records))


if __name__ == "__main__":
    main()
