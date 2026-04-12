from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from weather_patterns.config import DatasetConfig, SmoothingConfig
from weather_patterns.data.loading import apply_quality_masks, load_weather_dataset
from weather_patterns.signal.processing import build_signal_frame


PREDICTOR_NAMES = ("level", "velocity", "acceleration", "local_ar2")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental segmentation prototype based on local channel predictor stability.",
    )
    parser.add_argument("--csv", required=True, help="Path to source CSV.")
    parser.add_argument("--start", required=True, help="Inclusive interval start.")
    parser.add_argument("--end", required=True, help="Inclusive interval end.")
    parser.add_argument(
        "--history-window",
        type=int,
        default=24,
        help="Local history length used to decide the active predictor class.",
    )
    parser.add_argument(
        "--min-run-hours",
        type=int,
        default=3,
        help="Minimum per-channel predictor run before it is treated as a real switch.",
    )
    parser.add_argument(
        "--min-changed-channels",
        type=int,
        default=2,
        help="How many channels must change predictor class to emit a segment boundary.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the segmentation report as JSON.",
    )
    return parser.parse_args()


def _is_circular(channel: str) -> bool:
    return channel == "wind_direction"


def _unwrap(values: np.ndarray, channel: str) -> np.ndarray:
    if not _is_circular(channel):
        return values.astype(float, copy=True)
    return np.rad2deg(np.unwrap(np.deg2rad(values.astype(float))))


def _wrap(value: float, channel: str) -> float:
    if not _is_circular(channel):
        return float(value)
    return float(np.mod(value, 360.0))


def _prediction_error(actual: float, predicted: float, channel: str) -> float:
    if not _is_circular(channel):
        return float(abs(actual - predicted))
    delta = (actual - predicted + 180.0) % 360.0 - 180.0
    return float(abs(delta))


def _predict_level(history: np.ndarray) -> float:
    return float(history[-1])


def _predict_velocity(history: np.ndarray) -> float:
    return float(2.0 * history[-1] - history[-2])


def _predict_acceleration(history: np.ndarray) -> float:
    return float(3.0 * history[-1] - 3.0 * history[-2] + history[-3])


def _predict_local_ar2(history: np.ndarray, fit_window: int) -> float:
    usable = history[-max(fit_window, 3) :]
    design = np.column_stack(
        [
            np.ones((usable.size - 2,), dtype=float),
            usable[1:-1],
            usable[:-2],
        ]
    )
    target = usable[2:]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    intercept, phi1, phi2 = coeffs.tolist()
    return float(intercept + phi1 * usable[-1] + phi2 * usable[-2])


def _best_predictor(values: np.ndarray, actual: float, channel: str, fit_window: int) -> tuple[str, float]:
    history = values
    candidates: list[tuple[float, str]] = []
    if history.size >= 1:
        candidates.append((_prediction_error(actual, _wrap(_predict_level(history), channel), channel), "level"))
    if history.size >= 2:
        candidates.append((_prediction_error(actual, _wrap(_predict_velocity(history), channel), channel), "velocity"))
    if history.size >= 3:
        candidates.append(
            (_prediction_error(actual, _wrap(_predict_acceleration(history), channel), channel), "acceleration")
        )
        candidates.append(
            (_prediction_error(actual, _wrap(_predict_local_ar2(history, fit_window), channel), channel), "local_ar2")
        )
    best_error, best_name = min(candidates, key=lambda item: item[0])
    return best_name, float(best_error)


def _suppress_short_runs(labels: list[str], min_run_hours: int) -> list[str]:
    if not labels:
        return labels
    smoothed = labels[:]
    changed = True
    while changed:
        changed = False
        runs: list[tuple[int, int, str]] = []
        start = 0
        current = smoothed[0]
        for index in range(1, len(smoothed)):
            if smoothed[index] != current:
                runs.append((start, index - 1, current))
                start = index
                current = smoothed[index]
        runs.append((start, len(smoothed) - 1, current))
        for run_index, (run_start, run_end, run_label) in enumerate(runs):
            run_length = run_end - run_start + 1
            if run_length >= min_run_hours:
                continue
            left_label = runs[run_index - 1][2] if run_index > 0 else None
            right_label = runs[run_index + 1][2] if run_index + 1 < len(runs) else None
            replacement = left_label if left_label == right_label and left_label is not None else left_label or right_label
            if replacement is None or replacement == run_label:
                continue
            for position in range(run_start, run_end + 1):
                smoothed[position] = replacement
            changed = True
            break
    return smoothed


def _dominant_signature(label_map: dict[str, list[str]], index: int, channels: list[str]) -> dict[str, str]:
    return {channel: label_map[channel][index] for channel in channels}


def main() -> None:
    args = _parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    dataset = load_weather_dataset(args.csv, DatasetConfig())
    frame = apply_quality_masks(dataset)
    signal = build_signal_frame(frame, dataset.channel_columns, SmoothingConfig())
    interval = signal[(signal["date"] >= start) & (signal["date"] <= end)].copy()
    if interval.empty:
        raise ValueError(f"No data in [{start}, {end}].")

    label_map: dict[str, list[str]] = {}
    error_map: dict[str, list[float]] = {}
    usable_timestamps: list[pd.Timestamp] | None = None

    for channel in dataset.channel_columns:
        series = pd.to_numeric(interval[f"smoothed_{channel}"], errors="coerce").dropna().reset_index(drop=True)
        timestamps = interval.loc[series.index, "date"] if False else None
        valid_rows = interval[pd.to_numeric(interval[f"smoothed_{channel}"], errors="coerce").notna()][["date", f"smoothed_{channel}"]]
        channel_timestamps = valid_rows["date"].reset_index(drop=True)
        values = _unwrap(valid_rows[f"smoothed_{channel}"].to_numpy(dtype=float), channel)
        if values.size <= args.history_window + 1:
            continue
        winners: list[str] = []
        errors: list[float] = []
        winner_timestamps: list[pd.Timestamp] = []
        for index in range(args.history_window, values.size - 1):
            history = values[index - args.history_window + 1 : index + 1]
            actual = _wrap(values[index + 1], channel)
            winner, best_error = _best_predictor(history, actual, channel=channel, fit_window=args.history_window)
            winners.append(winner)
            errors.append(best_error)
            winner_timestamps.append(pd.Timestamp(channel_timestamps.iloc[index + 1]))
        winners = _suppress_short_runs(winners, min_run_hours=args.min_run_hours)
        label_map[channel] = winners
        error_map[channel] = errors
        if usable_timestamps is None:
            usable_timestamps = winner_timestamps

    if not label_map or usable_timestamps is None:
        raise ValueError("No channels had enough data for predictor segmentation.")

    channels = [channel for channel in dataset.channel_columns if channel in label_map]
    length = min(len(label_map[channel]) for channel in channels)
    timestamps = usable_timestamps[:length]
    for channel in channels:
        label_map[channel] = label_map[channel][:length]
        error_map[channel] = error_map[channel][:length]

    boundaries = [0]
    changed_channel_counts: list[int] = [0]
    for index in range(1, length):
        changed_channels = [
            channel
            for channel in channels
            if label_map[channel][index] != label_map[channel][index - 1]
        ]
        changed_count = len(changed_channels)
        changed_channel_counts.append(changed_count)
        if changed_count >= args.min_changed_channels:
            boundaries.append(index)
    boundaries.append(length)

    segments: list[dict[str, object]] = []
    signature_counter: Counter[tuple[tuple[str, str], ...]] = Counter()
    for segment_index in range(len(boundaries) - 1):
        left = boundaries[segment_index]
        right = boundaries[segment_index + 1]
        if right <= left:
            continue
        signature = _dominant_signature(label_map, left, channels)
        signature_key = tuple(sorted(signature.items()))
        signature_counter[signature_key] += right - left
        segments.append(
            {
                "segment_id": segment_index,
                "start": timestamps[left].isoformat(),
                "end": timestamps[right - 1].isoformat(),
                "hours": int(right - left),
                "signature": signature,
                "channel_mean_errors": {
                    channel: round(float(np.mean(error_map[channel][left:right])), 4)
                    for channel in channels
                },
            }
        )

    per_channel_shares = {
        channel: {
            predictor: round(count / len(label_map[channel]), 3)
            for predictor, count in Counter(label_map[channel]).most_common()
        }
        for channel in channels
    }

    top_signatures = [
        {
            "hours": int(hours),
            "signature": dict(signature_key),
        }
        for signature_key, hours in signature_counter.most_common(12)
    ]

    report = {
        "interval_start": timestamps[0].isoformat(),
        "interval_end": timestamps[-1].isoformat(),
        "interval_rows": int(length),
        "history_window": args.history_window,
        "min_run_hours": args.min_run_hours,
        "min_changed_channels": args.min_changed_channels,
        "channels": channels,
        "per_channel_predictor_shares": per_channel_shares,
        "segment_count": len(segments),
        "top_signatures": top_signatures,
        "segments_preview": segments[:20],
        "longest_segments": sorted(segments, key=lambda item: int(item["hours"]), reverse=True)[:20],
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
