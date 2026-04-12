from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from weather_patterns.config import DatasetConfig, SmoothingConfig
from weather_patterns.data.loading import apply_quality_masks, load_weather_dataset
from weather_patterns.signal.processing import build_signal_frame


@dataclass(frozen=True)
class PredictorScore:
    name: str
    mae: float
    valid_predictions: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental analysis of local channel predictor classes on the tail of the weather series.",
    )
    parser.add_argument("--csv", required=True, help="Path to source CSV.")
    parser.add_argument(
        "--anchor-timestamp",
        default="2026-03-01T00:00:00",
        help="End timestamp of the analysis interval.",
    )
    parser.add_argument(
        "--history-hours",
        type=int,
        default=24 * 14,
        help="How many trailing hourly steps to inspect before the anchor timestamp.",
    )
    parser.add_argument(
        "--fit-window",
        type=int,
        default=24,
        help="History length for local AR(2) fitting.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the experiment report as JSON.",
    )
    return parser.parse_args()


def _is_circular(channel: str) -> bool:
    return channel == "wind_direction"


def _unwrap_if_needed(values: np.ndarray, channel: str) -> np.ndarray:
    if not _is_circular(channel):
        return values.astype(float, copy=True)
    radians = np.deg2rad(values.astype(float))
    return np.rad2deg(np.unwrap(radians))


def _wrap_prediction(value: float, channel: str) -> float:
    if not _is_circular(channel):
        return float(value)
    return float(np.mod(value, 360.0))


def _prediction_error(actual: float, predicted: float, channel: str) -> float:
    if not _is_circular(channel):
        return float(abs(actual - predicted))
    delta = (actual - predicted + 180.0) % 360.0 - 180.0
    return float(abs(delta))


def _valid_tail(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def _predict_level(history: np.ndarray) -> float:
    return float(history[-1])


def _predict_velocity(history: np.ndarray) -> float:
    return float(2.0 * history[-1] - history[-2])


def _predict_acceleration(history: np.ndarray) -> float:
    return float(3.0 * history[-1] - 3.0 * history[-2] + history[-3])


def _fit_local_ar2(history: np.ndarray, fit_window: int) -> tuple[float, list[float]]:
    usable = history[-max(fit_window, 3) :]
    if usable.size < 3:
        raise ValueError("AR(2) requires at least 3 points.")
    left = usable[:-1]
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
    prediction = intercept + phi1 * usable[-1] + phi2 * usable[-2]
    return float(prediction), [float(intercept), float(phi1), float(phi2)]


def _score_predictor(
    values: np.ndarray,
    channel: str,
    predictor_name: str,
    fit_window: int,
) -> PredictorScore:
    errors: list[float] = []
    for end_index in range(values.size - 1):
        history = values[: end_index + 1]
        actual = values[end_index + 1]
        if predictor_name == "level":
            if history.size < 1:
                continue
            predicted = _predict_level(history)
        elif predictor_name == "velocity":
            if history.size < 2:
                continue
            predicted = _predict_velocity(history)
        elif predictor_name == "acceleration":
            if history.size < 3:
                continue
            predicted = _predict_acceleration(history)
        elif predictor_name == "local_ar2":
            if history.size < 3:
                continue
            predicted, _ = _fit_local_ar2(history, fit_window=fit_window)
        else:
            raise ValueError(f"Unsupported predictor: {predictor_name}")
        errors.append(_prediction_error(actual, _wrap_prediction(predicted, channel), channel))
    mae = float(np.mean(errors)) if errors else float("inf")
    return PredictorScore(name=predictor_name, mae=mae, valid_predictions=len(errors))


def _describe_current_predictor(
    values: np.ndarray,
    channel: str,
    best_name: str,
    fit_window: int,
) -> dict[str, object]:
    history = values
    if best_name == "level":
        next_value = _wrap_prediction(_predict_level(history), channel)
        return {
            "predictor": "level",
            "formula": "x[t+1] = x[t]",
            "next_step_prediction": next_value,
        }
    if best_name == "velocity":
        next_value = _wrap_prediction(_predict_velocity(history), channel)
        return {
            "predictor": "velocity",
            "formula": "x[t+1] = 2*x[t] - x[t-1]",
            "next_step_prediction": next_value,
        }
    if best_name == "acceleration":
        next_value = _wrap_prediction(_predict_acceleration(history), channel)
        return {
            "predictor": "acceleration",
            "formula": "x[t+1] = 3*x[t] - 3*x[t-1] + x[t-2]",
            "next_step_prediction": next_value,
        }
    predicted, coeffs = _fit_local_ar2(history, fit_window=fit_window)
    intercept, phi1, phi2 = coeffs
    return {
        "predictor": "local_ar2",
        "formula": "x[t+1] = c + phi1*x[t] + phi2*x[t-1]",
        "coefficients": {
            "c": intercept,
            "phi1": phi1,
            "phi2": phi2,
        },
        "next_step_prediction": _wrap_prediction(predicted, channel),
    }


def main() -> None:
    args = _parse_args()
    dataset = load_weather_dataset(args.csv, DatasetConfig())
    frame = apply_quality_masks(dataset)
    signal = build_signal_frame(frame, dataset.channel_columns, SmoothingConfig())

    anchor = pd.Timestamp(args.anchor_timestamp)
    start = anchor - pd.Timedelta(hours=args.history_hours - 1)
    segment = signal[(signal["date"] >= start) & (signal["date"] <= anchor)].copy()
    if segment.empty:
        raise ValueError(f"No data found in interval [{start}, {anchor}].")

    report: dict[str, object] = {
        "anchor_timestamp": anchor.isoformat(),
        "segment_start": pd.Timestamp(segment["date"].min()).isoformat(),
        "segment_end": pd.Timestamp(segment["date"].max()).isoformat(),
        "segment_rows": int(len(segment)),
        "note": (
            "The source CSV contains only one point in March 2026. "
            "This experiment therefore uses the trailing interval ending at the March anchor."
        ),
        "channels": {},
    }

    predictor_names = ["level", "velocity", "acceleration", "local_ar2"]
    for channel in dataset.channel_columns:
        values = _valid_tail(segment[f"smoothed_{channel}"])
        values = _unwrap_if_needed(values, channel)
        if values.size < 8:
            report["channels"][channel] = {
                "status": "insufficient_data",
                "valid_points": int(values.size),
            }
            continue

        scores = [
            _score_predictor(values, channel=channel, predictor_name=name, fit_window=args.fit_window)
            for name in predictor_names
        ]
        scores_sorted = sorted(scores, key=lambda item: item.mae)
        best = scores_sorted[0]
        report["channels"][channel] = {
            "status": "ok",
            "valid_points": int(values.size),
            "best_predictor": best.name,
            "scoreboard": [
                {
                    "predictor": score.name,
                    "mae": score.mae,
                    "valid_predictions": score.valid_predictions,
                }
                for score in scores_sorted
            ],
            "current_predictor_form": _describe_current_predictor(
                values,
                channel=channel,
                best_name=best.name,
                fit_window=args.fit_window,
            ),
        }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
