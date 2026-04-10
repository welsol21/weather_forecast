from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from weather_patterns.config import PipelineConfig
from weather_patterns.forecasting.dataset import build_forecast_training_dataset
from weather_patterns.forecasting.decoding import decode_forecast_result
from weather_patterns.forecasting.torch_sequence import TorchSequencePredictor
from weather_patterns.models import ForecastSample, PipelineArtifacts


def _safe_split_index(sample_count: int, fraction: float) -> int:
    return max(1, min(sample_count - 1, int(sample_count * fraction)))


def split_forecast_samples_chronologically(
    samples: list[ForecastSample],
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[list[ForecastSample], list[ForecastSample], list[ForecastSample]]:
    if len(samples) < 3:
        raise ValueError("At least three forecast samples are required for train/validation/test splitting.")

    train_end = _safe_split_index(len(samples), train_fraction)
    validation_end = _safe_split_index(len(samples), train_fraction + validation_fraction)
    if validation_end <= train_end:
        validation_end = min(len(samples) - 1, train_end + 1)

    train_samples = samples[:train_end]
    validation_samples = samples[train_end:validation_end]
    test_samples = samples[validation_end:]
    if not train_samples or not validation_samples or not test_samples:
        raise ValueError("Chronological split produced an empty partition.")
    return train_samples, validation_samples, test_samples


def _empty_stats() -> dict[str, float]:
    return {"count": 0.0, "sum_abs_error": 0.0, "sum_squared_error": 0.0}


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    count = int(stats["count"])
    if count == 0:
        return {"count": 0, "mae": math.nan, "rmse": math.nan}
    return {
        "count": count,
        "mae": float(stats["sum_abs_error"] / count),
        "rmse": float(math.sqrt(stats["sum_squared_error"] / count)),
    }


def _update_stats(stats: dict[str, float], error: float) -> None:
    stats["count"] += 1.0
    stats["sum_abs_error"] += abs(error)
    stats["sum_squared_error"] += error * error


def _evaluate_split(
    predictor: TorchSequencePredictor,
    samples: list[ForecastSample],
    artifacts: PipelineArtifacts,
    config: PipelineConfig,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    channels = artifacts.dataset.channel_columns
    pattern_windows_by_id = {
        window.window_id: window
        for window in artifacts.pattern_windows
    }
    actual_frame = artifacts.dataset.dataframe.set_index("date").sort_index()
    horizon_steps = config.window.forecast_horizon_steps

    per_channel_stats = {
        channel: _empty_stats()
        for channel in channels
    }
    per_channel_horizon_stats = {
        channel: {
            horizon: _empty_stats()
            for horizon in range(1, horizon_steps + 1)
        }
        for channel in channels
    }
    baseline_channel_stats = {
        channel: _empty_stats()
        for channel in channels
    }
    baseline_channel_horizon_stats = {
        channel: {
            horizon: _empty_stats()
            for horizon in range(1, horizon_steps + 1)
        }
        for channel in channels
    }

    evaluated_samples = 0
    used_samples = samples[:sample_limit] if sample_limit is not None else samples
    split_start_time = pattern_windows_by_id[used_samples[0].source_window_id].end_time
    split_end_time = pattern_windows_by_id[used_samples[-1].source_window_id].end_time

    for sample in used_samples:
        source_window = pattern_windows_by_id[sample.source_window_id]
        forecast_time = source_window.end_time
        current_row = actual_frame.loc[forecast_time]

        raw_result = predictor.predict(
            history_pattern_matrix=sample.history_pattern_matrix,
            forecast_time=forecast_time,
            horizon_steps=horizon_steps,
            prototypes=artifacts.discovery_result.prototypes,
        )
        decoded = decode_forecast_result(raw_result, channels=channels)
        evaluated_samples += 1

        for horizon in range(1, horizon_steps + 1):
            actual_timestamp = forecast_time + pd.to_timedelta(horizon, unit="h")
            if actual_timestamp not in actual_frame.index:
                continue
            actual_row = actual_frame.loc[actual_timestamp]
            for channel in channels:
                actual_value = actual_row[channel]
                if pd.isna(actual_value):
                    continue
                predicted_value = float(decoded.predicted_interval_values[channel][horizon - 1])
                if not np.isfinite(predicted_value):
                    continue
                error = predicted_value - float(actual_value)
                _update_stats(per_channel_stats[channel], error)
                _update_stats(per_channel_horizon_stats[channel][horizon], error)

                baseline_value = current_row[channel]
                if pd.isna(baseline_value):
                    continue
                baseline_error = float(baseline_value) - float(actual_value)
                _update_stats(baseline_channel_stats[channel], baseline_error)
                _update_stats(baseline_channel_horizon_stats[channel][horizon], baseline_error)

    return {
        "sample_count": len(used_samples),
        "evaluated_sample_count": evaluated_samples,
        "forecast_time_start": split_start_time.isoformat(),
        "forecast_time_end": split_end_time.isoformat(),
        "per_channel": {
            channel: {
                "overall": _finalize_stats(per_channel_stats[channel]),
                "baseline_overall": _finalize_stats(baseline_channel_stats[channel]),
                "per_horizon": {
                    str(horizon): {
                        "model": _finalize_stats(per_channel_horizon_stats[channel][horizon]),
                        "baseline_last_value": _finalize_stats(
                            baseline_channel_horizon_stats[channel][horizon]
                        ),
                    }
                    for horizon in range(1, horizon_steps + 1)
                },
            }
            for channel in channels
        },
    }


def _collect_hourly_scoreboard(split_summary: dict[str, Any], channels: list[str]) -> list[dict[str, float]]:
    first_channel = channels[0]
    horizon_keys = split_summary["per_channel"][first_channel]["per_horizon"].keys()
    scoreboard: list[dict[str, float]] = []
    for horizon_key in horizon_keys:
        model_mae: list[float] = []
        baseline_mae: list[float] = []
        model_rmse: list[float] = []
        baseline_rmse: list[float] = []
        for channel in channels:
            metrics = split_summary["per_channel"][channel]["per_horizon"][horizon_key]
            if metrics["model"]["count"] > 0:
                model_mae.append(metrics["model"]["mae"])
                model_rmse.append(metrics["model"]["rmse"])
            if metrics["baseline_last_value"]["count"] > 0:
                baseline_mae.append(metrics["baseline_last_value"]["mae"])
                baseline_rmse.append(metrics["baseline_last_value"]["rmse"])
        scoreboard.append(
            {
                "horizon_hour": int(horizon_key),
                "mean_channel_mae": float(np.mean(model_mae)) if model_mae else math.nan,
                "mean_channel_rmse": float(np.mean(model_rmse)) if model_rmse else math.nan,
                "baseline_mean_channel_mae": float(np.mean(baseline_mae)) if baseline_mae else math.nan,
                "baseline_mean_channel_rmse": float(np.mean(baseline_rmse)) if baseline_rmse else math.nan,
            }
        )
    return scoreboard


def evaluate_sequence_backtest(
    artifacts: PipelineArtifacts,
    config: PipelineConfig,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    train_samples, validation_samples, test_samples = split_forecast_samples_chronologically(
        artifacts.forecast_samples,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    training_dataset = build_forecast_training_dataset(train_samples)
    predictor = TorchSequencePredictor(
        model_config=config.model,
        compute_config=config.compute,
    )
    predictor.fit(training_dataset)

    validation_summary = _evaluate_split(
        predictor=predictor,
        samples=validation_samples,
        artifacts=artifacts,
        config=config,
        sample_limit=sample_limit,
    )
    test_summary = _evaluate_split(
        predictor=predictor,
        samples=test_samples,
        artifacts=artifacts,
        config=config,
        sample_limit=sample_limit,
    )
    channels = artifacts.dataset.channel_columns
    return {
        "split": {
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "test_samples": len(test_samples),
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1.0 - train_fraction - validation_fraction,
        },
        "dataset": {
            "rows": int(len(artifacts.dataset.dataframe)),
            "channels": channels,
            "start_timestamp": artifacts.dataset.dataframe["date"].min().isoformat(),
            "end_timestamp": artifacts.dataset.dataframe["date"].max().isoformat(),
        },
        "validation": validation_summary,
        "validation_hourly_scoreboard": _collect_hourly_scoreboard(validation_summary, channels),
        "test": test_summary,
        "test_hourly_scoreboard": _collect_hourly_scoreboard(test_summary, channels),
    }


def write_evaluation_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return destination
