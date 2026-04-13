from __future__ import annotations

import json
import logging
import math
import os
import resource
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from weather_patterns.config import PipelineConfig
from weather_patterns.forecasting.dataset import iter_forecast_samples_jsonl
from weather_patterns.forecasting.decoding import decode_forecast_result, decode_forecast_result_new_physics
from weather_patterns.forecasting.torch_sequence import TorchSequencePredictor
from weather_patterns.pipeline import load_pattern_window_end_times, load_pattern_window_new_physics_context
from weather_patterns.models import ForecastSample, PipelineArtifacts


def _log(logger: logging.Logger | None, message: str, **context: object) -> None:
    if logger is None:
        return
    details = ", ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    logger.info(f"{message}{' ' + details if details else ''}")


def _read_current_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024.0, 2)
    except OSError:
        pass
    return None


def _read_peak_rss_mb() -> float | None:
    try:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except OSError:
        return None
    return round(float(peak_kb) / 1024.0, 2)


def _read_cuda_memory_mb() -> tuple[float | None, float | None]:
    cuda_allocated_mb: float | None = None
    cuda_reserved_mb: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            cuda_allocated_mb = round(torch.cuda.memory_allocated(device_index) / (1024.0 * 1024.0), 2)
            cuda_reserved_mb = round(torch.cuda.memory_reserved(device_index) / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return cuda_allocated_mb, cuda_reserved_mb


def _memory_context() -> dict[str, object]:
    cuda_allocated_mb, cuda_reserved_mb = _read_cuda_memory_mb()
    return {
        "pid": os.getpid(),
        "rss_mb": _read_current_rss_mb(),
        "peak_rss_mb": _read_peak_rss_mb(),
        "cuda_allocated_mb": cuda_allocated_mb,
        "cuda_reserved_mb": cuda_reserved_mb,
    }


def _safe_split_index(sample_count: int, fraction: float) -> int:
    return max(1, min(sample_count - 1, int(sample_count * fraction)))


def _resolve_pattern_window_end_times(
    artifacts: PipelineArtifacts,
    prepared_pattern_windows_path: str | Path | None = None,
) -> dict[int, pd.Timestamp]:
    if artifacts.pattern_windows:
        return {
            window.window_id: window.end_time
            for window in artifacts.pattern_windows
        }
    if prepared_pattern_windows_path is None:
        raise ValueError(
            "Evaluation requires either loaded pattern windows or prepared_pattern_windows_path to resolve forecast timestamps."
        )
    return load_pattern_window_end_times(prepared_pattern_windows_path)


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
    logger: logging.Logger | None = None,
    split_name: str = "unknown",
    progress_interval: int = 100,
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
    _log(
        logger,
        "evaluation_split_start",
        split=split_name,
        sample_count=len(used_samples),
        forecast_time_start=split_start_time.isoformat(),
        forecast_time_end=split_end_time.isoformat(),
        **_memory_context(),
    )

    for sample in used_samples:
        source_window = pattern_windows_by_id[sample.source_window_id]
        forecast_time = source_window.end_time
        current_row = actual_frame.loc[forecast_time]

        raw_result = predictor.predict(
            history_pattern_matrix=sample.history_pattern_matrix,
            history_vector=sample.history_vector,
            forecast_time=forecast_time,
            horizon_steps=horizon_steps,
            prototypes=artifacts.discovery_result.prototypes,
        )
        if source_window.channel_stds:
            decoded = decode_forecast_result_new_physics(
                raw_result,
                channels=channels,
                initial_values=source_window.channel_end_values,
                channel_stds=source_window.channel_stds,
                stride_hours=config.window.stride_steps * config.time_step_hours,
            )
        else:
            decoded = decode_forecast_result(raw_result, channels=channels)
        evaluated_samples += 1
        if evaluated_samples == 1 or evaluated_samples % progress_interval == 0 or evaluated_samples == len(used_samples):
            _log(
                logger,
                "evaluation_split_progress",
                split=split_name,
                evaluated_samples=evaluated_samples,
                total_samples=len(used_samples),
                forecast_time=forecast_time.isoformat(),
                **_memory_context(),
            )

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

    summary = {
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
    _log(
        logger,
        "evaluation_split_end",
        split=split_name,
        evaluated_samples=evaluated_samples,
        **_memory_context(),
    )
    return summary


def _evaluate_split_iter(
    predictor: TorchSequencePredictor,
    samples: Iterable[ForecastSample],
    total_sample_count: int,
    actual_frame: pd.DataFrame,
    pattern_window_end_times: dict[int, pd.Timestamp],
    channels: list[str],
    prototypes,
    config: PipelineConfig,
    sample_limit: int | None = None,
    logger: logging.Logger | None = None,
    split_name: str = "unknown",
    progress_interval: int = 100,
    new_physics_context: dict[int, tuple[dict[str, float], dict[str, float]]] | None = None,
) -> dict[str, Any]:
    horizon_steps = config.window.forecast_horizon_steps

    per_channel_stats = {channel: _empty_stats() for channel in channels}
    per_channel_horizon_stats = {
        channel: {horizon: _empty_stats() for horizon in range(1, horizon_steps + 1)}
        for channel in channels
    }
    baseline_channel_stats = {channel: _empty_stats() for channel in channels}
    baseline_channel_horizon_stats = {
        channel: {horizon: _empty_stats() for horizon in range(1, horizon_steps + 1)}
        for channel in channels
    }

    evaluated_samples = 0
    used_samples = 0
    split_start_time: pd.Timestamp | None = None
    split_end_time: pd.Timestamp | None = None
    requested_sample_count = min(total_sample_count, sample_limit) if sample_limit is not None else total_sample_count
    _log(
        logger,
        "evaluation_split_start",
        split=split_name,
        sample_count=requested_sample_count,
        **_memory_context(),
    )

    for sample in samples:
        if sample_limit is not None and used_samples >= sample_limit:
            break
        forecast_time = pattern_window_end_times[sample.source_window_id]
        if split_start_time is None:
            split_start_time = forecast_time
        split_end_time = forecast_time
        current_row = actual_frame.loc[forecast_time]
        used_samples += 1

        raw_result = predictor.predict(
            history_pattern_matrix=sample.history_pattern_matrix,
            history_vector=sample.history_vector,
            forecast_time=forecast_time,
            horizon_steps=horizon_steps,
            prototypes=prototypes,
        )
        np_ctx = new_physics_context.get(sample.source_window_id) if new_physics_context else None
        if np_ctx is not None:
            ch_stds, ch_end_vals = np_ctx
            decoded = decode_forecast_result_new_physics(
                raw_result,
                channels=channels,
                initial_values=ch_end_vals,
                channel_stds=ch_stds,
                stride_hours=config.window.stride_steps * config.time_step_hours,
            )
        else:
            decoded = decode_forecast_result(raw_result, channels=channels)
        evaluated_samples += 1
        if evaluated_samples == 1 or evaluated_samples % progress_interval == 0 or evaluated_samples == requested_sample_count:
            _log(
                logger,
                "evaluation_split_progress",
                split=split_name,
                evaluated_samples=evaluated_samples,
                total_samples=requested_sample_count,
                forecast_time=forecast_time.isoformat(),
                **_memory_context(),
            )

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

    if split_start_time is None or split_end_time is None:
        raise ValueError("Expected at least one forecast sample for evaluation.")

    summary = {
        "sample_count": requested_sample_count,
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
    _log(
        logger,
        "evaluation_split_end",
        split=split_name,
        evaluated_samples=evaluated_samples,
        forecast_time_start=split_start_time.isoformat(),
        forecast_time_end=split_end_time.isoformat(),
        **_memory_context(),
    )
    return summary


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


def _summarize_split_overall(split_summary: dict[str, Any], channels: list[str]) -> dict[str, float]:
    model_mae: list[float] = []
    model_rmse: list[float] = []
    baseline_mae: list[float] = []
    baseline_rmse: list[float] = []
    for channel in channels:
        overall = split_summary["per_channel"][channel]["overall"]
        baseline = split_summary["per_channel"][channel]["baseline_overall"]
        if overall["count"] > 0:
            model_mae.append(overall["mae"])
            model_rmse.append(overall["rmse"])
        if baseline["count"] > 0:
            baseline_mae.append(baseline["mae"])
            baseline_rmse.append(baseline["rmse"])
    return {
        "mean_channel_mae": float(np.mean(model_mae)) if model_mae else math.nan,
        "mean_channel_rmse": float(np.mean(model_rmse)) if model_rmse else math.nan,
        "baseline_mean_channel_mae": float(np.mean(baseline_mae)) if baseline_mae else math.nan,
        "baseline_mean_channel_rmse": float(np.mean(baseline_rmse)) if baseline_rmse else math.nan,
    }


def summarize_evaluation_payload(summary: dict[str, Any], top_k_horizons: int = 5) -> dict[str, Any]:
    channels = list(summary["dataset"]["channels"])
    validation_hourly = list(summary.get("validation_hourly_scoreboard", []))
    test_hourly = list(summary.get("test_hourly_scoreboard", []))
    return {
        "split": summary["split"],
        "dataset": summary["dataset"],
        "validation": {
            "sample_count": summary["validation"]["sample_count"],
            "evaluated_sample_count": summary["validation"]["evaluated_sample_count"],
            "forecast_time_start": summary["validation"]["forecast_time_start"],
            "forecast_time_end": summary["validation"]["forecast_time_end"],
            "overall": _summarize_split_overall(summary["validation"], channels),
            "hourly_scoreboard_preview": validation_hourly[:top_k_horizons],
        },
        "test": {
            "sample_count": summary["test"]["sample_count"],
            "evaluated_sample_count": summary["test"]["evaluated_sample_count"],
            "forecast_time_start": summary["test"]["forecast_time_start"],
            "forecast_time_end": summary["test"]["forecast_time_end"],
            "overall": _summarize_split_overall(summary["test"], channels),
            "hourly_scoreboard_preview": test_hourly[:top_k_horizons],
        },
    }


def evaluate_sequence_backtest(
    artifacts: PipelineArtifacts,
    config: PipelineConfig,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    sample_limit: int | None = None,
    logger: logging.Logger | None = None,
    max_rss_mb: float | None = None,
) -> dict[str, Any]:
    _log(
        logger,
        "evaluation_start",
        mode="in_memory",
        total_samples=len(artifacts.forecast_samples),
        sample_limit=sample_limit,
        **_memory_context(),
    )
    train_samples, validation_samples, test_samples = split_forecast_samples_chronologically(
        artifacts.forecast_samples,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    _log(
        logger,
        "evaluation_split_plan",
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        test_samples=len(test_samples),
        **_memory_context(),
    )
    predictor = TorchSequencePredictor(
        model_config=config.model,
        compute_config=config.compute,
    )
    predictor.set_logger(logger)
    predictor.set_resource_limits(max_rss_mb=max_rss_mb)
    _log(logger, "evaluation_training_start", train_samples=len(train_samples), **_memory_context())
    predictor.fit_samples(train_samples, logger=logger)
    _log(logger, "evaluation_training_end", train_samples=len(train_samples), **_memory_context())

    validation_summary = _evaluate_split(
        predictor=predictor,
        samples=validation_samples,
        artifacts=artifacts,
        config=config,
        sample_limit=sample_limit,
        logger=logger,
        split_name="validation",
    )
    test_summary = _evaluate_split(
        predictor=predictor,
        samples=test_samples,
        artifacts=artifacts,
        config=config,
        sample_limit=sample_limit,
        logger=logger,
        split_name="test",
    )
    channels = artifacts.dataset.channel_columns
    summary = {
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
    _log(logger, "evaluation_complete", mode="in_memory", **_memory_context())
    return summary


def evaluate_sequence_backtest_from_saved_dataset(
    artifacts: PipelineArtifacts,
    sequence_dataset_path: str | Path,
    config: PipelineConfig,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    sample_limit: int | None = None,
    logger: logging.Logger | None = None,
    max_rss_mb: float | None = None,
    prepared_pattern_windows_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_path = str(sequence_dataset_path)
    channels = artifacts.dataset.channel_columns
    actual_frame = artifacts.dataset.dataframe.set_index("date").sort_index()
    pattern_window_end_times = _resolve_pattern_window_end_times(
        artifacts,
        prepared_pattern_windows_path=prepared_pattern_windows_path,
    )
    # Load new_physics context (channel_stds + channel_end_values) when available
    new_physics_ctx: dict[int, tuple[dict[str, float], dict[str, float]]] | None = None
    if prepared_pattern_windows_path is not None:
        new_physics_ctx = load_pattern_window_new_physics_context(prepared_pattern_windows_path) or None
    _log(
        logger,
        "evaluation_start",
        mode="streaming",
        dataset_path=dataset_path,
        sample_limit=sample_limit,
        **_memory_context(),
    )
    _log(logger, "evaluation_count_samples_start", dataset_path=dataset_path, **_memory_context())
    total_samples = sum(1 for _ in iter_forecast_samples_jsonl(dataset_path))
    _log(
        logger,
        "evaluation_count_samples_end",
        dataset_path=dataset_path,
        total_samples=total_samples,
        **_memory_context(),
    )
    if total_samples < 3:
        raise ValueError("At least three forecast samples are required for train/validation/test splitting.")

    train_end = _safe_split_index(total_samples, train_fraction)
    validation_end = _safe_split_index(total_samples, train_fraction + validation_fraction)
    if validation_end <= train_end:
        validation_end = min(total_samples - 1, train_end + 1)

    train_count = train_end
    validation_count = validation_end - train_end
    test_count = total_samples - validation_end
    if train_count <= 0 or validation_count <= 0 or test_count <= 0:
        raise ValueError("Chronological split produced an empty partition.")
    _log(
        logger,
        "evaluation_split_plan",
        train_samples=train_count,
        validation_samples=validation_count,
        test_samples=test_count,
        **_memory_context(),
    )

    def sample_range(start_index: int, end_index: int) -> Callable[[], Iterable[ForecastSample]]:
        def iterator() -> Iterable[ForecastSample]:
            for index, sample in enumerate(iter_forecast_samples_jsonl(dataset_path)):
                if index < start_index:
                    continue
                if index >= end_index:
                    break
                yield sample

        return iterator

    predictor = TorchSequencePredictor(
        model_config=config.model,
        compute_config=config.compute,
    )
    predictor.set_logger(logger)
    predictor.set_resource_limits(max_rss_mb=max_rss_mb)
    _log(
        logger,
        "evaluation_training_start",
        train_samples=train_count,
        dataset_path=dataset_path,
        **_memory_context(),
    )
    predictor.fit_sample_iterator(sample_range(0, train_end), train_count, logger=logger)
    _log(
        logger,
        "evaluation_training_end",
        train_samples=train_count,
        dataset_path=dataset_path,
        **_memory_context(),
    )

    validation_summary = _evaluate_split_iter(
        predictor=predictor,
        samples=sample_range(train_end, validation_end)(),
        total_sample_count=validation_count,
        actual_frame=actual_frame,
        pattern_window_end_times=pattern_window_end_times,
        channels=channels,
        prototypes=artifacts.discovery_result.prototypes,
        config=config,
        sample_limit=sample_limit,
        logger=logger,
        split_name="validation",
        new_physics_context=new_physics_ctx,
    )
    test_summary = _evaluate_split_iter(
        predictor=predictor,
        samples=sample_range(validation_end, total_samples)(),
        total_sample_count=test_count,
        actual_frame=actual_frame,
        pattern_window_end_times=pattern_window_end_times,
        channels=channels,
        prototypes=artifacts.discovery_result.prototypes,
        config=config,
        sample_limit=sample_limit,
        logger=logger,
        split_name="test",
        new_physics_context=new_physics_ctx,
    )
    summary = {
        "split": {
            "train_samples": train_count,
            "validation_samples": validation_count,
            "test_samples": test_count,
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
    _log(logger, "evaluation_complete", mode="streaming", dataset_path=dataset_path, **_memory_context())
    return summary


def write_evaluation_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return destination
