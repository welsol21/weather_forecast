from __future__ import annotations

import json
import logging
from typing import Any
from pathlib import Path

from weather_patterns.config import PipelineConfig
from weather_patterns.forecasting.dataset import (
    build_forecast_training_dataset,
    load_forecast_training_dataset_jsonl,
)
from weather_patterns.forecasting.torch_sequence import TorchSequencePredictor
from weather_patterns.models import ForecastTrainingDataset, PipelineArtifacts


def _log(logger: logging.Logger | None, message: str, **context: object) -> None:
    if logger is None:
        return
    details = ", ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    logger.info(f"{message}{' ' + details if details else ''}")


def train_sequence_predictor(
    artifacts: PipelineArtifacts,
    config: PipelineConfig,
) -> tuple[TorchSequencePredictor, ForecastTrainingDataset]:
    training_dataset = build_forecast_training_dataset(artifacts.forecast_samples)
    predictor = train_sequence_predictor_from_dataset(training_dataset, config)
    return predictor, training_dataset


def train_sequence_predictor_from_dataset(
    training_dataset: ForecastTrainingDataset,
    config: PipelineConfig,
    logger: logging.Logger | None = None,
) -> TorchSequencePredictor:
    _log(
        logger,
        "training_fit_start",
        sample_count=len(training_dataset.source_window_ids),
        history_window_count=training_dataset.history_window_count,
        forecast_window_count=training_dataset.forecast_window_count,
        feature_dim=training_dataset.feature_dim,
        device=config.compute.model_device,
    )
    predictor = TorchSequencePredictor(
        model_config=config.model,
        compute_config=config.compute,
    )
    predictor.fit(training_dataset)
    _log(logger, "training_fit_end")
    return predictor


def train_and_save_sequence_predictor(
    artifacts: PipelineArtifacts,
    config: PipelineConfig,
    checkpoint_path: str | Path,
) -> tuple[TorchSequencePredictor, ForecastTrainingDataset, Path]:
    predictor, training_dataset = train_sequence_predictor(artifacts, config)
    saved_path = predictor.save_checkpoint(checkpoint_path)
    return predictor, training_dataset, saved_path


def train_and_save_sequence_predictor_from_dataset(
    dataset_path: str | Path,
    config: PipelineConfig,
    checkpoint_path: str | Path,
    logger: logging.Logger | None = None,
) -> tuple[TorchSequencePredictor, ForecastTrainingDataset, Path]:
    _log(logger, "training_dataset_load_start", dataset_path=dataset_path)
    training_dataset = load_forecast_training_dataset_jsonl(str(dataset_path))
    _log(
        logger,
        "training_dataset_load_end",
        sample_count=len(training_dataset.source_window_ids),
        history_shape=training_dataset.history_pattern_tensor.shape,
        context_shape=training_dataset.history_vector_matrix.shape,
        target_shape=training_dataset.target_pattern_tensor.shape,
    )
    predictor = train_sequence_predictor_from_dataset(training_dataset, config, logger=logger)
    _log(logger, "training_checkpoint_save_start", checkpoint_path=checkpoint_path)
    saved_path = predictor.save_checkpoint(checkpoint_path)
    _log(logger, "training_checkpoint_save_end", checkpoint_path=saved_path)
    return predictor, training_dataset, saved_path


def summarize_training_dataset(dataset: ForecastTrainingDataset) -> dict[str, Any]:
    return {
        "sample_count": len(dataset.source_window_ids),
        "history_window_count": dataset.history_window_count,
        "forecast_window_count": dataset.forecast_window_count,
        "forecast_horizon_steps": dataset.forecast_horizon_steps,
        "feature_dim": dataset.feature_dim,
        "history_pattern_tensor_shape": list(dataset.history_pattern_tensor.shape),
        "target_pattern_tensor_shape": list(dataset.target_pattern_tensor.shape),
    }


def write_training_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return destination
