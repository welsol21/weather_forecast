from __future__ import annotations

from typing import Any
from pathlib import Path

from weather_patterns.config import PipelineConfig
from weather_patterns.forecasting.dataset import build_forecast_training_dataset
from weather_patterns.forecasting.torch_sequence import TorchSequencePredictor
from weather_patterns.models import ForecastTrainingDataset, PipelineArtifacts


def train_sequence_predictor(
    artifacts: PipelineArtifacts,
    config: PipelineConfig,
) -> tuple[TorchSequencePredictor, ForecastTrainingDataset]:
    training_dataset = build_forecast_training_dataset(artifacts.forecast_samples)
    predictor = TorchSequencePredictor(
        model_config=config.model,
        compute_config=config.compute,
    )
    predictor.fit(training_dataset)
    return predictor, training_dataset


def train_and_save_sequence_predictor(
    artifacts: PipelineArtifacts,
    config: PipelineConfig,
    checkpoint_path: str | Path,
) -> tuple[TorchSequencePredictor, ForecastTrainingDataset, Path]:
    predictor, training_dataset = train_sequence_predictor(artifacts, config)
    saved_path = predictor.save_checkpoint(checkpoint_path)
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
