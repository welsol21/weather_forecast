from __future__ import annotations

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
