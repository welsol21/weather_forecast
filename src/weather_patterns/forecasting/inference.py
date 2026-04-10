from __future__ import annotations

import numpy as np

from weather_patterns.config import PipelineConfig
from weather_patterns.forecasting.torch_sequence import TorchSequencePredictor
from weather_patterns.models import ForecastResult, PipelineArtifacts


def predict_future_pattern_sequence(
    predictor: TorchSequencePredictor,
    artifacts: PipelineArtifacts,
    config: PipelineConfig,
) -> ForecastResult:
    history_window_count = config.forecast.history_window_count
    if len(artifacts.pattern_windows) < history_window_count:
        raise ValueError("Not enough pattern windows to build an inference history sequence.")

    history_windows = artifacts.pattern_windows[-history_window_count:]
    history_pattern_matrix = np.vstack([window.feature_vector for window in history_windows])
    forecast_time = history_windows[-1].end_time
    return predictor.predict(
        history_pattern_matrix=history_pattern_matrix,
        forecast_time=forecast_time,
        horizon_steps=config.window.forecast_horizon_steps,
        prototypes=artifacts.discovery_result.prototypes,
    )
