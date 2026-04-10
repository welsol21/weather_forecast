from __future__ import annotations

import numpy as np
from typing import Any

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


def summarize_forecast_result(result: ForecastResult) -> dict[str, Any]:
    return {
        "forecast_time": result.forecast_time.isoformat(),
        "horizon_steps": result.horizon_steps,
        "predicted_window_count": result.predicted_window_count,
        "predicted_pattern_ids": result.predicted_pattern_ids,
        "predicted_pattern_matrix_shape": list(result.predicted_pattern_matrix.shape),
        "predicted_values_channels": sorted(result.predicted_values.keys()),
        "predicted_time_placeholder_count": len(result.predicted_time_placeholders),
        "predicted_peak_hazard_count": len(result.predicted_peak_hazard),
    }
