from __future__ import annotations

import numpy as np

from weather_patterns.config import ForecastConfig, WindowConfig
from weather_patterns.models import ForecastSample, PatternWindow


def build_forecast_samples(
    pattern_windows: list[PatternWindow],
    labels_by_window_id: dict[int, int],
    window_config: WindowConfig,
    forecast_config: ForecastConfig,
) -> list[ForecastSample]:
    if not pattern_windows:
        return []

    windows_by_start = {
        pattern_window.extrema_window.start_index: pattern_window
        for pattern_window in pattern_windows
    }
    samples: list[ForecastSample] = []
    for current_position, pattern_window in enumerate(pattern_windows):
        history_start = current_position - forecast_config.history_window_count + 1
        if history_start < 0:
            continue
        future_start = pattern_window.extrema_window.start_index + window_config.forecast_horizon_steps
        target_window = windows_by_start.get(future_start)
        if target_window is None:
            continue
        history_windows = pattern_windows[history_start : current_position + 1]
        history_vectors = [window.feature_vector for window in history_windows]
        history_vector = np.concatenate(
            history_vectors + [np.asarray([window_config.forecast_horizon_steps], dtype=float)]
        )
        samples.append(
            ForecastSample(
                source_window_id=pattern_window.window_id,
                history_window_ids=[window.window_id for window in history_windows],
                history_vector=history_vector,
                forecast_horizon_steps=window_config.forecast_horizon_steps,
                target_window_id=target_window.window_id,
                target_pattern_vector=target_window.feature_vector,
                target_pattern_id=labels_by_window_id.get(target_window.window_id),
            )
        )
    return samples
