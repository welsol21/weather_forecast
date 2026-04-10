from __future__ import annotations

import math

import numpy as np

from weather_patterns.config import ForecastConfig, WindowConfig
from weather_patterns.models import ForecastSample, PatternWindow


def resolve_target_window_count(
    window_config: WindowConfig,
    forecast_config: ForecastConfig,
) -> int:
    if forecast_config.target_window_count is not None:
        return max(1, forecast_config.target_window_count)
    return max(1, math.ceil(window_config.forecast_horizon_steps / window_config.stride_steps))


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
    target_window_count = resolve_target_window_count(window_config, forecast_config)
    samples: list[ForecastSample] = []
    for current_position, pattern_window in enumerate(pattern_windows):
        history_start = current_position - forecast_config.history_window_count + 1
        if history_start < 0:
            continue
        future_start = pattern_window.extrema_window.start_index + window_config.forecast_horizon_steps
        target_windows: list[PatternWindow] = []
        for offset in range(target_window_count):
            target_start = future_start + offset * window_config.stride_steps
            target_window = windows_by_start.get(target_start)
            if target_window is None:
                target_windows = []
                break
            target_windows.append(target_window)
        if not target_windows:
            continue
        history_windows = pattern_windows[history_start : current_position + 1]
        history_vectors = [window.feature_vector for window in history_windows]
        history_vector = np.concatenate(
            history_vectors
            + [
                np.asarray(
                    [
                        float(window_config.forecast_horizon_steps),
                        float(target_window_count),
                    ],
                    dtype=float,
                )
            ]
        )
        samples.append(
            ForecastSample(
                source_window_id=pattern_window.window_id,
                history_window_ids=[window.window_id for window in history_windows],
                history_pattern_matrix=np.vstack([window.feature_vector for window in history_windows]),
                history_vector=history_vector,
                forecast_horizon_steps=window_config.forecast_horizon_steps,
                forecast_window_count=target_window_count,
                target_window_ids=[window.window_id for window in target_windows],
                target_pattern_matrix=np.vstack([window.feature_vector for window in target_windows]),
                history_pattern_ids=[labels_by_window_id.get(window.window_id) for window in history_windows],
                target_pattern_ids=[labels_by_window_id.get(window.window_id) for window in target_windows],
            )
        )
    return samples
