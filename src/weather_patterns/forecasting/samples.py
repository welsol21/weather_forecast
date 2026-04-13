from __future__ import annotations

import math

import numpy as np
import pandas as pd

from weather_patterns.config import ForecastConfig, WindowConfig
from weather_patterns.models import ForecastSample, PatternWindow


def resolve_target_window_count(
    window_config: WindowConfig,
    forecast_config: ForecastConfig,
) -> int:
    if forecast_config.target_window_count is not None:
        return max(1, forecast_config.target_window_count)
    if window_config.segmentation_strategy == "predictor":
        return max(1, math.ceil(window_config.forecast_horizon_steps / max(window_config.predictor_min_window_steps, 1)))
    # "hierarchical" and "extrema" both use stride-based sliding windows of fixed length,
    # so the number of target windows equals horizon / stride.
    return max(1, math.ceil(window_config.forecast_horizon_steps / window_config.stride_steps))


def build_forecast_samples(
    pattern_windows: list[PatternWindow],
    labels_by_window_id: dict[int, int],
    window_config: WindowConfig,
    forecast_config: ForecastConfig,
) -> list[ForecastSample]:
    if not pattern_windows:
        return []

    target_window_count = resolve_target_window_count(window_config, forecast_config)
    samples: list[ForecastSample] = []
    windows_by_start = {
        pattern_window.extrema_window.start_index: pattern_window
        for pattern_window in pattern_windows
    }
    for current_position, pattern_window in enumerate(pattern_windows):
        history_start = current_position - forecast_config.history_window_count + 1
        if history_start < 0:
            continue
        if window_config.segmentation_strategy == "predictor":
            horizon_start = pattern_window.start_time + pd.Timedelta(hours=window_config.forecast_horizon_steps)
            first_target_position: int | None = None
            for future_position in range(current_position + 1, len(pattern_windows)):
                if pattern_windows[future_position].start_time >= horizon_start:
                    first_target_position = future_position
                    break
            if first_target_position is None:
                continue
            target_windows = pattern_windows[first_target_position : first_target_position + target_window_count]
            if len(target_windows) != target_window_count:
                continue
        elif window_config.segmentation_strategy == "hierarchical":
            # History and targets must belong to the same predictor regime block.
            # Cross-block history vectors are semantically incorrect: the dynamics
            # that generated those windows follow a *different* local predictor.
            history_windows_candidate = pattern_windows[history_start : current_position + 1]
            if any(w.parent_block_id != pattern_window.parent_block_id for w in history_windows_candidate):
                continue
            future_start = pattern_window.extrema_window.start_index + window_config.forecast_horizon_steps
            target_windows = []
            for offset in range(target_window_count):
                target_start = future_start + offset * window_config.stride_steps
                target_window = windows_by_start.get(target_start)
                if target_window is None:
                    target_windows = []
                    break
                if target_window.parent_block_id != pattern_window.parent_block_id:
                    target_windows = []
                    break
                target_windows.append(target_window)
        else:
            future_start = pattern_window.extrema_window.start_index + window_config.forecast_horizon_steps
            target_windows = []
            for offset in range(target_window_count):
                target_start = future_start + offset * window_config.stride_steps
                target_window = windows_by_start.get(target_start)
                if target_window is None:
                    target_windows = []
                    break
                target_windows.append(target_window)
        if not target_windows:
            continue
        history_windows = (
            history_windows_candidate
            if window_config.segmentation_strategy == "hierarchical"
            else pattern_windows[history_start : current_position + 1]
        )
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
