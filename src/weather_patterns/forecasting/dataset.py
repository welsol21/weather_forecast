from __future__ import annotations

import numpy as np

from weather_patterns.models import ForecastSample, ForecastTrainingDataset


def _encode_pattern_ids(pattern_ids: list[int | None], expected_length: int) -> np.ndarray:
    encoded = np.full(expected_length, -1, dtype=np.int64)
    for index, pattern_id in enumerate(pattern_ids[:expected_length]):
        if pattern_id is not None:
            encoded[index] = int(pattern_id)
    return encoded


def build_forecast_training_dataset(samples: list[ForecastSample]) -> ForecastTrainingDataset:
    if not samples:
        raise ValueError("At least one forecast sample is required to build a training dataset.")

    history_window_count = samples[0].history_pattern_matrix.shape[0]
    forecast_window_count = samples[0].target_pattern_matrix.shape[0]
    feature_dim = samples[0].target_pattern_matrix.shape[1]
    forecast_horizon_steps = samples[0].forecast_horizon_steps

    for sample in samples:
        if sample.history_pattern_matrix.shape != (history_window_count, feature_dim):
            raise ValueError("All history pattern matrices must share the same shape.")
        if sample.target_pattern_matrix.shape != (forecast_window_count, feature_dim):
            raise ValueError("All target pattern matrices must share the same shape.")
        if sample.forecast_horizon_steps != forecast_horizon_steps:
            raise ValueError("All forecast samples must share the same forecast horizon.")

    return ForecastTrainingDataset(
        source_window_ids=[sample.source_window_id for sample in samples],
        history_window_ids=[sample.history_window_ids for sample in samples],
        target_window_ids=[sample.target_window_ids for sample in samples],
        history_pattern_tensor=np.stack(
            [sample.history_pattern_matrix for sample in samples],
            axis=0,
        ).astype(np.float32),
        history_vector_matrix=np.stack(
            [sample.history_vector for sample in samples],
            axis=0,
        ).astype(np.float32),
        target_pattern_tensor=np.stack(
            [sample.target_pattern_matrix for sample in samples],
            axis=0,
        ).astype(np.float32),
        history_pattern_id_matrix=np.stack(
            [
                _encode_pattern_ids(sample.history_pattern_ids, history_window_count)
                for sample in samples
            ],
            axis=0,
        ),
        target_pattern_id_matrix=np.stack(
            [
                _encode_pattern_ids(sample.target_pattern_ids, forecast_window_count)
                for sample in samples
            ],
            axis=0,
        ),
        forecast_horizon_steps=forecast_horizon_steps,
        forecast_window_count=forecast_window_count,
        history_window_count=history_window_count,
        feature_dim=feature_dim,
    )
