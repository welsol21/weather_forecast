from __future__ import annotations

import numpy as np

from weather_patterns.io.artifacts import read_forecast_sequence_dataset_jsonl
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


def _decode_optional_pattern_ids(pattern_ids: list[object]) -> list[int | None]:
    decoded: list[int | None] = []
    for value in pattern_ids:
        if value is None:
            decoded.append(None)
        else:
            decoded.append(int(value))
    return decoded


def load_forecast_training_dataset_jsonl(path: str) -> ForecastTrainingDataset:
    records = read_forecast_sequence_dataset_jsonl(path)
    samples: list[ForecastSample] = []
    for record in records:
        samples.append(
            ForecastSample(
                source_window_id=int(record["source_window_id"]),
                history_window_ids=[int(value) for value in record["history_window_ids"]],
                history_pattern_matrix=np.asarray(record["history_pattern_matrix"], dtype=np.float32),
                history_vector=np.asarray(record["history_vector"], dtype=np.float32),
                forecast_horizon_steps=int(record["forecast_horizon_steps"]),
                forecast_window_count=int(record["forecast_window_count"]),
                target_window_ids=[int(value) for value in record["target_window_ids"]],
                target_pattern_matrix=np.asarray(record["target_pattern_matrix"], dtype=np.float32),
                history_pattern_ids=_decode_optional_pattern_ids(record["history_pattern_ids"]),
                target_pattern_ids=_decode_optional_pattern_ids(record["target_pattern_ids"]),
            )
        )
    return build_forecast_training_dataset(samples)
