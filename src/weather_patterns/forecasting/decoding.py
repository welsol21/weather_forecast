from __future__ import annotations

import numpy as np
import pandas as pd

from weather_patterns.models import ForecastResult, TimePlaceholders
from weather_patterns.pattern.representation import (
    INTER_FEATURES,
    INTRA_FEATURES,
    PEAK_HAZARD_FEATURES,
)

_CONVERGENCE_DIMS_PER_CHANNEL = 9


def _split_pattern_vector(
    pattern_vector: np.ndarray,
    channels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    channel_count = len(channels)
    intra_size = len(INTRA_FEATURES) * channel_count
    pair_labels = [
        f"{left}__{right}"
        for left_index, left in enumerate(channels)
        for right in channels[left_index + 1 :]
    ]
    inter_size = len(INTER_FEATURES) * len(pair_labels)
    peak_hazard_size = len(PEAK_HAZARD_FEATURES) * channel_count
    time_size = 8 + channel_count

    expected_size = intra_size + inter_size + peak_hazard_size + time_size
    if len(pattern_vector) != expected_size:
        raise ValueError(
            f"Unexpected pattern vector size: expected {expected_size}, got {len(pattern_vector)}."
        )

    intra_end = intra_size
    inter_end = intra_end + inter_size
    peak_hazard_end = inter_end + peak_hazard_size

    intra_matrix = pd.DataFrame(
        pattern_vector[:intra_end].reshape(len(INTRA_FEATURES), channel_count),
        index=INTRA_FEATURES,
        columns=channels,
    )
    inter_matrix = pd.DataFrame(
        pattern_vector[intra_end:inter_end].reshape(len(INTER_FEATURES), len(pair_labels)),
        index=INTER_FEATURES,
        columns=pair_labels,
    )
    peak_hazard_matrix = pd.DataFrame(
        pattern_vector[inter_end:peak_hazard_end].reshape(len(PEAK_HAZARD_FEATURES), channel_count),
        index=PEAK_HAZARD_FEATURES,
        columns=channels,
    )
    time_block = pattern_vector[peak_hazard_end:]
    return intra_matrix, inter_matrix, peak_hazard_matrix, time_block


def _decode_time_placeholders(time_block: np.ndarray, channels: list[str]) -> TimePlaceholders:
    base_values = time_block[:8]
    duration_values = time_block[8:]
    duration_map = {
        channel: float(duration_values[index])
        for index, channel in enumerate(channels)
    }
    normalized_event_count = max(int(round(float(base_values[7]))), 0)
    return TimePlaceholders(
        time_step_hours=float(base_values[0]),
        window_length_steps=max(int(round(float(base_values[1]))), 1),
        forecast_horizon_steps=max(int(round(float(base_values[2]))), 1),
        mean_inter_event_gap_steps=float(base_values[3]),
        var_inter_event_gap_steps=float(base_values[4]),
        mean_peak_width_steps=float(base_values[5]),
        max_peak_width_steps=float(base_values[6]),
        duration_over_threshold_by_channel=duration_map,
        normalized_event_positions=[0.0] * normalized_event_count,
    )


def _bounded(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        lower, upper = upper, lower
    return float(min(max(value, lower), upper))


def _decode_channel_value_sequence(
    intra_matrices: list[pd.DataFrame],
    channel: str,
) -> list[float]:
    decoded: list[float] = []
    total_steps = max(len(intra_matrices) - 1, 1)

    for index, intra_matrix in enumerate(intra_matrices):
        current_value = float(intra_matrix.loc["current_value", channel])
        delta_1 = float(intra_matrix.loc["delta_1", channel])
        delta_6 = float(intra_matrix.loc["delta_6", channel]) / 6.0
        delta_24 = float(intra_matrix.loc["delta_24", channel]) / 24.0
        second_diff = float(intra_matrix.loc["second_diff", channel])
        mean_value = float(intra_matrix.loc["mean_on_window", channel])
        min_value = float(intra_matrix.loc["min_on_window", channel])
        max_value = float(intra_matrix.loc["max_on_window", channel])
        value_range = abs(float(intra_matrix.loc["range_on_window", channel]))

        position = index / total_steps
        local_slope = 0.5 * delta_1 + 0.3 * delta_6 + 0.2 * delta_24
        curvature_adjustment = 0.5 * second_diff * position * (1.0 - position)
        mean_reversion = 0.15 * (mean_value - current_value)
        raw_value = current_value + local_slope * position + curvature_adjustment + mean_reversion

        if np.isclose(value_range, 0.0) and np.isclose(min_value, max_value):
            decoded.append(current_value if np.isclose(current_value, 0.0) is False else raw_value)
            continue

        padding = max(0.05 * value_range, 1e-6)
        decoded.append(_bounded(raw_value, min_value - padding, max_value + padding))

    return decoded


def _build_interval_timestamps(
    forecast_time: pd.Timestamp,
    horizon_steps: int,
    time_step_hours: float,
) -> list[pd.Timestamp]:
    step_hours = max(time_step_hours, 1e-6)
    return [
        forecast_time + pd.to_timedelta(step_index * step_hours, unit="h")
        for step_index in range(1, horizon_steps + 1)
    ]


def _interpolate_interval_values(
    source_timestamps: list[pd.Timestamp],
    source_values: list[float],
    interval_timestamps: list[pd.Timestamp],
) -> list[float]:
    if not source_timestamps or not source_values or not interval_timestamps:
        return []
    if len(source_timestamps) != len(source_values):
        raise ValueError("Source timestamps and values must have the same length.")
    if len(source_timestamps) == 1:
        return [float(source_values[0]) for _ in interval_timestamps]

    source_x = np.asarray(
        [(timestamp - source_timestamps[0]).total_seconds() / 3600.0 for timestamp in source_timestamps],
        dtype=float,
    )
    source_y = np.asarray(source_values, dtype=float)
    target_x = np.asarray(
        [(timestamp - source_timestamps[0]).total_seconds() / 3600.0 for timestamp in interval_timestamps],
        dtype=float,
    )
    interpolated = np.interp(
        target_x,
        source_x,
        source_y,
        left=source_y[0],
        right=source_y[-1],
    )
    return interpolated.astype(float).tolist()


def _interpolate_interval_peak_hazard(
    source_timestamps: list[pd.Timestamp],
    source_peak_hazard: list[dict[str, dict[str, float]]],
    interval_timestamps: list[pd.Timestamp],
    channels: list[str],
) -> list[dict[str, dict[str, float]]]:
    if not source_peak_hazard or not interval_timestamps:
        return []
    if len(source_peak_hazard) != len(source_timestamps):
        raise ValueError("Source hazard entries and timestamps must have the same length.")
    if len(source_peak_hazard) == 1:
        return [source_peak_hazard[0] for _ in interval_timestamps]

    source_x = np.asarray(
        [(timestamp - source_timestamps[0]).total_seconds() / 3600.0 for timestamp in source_timestamps],
        dtype=float,
    )
    target_x = np.asarray(
        [(timestamp - source_timestamps[0]).total_seconds() / 3600.0 for timestamp in interval_timestamps],
        dtype=float,
    )

    interval_peak_hazard: list[dict[str, dict[str, float]]] = []
    for target in target_x:
        interpolated_entry: dict[str, dict[str, float]] = {}
        for channel in channels:
            channel_features: dict[str, float] = {}
            for feature_name in PEAK_HAZARD_FEATURES:
                source_y = np.asarray(
                    [step_hazard[channel][feature_name] for step_hazard in source_peak_hazard],
                    dtype=float,
                )
                channel_features[feature_name] = float(
                    np.interp(target, source_x, source_y, left=source_y[0], right=source_y[-1])
                )
            interpolated_entry[channel] = channel_features
        interval_peak_hazard.append(interpolated_entry)
    return interval_peak_hazard


def _interpolate_interval_time_placeholders(
    source_timestamps: list[pd.Timestamp],
    source_time_placeholders: list[TimePlaceholders],
    interval_timestamps: list[pd.Timestamp],
    channels: list[str],
) -> list[TimePlaceholders]:
    if not source_time_placeholders or not interval_timestamps:
        return []
    if len(source_time_placeholders) != len(source_timestamps):
        raise ValueError("Source time placeholder entries and timestamps must have the same length.")
    if len(source_time_placeholders) == 1:
        return [source_time_placeholders[0] for _ in interval_timestamps]

    source_x = np.asarray(
        [(timestamp - source_timestamps[0]).total_seconds() / 3600.0 for timestamp in source_timestamps],
        dtype=float,
    )
    target_x = np.asarray(
        [(timestamp - source_timestamps[0]).total_seconds() / 3600.0 for timestamp in interval_timestamps],
        dtype=float,
    )

    def interpolate_scalar(values: list[float]) -> list[float]:
        source_y = np.asarray(values, dtype=float)
        return np.interp(target_x, source_x, source_y, left=source_y[0], right=source_y[-1]).astype(float).tolist()

    time_step_hours = interpolate_scalar([item.time_step_hours for item in source_time_placeholders])
    window_length_steps = interpolate_scalar([float(item.window_length_steps) for item in source_time_placeholders])
    forecast_horizon_steps = interpolate_scalar([float(item.forecast_horizon_steps) for item in source_time_placeholders])
    mean_inter_event_gap_steps = interpolate_scalar([item.mean_inter_event_gap_steps for item in source_time_placeholders])
    var_inter_event_gap_steps = interpolate_scalar([item.var_inter_event_gap_steps for item in source_time_placeholders])
    mean_peak_width_steps = interpolate_scalar([item.mean_peak_width_steps for item in source_time_placeholders])
    max_peak_width_steps = interpolate_scalar([item.max_peak_width_steps for item in source_time_placeholders])
    duration_maps = {
        channel: interpolate_scalar(
            [item.duration_over_threshold_by_channel.get(channel, 0.0) for item in source_time_placeholders]
        )
        for channel in channels
    }

    interval_time_placeholders: list[TimePlaceholders] = []
    for index in range(len(interval_timestamps)):
        interval_time_placeholders.append(
            TimePlaceholders(
                time_step_hours=float(time_step_hours[index]),
                window_length_steps=max(int(round(window_length_steps[index])), 1),
                forecast_horizon_steps=max(int(round(forecast_horizon_steps[index])), 1),
                mean_inter_event_gap_steps=float(mean_inter_event_gap_steps[index]),
                var_inter_event_gap_steps=float(var_inter_event_gap_steps[index]),
                mean_peak_width_steps=float(mean_peak_width_steps[index]),
                max_peak_width_steps=float(max_peak_width_steps[index]),
                duration_over_threshold_by_channel={
                    channel: float(duration_maps[channel][index])
                    for channel in channels
                },
                normalized_event_positions=[],
            )
        )
    return interval_time_placeholders


def decode_forecast_result_new_physics(
    result: ForecastResult,
    channels: list[str],
    initial_values: dict[str, float],
    channel_stds: dict[str, float],
    stride_hours: float,
) -> ForecastResult:
    """Decode a New Physics forecast result.

    Each predicted row is a 9×N_channels convergence feature vector.
    Values are reconstructed by chaining: the end of each pattern's local
    convergence function becomes the initial condition of the next.

    Parameters
    ----------
    initial_values : last observed channel values (placeholders)
    channel_stds   : per-channel std from the last observed window
    stride_hours   : hours per pattern window step (stride_steps × time_step_hours)
    """
    from weather_patterns.pattern.convergence import reconstruct_channel_sequence

    n_channels = len(channels)
    predicted_timestamps: list[pd.Timestamp] = []
    for step_index in range(result.predicted_window_count):
        predicted_timestamps.append(
            result.forecast_time + pd.to_timedelta((step_index + 1) * stride_hours, unit="h")
        )

    predicted_values: dict[str, list[float]] = {}
    for channel_idx, channel in enumerate(channels):
        initial = initial_values.get(channel, 0.0)
        std = channel_stds.get(channel, 1.0)
        predicted_values[channel] = reconstruct_channel_sequence(
            predicted_feature_matrix=result.predicted_pattern_matrix,
            channel_idx=channel_idx,
            n_channels=n_channels,
            initial_value=initial,
            channel_std=std,
            channel=channel,
        )

    # Build hourly interval timestamps (1-h resolution) by interpolation
    horizon_steps = result.horizon_steps
    time_step_hours = max(stride_hours / max(round(stride_hours), 1), 1e-6)
    interval_timestamps = _build_interval_timestamps(
        forecast_time=result.forecast_time,
        horizon_steps=horizon_steps,
        time_step_hours=time_step_hours,
    )
    predicted_interval_values = {
        channel: _interpolate_interval_values(
            source_timestamps=predicted_timestamps,
            source_values=predicted_values[channel],
            interval_timestamps=interval_timestamps,
        )
        for channel in channels
    }

    # Minimal time placeholders (stride-based)
    placeholder = TimePlaceholders(
        time_step_hours=stride_hours,
        window_length_steps=max(int(round(stride_hours)), 1),
        forecast_horizon_steps=horizon_steps,
        mean_inter_event_gap_steps=0.0,
        var_inter_event_gap_steps=0.0,
        mean_peak_width_steps=0.0,
        max_peak_width_steps=0.0,
        duration_over_threshold_by_channel={ch: 0.0 for ch in channels},
        normalized_event_positions=[],
    )
    predicted_time_placeholders = [placeholder] * len(predicted_timestamps)
    predicted_peak_hazard = [{ch: {f: 0.0 for f in PEAK_HAZARD_FEATURES} for ch in channels}] * len(predicted_timestamps)
    interval_placeholder = TimePlaceholders(
        time_step_hours=time_step_hours,
        window_length_steps=max(int(round(time_step_hours)), 1),
        forecast_horizon_steps=horizon_steps,
        mean_inter_event_gap_steps=0.0,
        var_inter_event_gap_steps=0.0,
        mean_peak_width_steps=0.0,
        max_peak_width_steps=0.0,
        duration_over_threshold_by_channel={ch: 0.0 for ch in channels},
        normalized_event_positions=[],
    )
    predicted_interval_time_placeholders = [interval_placeholder] * len(interval_timestamps)
    predicted_interval_peak_hazard = [{ch: {f: 0.0 for f in PEAK_HAZARD_FEATURES} for ch in channels}] * len(interval_timestamps)

    result.predicted_values = predicted_values
    result.predicted_timestamps = predicted_timestamps
    result.predicted_interval_timestamps = interval_timestamps
    result.predicted_interval_values = predicted_interval_values
    result.predicted_time_placeholders = predicted_time_placeholders
    result.predicted_peak_hazard = predicted_peak_hazard
    result.predicted_interval_time_placeholders = predicted_interval_time_placeholders
    result.predicted_interval_peak_hazard = predicted_interval_peak_hazard
    return result


def decode_forecast_result(
    result: ForecastResult,
    channels: list[str],
) -> ForecastResult:
    predicted_time_placeholders: list[TimePlaceholders] = []
    predicted_peak_hazard: list[dict[str, dict[str, float]]] = []
    predicted_timestamps: list[pd.Timestamp] = []
    intra_matrices: list[pd.DataFrame] = []

    for step_index, step_vector in enumerate(result.predicted_pattern_matrix):
        intra_matrix, _, peak_hazard_matrix, time_block = _split_pattern_vector(step_vector, channels)
        intra_matrices.append(intra_matrix)
        time_placeholders = _decode_time_placeholders(time_block, channels)
        predicted_time_placeholders.append(time_placeholders)
        step_hours = max(time_placeholders.time_step_hours, 1e-6)
        predicted_timestamps.append(
            result.forecast_time + pd.to_timedelta((step_index + 1) * step_hours, unit="h")
        )

        step_peak_hazard: dict[str, dict[str, float]] = {}
        for channel in channels:
            step_peak_hazard[channel] = {
                feature_name: float(peak_hazard_matrix.loc[feature_name, channel])
                for feature_name in PEAK_HAZARD_FEATURES
            }
        predicted_peak_hazard.append(step_peak_hazard)

    predicted_values = {
        channel: _decode_channel_value_sequence(intra_matrices, channel)
        for channel in channels
    }
    interval_timestamps = _build_interval_timestamps(
        forecast_time=result.forecast_time,
        horizon_steps=result.horizon_steps,
        time_step_hours=predicted_time_placeholders[0].time_step_hours if predicted_time_placeholders else 1.0,
    )
    predicted_interval_values = {
        channel: _interpolate_interval_values(
            source_timestamps=predicted_timestamps,
            source_values=predicted_values[channel],
            interval_timestamps=interval_timestamps,
        )
        for channel in channels
    }
    predicted_interval_time_placeholders = _interpolate_interval_time_placeholders(
        source_timestamps=predicted_timestamps,
        source_time_placeholders=predicted_time_placeholders,
        interval_timestamps=interval_timestamps,
        channels=channels,
    )
    predicted_interval_peak_hazard = _interpolate_interval_peak_hazard(
        source_timestamps=predicted_timestamps,
        source_peak_hazard=predicted_peak_hazard,
        interval_timestamps=interval_timestamps,
        channels=channels,
    )

    result.predicted_values = predicted_values
    result.predicted_timestamps = predicted_timestamps
    result.predicted_interval_timestamps = interval_timestamps
    result.predicted_interval_values = predicted_interval_values
    result.predicted_time_placeholders = predicted_time_placeholders
    result.predicted_peak_hazard = predicted_peak_hazard
    result.predicted_interval_time_placeholders = predicted_interval_time_placeholders
    result.predicted_interval_peak_hazard = predicted_interval_peak_hazard
    return result
