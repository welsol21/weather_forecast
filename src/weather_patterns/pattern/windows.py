from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from weather_patterns.config import WindowConfig
from weather_patterns.models import ExtremaEvent, ExtremaWindow, PeakEvent


def _build_windows_from_ranges(
    signal_frame: pd.DataFrame,
    ranges: Sequence[tuple[int, int]],
    extrema_events: list[ExtremaEvent],
    peak_events: list[PeakEvent],
    time_column: str = "date",
) -> list[ExtremaWindow]:
    windows: list[ExtremaWindow] = []
    event_left = 0
    event_right = 0
    peak_left = 0
    peak_right = 0

    for window_id, (start_index, end_index) in enumerate(ranges):
        if end_index < start_index:
            continue
        start_time = pd.Timestamp(signal_frame.iloc[start_index][time_column])
        end_time = pd.Timestamp(signal_frame.iloc[end_index][time_column])

        while event_left < len(extrema_events) and extrema_events[event_left].index < start_index:
            event_left += 1
        if event_right < event_left:
            event_right = event_left
        while event_right < len(extrema_events) and extrema_events[event_right].index <= end_index:
            event_right += 1
        events_in_window = extrema_events[event_left:event_right]

        while peak_left < len(peak_events) and peak_events[peak_left].index < start_index:
            peak_left += 1
        if peak_right < peak_left:
            peak_right = peak_left
        while peak_right < len(peak_events) and peak_events[peak_right].index <= end_index:
            peak_right += 1
        peaks_in_window = peak_events[peak_left:peak_right]

        windows.append(
            ExtremaWindow(
                window_id=window_id,
                start_time=start_time,
                end_time=end_time,
                start_index=start_index,
                end_index=end_index,
                events=events_in_window,
                peaks=peaks_in_window,
            )
        )
    return windows


def build_extrema_windows(
    signal_frame: pd.DataFrame,
    extrema_events: list[ExtremaEvent],
    peak_events: list[PeakEvent],
    config: WindowConfig,
    time_column: str = "date",
) -> list[ExtremaWindow]:
    max_start = len(signal_frame) - config.length_steps + 1
    if max_start <= 0:
        return []
    ranges = [
        (start_index, start_index + config.length_steps - 1)
        for start_index in range(0, max_start, config.stride_steps)
    ]
    return _build_windows_from_ranges(
        signal_frame,
        ranges,
        extrema_events,
        peak_events,
        time_column=time_column,
    )


def _is_circular_channel(channel: str) -> bool:
    return channel == "wind_direction"


def _unwrap_channel(values: np.ndarray, channel: str) -> np.ndarray:
    if not _is_circular_channel(channel):
        return values.astype(float, copy=True)
    return np.rad2deg(np.unwrap(np.deg2rad(values.astype(float))))


def _wrap_prediction(value: float, channel: str) -> float:
    if not _is_circular_channel(channel):
        return float(value)
    return float(np.mod(value, 360.0))


def _prediction_error(actual: float, predicted: float, channel: str) -> float:
    if not _is_circular_channel(channel):
        return float(abs(actual - predicted))
    delta = (actual - predicted + 180.0) % 360.0 - 180.0
    return float(abs(delta))


def _predict_level(history: np.ndarray) -> float:
    return float(history[-1])


def _predict_velocity(history: np.ndarray) -> float:
    return float(2.0 * history[-1] - history[-2])


def _predict_acceleration(history: np.ndarray) -> float:
    return float(3.0 * history[-1] - 3.0 * history[-2] + history[-3])


def _predict_local_ar2(history: np.ndarray, fit_window: int) -> float:
    usable = history[-max(fit_window, 3) :]
    design = np.column_stack(
        [
            np.ones((usable.size - 2,), dtype=float),
            usable[1:-1],
            usable[:-2],
        ]
    )
    target = usable[2:]
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    intercept, phi1, phi2 = coeffs.tolist()
    return float(intercept + phi1 * usable[-1] + phi2 * usable[-2])


def _best_predictor(
    history: np.ndarray,
    actual: float,
    channel: str,
    fit_window: int,
) -> str:
    candidates: list[tuple[float, str]] = [
        (_prediction_error(actual, _wrap_prediction(_predict_level(history), channel), channel), "level"),
    ]
    if history.size >= 2:
        candidates.append(
            (_prediction_error(actual, _wrap_prediction(_predict_velocity(history), channel), channel), "velocity")
        )
    if history.size >= 3:
        candidates.append(
            (
                _prediction_error(actual, _wrap_prediction(_predict_acceleration(history), channel), channel),
                "acceleration",
            )
        )
        candidates.append(
            (
                _prediction_error(actual, _wrap_prediction(_predict_local_ar2(history, fit_window), channel), channel),
                "local_ar2",
            )
        )
    return min(candidates, key=lambda item: item[0])[1]


def _suppress_short_runs(labels: list[str | None], min_run_steps: int) -> list[str | None]:
    if not labels:
        return labels
    smoothed = labels[:]
    changed = True
    while changed:
        changed = False
        run_start = 0
        current = smoothed[0]
        runs: list[tuple[int, int, str | None]] = []
        for index in range(1, len(smoothed)):
            if smoothed[index] != current:
                runs.append((run_start, index - 1, current))
                run_start = index
                current = smoothed[index]
        runs.append((run_start, len(smoothed) - 1, current))
        for run_index, (start_index, end_index, label) in enumerate(runs):
            if label is None:
                continue
            run_length = end_index - start_index + 1
            if run_length >= min_run_steps:
                continue
            left_label = runs[run_index - 1][2] if run_index > 0 else None
            right_label = runs[run_index + 1][2] if run_index + 1 < len(runs) else None
            replacement = left_label if left_label == right_label and left_label is not None else left_label or right_label
            if replacement is None or replacement == label:
                continue
            for position in range(start_index, end_index + 1):
                smoothed[position] = replacement
            changed = True
            break
    return smoothed


def _merge_short_ranges(ranges: list[tuple[int, int]], min_window_steps: int) -> list[tuple[int, int]]:
    if not ranges:
        return ranges
    merged: list[list[int]] = [[start, end] for start, end in ranges]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for index, (start_index, end_index) in enumerate(merged):
            if end_index - start_index + 1 >= min_window_steps:
                continue
            if index == 0:
                merged[index + 1][0] = start_index
            else:
                merged[index - 1][1] = end_index
            del merged[index]
            changed = True
            break
    return [(start, end) for start, end in merged]


def build_predictor_windows(
    signal_frame: pd.DataFrame,
    channels: list[str],
    extrema_events: list[ExtremaEvent],
    peak_events: list[PeakEvent],
    config: WindowConfig,
    time_column: str = "date",
) -> list[ExtremaWindow]:
    row_count = len(signal_frame)
    history_window = max(config.predictor_history_window_steps, 3)
    if row_count <= history_window + 1:
        return []

    channel_labels: dict[str, list[str | None]] = {}
    for channel in channels:
        series = pd.to_numeric(signal_frame[f"smoothed_{channel}"], errors="coerce").to_numpy(dtype=float)
        series = _unwrap_channel(series, channel)
        labels: list[str | None] = [None] * row_count
        for index in range(history_window, row_count):
            history = series[index - history_window : index]
            actual = series[index]
            if not np.isfinite(actual) or not np.isfinite(history).all():
                continue
            labels[index] = _best_predictor(
                history,
                _wrap_prediction(actual, channel),
                channel=channel,
                fit_window=config.predictor_fit_window_steps,
            )
        channel_labels[channel] = _suppress_short_runs(labels, min_run_steps=config.predictor_min_run_steps)

    boundaries = [history_window]
    for index in range(history_window + 1, row_count):
        changed_channels = 0
        for channel in channels:
            previous = channel_labels[channel][index - 1]
            current = channel_labels[channel][index]
            if previous is None or current is None:
                continue
            if previous != current:
                changed_channels += 1
        if changed_channels >= config.predictor_min_changed_channels:
            boundaries.append(index)
    if boundaries[-1] != row_count:
        boundaries.append(row_count)

    ranges: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        start_index = left
        end_index = right - 1
        if end_index >= start_index:
            ranges.append((start_index, end_index))
    ranges = _merge_short_ranges(ranges, min_window_steps=config.predictor_min_window_steps)

    return _build_windows_from_ranges(
        signal_frame,
        ranges,
        extrema_events,
        peak_events,
        time_column=time_column,
    )
