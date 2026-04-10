from __future__ import annotations

import math

import pandas as pd

from weather_patterns.models import PeakEvent


def _walk_left(series: list[float] | tuple[float, ...] | pd.Series, peak_index: int, sign: str) -> int:
    idx = peak_index
    if sign == "max":
        while idx > 0 and series[idx - 1] <= series[idx]:
            idx -= 1
    else:
        while idx > 0 and series[idx - 1] >= series[idx]:
            idx -= 1
    return idx


def _walk_right(series: list[float] | tuple[float, ...] | pd.Series, peak_index: int, sign: str) -> int:
    idx = peak_index
    if sign == "max":
        while idx < len(series) - 1 and series[idx + 1] <= series[idx]:
            idx += 1
    else:
        while idx < len(series) - 1 and series[idx + 1] >= series[idx]:
            idx += 1
    return idx


def detect_peaks(signal_frame: pd.DataFrame, channels: list[str], time_column: str = "date") -> list[PeakEvent]:
    peaks: list[PeakEvent] = []
    timestamps = pd.to_datetime(signal_frame[time_column], errors="coerce").to_numpy()
    for channel in channels:
        series = pd.to_numeric(signal_frame[f"smoothed_{channel}"], errors="coerce").to_numpy(dtype=float)
        for idx in range(1, len(series) - 1):
            prev_value = series[idx - 1]
            current_value = series[idx]
            next_value = series[idx + 1]
            if any(math.isnan(v) for v in (prev_value, current_value, next_value)):
                continue

            sign: str | None = None
            if prev_value < current_value >= next_value:
                sign = "max"
            elif prev_value > current_value <= next_value:
                sign = "min"
            if sign is None:
                continue

            left_index = _walk_left(series, idx, sign)
            right_index = _walk_right(series, idx, sign)
            left_base_value = float(series[left_index])
            right_base_value = float(series[right_index])

            if sign == "max":
                prominence = current_value - max(left_base_value, right_base_value)
            else:
                prominence = min(left_base_value, right_base_value) - current_value

            rise_duration = max(idx - left_index, 1)
            fall_duration = max(right_index - idx, 1)
            rise_slope = abs((current_value - left_base_value) / rise_duration)
            fall_slope = abs((current_value - right_base_value) / fall_duration)
            asymmetry = float(rise_duration / fall_duration)

            peaks.append(
                PeakEvent(
                    timestamp=pd.Timestamp(timestamps[idx]),
                    channel=channel,
                    sign=sign,
                    peak_value=float(current_value),
                    prominence=float(max(prominence, 0.0)),
                    width_steps=float(max(right_index - left_index, 1)),
                    rise_slope=float(rise_slope),
                    fall_slope=float(fall_slope),
                    asymmetry=asymmetry,
                    left_base_value=left_base_value,
                    right_base_value=right_base_value,
                    index=idx,
                    left_index=left_index,
                    right_index=right_index,
                )
            )
    peaks.sort(key=lambda peak: (peak.timestamp, peak.channel, peak.index))
    return peaks
