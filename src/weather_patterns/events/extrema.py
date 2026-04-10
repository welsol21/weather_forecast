from __future__ import annotations

import math

import pandas as pd

from weather_patterns.models import ExtremaEvent


def detect_extrema(signal_frame: pd.DataFrame, channels: list[str], time_column: str = "date") -> list[ExtremaEvent]:
    events: list[ExtremaEvent] = []
    timestamps = pd.to_datetime(signal_frame[time_column], errors="coerce").to_numpy()
    for channel in channels:
        diff_column = f"diff1_{channel}"
        second_diff_column = f"diff2_{channel}"
        values = pd.to_numeric(signal_frame[diff_column], errors="coerce").to_numpy(dtype=float)
        smoothed_values = pd.to_numeric(signal_frame[f"smoothed_{channel}"], errors="coerce").to_numpy(dtype=float)
        second_diff_values = pd.to_numeric(signal_frame[second_diff_column], errors="coerce").to_numpy(dtype=float)
        for idx in range(1, len(values) - 1):
            prev_value = values[idx - 1]
            current_value = values[idx]
            next_value = values[idx + 1]
            if any(math.isnan(v) for v in (prev_value, current_value, next_value)):
                continue

            sign: str | None = None
            if prev_value < current_value >= next_value:
                sign = "max"
            elif prev_value > current_value <= next_value:
                sign = "min"

            if sign is None:
                continue

            amplitude = abs(current_value - 0.5 * (prev_value + next_value))
            events.append(
                ExtremaEvent(
                    timestamp=pd.Timestamp(timestamps[idx]),
                    channel=channel,
                    sign=sign,
                    amplitude=float(amplitude),
                    value=float(smoothed_values[idx]),
                    first_diff_value=float(current_value),
                    second_diff_value=float(second_diff_values[idx]),
                    index=idx,
                )
            )
    events.sort(key=lambda event: (event.timestamp, event.channel, event.index))
    return events
