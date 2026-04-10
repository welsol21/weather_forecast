from __future__ import annotations

import math

import pandas as pd

from weather_patterns.models import ExtremaEvent


def detect_extrema(signal_frame: pd.DataFrame, channels: list[str], time_column: str = "date") -> list[ExtremaEvent]:
    events: list[ExtremaEvent] = []
    for channel in channels:
        diff_column = f"diff1_{channel}"
        second_diff_column = f"diff2_{channel}"
        values = signal_frame[diff_column].astype(float)
        for idx in range(1, len(signal_frame) - 1):
            prev_value = values.iloc[idx - 1]
            current_value = values.iloc[idx]
            next_value = values.iloc[idx + 1]
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
                    timestamp=pd.Timestamp(signal_frame.iloc[idx][time_column]),
                    channel=channel,
                    sign=sign,
                    amplitude=float(amplitude),
                    value=float(signal_frame.iloc[idx][f"smoothed_{channel}"]),
                    first_diff_value=float(current_value),
                    second_diff_value=float(signal_frame.iloc[idx][second_diff_column]),
                    index=idx,
                )
            )
    events.sort(key=lambda event: (event.timestamp, event.channel, event.index))
    return events
