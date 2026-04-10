from __future__ import annotations

import pandas as pd

from weather_patterns.config import WindowConfig
from weather_patterns.models import ExtremaEvent, ExtremaWindow, PeakEvent


def build_extrema_windows(
    signal_frame: pd.DataFrame,
    extrema_events: list[ExtremaEvent],
    peak_events: list[PeakEvent],
    config: WindowConfig,
    time_column: str = "date",
) -> list[ExtremaWindow]:
    windows: list[ExtremaWindow] = []
    max_start = len(signal_frame) - config.length_steps + 1
    if max_start <= 0:
        return windows

    event_left = 0
    event_right = 0
    peak_left = 0
    peak_right = 0

    for window_id, start_index in enumerate(range(0, max_start, config.stride_steps)):
        end_index = start_index + config.length_steps - 1
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
