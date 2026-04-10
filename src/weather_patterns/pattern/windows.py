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

    for window_id, start_index in enumerate(range(0, max_start, config.stride_steps)):
        end_index = start_index + config.length_steps - 1
        start_time = pd.Timestamp(signal_frame.iloc[start_index][time_column])
        end_time = pd.Timestamp(signal_frame.iloc[end_index][time_column])
        events_in_window = [
            event
            for event in extrema_events
            if start_index <= event.index <= end_index
        ]
        peaks_in_window = [
            peak
            for peak in peak_events
            if start_index <= peak.index <= end_index
        ]
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
