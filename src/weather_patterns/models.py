from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class LoadedWeatherDataset:
    dataframe: pd.DataFrame
    channel_columns: list[str]
    quality_columns: list[str]
    metadata_lines: list[str]
    source_path: Path


@dataclass(slots=True)
class ExtremaEvent:
    timestamp: pd.Timestamp
    channel: str
    sign: str
    amplitude: float
    value: float
    first_diff_value: float
    second_diff_value: float
    index: int


@dataclass(slots=True)
class PeakEvent:
    timestamp: pd.Timestamp
    channel: str
    sign: str
    peak_value: float
    prominence: float
    width_steps: float
    rise_slope: float
    fall_slope: float
    asymmetry: float
    left_base_value: float
    right_base_value: float
    index: int
    left_index: int
    right_index: int


@dataclass(slots=True)
class ExtremaWindow:
    window_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_index: int
    end_index: int
    events: list[ExtremaEvent] = field(default_factory=list)
    peaks: list[PeakEvent] = field(default_factory=list)


@dataclass(slots=True)
class TimePlaceholders:
    time_step_hours: float
    window_length_steps: int
    forecast_horizon_steps: int
    mean_inter_event_gap_steps: float
    var_inter_event_gap_steps: float
    mean_peak_width_steps: float
    max_peak_width_steps: float
    duration_over_threshold_by_channel: dict[str, float]
    normalized_event_positions: list[float] = field(default_factory=list)

    def to_feature_array(self, channel_order: list[str]) -> np.ndarray:
        vector: list[float] = [
            self.time_step_hours,
            float(self.window_length_steps),
            float(self.forecast_horizon_steps),
            self.mean_inter_event_gap_steps,
            self.var_inter_event_gap_steps,
            self.mean_peak_width_steps,
            self.max_peak_width_steps,
            float(len(self.normalized_event_positions)),
        ]
        vector.extend(
            float(self.duration_over_threshold_by_channel.get(channel, 0.0))
            for channel in channel_order
        )
        return np.asarray(vector, dtype=float)


@dataclass(slots=True)
class PatternWindow:
    window_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    channels: list[str]
    intra_matrix: pd.DataFrame
    inter_matrix: pd.DataFrame
    peak_hazard_matrix: pd.DataFrame
    time_placeholders: TimePlaceholders
    feature_vector: np.ndarray
    extrema_window: ExtremaWindow


@dataclass(slots=True)
class PatternPrototype:
    pattern_id: int
    centroid: np.ndarray
    member_window_ids: list[int]


@dataclass(slots=True)
class ForecastSample:
    source_window_id: int
    history_window_ids: list[int]
    history_vector: np.ndarray
    forecast_horizon_steps: int
    target_window_id: int
    target_pattern_vector: np.ndarray
    target_pattern_id: int | None = None


@dataclass(slots=True)
class ForecastResult:
    forecast_time: pd.Timestamp
    horizon_steps: int
    predicted_pattern_id: int | None
    predicted_pattern_vector: np.ndarray
    predicted_values: dict[str, float]
    predicted_time_placeholders: TimePlaceholders
    predicted_peak_hazard: dict[str, dict[str, float]]


@dataclass(slots=True)
class DiscoveryResult:
    labels_by_window_id: dict[int, int]
    prototypes: list[PatternPrototype]


@dataclass(slots=True)
class PipelineArtifacts:
    dataset: LoadedWeatherDataset
    signal_frame: pd.DataFrame
    extrema_events: list[ExtremaEvent]
    peak_events: list[PeakEvent]
    extrema_windows: list[ExtremaWindow]
    pattern_windows: list[PatternWindow]
    discovery_result: DiscoveryResult
    forecast_samples: list[ForecastSample]

    def summary(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.dataset.dataframe)),
            "channels": self.dataset.channel_columns,
            "extrema_events": len(self.extrema_events),
            "peak_events": len(self.peak_events),
            "extrema_windows": len(self.extrema_windows),
            "pattern_windows": len(self.pattern_windows),
            "forecast_samples": len(self.forecast_samples),
            "discovered_patterns": len(self.discovery_result.prototypes),
        }
