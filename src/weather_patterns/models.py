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
    intra_matrix: np.ndarray
    inter_matrix: np.ndarray
    peak_hazard_matrix: np.ndarray
    time_placeholders: TimePlaceholders
    feature_vector: np.ndarray
    extrema_window: ExtremaWindow


@dataclass(slots=True)
class PatternPrototype:
    pattern_id: int
    centroid: np.ndarray
    member_window_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ForecastSample:
    source_window_id: int
    history_window_ids: list[int]
    history_pattern_matrix: np.ndarray
    history_vector: np.ndarray
    forecast_horizon_steps: int
    forecast_window_count: int
    target_window_ids: list[int]
    target_pattern_matrix: np.ndarray
    history_pattern_ids: list[int | None] = field(default_factory=list)
    target_pattern_ids: list[int | None] = field(default_factory=list)


@dataclass(slots=True)
class ForecastResult:
    forecast_time: pd.Timestamp
    horizon_steps: int
    predicted_window_count: int
    predicted_pattern_ids: list[int | None]
    predicted_pattern_matrix: np.ndarray
    predicted_timestamps: list[pd.Timestamp]
    predicted_values: dict[str, list[float]]
    predicted_interval_timestamps: list[pd.Timestamp]
    predicted_interval_values: dict[str, list[float]]
    predicted_time_placeholders: list[TimePlaceholders]
    predicted_peak_hazard: list[dict[str, dict[str, float]]]
    predicted_interval_time_placeholders: list[TimePlaceholders]
    predicted_interval_peak_hazard: list[dict[str, dict[str, float]]]

    def to_value_frame(self) -> pd.DataFrame:
        if not self.predicted_values:
            return pd.DataFrame(index=pd.Index(self.predicted_timestamps, name="timestamp"))
        return pd.DataFrame(
            self.predicted_values,
            index=pd.Index(self.predicted_timestamps, name="timestamp"),
        )

    def to_peak_hazard_frame(self) -> pd.DataFrame:
        if not self.predicted_peak_hazard:
            return pd.DataFrame(index=pd.Index(self.predicted_timestamps, name="timestamp"))

        rows: list[dict[str, float]] = []
        for step_hazard in self.predicted_peak_hazard:
            row: dict[str, float] = {}
            for channel, features in step_hazard.items():
                for feature_name, value in features.items():
                    row[f"{channel}__{feature_name}"] = value
            rows.append(row)
        return pd.DataFrame(
            rows,
            index=pd.Index(self.predicted_timestamps, name="timestamp"),
        )

    def to_interval_value_frame(self) -> pd.DataFrame:
        if not self.predicted_interval_values:
            return pd.DataFrame(index=pd.Index(self.predicted_interval_timestamps, name="timestamp"))
        return pd.DataFrame(
            self.predicted_interval_values,
            index=pd.Index(self.predicted_interval_timestamps, name="timestamp"),
        )

    def to_interval_peak_hazard_frame(self) -> pd.DataFrame:
        if not self.predicted_interval_peak_hazard:
            return pd.DataFrame(index=pd.Index(self.predicted_interval_timestamps, name="timestamp"))

        rows: list[dict[str, float]] = []
        for step_hazard in self.predicted_interval_peak_hazard:
            row: dict[str, float] = {}
            for channel, features in step_hazard.items():
                for feature_name, value in features.items():
                    row[f"{channel}__{feature_name}"] = value
            rows.append(row)
        return pd.DataFrame(
            rows,
            index=pd.Index(self.predicted_interval_timestamps, name="timestamp"),
        )


@dataclass(slots=True)
class DiscoveryResult:
    labels_by_window_id: dict[int, int]
    prototypes: list[PatternPrototype]
    strategy: str = ""
    selected_cluster_count: int = 0
    selected_quality_score: float = 0.0
    selection_metric_name: str = ""
    candidate_quality: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ForecastTrainingDataset:
    source_window_ids: list[int]
    history_window_ids: list[list[int]]
    target_window_ids: list[list[int]]
    history_pattern_tensor: np.ndarray
    history_vector_matrix: np.ndarray
    target_pattern_tensor: np.ndarray
    history_pattern_id_matrix: np.ndarray
    target_pattern_id_matrix: np.ndarray
    forecast_horizon_steps: int
    forecast_window_count: int
    history_window_count: int
    feature_dim: int


@dataclass(slots=True)
class PreparedPatternWindowsArtifacts:
    dataset: LoadedWeatherDataset
    signal_frame: pd.DataFrame
    extrema_events: list[ExtremaEvent]
    peak_events: list[PeakEvent]
    extrema_windows: list[ExtremaWindow]
    pattern_windows: list[PatternWindow]

    def summary(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.dataset.dataframe)),
            "channels": self.dataset.channel_columns,
            "extrema_events": len(self.extrema_events),
            "peak_events": len(self.peak_events),
            "extrema_windows": len(self.extrema_windows),
            "pattern_windows": len(self.pattern_windows),
        }


@dataclass(slots=True)
class DiscoveryArtifacts:
    pattern_windows: list[PatternWindow]
    discovery_result: DiscoveryResult
    forecast_samples: list[ForecastSample]

    def summary(self) -> dict[str, Any]:
        target_window_counts = [sample.forecast_window_count for sample in self.forecast_samples]
        return {
            "pattern_windows": len(self.pattern_windows),
            "forecast_samples": len(self.forecast_samples),
            "forecast_target_window_count": max(target_window_counts) if target_window_counts else 0,
            "discovered_patterns": len(self.discovery_result.prototypes),
            "discovery_strategy": self.discovery_result.strategy,
            "selected_cluster_count": int(self.discovery_result.selected_cluster_count),
            "selected_cluster_quality": float(self.discovery_result.selected_quality_score),
            "discovery_selection_metric": self.discovery_result.selection_metric_name,
            "discovery_candidate_quality": {
                str(key): float(value)
                for key, value in sorted(self.discovery_result.candidate_quality.items())
            },
        }


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
        target_window_counts = [sample.forecast_window_count for sample in self.forecast_samples]
        return {
            "rows": int(len(self.dataset.dataframe)),
            "channels": self.dataset.channel_columns,
            "extrema_events": len(self.extrema_events),
            "peak_events": len(self.peak_events),
            "extrema_windows": len(self.extrema_windows),
            "pattern_windows": len(self.pattern_windows),
            "forecast_samples": len(self.forecast_samples),
            "forecast_target_window_count": max(target_window_counts) if target_window_counts else 0,
            "discovered_patterns": len(self.discovery_result.prototypes),
            "discovery_strategy": self.discovery_result.strategy,
            "selected_cluster_count": int(self.discovery_result.selected_cluster_count),
            "selected_cluster_quality": float(self.discovery_result.selected_quality_score),
            "discovery_selection_metric": self.discovery_result.selection_metric_name,
            "discovery_candidate_quality": {
                str(key): float(value)
                for key, value in sorted(self.discovery_result.candidate_quality.items())
            },
        }
