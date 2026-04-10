from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weather_patterns.config import PipelineConfig
from weather_patterns.data.loading import apply_quality_masks, load_weather_dataset
from weather_patterns.discovery.base import DiscoveryInput
from weather_patterns.discovery.kmeans import NumpyKMeansDiscovery
from weather_patterns.discovery.structural import StructuralPatternDiscovery
from weather_patterns.events.extrema import detect_extrema
from weather_patterns.events.peaks import detect_peaks
from weather_patterns.forecasting.samples import build_forecast_samples
from weather_patterns.models import PipelineArtifacts
from weather_patterns.pattern.representation import build_pattern_window, compute_channel_thresholds
from weather_patterns.pattern.representation import build_signal_channel_arrays, slice_signal_channel_arrays
from weather_patterns.pattern.windows import build_extrema_windows
from weather_patterns.signal.processing import build_signal_frame


def run_pipeline(csv_path: str | Path, config: PipelineConfig | None = None) -> PipelineArtifacts:
    active_config = config or PipelineConfig()
    dataset = load_weather_dataset(csv_path, active_config.dataset)
    cleaned_frame = apply_quality_masks(dataset)
    if active_config.max_rows is not None:
        cleaned_frame = cleaned_frame.iloc[: active_config.max_rows].copy()
        dataset.dataframe = cleaned_frame
    else:
        dataset.dataframe = cleaned_frame

    signal_frame = build_signal_frame(
        dataset.dataframe,
        dataset.channel_columns,
        active_config.smoothing,
    )
    extrema_events = detect_extrema(signal_frame, dataset.channel_columns)
    peak_events = detect_peaks(signal_frame, dataset.channel_columns)
    extrema_windows = build_extrema_windows(
        signal_frame,
        extrema_events,
        peak_events,
        active_config.window,
    )
    upper_thresholds, lower_thresholds = compute_channel_thresholds(
        dataset.dataframe,
        dataset.channel_columns,
        active_config.hazard,
    )
    raw_series, smoothed_series, diff1_series, diff2_series = build_signal_channel_arrays(
        signal_frame,
        dataset.channel_columns,
    )
    pattern_windows = [
        build_pattern_window(
            signal_frame,
            extrema_window,
            dataset.channel_columns,
            active_config,
            upper_thresholds,
            lower_thresholds,
            *slice_signal_channel_arrays(
                raw_series,
                smoothed_series,
                diff1_series,
                diff2_series,
                dataset.channel_columns,
                extrema_window.start_index,
                extrema_window.end_index,
            ),
        )
        for extrema_window in extrema_windows
    ]
    feature_matrix = (
        np.vstack([window.feature_vector for window in pattern_windows])
        if pattern_windows
        else np.empty((0, 0))
    )
    if active_config.discovery.strategy == "structural":
        discovery_strategy = StructuralPatternDiscovery(active_config.discovery)
    elif active_config.discovery.strategy == "kmeans":
        discovery_strategy = NumpyKMeansDiscovery(active_config.discovery)
    else:
        raise ValueError(f"Unsupported discovery strategy: {active_config.discovery.strategy}")

    discovery = discovery_strategy.fit_predict(
        DiscoveryInput(
            window_ids=[window.window_id for window in pattern_windows],
            feature_matrix=feature_matrix,
            pattern_windows=pattern_windows,
        )
    )
    forecast_samples = build_forecast_samples(
        pattern_windows,
        discovery.labels_by_window_id,
        active_config.window,
        active_config.forecast,
    )
    return PipelineArtifacts(
        dataset=dataset,
        signal_frame=signal_frame,
        extrema_events=extrema_events,
        peak_events=peak_events,
        extrema_windows=extrema_windows,
        pattern_windows=pattern_windows,
        discovery_result=discovery,
        forecast_samples=forecast_samples,
    )


def write_artifacts_summary(artifacts: PipelineArtifacts, output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(artifacts.summary(), indent=2),
        encoding="utf-8",
    )
    return summary_path


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _extrema_events_frame(artifacts: PipelineArtifacts) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": event.timestamp,
                "channel": event.channel,
                "sign": event.sign,
                "amplitude": event.amplitude,
                "value": event.value,
                "first_diff_value": event.first_diff_value,
                "second_diff_value": event.second_diff_value,
                "index": event.index,
            }
            for event in artifacts.extrema_events
        ]
    )


def _peak_events_frame(artifacts: PipelineArtifacts) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": peak.timestamp,
                "channel": peak.channel,
                "sign": peak.sign,
                "peak_value": peak.peak_value,
                "prominence": peak.prominence,
                "width_steps": peak.width_steps,
                "rise_slope": peak.rise_slope,
                "fall_slope": peak.fall_slope,
                "asymmetry": peak.asymmetry,
                "left_base_value": peak.left_base_value,
                "right_base_value": peak.right_base_value,
                "index": peak.index,
                "left_index": peak.left_index,
                "right_index": peak.right_index,
            }
            for peak in artifacts.peak_events
        ]
    )


def _pattern_windows_frame(artifacts: PipelineArtifacts) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in artifacts.pattern_windows:
        row: dict[str, object] = {
            "window_id": window.window_id,
            "start_time": window.start_time,
            "end_time": window.end_time,
            "start_index": window.extrema_window.start_index,
            "end_index": window.extrema_window.end_index,
            "event_count": len(window.extrema_window.events),
            "peak_count": len(window.extrema_window.peaks),
            "feature_vector_length": int(len(window.feature_vector)),
            "discovered_pattern_id": artifacts.discovery_result.labels_by_window_id.get(window.window_id),
            "time_step_hours": window.time_placeholders.time_step_hours,
            "window_length_steps": window.time_placeholders.window_length_steps,
            "forecast_horizon_steps": window.time_placeholders.forecast_horizon_steps,
            "mean_inter_event_gap_steps": window.time_placeholders.mean_inter_event_gap_steps,
            "var_inter_event_gap_steps": window.time_placeholders.var_inter_event_gap_steps,
            "mean_peak_width_steps": window.time_placeholders.mean_peak_width_steps,
            "max_peak_width_steps": window.time_placeholders.max_peak_width_steps,
        }
        for channel in window.channels:
            row[f"{channel}__duration_over_threshold"] = window.time_placeholders.duration_over_threshold_by_channel.get(
                channel,
                0.0,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _forecast_samples_frame(artifacts: PipelineArtifacts) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_window_id": sample.source_window_id,
                "history_window_ids": ",".join(str(value) for value in sample.history_window_ids),
                "target_window_ids": ",".join(str(value) for value in sample.target_window_ids),
                "history_pattern_ids": ",".join(
                    "" if value is None else str(value) for value in sample.history_pattern_ids
                ),
                "target_pattern_ids": ",".join(
                    "" if value is None else str(value) for value in sample.target_pattern_ids
                ),
                "forecast_horizon_steps": sample.forecast_horizon_steps,
                "forecast_window_count": sample.forecast_window_count,
                "history_window_count": len(sample.history_window_ids),
                "history_vector_length": int(len(sample.history_vector)),
                "feature_dim": int(sample.target_pattern_matrix.shape[1]),
            }
            for sample in artifacts.forecast_samples
        ]
    )


def write_pipeline_artifacts(artifacts: PipelineArtifacts, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_path = write_artifacts_summary(artifacts, destination)
    signal_path = destination / "signal_frame.csv"
    extrema_events_path = destination / "extrema_events.csv"
    peak_events_path = destination / "peak_events.csv"
    pattern_windows_path = destination / "pattern_windows.csv"
    forecast_samples_path = destination / "forecast_samples.csv"

    _write_frame(artifacts.signal_frame, signal_path)
    _write_frame(_extrema_events_frame(artifacts), extrema_events_path)
    _write_frame(_peak_events_frame(artifacts), peak_events_path)
    _write_frame(_pattern_windows_frame(artifacts), pattern_windows_path)
    _write_frame(_forecast_samples_frame(artifacts), forecast_samples_path)

    return {
        "summary_path": str(summary_path),
        "signal_frame_path": str(signal_path),
        "extrema_events_path": str(extrema_events_path),
        "peak_events_path": str(peak_events_path),
        "pattern_windows_path": str(pattern_windows_path),
        "forecast_samples_path": str(forecast_samples_path),
    }
