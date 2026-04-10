from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weather_patterns.config import PipelineConfig
from weather_patterns.data.loading import apply_quality_masks, load_weather_dataset
from weather_patterns.discovery.base import DiscoveryInput
from weather_patterns.discovery.kmeans import NumpyKMeansDiscovery
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
    discovery = NumpyKMeansDiscovery(active_config.discovery).fit_predict(
        DiscoveryInput(
            window_ids=[window.window_id for window in pattern_windows],
            feature_matrix=feature_matrix,
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
