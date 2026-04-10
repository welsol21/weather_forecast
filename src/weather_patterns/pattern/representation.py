from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from weather_patterns.config import HazardConfig, PipelineConfig
from weather_patterns.models import ExtremaEvent, ExtremaWindow, PatternWindow, PeakEvent, TimePlaceholders
INTRA_FEATURES = [
    "current_value",
    "delta_1",
    "delta_6",
    "delta_24",
    "second_diff",
    "mean_on_window",
    "variance_on_window",
    "integral_on_window",
    "max_on_window",
    "min_on_window",
    "range_on_window",
    "extrema_count",
    "extrema_amplitude_sum",
]

INTER_FEATURES = [
    "correlation",
    "lag_correlation",
    "slope_ratio",
    "synchronous_extrema_count",
    "mean_event_lag",
]

PEAK_HAZARD_FEATURES = [
    "number_of_peaks",
    "max_peak_value",
    "mean_prominence",
    "max_prominence",
    "mean_peak_width",
    "max_peak_width",
    "max_rise_slope",
    "duration_over_upper_threshold",
    "upper_tail_excess",
    "lower_tail_excess",
    "cumulative_risk",
    "hazard_flag",
    "compound_event_flag",
]


def compute_channel_thresholds(
    frame: pd.DataFrame,
    channels: list[str],
    config: HazardConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    upper = {
        channel: float(frame[channel].dropna().quantile(config.upper_quantile))
        for channel in channels
    }
    lower = {
        channel: float(frame[channel].dropna().quantile(config.lower_quantile))
        for channel in channels
    }
    return upper, lower


def _delta(series: pd.Series, steps: int) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    if len(clean) <= steps:
        return float(clean.iloc[-1] - clean.iloc[0])
    return float(clean.iloc[-1] - clean.iloc[-(steps + 1)])


def _delta_array(values: np.ndarray, steps: int) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 0.0
    if clean.size <= steps:
        return float(clean[-1] - clean[0])
    return float(clean[-1] - clean[-(steps + 1)])


def _variance_array(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return 0.0
    return float(np.var(clean))


def build_intra_matrix(
    window_frame: pd.DataFrame,
    window_events: list[ExtremaEvent],
    channels: list[str],
) -> pd.DataFrame:
    matrix = np.zeros((len(INTRA_FEATURES), len(channels)), dtype=float)
    feature_index = {name: index for index, name in enumerate(INTRA_FEATURES)}
    channel_index = {name: index for index, name in enumerate(channels)}
    event_stats: dict[str, tuple[float, float]] = {}
    for channel in channels:
        channel_events = [event for event in window_events if event.channel == channel]
        event_stats[channel] = (
            float(len(channel_events)),
            float(sum(abs(event.amplitude) for event in channel_events)),
        )

    for channel in channels:
        column_index = channel_index[channel]
        series = pd.to_numeric(window_frame[f"smoothed_{channel}"], errors="coerce").to_numpy(dtype=float)
        diff2 = pd.to_numeric(window_frame[f"diff2_{channel}"], errors="coerce").to_numpy(dtype=float)
        clean = series[np.isfinite(series)]
        clean_diff2 = diff2[np.isfinite(diff2)]
        extrema_count, extrema_amplitude_sum = event_stats[channel]
        maximum = float(clean.max()) if clean.size else 0.0
        minimum = float(clean.min()) if clean.size else 0.0

        matrix[feature_index["current_value"], column_index] = float(clean[-1]) if clean.size else 0.0
        matrix[feature_index["delta_1"], column_index] = _delta_array(series, 1)
        matrix[feature_index["delta_6"], column_index] = _delta_array(series, 6)
        matrix[feature_index["delta_24"], column_index] = _delta_array(series, 24)
        matrix[feature_index["second_diff"], column_index] = float(clean_diff2[-1]) if clean_diff2.size else 0.0
        matrix[feature_index["mean_on_window"], column_index] = float(clean.mean()) if clean.size else 0.0
        matrix[feature_index["variance_on_window"], column_index] = _variance_array(series)
        matrix[feature_index["integral_on_window"], column_index] = float(np.nansum(clean)) if clean.size else 0.0
        matrix[feature_index["max_on_window"], column_index] = maximum
        matrix[feature_index["min_on_window"], column_index] = minimum
        matrix[feature_index["range_on_window"], column_index] = maximum - minimum if clean.size else 0.0
        matrix[feature_index["extrema_count"], column_index] = extrema_count
        matrix[feature_index["extrema_amplitude_sum"], column_index] = extrema_amplitude_sum
    return pd.DataFrame(matrix, index=INTRA_FEATURES, columns=channels)


def _array_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid_mask = np.isfinite(a) & np.isfinite(b)
    if valid_mask.sum() < 2:
        return 0.0
    left = a[valid_mask]
    right = b[valid_mask]
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    left_scale = float(np.sqrt(np.dot(left_centered, left_centered)))
    right_scale = float(np.sqrt(np.dot(right_centered, right_centered)))
    if np.isclose(left_scale, 0.0) or np.isclose(right_scale, 0.0):
        return 0.0
    return float(np.dot(left_centered, right_centered) / (left_scale * right_scale))


def _max_lag_correlation_array(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    best = _array_corr(a, b)
    for lag in range(1, max_lag + 1):
        best = max(best, _array_corr(a[lag:], b[:-lag]), key=abs)
        best = max(best, _array_corr(a[:-lag], b[lag:]), key=abs)
    return float(best)


def _synchronous_extrema_metrics(
    events_a: list[ExtremaEvent],
    events_b: list[ExtremaEvent],
    tolerance_steps: int,
) -> tuple[float, float]:
    if not events_a or not events_b:
        return 0.0, 0.0
    lags: list[int] = []
    indices_b = [event.index for event in events_b]
    right_pointer = 0
    for event_a in events_a:
        while right_pointer < len(indices_b) and indices_b[right_pointer] < event_a.index - tolerance_steps:
            right_pointer += 1
        best_lag: int | None = None
        probe = right_pointer
        while probe < len(indices_b) and indices_b[probe] <= event_a.index + tolerance_steps:
            lag = abs(event_a.index - indices_b[probe])
            best_lag = lag if best_lag is None else min(best_lag, lag)
            probe += 1
        if best_lag is not None:
            lags.append(best_lag)
    if not lags:
        return 0.0, 0.0
    return float(len(lags)), float(np.mean(lags))


def build_inter_matrix(
    window_frame: pd.DataFrame,
    window_events: list[ExtremaEvent],
    channels: list[str],
    correlation_lag_steps: int,
    event_match_tolerance_steps: int,
) -> pd.DataFrame:
    pair_labels = [f"{left}__{right}" for left, right in combinations(channels, 2)]
    matrix = np.zeros((len(INTER_FEATURES), len(pair_labels)), dtype=float)
    feature_index = {name: index for index, name in enumerate(INTER_FEATURES)}
    series_by_channel = {
        channel: pd.to_numeric(window_frame[f"smoothed_{channel}"], errors="coerce").to_numpy(dtype=float)
        for channel in channels
    }
    diff_mean_by_channel = {
        channel: float(
            pd.to_numeric(window_frame[f"diff1_{channel}"], errors="coerce").abs().mean()
        )
        for channel in channels
    }
    events_by_channel = {
        channel: [event for event in window_events if event.channel == channel]
        for channel in channels
    }
    for pair_index, (left, right) in enumerate(combinations(channels, 2)):
        left_series = series_by_channel[left]
        right_series = series_by_channel[right]
        left_diff = diff_mean_by_channel[left]
        right_diff = diff_mean_by_channel[right]
        left_events = events_by_channel[left]
        right_events = events_by_channel[right]
        sync_count, mean_lag = _synchronous_extrema_metrics(
            left_events,
            right_events,
            event_match_tolerance_steps,
        )
        matrix[feature_index["correlation"], pair_index] = _array_corr(left_series, right_series)
        matrix[feature_index["lag_correlation"], pair_index] = _max_lag_correlation_array(
            left_series,
            right_series,
            correlation_lag_steps,
        )
        matrix[feature_index["slope_ratio"], pair_index] = float(left_diff / right_diff) if right_diff else 0.0
        matrix[feature_index["synchronous_extrema_count"], pair_index] = sync_count
        matrix[feature_index["mean_event_lag"], pair_index] = mean_lag
    return pd.DataFrame(matrix, index=INTER_FEATURES, columns=pair_labels)


def _compound_event_flag(
    window_frame: pd.DataFrame,
    upper_thresholds: dict[str, float],
) -> float:
    if "wind_speed" not in window_frame.columns or "rainfall" not in window_frame.columns:
        return 0.0
    wind_threshold = upper_thresholds.get("wind_speed", 0.0)
    rain_threshold = upper_thresholds.get("rainfall", 0.0)
    wind = pd.to_numeric(window_frame["wind_speed"], errors="coerce").to_numpy(dtype=float)
    rain = pd.to_numeric(window_frame["rainfall"], errors="coerce").to_numpy(dtype=float)
    wind = np.nan_to_num(wind, nan=0.0)
    rain = np.nan_to_num(rain, nan=0.0)
    return float(np.any((wind >= wind_threshold) & (rain >= rain_threshold)))


def build_peak_hazard_matrix(
    window_frame: pd.DataFrame,
    window_peaks: list[PeakEvent],
    channels: list[str],
    upper_thresholds: dict[str, float],
    lower_thresholds: dict[str, float],
) -> pd.DataFrame:
    matrix = np.zeros((len(PEAK_HAZARD_FEATURES), len(channels)), dtype=float)
    feature_index = {name: index for index, name in enumerate(PEAK_HAZARD_FEATURES)}
    channel_index = {name: index for index, name in enumerate(channels)}
    compound_flag = _compound_event_flag(window_frame, upper_thresholds)
    peaks_by_channel = {
        channel: [peak for peak in window_peaks if peak.channel == channel]
        for channel in channels
    }
    for channel in channels:
        column_index = channel_index[channel]
        series = pd.to_numeric(window_frame[channel], errors="coerce").to_numpy(dtype=float)
        clean_series = series[np.isfinite(series)]
        peaks = peaks_by_channel[channel]
        upper = upper_thresholds[channel]
        lower = lower_thresholds[channel]
        duration_over = float(np.sum(clean_series > upper)) if clean_series.size else 0.0
        upper_tail = float(np.sum(np.clip(clean_series - upper, a_min=0.0, a_max=None))) if clean_series.size else 0.0
        lower_tail = float(np.sum(np.clip(lower - clean_series, a_min=0.0, a_max=None))) if clean_series.size else 0.0
        cumulative_risk = upper_tail + lower_tail + duration_over
        prominences = np.asarray([peak.prominence for peak in peaks], dtype=float)
        widths = np.asarray([peak.width_steps for peak in peaks], dtype=float)
        rise_slopes = np.asarray([peak.rise_slope for peak in peaks], dtype=float)
        peak_values = np.asarray([peak.peak_value for peak in peaks], dtype=float)
        matrix[feature_index["number_of_peaks"], column_index] = float(len(peaks))
        matrix[feature_index["max_peak_value"], column_index] = float(peak_values.max()) if peak_values.size else 0.0
        matrix[feature_index["mean_prominence"], column_index] = float(prominences.mean()) if prominences.size else 0.0
        matrix[feature_index["max_prominence"], column_index] = float(prominences.max()) if prominences.size else 0.0
        matrix[feature_index["mean_peak_width"], column_index] = float(widths.mean()) if widths.size else 0.0
        matrix[feature_index["max_peak_width"], column_index] = float(widths.max()) if widths.size else 0.0
        matrix[feature_index["max_rise_slope"], column_index] = float(rise_slopes.max()) if rise_slopes.size else 0.0
        matrix[feature_index["duration_over_upper_threshold"], column_index] = duration_over
        matrix[feature_index["upper_tail_excess"], column_index] = upper_tail
        matrix[feature_index["lower_tail_excess"], column_index] = lower_tail
        matrix[feature_index["cumulative_risk"], column_index] = cumulative_risk
        matrix[feature_index["hazard_flag"], column_index] = float(duration_over > 0 or cumulative_risk > 0)
        matrix[feature_index["compound_event_flag"], column_index] = compound_flag
    return pd.DataFrame(matrix, index=PEAK_HAZARD_FEATURES, columns=channels)


def build_time_placeholders(
    extrema_window: ExtremaWindow,
    channels: list[str],
    forecast_horizon_steps: int,
    time_step_hours: float,
    peak_hazard_matrix: pd.DataFrame,
) -> TimePlaceholders:
    event_indices = sorted(event.index for event in extrema_window.events)
    gaps = np.diff(event_indices) if len(event_indices) > 1 else np.asarray([], dtype=float)
    peak_widths = np.asarray([peak.width_steps for peak in extrema_window.peaks], dtype=float)
    normalized_positions = []
    window_span = max(extrema_window.end_index - extrema_window.start_index, 1)
    for event in extrema_window.events:
        normalized_positions.append((event.index - extrema_window.start_index) / window_span)

    duration_row = peak_hazard_matrix.loc["duration_over_upper_threshold"]
    durations = {channel: float(duration_row[channel]) for channel in channels}
    return TimePlaceholders(
        time_step_hours=time_step_hours,
        window_length_steps=extrema_window.end_index - extrema_window.start_index + 1,
        forecast_horizon_steps=forecast_horizon_steps,
        mean_inter_event_gap_steps=float(gaps.mean()) if gaps.size else 0.0,
        var_inter_event_gap_steps=float(gaps.var()) if gaps.size else 0.0,
        mean_peak_width_steps=float(peak_widths.mean()) if peak_widths.size else 0.0,
        max_peak_width_steps=float(peak_widths.max()) if peak_widths.size else 0.0,
        duration_over_threshold_by_channel=durations,
        normalized_event_positions=normalized_positions,
    )


def flatten_pattern_representation(pattern_window: PatternWindow) -> np.ndarray:
    intra = pattern_window.intra_matrix.to_numpy(dtype=float).ravel(order="C")
    inter = pattern_window.inter_matrix.to_numpy(dtype=float).ravel(order="C")
    peak_hazard = pattern_window.peak_hazard_matrix.to_numpy(dtype=float).ravel(order="C")
    time_block = pattern_window.time_placeholders.to_feature_array(pattern_window.channels)
    return np.concatenate([intra, inter, peak_hazard, time_block]).astype(float)


def build_pattern_window(
    signal_frame: pd.DataFrame,
    extrema_window: ExtremaWindow,
    channels: list[str],
    config: PipelineConfig,
    upper_thresholds: dict[str, float],
    lower_thresholds: dict[str, float],
) -> PatternWindow:
    window_frame = signal_frame.iloc[extrema_window.start_index : extrema_window.end_index + 1]
    intra = build_intra_matrix(window_frame, extrema_window.events, channels)
    inter = build_inter_matrix(
        window_frame,
        extrema_window.events,
        channels,
        config.window.correlation_lag_steps,
        config.window.event_match_tolerance_steps,
    )
    peak_hazard = build_peak_hazard_matrix(
        window_frame,
        extrema_window.peaks,
        channels,
        upper_thresholds,
        lower_thresholds,
    )
    time_placeholders = build_time_placeholders(
        extrema_window=extrema_window,
        channels=channels,
        forecast_horizon_steps=config.window.forecast_horizon_steps,
        time_step_hours=config.time_step_hours,
        peak_hazard_matrix=peak_hazard,
    )
    pattern_window = PatternWindow(
        window_id=extrema_window.window_id,
        start_time=extrema_window.start_time,
        end_time=extrema_window.end_time,
        channels=channels,
        intra_matrix=intra,
        inter_matrix=inter,
        peak_hazard_matrix=peak_hazard,
        time_placeholders=time_placeholders,
        feature_vector=np.empty(0, dtype=float),
        extrema_window=extrema_window,
    )
    pattern_window.feature_vector = flatten_pattern_representation(pattern_window)
    return pattern_window
