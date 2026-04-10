from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from weather_patterns.config import HazardConfig, PipelineConfig
from weather_patterns.models import ExtremaEvent, ExtremaWindow, PatternWindow, PeakEvent, TimePlaceholders
from weather_patterns.signal.processing import safe_corr, safe_variance


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


def build_intra_matrix(
    window_frame: pd.DataFrame,
    window_events: list[ExtremaEvent],
    channels: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=INTRA_FEATURES, columns=channels, dtype=float)
    for channel in channels:
        series = pd.to_numeric(window_frame[f"smoothed_{channel}"], errors="coerce")
        diff2 = pd.to_numeric(window_frame[f"diff2_{channel}"], errors="coerce")
        channel_events = [event for event in window_events if event.channel == channel]
        clean = series.dropna()
        matrix.loc["current_value", channel] = float(clean.iloc[-1]) if not clean.empty else 0.0
        matrix.loc["delta_1", channel] = _delta(series, 1)
        matrix.loc["delta_6", channel] = _delta(series, 6)
        matrix.loc["delta_24", channel] = _delta(series, 24)
        matrix.loc["second_diff", channel] = float(diff2.dropna().iloc[-1]) if not diff2.dropna().empty else 0.0
        matrix.loc["mean_on_window", channel] = float(clean.mean()) if not clean.empty else 0.0
        matrix.loc["variance_on_window", channel] = safe_variance(series)
        matrix.loc["integral_on_window", channel] = float(np.nansum(clean.to_numpy(dtype=float))) if not clean.empty else 0.0
        matrix.loc["max_on_window", channel] = float(clean.max()) if not clean.empty else 0.0
        matrix.loc["min_on_window", channel] = float(clean.min()) if not clean.empty else 0.0
        matrix.loc["range_on_window", channel] = float(clean.max() - clean.min()) if not clean.empty else 0.0
        matrix.loc["extrema_count", channel] = float(len(channel_events))
        matrix.loc["extrema_amplitude_sum", channel] = float(sum(abs(event.amplitude) for event in channel_events))
    return matrix


def _max_lag_correlation(a: pd.Series, b: pd.Series, max_lag: int) -> float:
    correlations = [safe_corr(a, b)]
    for lag in range(1, max_lag + 1):
        correlations.append(safe_corr(a.iloc[lag:], b.iloc[:-lag]))
        correlations.append(safe_corr(a.iloc[:-lag], b.iloc[lag:]))
    return float(max(correlations, key=abs))


def _array_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid_mask = np.isfinite(a) & np.isfinite(b)
    if valid_mask.sum() < 2:
        return 0.0
    left = a[valid_mask]
    right = b[valid_mask]
    if np.isclose(left.std(), 0.0) or np.isclose(right.std(), 0.0):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


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
    for event_a in events_a:
        matching = [
            abs(event_a.index - event_b.index)
            for event_b in events_b
            if abs(event_a.index - event_b.index) <= tolerance_steps
        ]
        if matching:
            lags.append(min(matching))
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
    matrix = pd.DataFrame(index=INTER_FEATURES, columns=pair_labels, dtype=float)
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
    for left, right in combinations(channels, 2):
        label = f"{left}__{right}"
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
        matrix.loc["correlation", label] = _array_corr(left_series, right_series)
        matrix.loc["lag_correlation", label] = _max_lag_correlation_array(
            left_series,
            right_series,
            correlation_lag_steps,
        )
        matrix.loc["slope_ratio", label] = float(left_diff / right_diff) if right_diff else 0.0
        matrix.loc["synchronous_extrema_count", label] = sync_count
        matrix.loc["mean_event_lag", label] = mean_lag
    return matrix


def _compound_event_flag(
    window_frame: pd.DataFrame,
    upper_thresholds: dict[str, float],
) -> float:
    if "wind_speed" not in window_frame.columns or "rainfall" not in window_frame.columns:
        return 0.0
    wind_threshold = upper_thresholds.get("wind_speed", 0.0)
    rain_threshold = upper_thresholds.get("rainfall", 0.0)
    mask = (
        pd.to_numeric(window_frame["wind_speed"], errors="coerce").fillna(0.0) >= wind_threshold
    ) & (
        pd.to_numeric(window_frame["rainfall"], errors="coerce").fillna(0.0) >= rain_threshold
    )
    return float(mask.any())


def build_peak_hazard_matrix(
    window_frame: pd.DataFrame,
    window_peaks: list[PeakEvent],
    channels: list[str],
    upper_thresholds: dict[str, float],
    lower_thresholds: dict[str, float],
) -> pd.DataFrame:
    matrix = pd.DataFrame(index=PEAK_HAZARD_FEATURES, columns=channels, dtype=float)
    compound_flag = _compound_event_flag(window_frame, upper_thresholds)
    for channel in channels:
        series = pd.to_numeric(window_frame[channel], errors="coerce").dropna()
        peaks = [peak for peak in window_peaks if peak.channel == channel]
        upper = upper_thresholds[channel]
        lower = lower_thresholds[channel]
        duration_over = float((series > upper).sum()) if not series.empty else 0.0
        upper_tail = float((series - upper).clip(lower=0).sum()) if not series.empty else 0.0
        lower_tail = float((lower - series).clip(lower=0).sum()) if not series.empty else 0.0
        cumulative_risk = upper_tail + lower_tail + duration_over
        matrix.loc["number_of_peaks", channel] = float(len(peaks))
        matrix.loc["max_peak_value", channel] = float(max((peak.peak_value for peak in peaks), default=0.0))
        matrix.loc["mean_prominence", channel] = float(np.mean([peak.prominence for peak in peaks])) if peaks else 0.0
        matrix.loc["max_prominence", channel] = float(max((peak.prominence for peak in peaks), default=0.0))
        matrix.loc["mean_peak_width", channel] = float(np.mean([peak.width_steps for peak in peaks])) if peaks else 0.0
        matrix.loc["max_peak_width", channel] = float(max((peak.width_steps for peak in peaks), default=0.0))
        matrix.loc["max_rise_slope", channel] = float(max((peak.rise_slope for peak in peaks), default=0.0))
        matrix.loc["duration_over_upper_threshold", channel] = duration_over
        matrix.loc["upper_tail_excess", channel] = upper_tail
        matrix.loc["lower_tail_excess", channel] = lower_tail
        matrix.loc["cumulative_risk", channel] = cumulative_risk
        matrix.loc["hazard_flag", channel] = float(duration_over > 0 or cumulative_risk > 0)
        matrix.loc["compound_event_flag", channel] = compound_flag
    return matrix


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

    durations = {
        channel: float(peak_hazard_matrix.loc["duration_over_upper_threshold", channel])
        for channel in channels
    }
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
    window_frame = signal_frame.iloc[extrema_window.start_index : extrema_window.end_index + 1].copy()
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
