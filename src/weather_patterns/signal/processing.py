from __future__ import annotations

import numpy as np
import pandas as pd

from weather_patterns.config import SmoothingConfig


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _savgol_is_available() -> bool:
    try:
        from scipy.signal import savgol_filter  # noqa: F401
    except Exception:
        return False
    return True


def smooth_channels(
    frame: pd.DataFrame,
    channels: list[str],
    config: SmoothingConfig,
) -> pd.DataFrame:
    smoothed = pd.DataFrame(index=frame.index)
    for channel in channels:
        series = _coerce_numeric(frame[channel])
        if config.method == "rolling_mean":
            smoothed[channel] = series.rolling(
                window=config.window,
                min_periods=config.min_periods,
                center=config.center,
            ).mean()
            continue

        if config.method == "savitzky_golay" and _savgol_is_available():
            from scipy.signal import savgol_filter

            valid = series.interpolate(limit_direction="both").to_numpy(dtype=float)
            smoothed[channel] = savgol_filter(
                valid,
                window_length=config.savgol_window,
                polyorder=config.savgol_polyorder,
                mode="interp",
            )
            continue

        if config.method == "savitzky_golay" and config.fallback_to_rolling:
            smoothed[channel] = series.rolling(
                window=config.window,
                min_periods=config.min_periods,
                center=config.center,
            ).mean()
            continue

        raise ValueError(f"Unsupported smoothing method: {config.method}")
    return smoothed


def compute_derivatives(smoothed: pd.DataFrame, channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_diff = smoothed[channels].diff()
    second_diff = smoothed[channels].diff().diff()
    return first_diff, second_diff


def build_signal_frame(
    frame: pd.DataFrame,
    channels: list[str],
    config: SmoothingConfig,
) -> pd.DataFrame:
    smoothed = smooth_channels(frame, channels, config)
    first_diff, second_diff = compute_derivatives(smoothed, channels)
    signal_frame = frame.copy()
    for channel in channels:
        signal_frame[f"smoothed_{channel}"] = smoothed[channel]
        signal_frame[f"diff1_{channel}"] = first_diff[channel]
        signal_frame[f"diff2_{channel}"] = second_diff[channel]
    return signal_frame


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([_coerce_numeric(a), _coerce_numeric(b)], axis=1).dropna()
    if len(joined) < 2:
        return 0.0
    left = joined.iloc[:, 0].to_numpy(dtype=float)
    right = joined.iloc[:, 1].to_numpy(dtype=float)
    if np.isclose(np.std(left), 0.0) or np.isclose(np.std(right), 0.0):
        return 0.0
    value = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    if pd.isna(value):
        return 0.0
    return float(value)


def safe_variance(values: pd.Series) -> float:
    clean = _coerce_numeric(values).dropna()
    if clean.empty:
        return 0.0
    return float(np.var(clean.to_numpy(dtype=float)))
