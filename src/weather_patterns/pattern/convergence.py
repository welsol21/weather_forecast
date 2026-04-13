"""New Physics feature extraction: local convergence functions per channel.

A pattern is a set of local convergence functions — one per channel — each
describing how the channel evolves toward its local limit (lim) on the current
time interval.  Channels are treated independently.  The four function types
correspond to four regimes of convergence:

    level        — channel is at (or very near) its local limit; lim f(t) reached
    velocity     — linear approach to limit; constant rate of change
    acceleration — quadratic / accelerating approach; second derivative active
    local_ar2    — autoregressive convergence; channel oscillates or decays toward
                   its AR(2) fixed point  phi_1·X + phi_2·X[-1] + c

The pattern is NOT about absolute values — it is about the *shape* of convergence.
The absolute initial conditions (current channel values) are the placeholders that
are filled at prediction time.

Feature vector layout (per channel, 9 floats):
  [0..3]  type one-hot: [is_level, is_velocity, is_acceleration, is_ar2]
  [4]     rate_norm    : (predicted_next − current) / std  (scale-free rate)
  [5]     accel_norm   : (d²x/dt²) / std                  (0 for level/velocity)
  [6]     ar_phi1      : AR(2) coefficient φ₁              (0 for non-ar2)
  [7]     ar_phi2      : AR(2) coefficient φ₂              (0 for non-ar2)
  [8]     ar_stability : 1 − |φ₁+φ₂|  (1=stable, 0=unit-root, <0=explosive)

Total feature vector: 9 × N_channels floats.
"""

from __future__ import annotations

import numpy as np

from weather_patterns.config import WindowConfig
from weather_patterns.pattern.windows import (
    _is_circular_channel,
    _predict_level,
    _predict_velocity,
    _predict_acceleration,
    _predict_local_ar2,
    _prediction_error,
    _wrap_prediction,
    _unwrap_channel,
)

_DIMS_PER_CHANNEL = 9
_TYPE_LEVEL = 0
_TYPE_VELOCITY = 1
_TYPE_ACCELERATION = 2
_TYPE_AR2 = 3


def _fit_ar2_params(
    history: np.ndarray,
    fit_window: int,
) -> tuple[float, float, float]:
    """Fit AR(2) via OLS on the last `fit_window` points.

    Returns (phi1, phi2, intercept).  Falls back to (0, 0, history[-1]) when
    there are too few points or the design matrix is rank-deficient.
    """
    usable = history[-max(fit_window, 3):]
    if usable.size < 3:
        return 0.0, 0.0, float(history[-1])
    design = np.column_stack([
        np.ones((usable.size - 2,), dtype=float),
        usable[1:-1],
        usable[:-2],
    ])
    target = usable[2:]
    try:
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
        intercept, phi1, phi2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    except np.linalg.LinAlgError:
        return 0.0, 0.0, float(history[-1])
    return phi1, phi2, intercept


def _channel_convergence_vector(
    series: np.ndarray,
    channel: str,
    history_window: int,
    fit_window: int,
) -> np.ndarray:
    """Compute the 9-dim convergence feature vector for a single channel.

    `series` is the full (smoothed) channel array for the window.
    The last `history_window` points are used to fit local predictors.
    """
    vec = np.zeros(_DIMS_PER_CHANNEL, dtype=float)

    clean = series[np.isfinite(series)]
    if clean.size == 0:
        return vec

    # Normalise (scale-free representation)
    std = float(np.std(clean)) if clean.size > 1 else 1.0
    if std < 1e-8:
        std = 1.0
    mean = float(np.mean(clean))

    # Work on unwrapped / normalised history
    history_raw = series[-history_window:] if series.size >= history_window else series
    history_raw = _unwrap_channel(history_raw, channel)
    history_norm = (history_raw - mean) / std

    if history_norm.size == 0:
        return vec

    # Actual (last point) in normalised space — used only for error comparison
    actual_norm = history_norm[-1]

    # ── fit all applicable predictor types ────────────────────────────────────
    errors: list[tuple[float, int]] = []

    # level
    pred_level = _predict_level(history_norm)
    errors.append((_prediction_error(actual_norm, _wrap_prediction(pred_level, channel), channel), _TYPE_LEVEL))

    # velocity
    if history_norm.size >= 2:
        pred_vel = _predict_velocity(history_norm)
        errors.append((_prediction_error(actual_norm, _wrap_prediction(pred_vel, channel), channel), _TYPE_VELOCITY))

    # acceleration
    if history_norm.size >= 3:
        pred_acc = _predict_acceleration(history_norm)
        errors.append((_prediction_error(actual_norm, _wrap_prediction(pred_acc, channel), channel), _TYPE_ACCELERATION))
        try:
            pred_ar2 = _predict_local_ar2(history_norm, fit_window)
            errors.append((_prediction_error(actual_norm, _wrap_prediction(pred_ar2, channel), channel), _TYPE_AR2))
        except np.linalg.LinAlgError:
            pass  # SVD did not converge — skip AR2 candidate for this window

    best_type = min(errors, key=lambda item: item[0])[1]

    # ── encode one-hot type ────────────────────────────────────────────────────
    vec[best_type] = 1.0

    # ── compute rate (scale-free, one-step-ahead change) ──────────────────────
    if best_type == _TYPE_LEVEL:
        rate_norm = 0.0
        accel_norm = 0.0
        ar_phi1 = 0.0
        ar_phi2 = 0.0
        ar_stability = 1.0

    elif best_type == _TYPE_VELOCITY:
        rate_norm = float(history_norm[-1] - history_norm[-2]) if history_norm.size >= 2 else 0.0
        accel_norm = 0.0
        ar_phi1 = 0.0
        ar_phi2 = 0.0
        ar_stability = 1.0

    elif best_type == _TYPE_ACCELERATION:
        rate_norm = float(history_norm[-1] - history_norm[-2]) if history_norm.size >= 2 else 0.0
        accel_norm = float(history_norm[-1] - 2 * history_norm[-2] + history_norm[-3]) if history_norm.size >= 3 else 0.0
        ar_phi1 = 0.0
        ar_phi2 = 0.0
        ar_stability = 1.0

    else:  # _TYPE_AR2
        phi1, phi2, intercept = _fit_ar2_params(history_norm, fit_window)
        ar_phi1 = phi1
        ar_phi2 = phi2
        ar_stability = float(1.0 - abs(phi1 + phi2))
        # rate_norm = what the AR model predicts as next step minus current
        pred_next_norm = intercept + phi1 * history_norm[-1] + (phi2 * history_norm[-2] if history_norm.size >= 2 else 0.0)
        rate_norm = float(pred_next_norm - history_norm[-1])
        accel_norm = 0.0  # AR captures non-linearity through its coefficients

    vec[4] = rate_norm
    vec[5] = accel_norm
    vec[6] = ar_phi1
    vec[7] = ar_phi2
    vec[8] = ar_stability

    return vec


def compute_convergence_feature_vector(
    smoothed_series: dict[str, np.ndarray],
    channels: list[str],
    config: WindowConfig,
) -> np.ndarray:
    """Build the full New Physics feature vector for a pattern window.

    Concatenates 9-dim per-channel convergence vectors in `channels` order.
    Total length = 9 × len(channels).
    """
    parts: list[np.ndarray] = []
    history_window = max(config.predictor_history_window_steps, 3)
    fit_window = max(config.predictor_fit_window_steps, 3)
    for channel in channels:
        series = smoothed_series.get(channel)
        if series is None or series.size == 0:
            parts.append(np.zeros(_DIMS_PER_CHANNEL, dtype=float))
            continue
        parts.append(_channel_convergence_vector(series, channel, history_window, fit_window))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)


def channel_stds_from_window(
    smoothed_series: dict[str, np.ndarray],
    channels: list[str],
) -> dict[str, float]:
    """Return per-channel std (scale) from the smoothed window data.

    Used as placeholders for denormalising the convergence reconstruction.
    """
    stds: dict[str, float] = {}
    for channel in channels:
        series = smoothed_series.get(channel)
        if series is None or series.size == 0:
            stds[channel] = 1.0
            continue
        clean = series[np.isfinite(series)]
        std = float(np.std(clean)) if clean.size > 1 else 1.0
        stds[channel] = max(std, 1e-8)
    return stds


def channel_end_values_from_window(
    raw_series: dict[str, np.ndarray],
    channels: list[str],
) -> dict[str, float]:
    """Return the last finite observed value per channel (the placeholder)."""
    end_values: dict[str, float] = {}
    for channel in channels:
        series = raw_series.get(channel)
        if series is None or series.size == 0:
            end_values[channel] = 0.0
            continue
        finite = series[np.isfinite(series)]
        end_values[channel] = float(finite[-1]) if finite.size > 0 else 0.0
    return end_values


# ── Reconstruction ────────────────────────────────────────────────────────────

def _parse_channel_vector(vec: np.ndarray) -> tuple[int, float, float, float, float, float]:
    """Extract (type_idx, rate_norm, accel_norm, phi1, phi2, stability) from 9-dim slice."""
    type_idx = int(np.argmax(vec[:4]))
    return type_idx, float(vec[4]), float(vec[5]), float(vec[6]), float(vec[7]), float(vec[8])


def solve_convergence_forward(
    pred_type: int,
    rate_norm: float,
    accel_norm: float,
    ar_phi1: float,
    ar_phi2: float,
    initial_value: float,
    channel_std: float,
    channel: str,
    horizon_steps: int,
) -> list[float]:
    """Solve a local convergence function forward for `horizon_steps` steps.

    Parameters
    ----------
    pred_type        : 0=level, 1=velocity, 2=acceleration, 3=ar2
    rate_norm        : normalised one-step rate (rate_norm * std = actual rate)
    accel_norm       : normalised second derivative
    ar_phi1, ar_phi2 : AR(2) coefficients (normalised-space)
    initial_value    : last observed value of the channel (placeholder)
    channel_std      : std of the current window (scale factor)
    channel          : channel name (for circular handling)
    horizon_steps    : how many steps to project forward
    """
    if horizon_steps <= 0:
        return []

    rate = rate_norm * channel_std
    accel = accel_norm * channel_std

    if pred_type == _TYPE_LEVEL:
        values = [initial_value] * horizon_steps

    elif pred_type == _TYPE_VELOCITY:
        values = [initial_value + rate * t for t in range(1, horizon_steps + 1)]

    elif pred_type == _TYPE_ACCELERATION:
        values = [
            initial_value + rate * t + 0.5 * accel * t * (t - 1)
            for t in range(1, horizon_steps + 1)
        ]

    else:  # _TYPE_AR2
        # AR(2) in normalised space, seeded from initial conditions.
        # We estimate prev value as initial - rate.
        x0_norm = 0.0  # normalised current = 0 (mean-subtracted)
        x_prev_norm = -rate_norm  # normalised previous (estimate)
        history_norm = [x_prev_norm, x0_norm]
        # Intercept chosen so fixed point = 0 (the current mean level).
        # This means lim = initial_value; the AR shape governs convergence speed.
        c_norm = x0_norm * (1.0 - ar_phi1 - ar_phi2)
        for _ in range(horizon_steps):
            next_norm = c_norm + ar_phi1 * history_norm[-1] + ar_phi2 * history_norm[-2]
            history_norm.append(next_norm)
        values = [initial_value + v_norm * channel_std for v_norm in history_norm[2:]]

    # Wrap circular channels (wind direction)
    if _is_circular_channel(channel):
        values = [float(np.mod(v, 360.0)) for v in values]

    return values


def reconstruct_channel_sequence(
    predicted_feature_matrix: np.ndarray,
    channel_idx: int,
    n_channels: int,
    initial_value: float,
    channel_std: float,
    channel: str,
) -> list[float]:
    """Reconstruct a channel's value sequence from a predicted pattern matrix.

    Each row of `predicted_feature_matrix` is the feature vector of one
    predicted pattern window.  We solve each pattern's local function one step
    forward, chaining the end value as the next pattern's initial condition.

    Parameters
    ----------
    predicted_feature_matrix : shape (n_windows, 9 * n_channels)
    channel_idx              : index of this channel in the channel list
    n_channels               : total number of channels
    initial_value            : current observed value (placeholder)
    channel_std              : std of current window (scale)
    channel                  : channel name (for circular handling)
    """
    values: list[float] = []
    current = initial_value

    for row in predicted_feature_matrix:
        start = channel_idx * _DIMS_PER_CHANNEL
        ch_vec = row[start: start + _DIMS_PER_CHANNEL]
        pred_type, rate_norm, accel_norm, phi1, phi2, _ = _parse_channel_vector(ch_vec)

        # Each pattern window contributes ONE predicted step forward.
        step_values = solve_convergence_forward(
            pred_type=pred_type,
            rate_norm=rate_norm,
            accel_norm=accel_norm,
            ar_phi1=phi1,
            ar_phi2=phi2,
            initial_value=current,
            channel_std=channel_std,
            channel=channel,
            horizon_steps=1,
        )
        if step_values:
            current = step_values[0]
            values.append(current)

    return values
