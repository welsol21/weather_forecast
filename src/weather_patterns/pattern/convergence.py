"""ODE-based local convergence functions for pattern feature extraction.

A channel pattern is the best-fitting local differential equation on a time
segment.  Four types:

    level        dx/dt = 0              x(t) = L
    linear       dx/dt = c              x(t) = x0 + c*t
    exponential  dx/dt = -λ(x - L)     x(t) = L + (x0-L)*exp(-λ*t)
    oscillatory  d²x/dt² + 2λẋ + ω²(x-L) = 0
                 x(t) = L + exp(-λ*t)*(A*cos(ω*t) + B*sin(ω*t))

All parameters (L, λ, ω, A, B, c) are fitted jointly by nonlinear least
squares on the observed segment.  L is the mathematical attractor of the
equation — a fitted parameter, not an observed value.

Placeholder: only x0 (the actual observed channel value at the window start)
is stored separately and substituted at forecast time.

Feature vector per channel (6 floats):
    [0]  L      — fitted limit (normalised by channel std)
    [1]  c      — linear rate (normalised; 0 for non-linear types)
    [2]  λ      — convergence / damping rate  (0 for level/linear)
    [3]  ω      — oscillation frequency       (0 for non-oscillatory)
    [4]  A      — oscillation cosine coeff    (normalised; 0 for non-oscillatory)
    [5]  B      — oscillation sine coeff      (normalised; 0 for non-oscillatory)

Type is encoded separately as an integer (0–3) and stored alongside the vector.

Total feature vector including type one-hot (4) + 6 params = 10 floats per channel.
For 9 physical channels + 1 structural (10th channel = boundary synchrony count):
the structural channel contributes 1 float only (no ODE).
Total: 9 * 10 + 1 = 91 floats.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TYPE_LEVEL = 0
TYPE_LINEAR = 1
TYPE_EXPONENTIAL = 2
TYPE_OSCILLATORY = 3

DIMS_TYPE_ONEHOT = 4
DIMS_PARAMS = 6          # L, c, λ, ω, A, B
DIMS_PER_CHANNEL = DIMS_TYPE_ONEHOT + DIMS_PARAMS   # 10
DIMS_STRUCTURAL = 1      # 10th channel: boundary synchrony count

# Minimum segment length (points) required to attempt each fit
MIN_POINTS = {
    TYPE_LEVEL: 1,
    TYPE_LINEAR: 2,
    TYPE_EXPONENTIAL: 3,
    TYPE_OSCILLATORY: 10,
}

# Free parameters per type (excluding x0 which is pinned to y[0]).
# Used for complexity penalty: adjusted_mse = mse * (1 + FREE_PARAMS * penalty).
# Prevents overfitting simpler dynamics with high-parameter models.
_FREE_PARAMS = {
    TYPE_LEVEL: 1,        # L
    TYPE_LINEAR: 1,       # c
    TYPE_EXPONENTIAL: 2,  # L, λ
    TYPE_OSCILLATORY: 5,  # L, λ, ω, A, B
}
_COMPLEXITY_PENALTY = 0.15   # 15% MSE markup per free parameter above TYPE_LEVEL

# Regularisation: penalise explosive solutions
_LAMBDA_MAX = 10.0
_OMEGA_MAX = np.pi   # Nyquist for unit time-step


# ── ODE solutions (normalised time t = 0, 1, 2, ...) ─────────────────────────

def _solve_level(t: np.ndarray, L: float) -> np.ndarray:
    return np.full_like(t, L, dtype=float)


def _solve_linear(t: np.ndarray, x0: float, c: float) -> np.ndarray:
    return x0 + c * t


def _solve_exponential(t: np.ndarray, x0: float, L: float, lam: float) -> np.ndarray:
    return L + (x0 - L) * np.exp(-lam * t)


def _solve_oscillatory(
    t: np.ndarray,
    x0: float,
    L: float,
    lam: float,
    omega: float,
    A: float,
    B: float,
) -> np.ndarray:
    return L + np.exp(-lam * t) * (A * np.cos(omega * t) + B * np.sin(omega * t))


# ── Fitting routines ──────────────────────────────────────────────────────────

def _fit_level(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Returns (L, residual_mse)."""
    L = float(np.mean(y))
    mse = float(np.mean((y - L) ** 2))
    return L, mse


def _fit_linear(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    """Returns (x0, c, residual_mse)."""
    # Linear regression y = x0 + c*t
    A = np.column_stack([np.ones_like(t), t])
    result, *_ = np.linalg.lstsq(A, y, rcond=None)
    x0, c = float(result[0]), float(result[1])
    pred = _solve_linear(t, x0, c)
    mse = float(np.mean((y - pred) ** 2))
    return x0, c, mse


def _fit_exponential(
    y: np.ndarray,
    t: np.ndarray,
) -> tuple[float, float, float, float]:
    """Returns (x0, L, λ, residual_mse) via nonlinear least squares."""
    x0_obs = float(y[0])
    L0 = float(np.mean(y[-max(1, len(y) // 4):]))   # rough asymptote guess
    lam0 = 0.1

    def residuals(params: np.ndarray) -> np.ndarray:
        L, lam = params
        lam_c = np.clip(lam, 1e-6, _LAMBDA_MAX)
        return y - _solve_exponential(t, x0_obs, L, lam_c)

    try:
        res = least_squares(
            residuals,
            x0=[L0, lam0],
            bounds=([-np.inf, 1e-6], [np.inf, _LAMBDA_MAX]),
            method="trf",
            max_nfev=200,
        )
        L, lam = float(res.x[0]), float(res.x[1])
    except Exception:
        L, lam = L0, lam0

    pred = _solve_exponential(t, x0_obs, L, lam)
    mse = float(np.mean((y - pred) ** 2))
    return x0_obs, L, lam, mse


def _estimate_omega(y: np.ndarray, t: np.ndarray) -> float:
    """Estimate dominant oscillation frequency via FFT on de-trended signal."""
    if len(y) < 4:
        return 0.1
    detrended = y - np.mean(y)
    fft = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(len(y))
    # Exclude DC (index 0)
    if len(fft) < 2:
        return 0.1
    dominant_idx = int(np.argmax(fft[1:]) + 1)
    omega = float(2 * np.pi * freqs[dominant_idx])
    return np.clip(omega, 1e-3, _OMEGA_MAX)


def _fit_oscillatory(
    y: np.ndarray,
    t: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    """Returns (x0, L, λ, ω, A, B, residual_mse) via nonlinear least squares."""
    x0_obs = float(y[0])
    L0 = float(np.mean(y))
    omega0 = _estimate_omega(y, t)
    lam0 = 0.05
    # With L and ω fixed, A and B are linear — use two-stage fit
    A0 = float(y[0] - L0)
    B0 = 0.0

    def residuals(params: np.ndarray) -> np.ndarray:
        L, lam, omega, A, B = params
        lam_c = np.clip(lam, 1e-6, _LAMBDA_MAX)
        omega_c = np.clip(omega, 1e-3, _OMEGA_MAX)
        return y - _solve_oscillatory(t, x0_obs, L, lam_c, omega_c, A, B)

    try:
        res = least_squares(
            residuals,
            x0=[L0, lam0, omega0, A0, B0],
            bounds=(
                [-np.inf, 1e-6, 1e-3, -np.inf, -np.inf],
                [np.inf, _LAMBDA_MAX, _OMEGA_MAX, np.inf, np.inf],
            ),
            method="trf",
            max_nfev=400,
        )
        L, lam, omega, A, B = (
            float(res.x[0]), float(res.x[1]), float(res.x[2]),
            float(res.x[3]), float(res.x[4]),
        )
    except Exception:
        L, lam, omega, A, B = L0, lam0, omega0, A0, B0

    pred = _solve_oscillatory(t, x0_obs, L, lam, omega, A, B)
    mse = float(np.mean((y - pred) ** 2))
    return x0_obs, L, lam, omega, A, B, mse


# ── Channel feature vector ────────────────────────────────────────────────────

@dataclass
class ChannelFit:
    """Result of fitting one channel on one segment."""
    type_idx: int          # TYPE_LEVEL / LINEAR / EXPONENTIAL / OSCILLATORY
    L: float               # fitted limit (in normalised units)
    c: float               # linear rate (normalised)
    lam: float             # convergence rate λ
    omega: float           # oscillation frequency ω
    A: float               # oscillation cosine amplitude (normalised)
    B: float               # oscillation sine amplitude (normalised)
    mse: float             # residual MSE on normalised series
    x0: float              # actual observed value at segment start (placeholder)
    std: float             # channel std on this segment (for denormalisation)


def fit_channel_segment(
    y_raw: np.ndarray,
    channel: str,
) -> ChannelFit:
    """Fit the best ODE type to a raw channel segment.

    Returns a ChannelFit with all parameters in normalised (std-scaled) space
    except x0 and std which are in original units.
    """
    t_start = time.perf_counter()

    finite = y_raw[np.isfinite(y_raw)]
    if finite.size == 0:
        return ChannelFit(TYPE_LEVEL, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    x0_raw = float(finite[0])
    std = float(np.std(finite)) if finite.size > 1 else 1.0
    if std < 1e-8:
        std = 1.0
    mean = float(np.mean(finite))

    # Normalise
    y = (finite - mean) / std
    t = np.arange(len(y), dtype=float)

    n = len(y)
    best: ChannelFit | None = None

    def _adjusted(mse: float, type_idx: int) -> float:
        extra = _FREE_PARAMS[type_idx] - _FREE_PARAMS[TYPE_LEVEL]
        return mse * (1.0 + extra * _COMPLEXITY_PENALTY)

    # ── level ──────────────────────────────────────────────────────────────
    if n >= MIN_POINTS[TYPE_LEVEL]:
        L, mse = _fit_level(y, t)
        best = ChannelFit(TYPE_LEVEL, L, 0.0, 0.0, 0.0, 0.0, 0.0, _adjusted(mse, TYPE_LEVEL), x0_raw, std)

    # ── linear ─────────────────────────────────────────────────────────────
    if n >= MIN_POINTS[TYPE_LINEAR]:
        _, c, mse = _fit_linear(y, t)
        L_lin = float(y[0] + c * (n - 1))
        adj = _adjusted(mse, TYPE_LINEAR)
        if best is None or adj < best.mse:
            best = ChannelFit(TYPE_LINEAR, L_lin, c, 0.0, 0.0, 0.0, 0.0, adj, x0_raw, std)

    # ── exponential ────────────────────────────────────────────────────────
    if n >= MIN_POINTS[TYPE_EXPONENTIAL]:
        _, L, lam, mse = _fit_exponential(y, t)
        adj = _adjusted(mse, TYPE_EXPONENTIAL)
        if best is None or adj < best.mse:
            best = ChannelFit(TYPE_EXPONENTIAL, L, 0.0, lam, 0.0, 0.0, 0.0, adj, x0_raw, std)

    # ── oscillatory ────────────────────────────────────────────────────────
    if n >= MIN_POINTS[TYPE_OSCILLATORY]:
        _, L, lam, omega, A, B, mse = _fit_oscillatory(y, t)
        adj = _adjusted(mse, TYPE_OSCILLATORY)
        if best is None or adj < best.mse:
            best = ChannelFit(TYPE_OSCILLATORY, L, 0.0, lam, omega, A, B, adj, x0_raw, std)

    elapsed = time.perf_counter() - t_start
    logger.debug(
        "fit_channel channel=%s n=%d best_type=%d mse=%.6f elapsed_ms=%.1f",
        channel, n, best.type_idx, best.mse, elapsed * 1000,
    )
    return best  # type: ignore[return-value]


def channel_fit_to_vector(fit: ChannelFit) -> np.ndarray:
    """Encode a ChannelFit as a 10-dim feature vector.

    Layout: [is_level, is_linear, is_exp, is_osc, L, c, λ, ω, A, B]
    """
    vec = np.zeros(DIMS_PER_CHANNEL, dtype=float)
    vec[fit.type_idx] = 1.0   # one-hot type
    vec[4] = fit.L
    vec[5] = fit.c
    vec[6] = fit.lam
    vec[7] = fit.omega
    vec[8] = fit.A
    vec[9] = fit.B
    return vec


def channel_fit_from_vector(vec: np.ndarray) -> ChannelFit:
    """Reconstruct a ChannelFit from a 10-dim feature vector (no x0/std)."""
    type_idx = int(np.argmax(vec[:4]))
    return ChannelFit(
        type_idx=type_idx,
        L=float(vec[4]),
        c=float(vec[5]),
        lam=float(vec[6]),
        omega=float(vec[7]),
        A=float(vec[8]),
        B=float(vec[9]),
        mse=0.0,
        x0=0.0,
        std=1.0,
    )


# ── Prediction error for boundary detection ───────────────────────────────────

def predict_next(fit: ChannelFit, y_raw: np.ndarray) -> float:
    """Predict the next raw value given the current fitted ODE and the last
    observed raw value.  Used by the segmentation algorithm to detect
    boundaries.
    """
    if len(y_raw) == 0:
        return fit.x0

    std = fit.std if fit.std > 1e-8 else 1.0
    mean_approx = float(np.mean(y_raw[np.isfinite(y_raw)])) if len(y_raw) > 0 else 0.0
    n = len(y_raw)
    t_next = float(n)

    if fit.type_idx == TYPE_LEVEL:
        y_norm_next = fit.L
    elif fit.type_idx == TYPE_LINEAR:
        y_norm_next = (y_raw[0] - mean_approx) / std + fit.c * t_next
    elif fit.type_idx == TYPE_EXPONENTIAL:
        x0_norm = (y_raw[0] - mean_approx) / std
        y_norm_next = fit.L + (x0_norm - fit.L) * np.exp(-fit.lam * t_next)
    else:  # oscillatory
        x0_norm = (y_raw[0] - mean_approx) / std
        y_norm_next = fit.L + np.exp(-fit.lam * t_next) * (
            fit.A * np.cos(fit.omega * t_next) + fit.B * np.sin(fit.omega * t_next)
        )

    return float(y_norm_next * std + mean_approx)


# ── Reconstruction (forecast decoding) ───────────────────────────────────────

def solve_forward(
    fit: ChannelFit,
    x0_actual: float,
    horizon_steps: int,
) -> list[float]:
    """Project a fitted ODE forward for `horizon_steps` steps.

    x0_actual is the placeholder — the real observed value at the start of the
    window, used to anchor the absolute level of the forecast.
    """
    if horizon_steps <= 0:
        return []

    std = fit.std if fit.std > 1e-8 else 1.0
    # In normalised space x0 corresponds to 0 (mean-subtracted seed)
    x0_norm = 0.0
    t = np.arange(1, horizon_steps + 1, dtype=float)

    if fit.type_idx == TYPE_LEVEL:
        y_norm = np.full(horizon_steps, fit.L)
    elif fit.type_idx == TYPE_LINEAR:
        y_norm = x0_norm + fit.c * t
    elif fit.type_idx == TYPE_EXPONENTIAL:
        y_norm = fit.L + (x0_norm - fit.L) * np.exp(-fit.lam * t)
    else:  # oscillatory
        y_norm = fit.L + np.exp(-fit.lam * t) * (
            fit.A * np.cos(fit.omega * t) + fit.B * np.sin(fit.omega * t)
        )

    # Denormalise: add x0_actual as the absolute anchor
    values = [float(x0_actual + v * std) for v in y_norm]

    return values


def reconstruct_channel_sequence(
    feature_matrix: np.ndarray,
    channel_idx: int,
    x0_actual: float,
) -> list[float]:
    """Reconstruct a channel's value sequence from a predicted pattern matrix.

    Each row of feature_matrix is the feature vector of one predicted pattern
    (full vector for all channels).  We chain: end of pattern i → x0 of
    pattern i+1.
    """
    values: list[float] = []
    current_x0 = x0_actual

    for row in feature_matrix:
        start = channel_idx * DIMS_PER_CHANNEL
        ch_vec = row[start: start + DIMS_PER_CHANNEL]
        fit = channel_fit_from_vector(ch_vec)
        step = solve_forward(fit, current_x0, horizon_steps=1)
        if step:
            current_x0 = step[0]
            values.append(current_x0)

    return values


# ── Functional distance (for k-medoids) ──────────────────────────────────────

def _channel_integral_distance(fit1: ChannelFit, fit2: ChannelFit) -> float:
    """∫₀¹ (f1(t) - f2(t))² dt on normalised [0,1] interval, 100-point quad."""
    t = np.linspace(0.0, 1.0, 100)

    def _eval(fit: ChannelFit) -> np.ndarray:
        if fit.type_idx == TYPE_LEVEL:
            return _solve_level(t, fit.L)
        elif fit.type_idx == TYPE_LINEAR:
            return _solve_linear(t, 0.0, fit.c)
        elif fit.type_idx == TYPE_EXPONENTIAL:
            return _solve_exponential(t, 0.0, fit.L, fit.lam)
        else:
            return _solve_oscillatory(t, 0.0, fit.L, fit.lam, fit.omega, fit.A, fit.B)

    diff = _eval(fit1) - _eval(fit2)
    return float(np.trapz(diff ** 2, t))


def pattern_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    n_channels: int,
) -> float:
    """Sum of per-channel integral distances between two pattern feature vectors.

    Vectors have shape (n_channels * DIMS_PER_CHANNEL + DIMS_STRUCTURAL,).
    The structural (10th) channel contributes a simple squared difference.
    """
    total = 0.0
    for ch in range(n_channels):
        s = ch * DIMS_PER_CHANNEL
        f1 = channel_fit_from_vector(vec1[s: s + DIMS_PER_CHANNEL])
        f2 = channel_fit_from_vector(vec2[s: s + DIMS_PER_CHANNEL])
        total += _channel_integral_distance(f1, f2)

    # Structural channel: last element
    if len(vec1) > n_channels * DIMS_PER_CHANNEL:
        s1 = float(vec1[-1])
        s2 = float(vec2[-1])
        total += (s1 - s2) ** 2

    return total
