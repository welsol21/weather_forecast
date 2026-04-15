"""Temperature-only segmentation using equation-fitting with natural boundaries.

Algorithm:
1. Start at position 0. Collect 24 points (one full solar cycle), fit harmonic.
2. Walk forward hour by hour. At each step predict the current value using the
   fitted equation. If |actual - predicted| > TOLERANCE_CELSIUS → boundary.
3. At a boundary: try all 6 candidate equations on the accumulated points of
   the NEW segment (minimum points required per type). Pick the one with the
   lowest RMS error on those points. Harmonic is tried first.
4. If no equation fits within tolerance — fall back to constant (always works).
5. Continue walking with the new equation.

Equation types for temperature (in try order):
  0 - harmonic:        x(t) = L + A·cos(ωt) + B·sin(ωt),  ω = 2π/24
  1 - linear+harmonic: x(t) = x₀ + c·t + A·cos(ωt) + B·sin(ωt)
  2 - damped harmonic: x(t) = L + e^(-λt)·(A·cos(ωt) + B·sin(ωt))
  3 - exponential:     x(t) = L + (x₀ - L)·e^(-λt)
  4 - linear:          x(t) = x₀ + c·t
  5 - constant:        x(t) = L

Tolerance: 1°C absolute.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

TOLERANCE_CELSIUS = 1.0
OMEGA = 2 * np.pi / 24.0   # fixed solar-cycle frequency

EQ_HARMONIC        = "harmonic"
EQ_LINEAR_HARMONIC = "linear_harmonic"
EQ_DAMPED_HARMONIC = "damped_harmonic"
EQ_EXPONENTIAL     = "exponential"
EQ_LINEAR          = "linear"
EQ_CONSTANT        = "constant"

# Minimum points needed to fit each type
MIN_POINTS: dict[str, int] = {
    EQ_HARMONIC:        24,
    EQ_LINEAR_HARMONIC: 24,
    EQ_DAMPED_HARMONIC: 24,
    EQ_EXPONENTIAL:     3,
    EQ_LINEAR:          2,
    EQ_CONSTANT:        1,
}

TRY_ORDER = [
    EQ_HARMONIC,
    EQ_LINEAR_HARMONIC,
    EQ_DAMPED_HARMONIC,
    EQ_EXPONENTIAL,
    EQ_LINEAR,
    EQ_CONSTANT,
]


# ── Fit data structures ────────────────────────────────────────────────────────

@dataclass
class TemperatureFit:
    eq_type: str
    L: float = 0.0
    c: float = 0.0
    lam: float = 0.0
    A: float = 0.0
    B: float = 0.0
    rms: float = 0.0      # root-mean-square error on fitting points
    n_points: int = 0     # number of points used for fitting


@dataclass
class TemperatureSegment:
    segment_id: int
    start_index: int
    end_index: int         # inclusive
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: int
    fit: TemperatureFit
    x0: float              # actual observed temperature at start (placeholder)


# ── Equation solvers ───────────────────────────────────────────────────────────

def _predict_value(fit: TemperatureFit, t: float) -> float:
    if fit.eq_type == EQ_CONSTANT:
        return fit.L
    if fit.eq_type == EQ_LINEAR:
        return fit.L + fit.c * t
    if fit.eq_type == EQ_EXPONENTIAL:
        return fit.L + (fit.A) * np.exp(-fit.lam * t)   # A = x0 - L
    if fit.eq_type == EQ_HARMONIC:
        return fit.L + fit.A * np.cos(OMEGA * t) + fit.B * np.sin(OMEGA * t)
    if fit.eq_type == EQ_LINEAR_HARMONIC:
        return fit.L + fit.c * t + fit.A * np.cos(OMEGA * t) + fit.B * np.sin(OMEGA * t)
    if fit.eq_type == EQ_DAMPED_HARMONIC:
        return fit.L + np.exp(-fit.lam * t) * (fit.A * np.cos(OMEGA * t) + fit.B * np.sin(OMEGA * t))
    return fit.L


def _predict_array(fit: TemperatureFit, t_arr: np.ndarray) -> np.ndarray:
    return np.array([_predict_value(fit, float(t)) for t in t_arr])


# ── Fitters ────────────────────────────────────────────────────────────────────

def _fit_constant(t: np.ndarray, y: np.ndarray) -> TemperatureFit:
    L = float(np.mean(y))
    rms = float(np.sqrt(np.mean((y - L) ** 2)))
    return TemperatureFit(EQ_CONSTANT, L=L, rms=rms, n_points=len(y))


def _fit_linear(t: np.ndarray, y: np.ndarray) -> TemperatureFit:
    A_mat = np.column_stack([np.ones_like(t), t])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    L, c = float(coeffs[0]), float(coeffs[1])
    pred = L + c * t
    rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    return TemperatureFit(EQ_LINEAR, L=L, c=c, rms=rms, n_points=len(y))


def _fit_exponential(t: np.ndarray, y: np.ndarray) -> TemperatureFit:
    x0 = float(y[0])
    L0 = float(np.mean(y[-max(1, len(y) // 4):]))

    def residuals(p):
        L, lam = p
        if lam <= 0:
            return np.full_like(y, 1e6)
        return y - (L + (x0 - L) * np.exp(-lam * t))

    try:
        result = least_squares(residuals, [L0, 0.1], bounds=([-np.inf, 1e-6], [np.inf, 10.0]), method="trf")
        L, lam = float(result.x[0]), float(result.x[1])
        pred = L + (x0 - L) * np.exp(-lam * t)
        rms = float(np.sqrt(np.mean((y - pred) ** 2)))
        return TemperatureFit(EQ_EXPONENTIAL, L=L, lam=lam, A=x0 - L, rms=rms, n_points=len(y))
    except Exception:
        return _fit_constant(t, y)


def _fit_harmonic(t: np.ndarray, y: np.ndarray) -> TemperatureFit:
    # Linear in L, A, B: design matrix [1, cos(ωt), sin(ωt)]
    A_mat = np.column_stack([np.ones_like(t), np.cos(OMEGA * t), np.sin(OMEGA * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    L, A, B = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    pred = L + A * np.cos(OMEGA * t) + B * np.sin(OMEGA * t)
    rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    return TemperatureFit(EQ_HARMONIC, L=L, A=A, B=B, rms=rms, n_points=len(y))


def _fit_linear_harmonic(t: np.ndarray, y: np.ndarray) -> TemperatureFit:
    # Linear in L, c, A, B: design matrix [1, t, cos(ωt), sin(ωt)]
    A_mat = np.column_stack([np.ones_like(t), t, np.cos(OMEGA * t), np.sin(OMEGA * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    L, c, A, B = float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])
    pred = L + c * t + A * np.cos(OMEGA * t) + B * np.sin(OMEGA * t)
    rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    return TemperatureFit(EQ_LINEAR_HARMONIC, L=L, c=c, A=A, B=B, rms=rms, n_points=len(y))


def _fit_damped_harmonic(t: np.ndarray, y: np.ndarray) -> TemperatureFit:
    # Nonlinear in λ: x(t) = L + e^(-λt)·(A·cos(ωt) + B·sin(ωt))
    # For fixed λ, linear in L, A, B
    def residuals(p):
        L, lam, A, B = p
        if lam < 0:
            return np.full_like(y, 1e6)
        return y - (L + np.exp(-lam * t) * (A * np.cos(OMEGA * t) + B * np.sin(OMEGA * t)))

    harm_fit = _fit_harmonic(t, y)
    p0 = [harm_fit.L, 0.01, harm_fit.A, harm_fit.B]
    try:
        result = least_squares(residuals, p0, bounds=([-np.inf, 0, -np.inf, -np.inf], [np.inf, 5.0, np.inf, np.inf]), method="trf")
        L, lam, A, B = [float(v) for v in result.x]
        pred = L + np.exp(-lam * t) * (A * np.cos(OMEGA * t) + B * np.sin(OMEGA * t))
        rms = float(np.sqrt(np.mean((y - pred) ** 2)))
        return TemperatureFit(EQ_DAMPED_HARMONIC, L=L, lam=lam, A=A, B=B, rms=rms, n_points=len(y))
    except Exception:
        return harm_fit


_FITTERS = {
    EQ_HARMONIC:        _fit_harmonic,
    EQ_LINEAR_HARMONIC: _fit_linear_harmonic,
    EQ_DAMPED_HARMONIC: _fit_damped_harmonic,
    EQ_EXPONENTIAL:     _fit_exponential,
    EQ_LINEAR:          _fit_linear,
    EQ_CONSTANT:        _fit_constant,
}


def _best_fit(t: np.ndarray, y: np.ndarray) -> TemperatureFit:
    """Try all equations in TRY_ORDER, return the one with lowest RMS."""
    best: TemperatureFit | None = None
    for eq_type in TRY_ORDER:
        if len(y) < MIN_POINTS[eq_type]:
            continue
        fit = _FITTERS[eq_type](t, y)
        if best is None or fit.rms < best.rms:
            best = fit
    return best or _fit_constant(t, y)


# ── Main segmentation ──────────────────────────────────────────────────────────

def segment_temperature(
    series: np.ndarray,
    timestamps: pd.DatetimeIndex,
    tolerance: float = TOLERANCE_CELSIUS,
) -> tuple[list[TemperatureSegment], dict[str, int]]:
    """Find natural pattern boundaries in the temperature series.

    Returns:
        segments: list of TemperatureSegment in chronological order
        eq_counts: dict[equation_type → count of segments]
    """
    n = len(series)
    if n == 0:
        return [], {}

    t0_wall = time.perf_counter()
    logger.info("temperature_segmentation_start n=%d tolerance=%.1f°C", n, tolerance)

    segments: list[TemperatureSegment] = []
    eq_counts: dict[str, int] = {eq: 0 for eq in TRY_ORDER}

    seg_start = 0
    seg_id = 0

    while seg_start < n:
        # --- Initial fit on first 24 points (or fewer if near end) ---
        init_end = min(seg_start + 24, n)
        t_init = np.arange(init_end - seg_start, dtype=float)
        y_init = series[seg_start:init_end]
        fit = _best_fit(t_init, y_init)

        # --- Walk forward from init_end ---
        boundary = n   # default: segment runs to end
        for i in range(init_end, n):
            t_i = float(i - seg_start)
            pred = _predict_value(fit, t_i)
            actual = float(series[i])
            if not np.isfinite(actual):
                continue
            if abs(actual - pred) > tolerance:
                # Refit on everything from seg_start to i (accumulated data)
                t_acc = np.arange(i - seg_start + 1, dtype=float)
                y_acc = series[seg_start:i + 1]
                new_fit = _best_fit(t_acc, y_acc)
                # Check if new fit also fails immediately
                pred_next = _predict_value(new_fit, float(i - seg_start))
                if abs(float(series[i]) - pred_next) <= tolerance:
                    # New equation describes the transition point — keep walking with it
                    fit = new_fit
                    continue
                # Hard boundary
                boundary = i
                break

        end_idx = boundary - 1
        duration = end_idx - seg_start + 1

        seg = TemperatureSegment(
            segment_id=seg_id,
            start_index=seg_start,
            end_index=end_idx,
            start_time=timestamps[seg_start],
            end_time=timestamps[end_idx],
            duration_hours=duration,
            fit=fit,
            x0=float(series[seg_start]),
        )
        segments.append(seg)
        eq_counts[fit.eq_type] = eq_counts.get(fit.eq_type, 0) + 1

        logger.info(
            "segment id=%d eq=%s duration_h=%d start=%s rms=%.3f°C",
            seg_id, fit.eq_type, duration,
            timestamps[seg_start].strftime("%Y-%m-%d %H:%M"), fit.rms,
        )

        seg_start = boundary
        seg_id += 1

    elapsed = time.perf_counter() - t0_wall
    logger.info(
        "temperature_segmentation_complete segments=%d elapsed_s=%.1f",
        len(segments), elapsed,
    )
    return segments, eq_counts


# ── Report ─────────────────────────────────────────────────────────────────────

def build_report(segments: list[TemperatureSegment], eq_counts: dict[str, int]) -> dict:
    total = len(segments)
    durations = [s.duration_hours for s in segments]
    rms_values = [s.fit.rms for s in segments]

    by_type = {}
    for eq in TRY_ORDER:
        segs_of_type = [s for s in segments if s.fit.eq_type == eq]
        count = len(segs_of_type)
        if count == 0:
            continue
        by_type[eq] = {
            "count": count,
            "pct": round(100 * count / total, 1),
            "mean_duration_h": round(float(np.mean([s.duration_hours for s in segs_of_type])), 1),
            "min_duration_h": min(s.duration_hours for s in segs_of_type),
            "max_duration_h": max(s.duration_hours for s in segs_of_type),
            "mean_rms_celsius": round(float(np.mean([s.fit.rms for s in segs_of_type])), 3),
        }

    return {
        "total_segments": total,
        "total_hours": sum(durations),
        "mean_duration_h": round(float(np.mean(durations)), 1),
        "min_duration_h": min(durations),
        "max_duration_h": max(durations),
        "overall_mean_rms_celsius": round(float(np.mean(rms_values)), 3),
        "by_equation_type": by_type,
    }
