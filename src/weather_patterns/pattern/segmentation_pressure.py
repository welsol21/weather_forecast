"""Pressure segmentation using equation-fitting with natural boundaries.

Same walk-forward algorithm as segmentation_temperature.py.

Equation types for pressure (synoptic-scale dynamics, no solar harmonic):
  0 - exponential: x(t) = L + (x₀ − L)·e^(−λt)  — approach to equilibrium
  1 - linear:      x(t) = L + c·t                 — steady rise or fall
  2 - constant:    x(t) = L                        — stable system

Boundary = point where current equation prediction exceeds tolerance.
Tolerance: 1 hPa absolute.
Initial window: 24 hours.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

TOLERANCE_HPA    = 1.0
INITIAL_WINDOW_H = 24

EQ_EXPONENTIAL = "exponential"
EQ_LINEAR      = "linear"
EQ_CONSTANT    = "constant"

TRY_ORDER = [EQ_EXPONENTIAL, EQ_LINEAR, EQ_CONSTANT]

MIN_POINTS: dict[str, int] = {
    EQ_EXPONENTIAL: 3,
    EQ_LINEAR:      2,
    EQ_CONSTANT:    1,
}


# ── Fit data structures ────────────────────────────────────────────────────────

@dataclass
class PressureFit:
    eq_type: str
    L:   float = 0.0   # equilibrium / level / intercept
    c:   float = 0.0   # linear rate (EQ_LINEAR only)
    lam: float = 0.0   # decay rate  (EQ_EXPONENTIAL only)
    A:   float = 0.0   # departure x₀ − L (EQ_EXPONENTIAL only)
    rms: float = 0.0
    n_points: int = 0


@dataclass
class PressureSegment:
    segment_id:    int
    start_index:   int
    end_index:     int
    start_time:    pd.Timestamp
    end_time:      pd.Timestamp
    duration_hours: int
    fit:           PressureFit
    x0:            float


# ── Predictors ─────────────────────────────────────────────────────────────────

def _predict_value(fit: PressureFit, t: float) -> float:
    if fit.eq_type == EQ_CONSTANT:
        return fit.L
    if fit.eq_type == EQ_LINEAR:
        return fit.L + fit.c * t
    if fit.eq_type == EQ_EXPONENTIAL:
        return fit.L + fit.A * np.exp(-fit.lam * t)
    return fit.L


# ── Fitters ────────────────────────────────────────────────────────────────────

def _fit_constant(t: np.ndarray, y: np.ndarray) -> PressureFit:
    L   = float(np.mean(y))
    rms = float(np.sqrt(np.mean((y - L) ** 2)))
    return PressureFit(EQ_CONSTANT, L=L, rms=rms, n_points=len(y))


def _fit_linear(t: np.ndarray, y: np.ndarray) -> PressureFit:
    A_mat = np.column_stack([np.ones_like(t), t])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    L, c = float(coeffs[0]), float(coeffs[1])
    rms  = float(np.sqrt(np.mean((y - (L + c * t)) ** 2)))
    return PressureFit(EQ_LINEAR, L=L, c=c, rms=rms, n_points=len(y))


def _fit_exponential(t: np.ndarray, y: np.ndarray) -> PressureFit:
    x0 = float(y[0])
    # Estimate equilibrium as mean of last quarter of points
    L0 = float(np.mean(y[-max(1, len(y) // 4):]))

    def residuals(p):
        L, lam = p
        if lam <= 0:
            return np.full_like(y, 1e6)
        return y - (L + (x0 - L) * np.exp(-lam * t))

    try:
        res = least_squares(
            residuals, [L0, 0.05],
            bounds=([-np.inf, 1e-6], [np.inf, 5.0]),
            method="trf",
        )
        L, lam = float(res.x[0]), float(res.x[1])
        A   = x0 - L
        pred = L + A * np.exp(-lam * t)
        rms  = float(np.sqrt(np.mean((y - pred) ** 2)))
        return PressureFit(EQ_EXPONENTIAL, L=L, lam=lam, A=A, rms=rms, n_points=len(y))
    except Exception:
        return _fit_constant(t, y)


_FITTERS = {
    EQ_EXPONENTIAL: _fit_exponential,
    EQ_LINEAR:      _fit_linear,
    EQ_CONSTANT:    _fit_constant,
}


def _best_fit(t: np.ndarray, y: np.ndarray) -> PressureFit:
    best: PressureFit | None = None
    for eq_type in TRY_ORDER:
        if len(y) < MIN_POINTS[eq_type]:
            continue
        fit = _FITTERS[eq_type](t, y)
        if best is None or fit.rms < best.rms:
            best = fit
    return best or _fit_constant(t, y)


# ── Main segmentation ──────────────────────────────────────────────────────────

def segment_pressure(
    series: np.ndarray,
    timestamps: pd.DatetimeIndex,
    tolerance: float = TOLERANCE_HPA,
) -> tuple[list[PressureSegment], dict[str, int]]:
    n = len(series)
    if n == 0:
        return [], {}

    t0_wall = time.perf_counter()
    logger.info("pressure_segmentation_start n=%d tolerance=%.1f hPa", n, tolerance)

    segments:  list[PressureSegment] = []
    eq_counts: dict[str, int]        = {eq: 0 for eq in TRY_ORDER}

    seg_start = 0
    seg_id    = 0

    while seg_start < n:
        init_end = min(seg_start + INITIAL_WINDOW_H, n)
        t_init   = np.arange(init_end - seg_start, dtype=float)
        y_init   = series[seg_start:init_end]
        fit      = _best_fit(t_init, y_init)

        boundary = n
        for i in range(init_end, n):
            t_i    = float(i - seg_start)
            pred   = _predict_value(fit, t_i)
            actual = float(series[i])
            if not np.isfinite(actual):
                continue
            if abs(actual - pred) > tolerance:
                t_acc   = np.arange(i - seg_start + 1, dtype=float)
                y_acc   = series[seg_start:i + 1]
                new_fit = _best_fit(t_acc, y_acc)
                if abs(float(series[i]) - _predict_value(new_fit, t_acc[-1])) <= tolerance:
                    fit = new_fit
                    continue
                boundary = i
                break

        end_idx  = boundary - 1
        duration = end_idx - seg_start + 1

        seg = PressureSegment(
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
            "segment id=%d eq=%s duration_h=%d start=%s rms=%.3f hPa",
            seg_id, fit.eq_type, duration,
            timestamps[seg_start].strftime("%Y-%m-%d %H:%M"), fit.rms,
        )

        seg_start = boundary
        seg_id   += 1

    elapsed = time.perf_counter() - t0_wall
    logger.info("pressure_segmentation_complete segments=%d elapsed_s=%.1f",
                len(segments), elapsed)
    return segments, eq_counts


# ── Report ─────────────────────────────────────────────────────────────────────

def build_report(segments: list[PressureSegment], eq_counts: dict[str, int]) -> dict:
    total      = len(segments)
    durations  = [s.duration_hours for s in segments]
    rms_values = [s.fit.rms for s in segments]

    by_type = {}
    for eq in TRY_ORDER:
        segs_of_type = [s for s in segments if s.fit.eq_type == eq]
        count = len(segs_of_type)
        if count == 0:
            continue
        by_type[eq] = {
            "count":         count,
            "pct":           round(100 * count / total, 1),
            "mean_duration_h": round(float(np.mean([s.duration_hours for s in segs_of_type])), 1),
            "min_duration_h":  min(s.duration_hours for s in segs_of_type),
            "max_duration_h":  max(s.duration_hours for s in segs_of_type),
            "mean_rms_hpa":    round(float(np.mean([s.fit.rms for s in segs_of_type])), 3),
        }

    return {
        "total_segments":       total,
        "total_hours":          sum(durations),
        "mean_duration_h":      round(float(np.mean(durations)), 1),
        "min_duration_h":       min(durations),
        "max_duration_h":       max(durations),
        "overall_mean_rms_hpa": round(float(np.mean(rms_values)), 3),
        "by_equation_type":     by_type,
    }
