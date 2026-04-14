"""Channel-first pattern segmentation for Run 8.

Algorithm:
1. For each of the 9 channels independently, walk the time series forward.
   At each step, check whether the current fitted ODE still predicts the
   next observed value within tolerance.  When it fails — that is a channel
   boundary.  Re-fit a new ODE from the current point.

2. Collect all channel boundaries.  Take their union.

3. On every interval between consecutive union-boundaries, build the final
   PatternSegment: re-fit each channel on that interval, encode the feature
   vector, record x0 (placeholder), and count how many channels changed
   their ODE simultaneously at the left boundary (10th channel).

Tolerance is expressed as a multiple of the channel std on the current
segment.  Default: error > tolerance_sigma * std → boundary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from weather_patterns.pattern.convergence import (
    ChannelFit,
    DIMS_PER_CHANNEL,
    DIMS_STRUCTURAL,
    channel_fit_to_vector,
    fit_channel_segment,
)

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PatternSegment:
    """One final pattern: a time interval where all channels are stable."""
    segment_id: int
    start_index: int          # inclusive, row index in signal_frame
    end_index: int            # inclusive
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    channels: list[str]
    channel_fits: dict[str, ChannelFit]   # per channel
    boundary_synchrony: int               # 10th channel value
    feature_vector: np.ndarray            # shape: (n_channels*10 + 1,)
    channel_x0: dict[str, float]          # placeholder: x0 per channel


# ── Step 1: find channel boundaries ──────────────────────────────────────────

def _find_channel_boundaries(
    series: np.ndarray,
    channel: str,
    tolerance_sigma: float,
    min_segment_length: int,
) -> list[int]:
    """Return sorted list of boundary indices for one channel.

    A boundary at index i means the segment [prev_boundary, i-1] ends and
    [i, next_boundary-1] begins.  First boundary is always 0.
    """
    n = len(series)
    if n == 0:
        return [0]

    boundaries: list[int] = [0]
    seg_start = 0

    while seg_start < n:
        seg = series[seg_start:]
        fit = fit_channel_segment(seg, channel)

        std = fit.std if fit.std > 1e-8 else 1.0
        threshold = tolerance_sigma * std

        # Walk forward from seg_start + min_segment_length
        boundary_found = False
        for i in range(min_segment_length, len(seg)):
            predicted = _predict_at_step(fit, seg, i)
            actual = float(series[seg_start + i]) if np.isfinite(series[seg_start + i]) else predicted
            error = abs(actual - predicted)
            if error > threshold:
                new_boundary = seg_start + i
                boundaries.append(new_boundary)
                seg_start = new_boundary
                boundary_found = True
                break

        if not boundary_found:
            break

    return sorted(set(boundaries))


def _predict_at_step(fit: ChannelFit, seg: np.ndarray, step: int) -> float:
    """Predict value at position `step` within `seg` using the fitted ODE."""
    from weather_patterns.pattern.convergence import (
        TYPE_LEVEL, TYPE_LINEAR, TYPE_EXPONENTIAL, TYPE_OSCILLATORY,
        _solve_level, _solve_linear, _solve_exponential, _solve_oscillatory,
    )
    std = fit.std if fit.std > 1e-8 else 1.0
    finite = seg[np.isfinite(seg)]
    mean = float(np.mean(finite)) if finite.size > 0 else 0.0
    x0_norm = (float(seg[0]) - mean) / std if np.isfinite(seg[0]) else 0.0
    t = float(step)

    if fit.type_idx == TYPE_LEVEL:
        y_norm = fit.L
    elif fit.type_idx == TYPE_LINEAR:
        y_norm = x0_norm + fit.c * t
    elif fit.type_idx == TYPE_EXPONENTIAL:
        y_norm = fit.L + (x0_norm - fit.L) * np.exp(-fit.lam * t)
    else:
        y_norm = fit.L + np.exp(-fit.lam * t) * (
            fit.A * np.cos(fit.omega * t) + fit.B * np.sin(fit.omega * t)
        )

    return float(y_norm * std + mean)


# ── Step 2: union of boundaries ───────────────────────────────────────────────

def _union_boundaries(
    channel_boundaries: dict[str, list[int]],
    n: int,
) -> tuple[list[int], dict[int, int]]:
    """Union all channel boundaries.  Also compute synchrony count per boundary.

    Returns:
        union_boundaries: sorted list of boundary indices
        synchrony: dict[boundary_index → count of channels with boundary here]
    """
    from collections import Counter
    counts: Counter[int] = Counter()
    for blist in channel_boundaries.values():
        counts.update(blist)

    union = sorted(counts.keys())
    # Ensure 0 and n are present
    if not union or union[0] != 0:
        union = [0] + union
    if union[-1] != n:
        union.append(n)

    synchrony = {b: counts.get(b, 0) for b in union}
    return union, synchrony


# ── Step 3: build final PatternSegments ──────────────────────────────────────

def _build_segment(
    segment_id: int,
    start_idx: int,
    end_idx: int,
    signal_frame: pd.DataFrame,
    channel_arrays: dict[str, np.ndarray],
    channels: list[str],
    synchrony_count: int,
    time_column: str,
) -> PatternSegment:
    t_start = time.perf_counter()

    start_time = pd.Timestamp(signal_frame.iloc[start_idx][time_column])
    end_time = pd.Timestamp(signal_frame.iloc[min(end_idx, len(signal_frame) - 1)][time_column])

    channel_fits: dict[str, ChannelFit] = {}
    channel_x0: dict[str, float] = {}
    vec_parts: list[np.ndarray] = []

    for ch in channels:
        arr = channel_arrays[ch][start_idx: end_idx + 1]
        fit = fit_channel_segment(arr, ch)
        channel_fits[ch] = fit
        channel_x0[ch] = fit.x0
        vec_parts.append(channel_fit_to_vector(fit))

    # 10th channel: boundary synchrony
    structural = np.array([float(synchrony_count)], dtype=float)
    feature_vector = np.concatenate(vec_parts + [structural])

    elapsed = time.perf_counter() - t_start
    logger.debug(
        "segment_built id=%d start=%d end=%d len=%d synchrony=%d elapsed_ms=%.1f",
        segment_id, start_idx, end_idx, end_idx - start_idx + 1,
        synchrony_count, elapsed * 1000,
    )

    return PatternSegment(
        segment_id=segment_id,
        start_index=start_idx,
        end_index=end_idx,
        start_time=start_time,
        end_time=end_time,
        channels=channels,
        channel_fits=channel_fits,
        boundary_synchrony=synchrony_count,
        feature_vector=feature_vector,
        channel_x0=channel_x0,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def segment_time_series(
    signal_frame: pd.DataFrame,
    channel_arrays: dict[str, np.ndarray],
    channels: list[str],
    tolerance_sigma: float = 2.0,
    min_segment_length: int = 3,
    time_column: str = "date",
) -> list[PatternSegment]:
    """Find natural pattern boundaries and build PatternSegments.

    Parameters
    ----------
    signal_frame      : full dataframe with timestamps
    channel_arrays    : dict channel → raw numpy array (length = len(signal_frame))
    channels          : ordered list of channel names (9 physical channels)
    tolerance_sigma   : boundary threshold in units of channel std
    min_segment_length: minimum number of points before a boundary can occur
    time_column       : name of the datetime column in signal_frame

    Returns
    -------
    List of PatternSegment ordered by start_index.
    """
    wall_start = time.perf_counter()
    n = len(signal_frame)
    logger.info(
        "segmentation_start n=%d channels=%d tolerance_sigma=%.2f min_segment=%d",
        n, len(channels), tolerance_sigma, min_segment_length,
    )

    # Step 1: find boundaries per channel
    channel_boundaries: dict[str, list[int]] = {}
    for ch in channels:
        t0 = time.perf_counter()
        arr = channel_arrays.get(ch)
        if arr is None or len(arr) == 0:
            channel_boundaries[ch] = [0]
            continue
        blist = _find_channel_boundaries(arr, ch, tolerance_sigma, min_segment_length)
        channel_boundaries[ch] = blist
        logger.info(
            "channel_boundaries channel=%s boundaries=%d elapsed_ms=%.0f",
            ch, len(blist), (time.perf_counter() - t0) * 1000,
        )

    # Step 2: union
    union, synchrony = _union_boundaries(channel_boundaries, n)
    logger.info("union_boundaries total=%d", len(union) - 1)

    # Step 3: build segments
    segments: list[PatternSegment] = []
    for i in range(len(union) - 1):
        start_idx = union[i]
        end_idx = union[i + 1] - 1
        if end_idx < start_idx:
            continue
        seg = _build_segment(
            segment_id=i,
            start_idx=start_idx,
            end_idx=end_idx,
            signal_frame=signal_frame,
            channel_arrays=channel_arrays,
            channels=channels,
            synchrony_count=synchrony.get(start_idx, 0),
            time_column=time_column,
        )
        segments.append(seg)

    elapsed = time.perf_counter() - wall_start
    logger.info(
        "segmentation_complete segments=%d elapsed_s=%.1f",
        len(segments), elapsed,
    )
    return segments
