# Weather Forecast ML: Theory Evolution and Experimental Runs

Station: Knock Airport Ireland, EIKN / Met Éireann station 4935
Training data: 2020–2026 (hourly, 9 channels)
Channels: temperature, relative_humidity, pressure, wind_speed, rainfall, dew_point, wet_bulb, vapour_pressure, wind_direction

---

## Starting Assumptions (Pre-Run 1)

The foundational bet: weather has **recurring local patterns** — short time windows where the joint dynamics of all channels are recognisably similar across different dates. If those patterns can be clustered, the sequence of past patterns should predict the sequence of future patterns, and the predicted future patterns can be decoded back into physical values.

Early assumptions:
- A **pattern** is a statistical fingerprint of a window: what values are typical, how fast they're changing, how correlated channels are.
- Clustering by those fingerprints groups "similar atmospheric states".
- Forecasting = predicting which cluster sequence will follow the current window sequence.

---

## Run 1 — Baseline: Extrema-Based Windows, Statistical Features

**Segmentation**: extrema-driven variable-length windows (window boundaries = local extrema events)  
**Feature vector**: concatenation of intra-channel statistics (current value, deltas at 1/6/24 steps, mean, variance, integral, extrema count/amplitude), inter-channel correlations and lag-correlations, peak hazard metrics (prominence, duration over threshold, cumulative risk), and time placeholders  
**Discovery**: k-means over these vectors → pattern prototypes  
**Forecasting**: Transformer/GRU sequence model predicting the next N pattern vectors given the last K

Results (24-h horizon, test set):
- Mean channel MAE: ~1.8× persistence at 1-3h, ~1.0× at 6h, ~0.85× at 12-24h
- Temperature: reasonable at longer horizons, poor near-term
- Wind direction: poor throughout (circular topology not handled)

What we learned: the model beats the last-value baseline at longer horizons but is worse near-term. This is physically expected — the model trades short-term accuracy for long-term structure. But the feature vector has a deeper problem.

---

## The Fundamental Problem: Structural vs. Absolute Features

After Run 1, we identified the core flaw in the feature representation:

> The statistical features mix **structure** (how the channel is evolving) with **scale** (what the absolute value is). When two temperature windows have the same dynamics but different base temperatures (e.g., +5°C in winter, +15°C in summer), they appear far apart in feature space. The clustering groups windows by absolute value, not by dynamic structure.

This means:
- Similar atmospheric dynamics in different seasons get different cluster IDs
- The model learns season-confounded patterns, not physical dynamics
- The feature vector is not **scale-free**

---

## Runs 2–4 — Incremental Refinements (Not Fully Documented)

Experiments explored: smoothing window changes, different cluster counts, modified inter-channel feature weights. All used the same statistical feature framework. Marginal improvements, but the root problem remained.

---

## Run 5 — Hierarchical Segmentation

**Key insight**: the signal is **non-stationary** — there are regime shifts where the local dynamics change qualitatively (frontal passage, pressure system transition). A sliding window over the whole time series mixes windows from different regimes.

**Solution**: two-level segmentation:
1. **Level 1 (predictor blocks)**: run the predictor segmentation — find contiguous intervals where a single local model fits well. Block boundaries = regime changes.
2. **Level 2 (sliding windows)**: within each block, cut fixed-stride sliding windows.

Cross-block training pairs are excluded: history from block A + target from block B is semantically wrong because the dynamics changed.

Results vs. Run 1:
- Consistently better across all channels and horizons (no regressions)
- Larger improvements at longer horizons (12-24h)
- Temperature MAE/persistence ratio: 0.91→0.83 at 24h

Why: the sequence model now sees temporally coherent training pairs. The features still have the scale-mixing problem, but the structural signal is cleaner.

---

## Run 6 — New Physics: Convergence Function Representation

### The New Theory

A pattern is not a statistical fingerprint. A pattern is a **set of local convergence functions** — one per channel — each describing how that channel evolves toward its local limit on the current time interval.

Formally: on time interval $[t_0, t_0 + W]$, channel $c$ follows one of four local function types:

| Type | Name | Description |
|------|------|-------------|
| 0 | level | Channel is at or very near its local limit: $x(t) = \text{lim}$ |
| 1 | velocity | Linear approach: $x(t) = x_0 + v \cdot t$ |
| 2 | acceleration | Quadratic: $x(t) = x_0 + v \cdot t + \frac{1}{2} a \cdot t^2$ |
| 3 | local AR(2) | Autoregressive convergence: $x_t = \phi_1 x_{t-1} + \phi_2 x_{t-2} + c$, fixed point = $c/(1-\phi_1-\phi_2)$ |

Key design principles:

1. **Channels are independent**: no inter-channel coupling in the local functions. Wind speed and pressure may be correlated globally, but within a short window each follows its own local equation. (Inter-channel coupling is implicit in the sequence model that predicts which pattern follows which.)

2. **Scale-free representation**: all parameters are normalized by the local std of that channel on that window. A rapid temperature drop in January and a rapid drop in July encode the same $r_\text{norm}$ if the normalized rate is the same. Clustering now groups structurally similar dynamics, not absolute-value-similar windows.

3. **lim, not constant**: for the "level" type we say the channel has reached its local *limit* — the dynamic interpretation is more precise than saying "constant", because pressure, humidity, or rainfall can be at a quasi-stationary attractor for the duration of the window.

4. **Forecast = sequence of patterns**: predicting the next $K$ patterns IS the forecast. Each pattern's local convergence function has an analytical solution — no iterative accumulation of one-step errors. The sequence model predicts the pattern sequence; decoding chains each pattern's solution, using the end of pattern $i$ as the initial condition for pattern $i+1$.

### Feature Vector Layout (per channel, 9 floats)

```
[0..3]  type one-hot:   [is_level, is_velocity, is_acceleration, is_ar2]
[4]     rate_norm:      (predicted_next - current) / std
[5]     accel_norm:     (d²x/dt²) / std                (0 for level/velocity/ar2)
[6]     ar_phi1:        AR(2) coefficient φ₁             (0 for non-ar2)
[7]     ar_phi2:        AR(2) coefficient φ₂             (0 for non-ar2)
[8]     ar_stability:   1 - |φ₁ + φ₂|  (1=stable, 0=unit-root, <0=explosive)
```

Total: 9 × 9 channels = **81 floats** (vs. ~431 floats in the statistical representation).

The type is selected by fitting all applicable predictors to the last `history_window_steps` of the normalized channel series and picking the one with minimum one-step prediction error.

### Decoding: Analytical Reconstruction

Given the predicted pattern sequence and the last observed channel values (placeholders), reconstruction is:

```
For each predicted pattern i:
    x_{i+1} = solve_forward(type, rate_norm, accel_norm, ar_phi1, ar_phi2,
                             initial_value=x_i, channel_std, horizon_steps=1)
```

For AR(2), the forward iteration runs in normalized space:
- Seed: $x_0^\text{norm} = 0$, $x_{-1}^\text{norm} = -r_\text{norm}$
- Intercept: $c^\text{norm} = x_0^\text{norm}(1 - \phi_1 - \phi_2)$ → fixed point = $x_i$ (the initial value)
- Unscale: $x_{i+1} = x_i + x_\text{next}^\text{norm} \cdot \sigma$

For velocity: $x_{i+1} = x_i + r_\text{norm} \cdot \sigma$  
For level: $x_{i+1} = x_i$

### Segmentation

New Physics does NOT use hierarchical segmentation or predictor blocks. A pattern describes the behaviour of each channel on a given time interval independently of absolute values and season. The same pattern type (e.g. temperature falling linearly) is the same pattern in January and July. Therefore the entire 5-year history is one continuous sequence of patterns — simple sliding windows over the full history, no blocks, no cross-block filtering.

Run 6 incorrectly inherited the hierarchical segmentation from Run 5. This was wrong: the blocks were a workaround for the fact that old statistical features mixed regimes. New Physics features are scale-free and self-contained per window — regime blocks are not needed.

Run 7 corrects this: simple sliding windows over the full history, same as Run 1, but with New Physics feature vectors instead of statistical ones.

### Implementation Notes

- `src/weather_patterns/pattern/convergence.py`: core extraction and reconstruction module
- `PatternWindow.channel_stds`: per-channel std (scale placeholder for decoding)
- `PatternWindow.channel_end_values`: last observed raw value per channel (initial condition)
- `intra_matrix`, `inter_matrix`, `peak_hazard_matrix` are empty for new_physics windows
- Evaluation auto-detects the mode by checking feature vector size (81 vs. ~431)

---

## Run 8 — Proper Differential Equations, Natural Boundaries, Channel-First Discovery

### What was wrong in Runs 6–7

Runs 6 and 7 used fixed-length sliding windows (stride = 1 hour). This is cutting, not finding. A window boundary placed at an arbitrary clock tick may fall in the middle of a real dynamic regime — the fitted function then describes a transition, not a coherent pattern. The result is noise in the feature space and the model learns nothing (ratio = 1.00).

### Natural boundaries

A pattern boundary is a point where the current equation of a channel stops fitting the data. This is determined by the mathematics, not by the clock.

The algorithm for each channel:
1. Fit the base equation on an initial window (24 hours for temperature)
2. Walk forward one step at a time — check whether the fitted equation still describes the data
3. When it stops fitting — that is the boundary
4. At the boundary, find the best equation for the next segment from the candidate set
5. If the same equation continues to work past the initial window boundary (e.g. 48h, 72h) — there is no pattern boundary. The pattern continues as long as the equation holds.

The 24-hour initial window is a fitting unit, not a hard boundary. Pattern length is determined entirely by the mathematics.

### Function types per channel

Different channels have different physical dynamics. The candidate equation set is chosen per channel based on its physical nature.

**Temperature** — governed by the 24-hour solar cycle plus synoptic-scale disturbances. Six candidate equations:

| Type | Equation |
|------|----------|
| constant | `x(t) = L` |
| linear | `x(t) = x₀ + c·t` |
| exponential | `x(t) = L + (x₀ − L)·e^(−λt)` |
| harmonic | `x(t) = L + A·cos(ωt) + B·sin(ωt)`,  ω = 2π/24 fixed |
| linear + harmonic | `x(t) = x₀ + c·t + A·cos(ωt) + B·sin(ωt)` |
| damped harmonic | `x(t) = L + e^(−λt)·(A·cos(ωt) + B·sin(ωt))` |

The base (default) equation for temperature is the **harmonic** — a full 24-hour solar cycle. The algorithm starts with it. If it fails, the other types are tried.

### Parameters: what goes where

- **Feature vector** (describes the shape of dynamics, used for clustering): type, L, λ, ω, A, B
- **Placeholder** (actual observed value, substituted at forecast time): x₀ — the real channel value at the start of the window

L is a fitted parameter of the equation (the mathematical attractor), not an observed value. It belongs in the feature vector.

L and all other parameters are found jointly via nonlinear optimisation (e.g. least squares) — not estimated separately or from heuristics like window mean or inflection point.

### Channel-first discovery

Patterns are found in two stages:

1. **Channel patterns**: for each of the 9 channels independently, walk the 5-year series and find segments where a single differential equation holds. Each segment is a channel pattern with its equation parameters.

2. **Final patterns**: take the union of all boundaries from all 9 channel sequences. Each interval between consecutive boundaries is a final pattern — a period where all 9 channels are simultaneously stable under their local equations.

### The 10th channel

The number of channels that simultaneously change their equation at a given boundary (0–9) is recorded as a 10th structural channel. A high value means a sharp synchronised regime transition (e.g. frontal passage). A low value means a quiet single-channel shift. This participates in clustering and sequence prediction alongside the physical channels.

### Clustering

Because patterns have variable length and their features are functions (not fixed-length vectors), clustering uses a **functional distance metric**:

`d(p₁, p₂) = Σ_channels ∫₀¹ (f₁_c(t) − f₂_c(t))² dt`

where both functions are evaluated on a normalised [0, 1] time interval. This allows direct comparison between, say, an exponential and an oscillatory pattern without padding or approximation.

Clustering method: **k-medoids** on the pairwise distance matrix.

---

## Temperature Channel Prototype — Two-Pass Sequence Model

### Context

Before applying the full 9-channel pipeline, the temperature channel is used as a prototype to develop and validate the sequence forecasting approach. Five years of hourly temperature data (2020–2026-01-31) are segmented into equation-based patterns using the walk-forward algorithm from Run 8. This produces 1707 segments with a mean duration of ~30 hours. February 2026 is held out for evaluation.

### Why a prototype

The full 9-channel pipeline has a compound complexity: segmentation, cross-channel boundary union, clustering, and sequence prediction all interact. Working on a single channel isolates the forecasting problem: given a history of equation-based patterns, predict future patterns correctly.

### Experiments

| Approach | MAE (Feb 1, first 24h) | Notes |
|----------|----------------------|-------|
| GRU, history=14 segments, autoregressive | 3.92°C (full Feb) | Autoregressive: one segment predicted at a time, history window of 14 |
| Transformer, full history, autoregressive | 13.5°C (full Feb) | Full 1707-segment context; error accumulates over 21+ autoregressive steps |
| Transformer, direct 24h value prediction | 1.55°C | Bypasses segment parameters; predicts hourly values directly — not the right direction |

### The right direction: two-pass architecture

The goal is to predict **sequences of equation-based patterns**, not raw values. Predicting patterns preserves the physical meaning of the representation and allows arbitrary-horizon forecasting by chaining pattern solutions analytically.

The failure mode of the autoregressive Transformer (MAE 13.5°C) is not a fundamental flaw of pattern-based prediction — it is a consequence of generating one segment at a time and feeding each predicted segment back as input. Error accumulates with each step.

**Two-pass architecture**:

- **Pass 1 — Length**: given the full history of segments (5 years of context, including seasonal patterns) and a desired time interval T, predict K — how many segments will occur in that interval. Seasonality matters: segment duration is not uniform across the year (winter vs. summer dynamics differ).

- **Pass 2 — Content**: given the full history + T + K, predict the K segment vectors (equation type + coefficients). Again, seasonality is fully present in the history context.

Both passes use the same encoder (full 5-year segment history with temporal features). Only the query/head differs.

Inference for any horizon T:
1. Pass 1 → K
2. Pass 2 → [seg_1, ..., seg_K] (equation parameters)
3. Decode: chain segment solutions analytically, x₀ of each segment = end value of previous
4. Return the first T hours of the decoded series

This avoids autoregressive error accumulation: both passes are single forward passes over the fixed historical context.

---

## Process of Discovery — Summary

```
Run 1: extrema windows + statistical features → baseline established
       → problem: scale-mixed features, regime-mixing windows

Run 5: hierarchical segmentation → cleaner training pairs, consistent improvement
       → problem: feature representation still structurally wrong

Insight: patterns should describe *dynamics*, not *absolute state*
       → need scale-free, structure-first representation

New Physics theory (Runs 6–7): pattern = set of local convergence functions per channel
       → scale-free (normalized by local std)
       → analytically solvable forward
       → but: fixed-stride sliding windows = cutting, not finding
       → result: model learns nothing (ratio = 1.00)

Insight: boundaries must come from the mathematics, not the clock
       → boundary = point where local equation stops fitting
       → patterns are found, not cut

Run 8: proper differential equations (exponential, oscillatory with complex roots)
       + natural boundaries from prediction error
       + channel-first discovery → union of channel boundaries = final patterns
       + 10th channel = synchrony of boundary events
       + k-medoids on functional distance metric
```

The journey reflects a deepening from statistical summarisation → scale-free dynamics → full differential equation representation with mathematically determined boundaries. The key shift: we find patterns, we do not cut them.
