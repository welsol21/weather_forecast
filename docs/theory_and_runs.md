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

Run 6 uses the same hierarchical segmentation as Run 5 (predictor blocks → sliding windows within blocks). The segmentation strategy is `new_physics`. Cross-block sample pairs are excluded from training, same as `hierarchical`.

### Implementation Notes

- `src/weather_patterns/pattern/convergence.py`: core extraction and reconstruction module
- `PatternWindow.channel_stds`: per-channel std (scale placeholder for decoding)
- `PatternWindow.channel_end_values`: last observed raw value per channel (initial condition)
- `intra_matrix`, `inter_matrix`, `peak_hazard_matrix` are empty for new_physics windows
- Evaluation auto-detects the mode by checking feature vector size (81 vs. ~431)

---

## Process of Discovery — Summary

```
Run 1: sliding extrema windows + statistical features → baseline established
       → problem: scale-mixed features, regime-mixing windows

Run 5: hierarchical segmentation → cleaner training pairs, consistent improvement
       → problem: feature representation still structurally wrong

Insight: patterns should describe *dynamics*, not *absolute state*
       → need scale-free, structure-first representation

New Physics theory: pattern = set of local convergence functions per channel
       → scale-free (normalized by local std)
       → analytically solvable forward
       → forecast = sequence of patterns, not sequence of point predictions

Run 6: new_physics segmentation + convergence feature vectors
```

The journey reflects a shift from a data-engineering mindset (summarize the window statistically) to a physics-inspired mindset (identify the local dynamic regime). The scale-free representation allows clustering to group structurally equivalent dynamics regardless of season or baseline level — which is what physical patterns actually are.
