# ODE-Segment Sequence Forecasting

**Learning Weather Pattern Grammar from Single-Station Observations**

> A transformer-based approach to single-station deterministic weather forecasting
> via walk-forward analytical segmentation and symmetric-window sequence modelling.

**Authors:** Vladyslav Rastvorov (Final Year Student) · Dr. Nasir Ahmad (Supervisor)  
**Institution:** Munster Technological University, Department of Computer Science  
**Repository:** [github.com/welsol21/weather_forecast](https://github.com/welsol21/weather_forecast)

---

## Abstract

Raw hourly weather observations are compressed into sequences of analytical ODE segments —
each segment described by an equation type and its fitted parameters. A transformer encoder
is then trained to predict the future sequence of segments from a symmetric window of
historical ones. Unlike NWP-based methods, the approach uses no atmospheric physics or
spatial information; unlike standard time-series models, it operates on an interpretable,
compressed representation rather than raw sensor values.

On a 7-day evaluation period (February 2026, Knock Airport, Ireland), the ensemble achieves:

| Horizon | Temperature MAE | Pressure MAE | Wind MAE |
|---------|----------------|--------------|----------|
| h 1–12  | 1.42°C         | **0.89 hPa** | 2.48 kt  |
| h 13–24 | 2.31°C         | 3.74 hPa     | 1.67 kt  |
| h 25–168 (days 2–7) | 1.79°C | 7.04 hPa | 6.22 kt |
| **Overall (168h)** | **1.80°C** | **6.37 hPa** | **5.62 kt** |

Pressure at 12 h beats naive persistence by **34%** (0.89 vs 1.35 hPa).

---

## Key Idea

Instead of predicting raw time-series values, we predict the **sequence of analytical
equations** that describe upcoming weather. A "rising pressure" event becomes an
exponential-decay segment with fitted rate and amplitude; a diurnal temperature cycle
becomes a harmonic segment. The transformer learns the grammar of how these patterns
follow one another.

```
Raw observations
       │
       ▼
ODE Segmentation (per channel)
  temp:  6 equation types (harmonic, linear-harmonic, damped-harmonic,
                           exponential, linear, constant)
  pres:  3 equation types (exponential, linear, constant)
  wind:  3 equation types (exponential, linear, constant)
       │
       ▼
Joint boundary unification  →  10,055 segments over 6 years (mean 5.3h)
       │
       ▼
Transformer Encoder
  Input:  tail window of segment vectors (right-aligned, zero-padded to 256)
  Output: head window of segment vectors (future 256 positions)
       │
       ▼
ODE decoding  →  hourly forecast
```

---

## Results vs Baselines

| Method | T MAE @24h | T MAE @168h | P MAE @12h | P MAE @168h |
|--------|-----------|-------------|-----------|-------------|
| Persistence | 1.47°C | 1.14°C | 1.35 hPa | 8.00 hPa |
| Climatology | 1.71°C | 1.81°C | 17.4 hPa | 28.1 hPa |
| **Ours (Ensemble)** | **1.42°C** | **1.79°C** | **0.89 hPa** | **7.04 hPa** |
| ECMWF-IFS (lit.) | ~1.2°C | ~2.5°C | ~1.0 hPa | ~6.0 hPa |

---

## Window-Size Ablation

The optimal training window approximately equals the target forecast horizon,
motivating an ensemble of models:

| Window | T MAE @24h | T MAE @168h | Used for |
|--------|-----------|-------------|----------|
| 12h | 3.58°C | 4.29°C | **h 1–12** (best pressure at 0.89 hPa) |
| 15d | 1.89°C | **1.80°C** | **h 25–168** |
| 45d | **1.99°C** | 1.78°C | **h 13–24** |

---

## Repository Structure

```
weather_forecast/
├── src/weather_patterns/
│   └── pattern/
│       ├── segmentation_temperature.py   # 6 equation types, tolerance 1°C
│       ├── segmentation_pressure.py      # 3 equation types, tolerance 1 hPa
│       └── segmentation_windspeed.py     # 3 equation types, tolerance 2 kt
├── scripts/
│   ├── run_joint_segmentation.py         # Build joint_segments.json
│   ├── train_joint_two_pass.py           # Train transformer (--window-days N)
│   └── forecast_february.py             # Ensemble inference + HTML report
├── segments/
│   ├── temperature_segments.json         # 1,707 segments
│   ├── pressure_segments.json            # 1,798 segments
│   ├── windspeed_segments.json           # 1,951 segments
│   └── joint_segments.json              # 10,055 joint segments
├── artifacts/                            # symlink → /mnt/ml/ (model checkpoints)
│   ├── joint_two_pass_w12h.pt
│   ├── joint_two_pass_w15d.pt
│   ├── joint_two_pass.pt                 # 45d model (legacy name)
│   ├── february_forecast_report.html     # Interactive forecast report
│   └── technical_report.html            # Full technical report (paper draft)
├── docs/
│   ├── project_journey.md               # Development history and process notes
│   ├── TZ.docx                          # Original project specification
│   ├── project_formulation.md
│   └── theory_and_runs.md
└── hly4935_subset.csv                   # Met Éireann HLY station 4935
```

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

**Step 1 — Build joint segments** (requires pre-built per-channel JSONs):
```bash
python scripts/run_joint_segmentation.py
```

**Step 2 — Train a model** (GPU recommended, ~2h per run):
```bash
python scripts/train_joint_two_pass.py --window-days 15
python scripts/train_joint_two_pass.py --window-days 45
python scripts/train_joint_two_pass.py --window-hours 12
```

**Step 3 — Generate forecast report**:
```bash
python scripts/forecast_february.py
# Opens: artifacts/february_forecast_report.html
```

Artifacts (model checkpoints) are stored outside the repo on `/mnt/ml/` and
exposed via the `artifacts/` symlink. See `docs/project_journey.md` for
storage setup details.

---

## Future Work

- Extended evaluation: 90-day rolling test period across all seasons
- Multi-station generalisation (Shannon, Dublin Airport, Valentia)
- Wind vector decomposition: replace scalar speed with (u, v) components
- Relative humidity as a fourth channel
- Formal ECMWF baseline on identical dates
- Probabilistic forecasting over segment sequences

---

## Technical Report

A full paper-style technical report (methodology, results, baseline comparison,
proposed research agenda) is available at
[`artifacts/technical_report.html`](artifacts/technical_report.html).

---

*Munster Technological University · Department of Computer Science · Final Year Project 2026*
