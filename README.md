# ODE-Segment Sequence Forecasting

**Learning Weather Pattern Grammar from Single-Station Observations**

**Authors:** Vladyslav Rastvorov (Final Year Student) · Dr. Nasir Ahmad (Supervisor)  
**Institution:** Munster Technological University, Department of Computer Science  
**Repository:** [github.com/welsol21/weather_forecast](https://github.com/welsol21/weather_forecast)  
**Technical Report:** [docs/technical_report.html](docs/technical_report.html)

---

## Motivation

This work did not originate in meteorology. It originated in a broader question about the nature of sequential data — specifically, in prior work on linguistic pattern analysis. There, we observed that natural language is not simply a stream of tokens: it is a stream of *regimes*, each with internal structure, characteristic duration, and predictable transitions to the next.

The intellectual precedent we find most clarifying is **Mendeleev's Periodic Table**. Mendeleev did not discover new elements — he discovered that elements form repeating patterns, and that those patterns are predictive. The method was: identify the minimal unit of description, classify it, study the grammar of its sequences, use that grammar to forecast.

We propose this is a **general scientific method**. Language was the domain where it was first formalised at the neural-network level. **Weather was the next test** — and the results suggest the approach transfers.

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
| h 1–12  | 1.42°C | **0.89 hPa** | 2.48 kt |
| h 13–24 | 2.31°C | 3.74 hPa | 1.67 kt |
| h 25–168 (days 2–7) | 1.79°C | 7.04 hPa | 6.22 kt |

Pressure at 12 h beats naive persistence by **34%** (0.89 vs 1.35 hPa).

---

## Key Idea

Instead of predicting raw time-series values, we predict the **sequence of analytical
equations** that describe upcoming weather. A "rising pressure" event becomes an
exponential-decay segment; a diurnal temperature cycle becomes a harmonic segment.
The transformer learns the grammar of how these regimes follow one another.

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

| Method | T MAE @24h | T MAE @168h | P MAE @12h | P MAE @168h | W MAE @168h |
|--------|-----------|-------------|-----------|-------------|-------------|
| Persistence | 1.47°C | 1.14°C | 1.35 hPa | 8.00 hPa | 7.21 kt |
| Climatology | 1.71°C | 1.81°C | 17.4 hPa | 28.1 hPa | 4.41 kt |
| ECMWF IFS 0.25° | 0.73°C | 0.44°C | 0.13 hPa | 0.59 hPa | 1.48 kt |
| GFS Seamless | 0.79°C | 0.67°C | 0.58 hPa | 0.83 hPa | 1.78 kt |
| **Ours (Ensemble)** | **1.42°C** | **1.79°C** | **0.89 hPa** | **7.04 hPa** | **6.22 kt** |

ECMWF and GFS figures are actual operational forecasts for the same station and dates,
fetched from the Open-Meteo Historical Forecast API — not literature approximations.

---

## Window-Size Ablation

The optimal training window approximately equals the target forecast horizon,
motivating an ensemble of models:

| Window | T MAE @24h | T MAE @168h | Selected for |
|--------|-----------|-------------|--------------|
| 12h | 3.58°C | 4.29°C | **h 1–12** (best pressure: 0.89 hPa) |
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
├── docs/
│   ├── technical_report.html            # Full paper-draft technical report
│   ├── project_journey.md               # Development history and process notes
│   ├── TZ.docx                          # Original project specification
│   ├── project_formulation.md
│   └── theory_and_runs.md
├── artifacts/                            # symlink → /mnt/ml/ (model checkpoints)
│   ├── joint_two_pass_w12h.pt
│   ├── joint_two_pass_w15d.pt
│   └── joint_two_pass.pt                # 45d model (legacy name)
└── hly4935_subset.csv                   # Met Éireann HLY station 4935
```

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

**Step 1 — Build joint segments:**
```bash
python scripts/run_joint_segmentation.py
```

**Step 2 — Train models** (GPU recommended, ~2h per run):
```bash
python scripts/train_joint_two_pass.py --window-hours 12
python scripts/train_joint_two_pass.py --window-days 15
python scripts/train_joint_two_pass.py --window-days 45
```

**Step 3 — Generate forecast report:**
```bash
python scripts/forecast_february.py
# Output: artifacts/february_forecast_report.html
```

Artifacts (model checkpoints) are stored on `/mnt/ml/` and exposed via the
`artifacts/` symlink. See `docs/project_journey.md` for storage setup details.

---

## Future Work

- Extended evaluation: 90-day rolling test period across all seasons
- Multi-station generalisation (Shannon, Dublin Airport, Valentia)
- Wind vector decomposition: replace scalar speed with (u, v) components
- Relative humidity as a fourth channel
- Probabilistic forecasting over segment sequences
- Application to other sequential physical domains beyond meteorology

---

*Munster Technological University · Department of Computer Science · Final Year Project 2026*
