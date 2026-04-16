# Paper Artifacts — ODE-Segment Sequence Forecasting

**Authors:** Vladyslav Rastvorov · Dr. Nasir Ahmad  
**Institution:** Munster Technological University, Department of Computer Science  
**Code repository:** https://github.com/welsol21/weather_forecast  
**Date:** April 2026

---

## Contents

```
paper_artifacts/
├── README.md
├── report/                               # Technical report (final document)
│   ├── technical_report.html
│   └── technical_report.pdf
├── forecast/                             # Interactive February 2026 forecast report
│   └── february_forecast_report.html
├── figures/                              # Exploratory analysis figures (pressure channel)
│   ├── pressure_january_2026.png
│   ├── pressure_derivatives.png
│   ├── pressure_extrema_segments.png
│   └── pressure_sigma_comparison.png
├── models/                               # Trained transformer checkpoints (PyTorch)
│   ├── model_w12h.pt                     #   window = 12 h  → ensemble h 1–12
│   ├── model_w45d.pt                     #   window = 45 d  → ensemble h 13–24
│   └── model_w15d.pt                     #   window = 15 d  → ensemble h 25–168
├── training_reports/                     # Training summary JSON per window-size run
│   ├── model_w12h.json
│   ├── model_w24h.json
│   ├── model_w7d.json
│   ├── model_w15d.json
│   ├── model_w30d.json
│   └── model_w45d.json
├── segmentation_reports/                 # ODE segmentation statistics per channel
│   ├── temperature_segmentation.json
│   └── pressure_segmentation.json
└── segments/                             # Raw ODE segment data (training corpus)
    ├── temperature_segments.json         #   1,707 segments
    ├── pressure_segments.json            #   1,798 segments
    ├── windspeed_segments.json           #   1,951 segments
    └── joint_segments.json              #   10,055 joint segments (transformer input)
```

---

## 1. `report/` — Technical Report

| File | Description |
|------|-------------|
| `technical_report.html` | Full academic-style paper. Open in any browser. Contains interactive Chart.js charts, full methodology, results tables, and comparison against ECMWF IFS / GFS operational forecasts. |
| `technical_report.pdf`  | A4 print version of the same report, with matplotlib-rendered charts suitable for publication. Generated via `scripts/export_pdf.py`. |

**How it was produced:**  
The HTML report was written by hand (using `scripts/forecast_february.py` for data) and iterated in collaboration with the supervisor. The PDF was generated via Chrome headless print from a static-chart variant of the HTML (matplotlib PNGs replacing Chart.js canvases, to avoid headless rendering artefacts).

---

## 2. `forecast/` — February 2026 Forecast Report

| File | Description |
|------|-------------|
| `february_forecast_report.html` | Interactive forecast viewer for Feb 1–7 2026, Knock Airport (Met Éireann station 4935). Shows actual observations vs. ensemble model forecast for temperature, pressure, and wind speed. Open in a browser. |

**How it was produced:**  
`scripts/forecast_february.py` loads the three ensemble models (`w12h`, `w45d`, `w15d`), runs inference on the joint segment sequence up to 2026-01-31, decodes predicted ODE segments to hourly values, stitches the ensemble (h1–12 from w12h, h13–24 from w45d, h25–168 from w15d), and renders the result as a self-contained HTML page with Chart.js.

**Ensemble configuration:**

| Horizon | Model | Rationale |
|---------|-------|-----------|
| h 1–12  | `model_w12h` (window = 12 h) | Best short-range pressure MAE: 0.89 hPa |
| h 13–24 | `model_w45d` (window = 45 d) | Best temperature MAE at 24 h: 1.42°C |
| h 25–168 | `model_w15d` (window = 15 d) | Best temperature MAE at 168 h: 1.79°C |

---

## 3. `figures/` — Pressure Channel Analysis

These were produced during the exploratory analysis phase (mid-April 2026) to validate the segmentation algorithm on the pressure channel before extending it to all three channels.

| File | Description |
|------|-------------|
| `pressure_january_2026.png` | Raw hourly pressure (hPa) for January 2026 at Knock Airport, overlaid with the fitted ODE segments. Shows how the segmentation algorithm covers a typical winter month. |
| `pressure_derivatives.png` | First-derivative signal used to detect candidate segment boundaries. Peaks indicate rapid pressure changes (frontal passages). |
| `pressure_extrema_segments.png` | Detected local extrema (maxima/minima) used as hard boundary anchors, with the resulting segments highlighted. |
| `pressure_sigma_comparison.png` | Tolerance sensitivity study: residual RMS as a function of the walk-forward fitting tolerance parameter (σ). Used to select the final tolerance of 1 hPa. |

**How they were produced:**  
`scripts/run_joint_segmentation.py` (pressure analysis sub-section) calls the ODE fitter on the pressure channel and saves exploratory plots to the artifacts directory.

---

## 4. `models/` — Trained Transformer Checkpoints

These are the three PyTorch checkpoint files used to produce the ensemble forecast. All three share an identical model architecture (transformer encoder, ~2.4 M parameters); only the training context window differs.

| File | Window | Ensemble role | Final training loss |
|------|--------|---------------|-------------------|
| `model_w12h.pt` | 12 h | h 1–12  | 0.0695 |
| `model_w45d.pt` | 45 d | h 13–24 | 0.1039 |
| `model_w15d.pt` | 15 d | h 25–168 | 0.0931 |

**How to load a checkpoint:**
```python
import torch
from scripts.train_joint_two_pass import JointTransformer, N_TYPES, MAX_PARAMS

model = JointTransformer(N_TYPES, MAX_PARAMS)
model.load_state_dict(torch.load("models/model_w12h.pt", map_location="cpu"))
model.eval()
```

See `scripts/forecast_february.py` for the full inference pipeline.

---

## 5. `training_reports/` — Model Training Summaries

Each JSON file is the summary written by `scripts/train_joint_two_pass.py` at the end of training.

| File | Window | Role | Final Loss |
|------|--------|------|-----------|
| `model_w12h.json` | 12 h | Ensemble h 1–12 | 0.0695 |
| `model_w24h.json` | 24 h | Ablation only | 0.0373 |
| `model_w7d.json`  | 7 d  | Ablation only | — |
| `model_w15d.json` | 15 d | Ensemble h 25–168 | 0.0931 |
| `model_w30d.json` | 30 d | Ablation only | 0.1007 |
| `model_w45d.json` | 45 d | Ensemble h 13–24 | 0.1039 |

**Key observations:**
- Loss is a per-segment mixed MSE over equation type (cross-entropy) + parameter regression.
- The 24 h window achieves the lowest loss on the training distribution but the worst held-out forecast MAE — evidence of overfitting to recent segment statistics.
- The ablation result (`w24h` loss 0.037 vs `w12h` loss 0.070) motivated choosing the 12 h model for short-range rather than the ostensibly "better" 24 h model.

---

## 6. `segmentation_reports/` — ODE Segmentation Statistics

| File | Description |
|------|-------------|
| `temperature_segmentation.json` | Per-equation-type breakdown for the temperature channel: segment counts, mean/min/max duration, mean residual RMS. |
| `pressure_segmentation.json` | Same for the pressure channel. |

**Temperature channel summary (from JSON):**
- 1,707 segments over 53,352 hours (6 years)
- Mean segment duration: 31.3 h
- Dominant type: `linear_harmonic` (72.3 % of segments)
- Overall mean RMS: 0.566 °C (within 1 °C tolerance)

**Pressure channel summary (from JSON):**
- 1,798 segments over 53,352 hours
- Mean segment duration: 29.7 h
- Split almost evenly: `exponential` 49.8 %, `linear` 50.2 %
- Overall mean RMS: 0.883 hPa (within 1 hPa tolerance)

Wind speed segmentation report is not included separately (statistics are embedded in the joint segmentation log).

---

## 7. `segments/` — Full Segment Data Files

These are the raw output of the ODE segmentation pipeline. Each JSON array contains one record per fitted segment.

| File | Segments | Size | Description |
|------|----------|------|-------------|
| `temperature_segments.json` | 1,707 | 760 KB | Temperature ODE fits (6 equation types) |
| `pressure_segments.json` | 1,798 | 724 KB | Pressure ODE fits (3 equation types) |
| `windspeed_segments.json` | 1,951 | 779 KB | Wind speed ODE fits (3 equation types) |
| `joint_segments.json` | 10,055 | 7.6 MB | Union of all three channels after boundary unification. This is the actual training corpus for the transformer. |

**Schema (one record in `joint_segments.json`):**
```json
{
  "start": "2020-01-01T00:00:00",
  "end":   "2020-01-02T05:00:00",
  "duration_h": 29,
  "temp_eq":  "linear_harmonic",
  "temp_params": [A, omega, phi, m, b],
  "pres_eq":  "exponential",
  "pres_params": [A, tau, b],
  "wind_eq":  "linear",
  "wind_params": [m, b]
}
```

**How they were produced:**  
`scripts/run_joint_segmentation.py` runs the three per-channel ODE fitters (in `src/weather_patterns/pattern/`) on `hly4935_subset.csv` (Met Éireann HLY station 4935, January 2020 – January 2026), collects the union of all segment boundaries, and re-fits each channel on the joint grid.

---

## Data Source

All observations: **Met Éireann HLY dataset, station 4935 (Knock Airport, Co. Mayo, Ireland)**  
Period: January 2020 – February 2026  
Variables used: dry-bulb temperature (°C), mean sea-level pressure (hPa), wind speed (knots)  
File in repository: `hly4935_subset.csv`

Operational NWP comparison data (ECMWF IFS 0.25°, GFS Seamless) were fetched from the **Open-Meteo Historical Forecast API** for the same station coordinates and the February 1–7 2026 evaluation period.

---

*Munster Technological University · Department of Computer Science · Final Year Project 2026*
