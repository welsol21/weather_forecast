"""Ensemble forecast for February 1–7 2026.

Uses three models optimised for different horizons:
  - w12h  → hours  1–12   (best short-range)
  - w1080h (45d) → hours 13–24  (best for ~24h)
  - w360h (15d)  → hours 25–168 (best for 1–7 days)

Compares predictions against actual observations and
generates an HTML report with matplotlib charts.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from weather_patterns.pattern.segmentation_temperature import (
    TRY_ORDER as TEMP_EQ_ORDER, _predict_value as temp_predict, TemperatureFit,
)
from weather_patterns.pattern.segmentation_pressure import (
    TRY_ORDER as PRES_EQ_ORDER, _predict_value as pres_predict, PressureFit,
)
from weather_patterns.pattern.segmentation_windspeed import (
    TRY_ORDER as WIND_EQ_ORDER, _predict_value as wind_predict, WindspeedFit,
)
from train_joint_two_pass import (
    JointForecaster, _global_scales, segment_to_vector, vector_to_fits,
    decode_to_series, FEAT_DIM, ODE_DIM, MAX_SEQ_LEN,
)

SEGMENTS_PATH = Path("segments/joint_segments.json")
CSV_PATH      = Path("hly4935_subset.csv")
OUT_HTML      = Path("artifacts/february_forecast_report.html")

MODELS = {
    "12h":  (Path("artifacts/joint_two_pass_w12h.pt"),  12),
    "45d":  (Path("artifacts/joint_two_pass.pt"),        1080),
    "15d":  (Path("artifacts/joint_two_pass_w15d.pt"),   360),
}

FEB1 = pd.Timestamp("2026-02-01 00:00")
N_HOURS = 168  # 7 days


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(path: Path, half_window_h: int, device: str) -> JointForecaster:
    model = JointForecaster()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def run_forecast(
    model: JointForecaster,
    segments: list[dict],
    sc: dict,
    half_window_h: int,
    center: pd.Timestamp,
    device: str,
    n_hours: int,
    temp_x0: float,
    pres_x0: float,
    wind_x0: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    starts  = np.array([pd.Timestamp(s["start_time"]).value for s in segments], dtype=np.int64)
    vectors = np.array([segment_to_vector(s, sc) for s in segments], dtype=np.float32)

    cv  = center.value
    wv  = int(half_window_h * pd.Timedelta(hours=1).value)
    idx = np.where((starts >= cv - wv) & (starts < cv))[0]

    tail = np.zeros((1, MAX_SEQ_LEN, FEAT_DIM), dtype=np.float32)
    mask = np.zeros((1, MAX_SEQ_LEN), dtype=bool)
    t_len = min(len(idx), MAX_SEQ_LEN)
    if t_len > 0:
        tail[0, MAX_SEQ_LEN - t_len:] = vectors[idx[-t_len:]]
        mask[0, MAX_SEQ_LEN - t_len:] = True

    with torch.no_grad():
        tail_t = torch.tensor(tail, dtype=torch.float32).to(device)
        mask_t = torch.tensor(mask, dtype=torch.bool).to(device)
        pred   = model(tail_t, mask_t)[0].cpu().numpy()

    predicted = []
    for v in pred:
        tf, pf, wf, dur = vector_to_fits(v, sc)
        if dur > 0:
            predicted.append((tf, pf, wf, dur))

    return decode_to_series(predicted, temp_x0, pres_x0, wind_x0, center, n_hours)


# ── Stitch ensemble ────────────────────────────────────────────────────────────

def stitch(
    pred_12h:  tuple[pd.Series, pd.Series, pd.Series],
    pred_45d:  tuple[pd.Series, pd.Series, pd.Series],
    pred_15d:  tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    h 1–12:   12h model
    h 13–24:  45d model
    h 25–168: 15d model
    """
    result_temp, result_pres, result_wind = {}, {}, {}

    for i, (pt, pp, pw) in enumerate(zip(*pred_12h)):
        ts = FEB1 + pd.Timedelta(hours=i)
        if i < 12:
            result_temp[ts] = pt
            result_pres[ts] = pp
            result_wind[ts] = pw

    for i, (pt, pp, pw) in enumerate(zip(pred_45d[0], pred_45d[1], pred_45d[2])):
        ts = FEB1 + pd.Timedelta(hours=i)
        if 12 <= i < 24:
            result_temp[ts] = pt
            result_pres[ts] = pp
            result_wind[ts] = pw

    for i, (pt, pp, pw) in enumerate(zip(pred_15d[0], pred_15d[1], pred_15d[2])):
        ts = FEB1 + pd.Timedelta(hours=i)
        if i >= 24:
            result_temp[ts] = pt
            result_pres[ts] = pp
            result_wind[ts] = pw

    idx = sorted(result_temp.keys())
    return (
        pd.Series([result_temp[ts] for ts in idx], index=idx),
        pd.Series([result_pres[ts] for ts in idx], index=idx),
        pd.Series([result_wind[ts] for ts in idx], index=idx),
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def mae_rmse(actual: pd.Series, pred: pd.Series) -> tuple[float, float]:
    common = actual.index.intersection(pred.index)
    if len(common) == 0:
        return float("nan"), float("nan")
    e = actual.loc[common].values - pred.loc[common].values
    e = e[np.isfinite(e)]
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2)))


# ── Chart data builder ─────────────────────────────────────────────────────────

def _ser_to_js(s: pd.Series) -> str:
    """Convert a pandas Series with DatetimeIndex to a JS array of {x,y} objects."""
    pts = []
    for ts, v in s.items():
        if np.isfinite(v):
            pts.append(f'{{"x":"{ts.isoformat()}","y":{v:.3f}}}')
        else:
            pts.append(f'{{"x":"{ts.isoformat()}","y":null}}')
    return "[" + ",".join(pts) + "]"


def build_html(
    actual_temp: pd.Series, actual_pres: pd.Series, actual_wind: pd.Series,
    ens_temp: pd.Series, ens_pres: pd.Series, ens_wind: pd.Series,
    pred_12h: tuple, pred_45d: tuple, pred_15d: tuple,
    mae_t: float, rmse_t: float,
    mae_p: float, rmse_p: float,
    mae_w: float, rmse_w: float,
    mae_t1: float, mae_p1: float, mae_w1: float,
    mae_t2: float, mae_p2: float, mae_w2: float,
    mae_t3: float, mae_p3: float, mae_w3: float,
) -> str:

    def cls(v, good, ok):
        return "good" if v <= good else ("ok" if v <= ok else "bad")

    # Build per-range slice series (only the hours each model is responsible for)
    h12_end  = FEB1 + pd.Timedelta(hours=12)
    h24_end  = FEB1 + pd.Timedelta(hours=24)
    h168_end = FEB1 + pd.Timedelta(hours=168)

    def slice_ser(s, start, end):
        return s[(s.index >= start) & (s.index < end)]

    p12_t = slice_ser(pred_12h[0], FEB1, h12_end)
    p12_p = slice_ser(pred_12h[1], FEB1, h12_end)
    p12_w = slice_ser(pred_12h[2], FEB1, h12_end)

    p45_t = slice_ser(pred_45d[0], h12_end, h24_end)
    p45_p = slice_ser(pred_45d[1], h12_end, h24_end)
    p45_w = slice_ser(pred_45d[2], h12_end, h24_end)

    p15_t = slice_ser(pred_15d[0], h24_end, h168_end)
    p15_p = slice_ser(pred_15d[1], h24_end, h168_end)
    p15_w = slice_ser(pred_15d[2], h24_end, h168_end)

    feb1_iso  = FEB1.isoformat()
    h12_iso   = h12_end.isoformat()
    h24_iso   = h24_end.isoformat()
    h168_iso  = h168_end.isoformat()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>February 2026 Weather Forecast — EIKN</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 0 0 48px;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
    border-bottom: 1px solid #1e2535;
    padding: 32px 40px 28px;
    margin-bottom: 32px;
  }}
  .header h1 {{
    font-size: 26px; font-weight: 700; letter-spacing: -.3px;
    color: #f1f5f9;
  }}
  .header h1 span {{ color: #60a5fa; }}
  .header p {{
    margin-top: 6px; font-size: 13px; color: #64748b; line-height: 1.5;
  }}
  .badge {{
    display: inline-block;
    background: #1e2535; border: 1px solid #2d3748;
    border-radius: 20px; padding: 2px 10px;
    font-size: 11px; color: #94a3b8; margin-right: 6px; margin-top: 8px;
  }}

  /* ── Layout ── */
  .page {{ max-width: 1280px; margin: 0 auto; padding: 0 32px; }}

  /* ── KPI row ── */
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }}
  .kpi {{
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
  }}
  .kpi::after {{
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 2px 0 0 2px;
  }}
  .kpi.temp::after {{ background: #f87171; }}
  .kpi.pres::after {{ background: #60a5fa; }}
  .kpi.wind::after {{ background: #34d399; }}
  .kpi .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #64748b; }}
  .kpi .mae   {{ font-size: 36px; font-weight: 800; line-height: 1.1; margin: 6px 0 2px; }}
  .kpi.temp .mae {{ color: #fca5a5; }}
  .kpi.pres .mae {{ color: #93c5fd; }}
  .kpi.wind .mae {{ color: #6ee7b7; }}
  .kpi .rmse  {{ font-size: 12px; color: #475569; }}
  .kpi .sub   {{ font-size: 11px; color: #475569; margin-top: 4px; }}

  /* ── Chart cards ── */
  .card {{
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
  }}
  .card-header {{
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 16px;
  }}
  .card-header h2 {{
    font-size: 14px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .06em; color: #94a3b8;
  }}
  .card-header .card-mae {{
    font-size: 12px; color: #475569;
  }}
  .chart-wrap {{ position: relative; height: 260px; }}

  /* ── Range legend ── */
  .range-legend {{
    display: flex; gap: 20px; margin-top: 14px; flex-wrap: wrap;
  }}
  .rl-item {{
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: #64748b;
  }}
  .rl-swatch {{
    width: 24px; height: 3px; border-radius: 2px;
    display: inline-block; flex-shrink: 0;
  }}
  .rl-swatch.dashed {{ background: repeating-linear-gradient(90deg,currentColor 0,currentColor 4px,transparent 4px,transparent 8px); height:2px; }}

  /* ── Table ── */
  .table-card {{
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 20px;
  }}
  .table-card h2 {{
    font-size: 14px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .06em; color: #94a3b8;
    padding: 20px 24px 14px;
    border-bottom: 1px solid #1e2535;
  }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
  }}
  th {{
    background: #1a1f2e; padding: 10px 20px;
    text-align: left; font-weight: 600; color: #64748b;
    font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
  }}
  td {{ padding: 11px 20px; border-top: 1px solid #1e2535; color: #cbd5e1; }}
  tr:hover td {{ background: #1a1f2e; }}
  .good {{ color: #4ade80; font-weight: 600; }}
  .ok   {{ color: #fb923c; font-weight: 600; }}
  .bad  {{ color: #f87171; font-weight: 600; }}
  .model-badge {{
    display: inline-block; background: #1e2535; border-radius: 4px;
    padding: 2px 7px; font-size: 11px; color: #94a3b8;
  }}

  /* ── Footer ── */
  .footer {{
    margin-top: 32px; padding: 20px 24px;
    background: #161b27; border: 1px solid #1e2535; border-radius: 12px;
    font-size: 11px; color: #475569; line-height: 1.7;
  }}
  .footer b {{ color: #64748b; }}
</style>
</head>
<body>

<div class="header">
  <h1>February 2026 — <span>7-Day Forecast</span></h1>
  <p>
    <span class="badge">EIKN / Knock Airport, Ireland</span>
    <span class="badge">ODE Segmentation + Transformer Ensemble</span>
    <span class="badge">Training: 2020-01 – 2026-01</span>
    <span class="badge">Test: Feb 1–7 2026</span>
  </p>
</div>

<div class="page">

  <!-- KPI -->
  <div class="kpi-row">
    <div class="kpi temp">
      <div class="label">Temperature — overall MAE</div>
      <div class="mae">{mae_t:.2f}°C</div>
      <div class="rmse">RMSE {rmse_t:.2f}°C</div>
    </div>
    <div class="kpi pres">
      <div class="label">Pressure — overall MAE</div>
      <div class="mae">{mae_p:.2f} hPa</div>
      <div class="rmse">RMSE {rmse_p:.2f} hPa</div>
    </div>
    <div class="kpi wind">
      <div class="label">Wind Speed — overall MAE</div>
      <div class="mae">{mae_w:.2f} kt</div>
      <div class="rmse">RMSE {rmse_w:.2f} kt</div>
    </div>
  </div>

  <!-- Temperature chart -->
  <div class="card">
    <div class="card-header">
      <h2>Temperature</h2>
      <span class="card-mae">Overall MAE {mae_t:.2f}°C &nbsp;|&nbsp; RMSE {rmse_t:.2f}°C</span>
    </div>
    <div class="chart-wrap"><canvas id="ch_temp"></canvas></div>
    <div class="range-legend">
      <span class="rl-item"><span class="rl-swatch" style="background:#f1f5f9;"></span>Actual</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#f87171;"></span>Ensemble forecast</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#60a5fa;opacity:.7"></span>12h model (h1–12)</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#c084fc;opacity:.7"></span>45d model (h13–24)</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#fb923c;opacity:.7"></span>15d model (h25–168)</span>
    </div>
  </div>

  <!-- Pressure chart -->
  <div class="card">
    <div class="card-header">
      <h2>Pressure (MSL)</h2>
      <span class="card-mae">Overall MAE {mae_p:.2f} hPa &nbsp;|&nbsp; RMSE {rmse_p:.2f} hPa</span>
    </div>
    <div class="chart-wrap"><canvas id="ch_pres"></canvas></div>
    <div class="range-legend">
      <span class="rl-item"><span class="rl-swatch" style="background:#f1f5f9;"></span>Actual</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#60a5fa;"></span>Ensemble forecast</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#60a5fa;opacity:.5"></span>12h model (h1–12)</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#c084fc;opacity:.5"></span>45d model (h13–24)</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#fb923c;opacity:.5"></span>15d model (h25–168)</span>
    </div>
  </div>

  <!-- Wind chart -->
  <div class="card">
    <div class="card-header">
      <h2>Wind Speed</h2>
      <span class="card-mae">Overall MAE {mae_w:.2f} kt &nbsp;|&nbsp; RMSE {rmse_w:.2f} kt</span>
    </div>
    <div class="chart-wrap"><canvas id="ch_wind"></canvas></div>
    <div class="range-legend">
      <span class="rl-item"><span class="rl-swatch" style="background:#f1f5f9;"></span>Actual</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#34d399;"></span>Ensemble forecast</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#34d399;opacity:.5"></span>12h model (h1–12)</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#c084fc;opacity:.5"></span>45d model (h13–24)</span>
      <span class="rl-item"><span class="rl-swatch" style="background:#fb923c;opacity:.5"></span>15d model (h25–168)</span>
    </div>
  </div>

  <!-- Accuracy table -->
  <div class="table-card">
    <h2>Accuracy by Forecast Range</h2>
    <table>
      <tr>
        <th>Horizon</th>
        <th>Model used</th>
        <th>Temp MAE</th>
        <th>Pres MAE</th>
        <th>Wind MAE</th>
      </tr>
      <tr>
        <td>h 1–12 &nbsp;<small style="color:#475569">(first 12 h)</small></td>
        <td><span class="model-badge">12h window</span></td>
        <td class="{cls(mae_t1,1.5,3)}">{mae_t1:.2f}°C</td>
        <td class="{cls(mae_p1,2,5)}">{mae_p1:.2f} hPa</td>
        <td class="{cls(mae_w1,2,4)}">{mae_w1:.2f} kt</td>
      </tr>
      <tr>
        <td>h 13–24 &nbsp;<small style="color:#475569">(hours 13–24)</small></td>
        <td><span class="model-badge">45d window</span></td>
        <td class="{cls(mae_t2,1.5,3)}">{mae_t2:.2f}°C</td>
        <td class="{cls(mae_p2,2,5)}">{mae_p2:.2f} hPa</td>
        <td class="{cls(mae_w2,2,4)}">{mae_w2:.2f} kt</td>
      </tr>
      <tr>
        <td>h 25–168 &nbsp;<small style="color:#475569">(days 2–7)</small></td>
        <td><span class="model-badge">15d window</span></td>
        <td class="{cls(mae_t3,2,4)}">{mae_t3:.2f}°C</td>
        <td class="{cls(mae_p3,5,10)}">{mae_p3:.2f} hPa</td>
        <td class="{cls(mae_w3,3,6)}">{mae_w3:.2f} kt</td>
      </tr>
    </table>
  </div>

  <!-- Footer -->
  <div class="footer">
    <b>Method:</b>
    Walk-forward ODE segmentation across 3 channels (temperature · pressure · wind speed).
    Segment boundaries defined where residual error exceeds channel tolerance (temp 1°C, pres 1 hPa, wind 2 kt).
    Joint boundary set = union of all per-channel boundaries.
    Transformer encoder over a symmetric historical window → predicts the future pattern sequence.
    Temperature uses 6 equation types (harmonic, linear-harmonic, damped-harmonic, exponential, linear, constant);
    pressure and wind use 3 (exponential, linear, constant).
    Ensemble selects the best-performing model per horizon based on cross-validation results.
  </div>

</div><!-- /page -->

<script>
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = '#1e2535';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
Chart.defaults.font.size = 11;

const ZONE_12  = '{feb1_iso}';
const ZONE_45  = '{h12_iso}';
const ZONE_15  = '{h24_iso}';
const ZONE_END = '{h168_iso}';

// Vertical band annotation via beforeDraw plugin
const bandPlugin = {{
  id: 'bands',
  beforeDraw(chart) {{
    const {{ctx, scales: {{x, y}}}} = chart;
    if (!x || !y) return;
    const bands = [
      [ZONE_12,  ZONE_45,  'rgba(96,165,250,0.06)'],
      [ZONE_45,  ZONE_15,  'rgba(192,132,252,0.06)'],
      [ZONE_15,  ZONE_END, 'rgba(251,146,60,0.06)'],
    ];
    ctx.save();
    for (const [s, e, c] of bands) {{
      const x0 = x.getPixelForValue(new Date(s).getTime());
      const x1 = x.getPixelForValue(new Date(e).getTime());
      ctx.fillStyle = c;
      ctx.fillRect(x0, y.top, x1 - x0, y.bottom - y.top);
    }}
    ctx.restore();
  }}
}};

function makeChart(id, label, unit, mainColor, data) {{
  const ctx = document.getElementById(id).getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    plugins: [bandPlugin],
    data: {{
      datasets: [
        {{
          label: 'Actual',
          data: data.actual,
          borderColor: '#e2e8f0',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          order: 1,
        }},
        {{
          label: 'Ensemble forecast',
          data: data.ensemble,
          borderColor: mainColor,
          borderWidth: 2.5,
          pointRadius: 0,
          tension: 0.3,
          order: 2,
        }},
        {{
          label: 'Model 12h  (h1–12)',
          data: data.m12,
          borderColor: '#60a5fa',
          borderWidth: 1.5,
          borderDash: [5, 4],
          pointRadius: 0,
          tension: 0.3,
          order: 3,
        }},
        {{
          label: 'Model 45d  (h13–24)',
          data: data.m45,
          borderColor: '#c084fc',
          borderWidth: 1.5,
          borderDash: [5, 4],
          pointRadius: 0,
          tension: 0.3,
          order: 4,
        }},
        {{
          label: 'Model 15d  (h25–168)',
          data: data.m15,
          borderColor: '#fb923c',
          borderWidth: 1.5,
          borderDash: [5, 4],
          pointRadius: 0,
          tension: 0.3,
          order: 5,
        }},
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{
          display: false,   // custom legend in HTML
        }},
        tooltip: {{
          backgroundColor: '#1a1f2e',
          borderColor: '#2d3748',
          borderWidth: 1,
          titleColor: '#94a3b8',
          bodyColor: '#e2e8f0',
          padding: 10,
          callbacks: {{
            title: items => new Date(items[0].parsed.x).toLocaleString('en-IE', {{
              month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            }}),
            label: item => ` ${{item.dataset.label}}: ${{item.parsed.y !== null ? item.parsed.y.toFixed(2) : '—'}} ${{unit}}`,
          }}
        }}
      }},
      scales: {{
        x: {{
          type: 'time',
          time: {{
            unit: 'day',
            displayFormats: {{ day: 'MMM d' }}
          }},
          grid: {{ color: '#1e2535' }},
          ticks: {{ color: '#475569', maxRotation: 0 }},
        }},
        y: {{
          grid: {{ color: '#1e2535' }},
          ticks: {{ color: '#475569', callback: v => v + ' ' + unit }},
        }}
      }}
    }}
  }});
}}

makeChart('ch_temp', 'Temperature', '°C',  '#f87171', {{
  actual:   {_ser_to_js(actual_temp)},
  ensemble: {_ser_to_js(ens_temp)},
  m12:      {_ser_to_js(p12_t)},
  m45:      {_ser_to_js(p45_t)},
  m15:      {_ser_to_js(p15_t)},
}});

makeChart('ch_pres', 'Pressure', 'hPa', '#60a5fa', {{
  actual:   {_ser_to_js(actual_pres)},
  ensemble: {_ser_to_js(ens_pres)},
  m12:      {_ser_to_js(p12_p)},
  m45:      {_ser_to_js(p45_p)},
  m15:      {_ser_to_js(p15_p)},
}});

makeChart('ch_wind', 'Wind Speed', 'kt', '#34d399', {{
  actual:   {_ser_to_js(actual_wind)},
  ensemble: {_ser_to_js(ens_wind)},
  m12:      {_ser_to_js(p12_w)},
  m45:      {_ser_to_js(p45_w)},
  m15:      {_ser_to_js(p15_w)},
}});
</script>
</body>
</html>"""


def cls_mae(mae: float, good: float, ok: float) -> str:
    if mae <= good: return "good"
    if mae <= ok:   return "ok"
    return "bad"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    Path("artifacts").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    segments = json.loads(SEGMENTS_PATH.read_text(encoding="utf-8"))
    sc = _global_scales(segments)

    df = pd.read_csv(CSV_PATH, skiprows=23)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.set_index("date").sort_index()

    last_known = df.loc[:"2026-01-31 23:00"].iloc[-1]
    temp_x0 = float(last_known["temp"])
    pres_x0 = float(last_known["msl"])
    wind_x0 = float(last_known["wdsp"])

    actual_temp = df["temp"].loc["2026-02-01":"2026-02-07 23:00"]
    actual_pres = df["msl"].loc["2026-02-01":"2026-02-07 23:00"]
    actual_wind = df["wdsp"].loc["2026-02-01":"2026-02-07 23:00"]

    preds = {}
    for name, (path, hw) in MODELS.items():
        print(f"Loading {name} model ({path.name}) …")
        model = load_model(path, hw, device)
        pt, pp, pw = run_forecast(model, segments, sc, hw, FEB1, device,
                                  N_HOURS, temp_x0, pres_x0, wind_x0)
        preds[name] = (pt, pp, pw)
        print(f"  {name}: {len(pt)} hours predicted")

    ens_temp, ens_pres, ens_wind = stitch(preds["12h"], preds["45d"], preds["15d"])

    # Overall metrics
    mae_t, rmse_t = mae_rmse(actual_temp, ens_temp)
    mae_p, rmse_p = mae_rmse(actual_pres, ens_pres)
    mae_w, rmse_w = mae_rmse(actual_wind, ens_wind)

    # Per-range metrics
    def range_mae(actual, ens, h_start, h_end):
        ts_range = [FEB1 + pd.Timedelta(hours=h) for h in range(h_start, h_end)]
        idx = pd.DatetimeIndex(ts_range)
        return mae_rmse(actual.reindex(idx), ens.reindex(idx))

    mae_t1, _ = range_mae(actual_temp, ens_temp, 0, 12)
    mae_p1, _ = range_mae(actual_pres, ens_pres, 0, 12)
    mae_w1, _ = range_mae(actual_wind, ens_wind, 0, 12)

    mae_t2, _ = range_mae(actual_temp, ens_temp, 12, 24)
    mae_p2, _ = range_mae(actual_pres, ens_pres, 12, 24)
    mae_w2, _ = range_mae(actual_wind, ens_wind, 12, 24)

    mae_t3, _ = range_mae(actual_temp, ens_temp, 24, 168)
    mae_p3, _ = range_mae(actual_pres, ens_pres, 24, 168)
    mae_w3, _ = range_mae(actual_wind, ens_wind, 24, 168)

    print(f"\nOverall MAE: temp={mae_t:.2f}°C  pres={mae_p:.2f} hPa  wind={mae_w:.2f} kt")
    print(f"h1–12:    temp={mae_t1:.2f}  pres={mae_p1:.2f}  wind={mae_w1:.2f}")
    print(f"h13–24:   temp={mae_t2:.2f}  pres={mae_p2:.2f}  wind={mae_w2:.2f}")
    print(f"h25–168:  temp={mae_t3:.2f}  pres={mae_p3:.2f}  wind={mae_w3:.2f}")

    html = build_html(
        actual_temp, actual_pres, actual_wind,
        ens_temp, ens_pres, ens_wind,
        preds["12h"], preds["45d"], preds["15d"],
        mae_t, rmse_t, mae_p, rmse_p, mae_w, rmse_w,
        mae_t1, mae_p1, mae_w1,
        mae_t2, mae_p2, mae_w2,
        mae_t3, mae_p3, mae_w3,
    )

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport saved → {OUT_HTML}")


if __name__ == "__main__":
    main()
