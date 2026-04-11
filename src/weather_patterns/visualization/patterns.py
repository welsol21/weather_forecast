from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from weather_patterns.config import DatasetConfig
from weather_patterns.data.loading import apply_quality_masks, load_weather_dataset
from weather_patterns.io.artifacts import read_pattern_flow_jsonl, read_pattern_prototypes_jsonl


def _load_matplotlib() -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    return plt, np


def _pattern_flow_frame(artifacts_dir: Path) -> pd.DataFrame:
    records = read_pattern_flow_jsonl(artifacts_dir / "pattern_flow.jsonl")
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    return frame.sort_values("start_time").reset_index(drop=True)


def _pattern_prototypes_frame(artifacts_dir: Path) -> pd.DataFrame:
    records = read_pattern_prototypes_jsonl(artifacts_dir / "pattern_prototypes.jsonl")
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values("pattern_id").reset_index(drop=True)


def _render_pattern_flow_timeline(flow_frame: pd.DataFrame, output_path: Path) -> None:
    plt, _ = _load_matplotlib()
    figure, axis = plt.subplots(figsize=(14, 4))
    axis.scatter(
        flow_frame["start_time"],
        flow_frame["pattern_id"],
        c=flow_frame["pattern_id"],
        cmap="tab20",
        s=16,
    )
    axis.set_title("Pattern Flow Timeline")
    axis.set_xlabel("Time")
    axis.set_ylabel("Pattern ID")
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _render_pattern_prototype_heatmap(prototypes_frame: pd.DataFrame, output_path: Path) -> None:
    plt, np = _load_matplotlib()
    matrix = np.asarray(prototypes_frame["centroid"].tolist(), dtype=float)
    figure, axis = plt.subplots(figsize=(14, max(4, len(prototypes_frame) * 0.6)))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_title("Pattern Prototype Centroids")
    axis.set_xlabel("Feature Index")
    axis.set_ylabel("Pattern ID")
    axis.set_yticks(range(len(prototypes_frame)))
    axis.set_yticklabels([str(value) for value in prototypes_frame["pattern_id"].tolist()])
    figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _render_weather_overlay(flow_frame: pd.DataFrame, csv_path: str | Path, output_path: Path) -> None:
    plt, _ = _load_matplotlib()
    dataset_config = DatasetConfig()
    dataset = load_weather_dataset(csv_path, dataset_config)
    cleaned_frame = apply_quality_masks(dataset).sort_values("date").reset_index(drop=True)
    channels = ["temperature", "relative_humidity", "pressure", "wind_speed", "rainfall"]

    figure, axes = plt.subplots(len(channels), 1, figsize=(14, 10), sharex=True)
    for axis, channel in zip(axes, channels):
        axis.plot(cleaned_frame["date"], cleaned_frame[channel], linewidth=0.9, color="black")
        axis.set_ylabel(channel)
        for row in flow_frame.itertuples(index=False):
            axis.axvspan(row.start_time, row.end_time, color=plt.cm.tab20(row.pattern_id % 20), alpha=0.12)
        axis.grid(alpha=0.2)
    axes[0].set_title("Weather Channels with Pattern Overlay")
    axes[-1].set_xlabel("Time")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def render_pattern_diagnostics(
    artifacts_dir: str | Path,
    output_dir: str | Path,
    csv_path: str | Path | None = None,
) -> dict[str, str]:
    artifacts_root = Path(artifacts_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    flow_frame = _pattern_flow_frame(artifacts_root)
    prototypes_frame = _pattern_prototypes_frame(artifacts_root)
    if flow_frame.empty:
        raise ValueError(f"No pattern flow records found in {artifacts_root / 'pattern_flow.jsonl'}.")
    if prototypes_frame.empty:
        raise ValueError(f"No pattern prototype records found in {artifacts_root / 'pattern_prototypes.jsonl'}.")

    pattern_flow_timeline_path = destination / "pattern_flow_timeline.png"
    pattern_prototypes_heatmap_path = destination / "pattern_prototypes_heatmap.png"
    _render_pattern_flow_timeline(flow_frame, pattern_flow_timeline_path)
    _render_pattern_prototype_heatmap(prototypes_frame, pattern_prototypes_heatmap_path)

    payload = {
        "pattern_flow_timeline_path": str(pattern_flow_timeline_path),
        "pattern_prototypes_heatmap_path": str(pattern_prototypes_heatmap_path),
    }
    if csv_path is not None:
        weather_overlay_path = destination / "weather_pattern_overlay.png"
        _render_weather_overlay(flow_frame, csv_path, weather_overlay_path)
        payload["weather_overlay_path"] = str(weather_overlay_path)
    return payload
