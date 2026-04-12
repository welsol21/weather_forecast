from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from weather_patterns.config import DatasetConfig
from weather_patterns.data.loading import apply_quality_masks, load_weather_dataset
from weather_patterns.io.artifacts import (
    read_forecast_sequence_dataset_jsonl,
    read_pattern_flow_jsonl,
    read_pattern_prototypes_jsonl,
    resolve_artifact_path,
)


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


def _sequence_dataset_frame(artifacts_dir: Path) -> pd.DataFrame:
    records = read_forecast_sequence_dataset_jsonl(artifacts_dir / "forecast_sequence_dataset.jsonl")
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values("source_window_id").reset_index(drop=True)


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
        values = pd.to_numeric(cleaned_frame[channel], errors="coerce")
        valid = values.notna()
        axis.plot(cleaned_frame.loc[valid, "date"], values.loc[valid], linewidth=0.9, color="black")
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


def _render_sequence_matrix(sequence_frame: pd.DataFrame, output_path: Path, max_rows: int = 32) -> None:
    plt, np = _load_matplotlib()
    preview = sequence_frame.head(max_rows).copy()
    if preview.empty:
        raise ValueError("No sequence dataset rows available for plotting.")

    matrix_rows: list[list[float]] = []
    history_length = 0
    target_length = 0
    for row in preview.itertuples(index=False):
        history_ids = [(-1 if value is None else int(value)) for value in row.history_pattern_ids]
        target_ids = [(-1 if value is None else int(value)) for value in row.target_pattern_ids]
        history_length = max(history_length, len(history_ids))
        target_length = max(target_length, len(target_ids))
        matrix_rows.append(history_ids + target_ids)

    matrix = np.asarray(matrix_rows, dtype=float)
    figure, axis = plt.subplots(figsize=(14, max(4, len(preview) * 0.3)))
    image = axis.imshow(matrix, aspect="auto", cmap="tab20", interpolation="nearest")
    axis.axvline(history_length - 0.5, color="white", linewidth=1.2, alpha=0.8)
    axis.set_title("Forecast Sequence Dataset: History -> Target Pattern IDs")
    axis.set_xlabel("Sequence Step")
    axis.set_ylabel("Sample Index")
    axis.set_xticks([0, max(history_length - 1, 0), history_length, history_length + max(target_length - 1, 0)])
    axis.set_xticklabels(
        [
            "history[0]",
            f"history[{max(history_length - 1, 0)}]",
            "target[0]",
            f"target[{max(target_length - 1, 0)}]",
        ]
    )
    figure.colorbar(image, ax=axis, shrink=0.85)
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
    sequence_frame = _sequence_dataset_frame(artifacts_root)
    if flow_frame.empty:
        raise ValueError(f"No pattern flow records found in {resolve_artifact_path(artifacts_root / 'pattern_flow.jsonl')}.")
    if prototypes_frame.empty:
        raise ValueError(
            f"No pattern prototype records found in {resolve_artifact_path(artifacts_root / 'pattern_prototypes.jsonl')}."
        )
    if sequence_frame.empty:
        raise ValueError(
            "No forecast sequence dataset records found in "
            f"{resolve_artifact_path(artifacts_root / 'forecast_sequence_dataset.jsonl')}."
        )

    pattern_flow_timeline_path = destination / "pattern_flow_timeline.png"
    pattern_prototypes_heatmap_path = destination / "pattern_prototypes_heatmap.png"
    sequence_matrix_path = destination / "forecast_sequence_matrix.png"
    _render_pattern_flow_timeline(flow_frame, pattern_flow_timeline_path)
    _render_pattern_prototype_heatmap(prototypes_frame, pattern_prototypes_heatmap_path)
    _render_sequence_matrix(sequence_frame, sequence_matrix_path)

    payload = {
        "pattern_flow_timeline_path": str(pattern_flow_timeline_path),
        "pattern_prototypes_heatmap_path": str(pattern_prototypes_heatmap_path),
        "forecast_sequence_matrix_path": str(sequence_matrix_path),
    }
    if csv_path is not None:
        weather_overlay_path = destination / "weather_pattern_overlay.png"
        _render_weather_overlay(flow_frame, csv_path, weather_overlay_path)
        payload["weather_overlay_path"] = str(weather_overlay_path)
    return payload
