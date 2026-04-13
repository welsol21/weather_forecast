from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from weather_patterns.config import PipelineConfig
from weather_patterns.data.loading import apply_quality_masks, load_weather_dataset
from weather_patterns.discovery.base import DiscoveryInput
from weather_patterns.discovery.kmeans import NumpyKMeansDiscovery
from weather_patterns.discovery.structural import StructuralPatternDiscovery
from weather_patterns.events.extrema import detect_extrema
from weather_patterns.events.peaks import detect_peaks
from weather_patterns.forecasting.dataset import load_forecast_samples_jsonl
from weather_patterns.forecasting.samples import build_forecast_samples
from weather_patterns.io.artifacts import (
    iter_jsonl,
    read_prepared_pattern_windows_jsonl,
    read_pattern_prototypes_jsonl,
    write_forecast_sequence_dataset_jsonl,
    write_prepared_pattern_windows_jsonl,
    write_pattern_flow_jsonl,
    write_pattern_prototypes_jsonl,
)
from weather_patterns.models import (
    DiscoveryArtifacts,
    DiscoveryResult,
    ExtremaEvent,
    ExtremaWindow,
    PatternWindow,
    PeakEvent,
    PipelineArtifacts,
    PatternPrototype,
    PreparedPatternWindowsArtifacts,
    TimePlaceholders,
)
from weather_patterns.pattern.representation import (
    build_convergence_pattern_window,
    build_pattern_window,
    build_signal_channel_arrays,
    compute_channel_thresholds,
    slice_signal_channel_arrays,
)
from weather_patterns.pattern.windows import build_extrema_windows, build_hierarchical_windows, build_predictor_windows
from weather_patterns.signal.processing import build_signal_frame


def prepare_pattern_windows(
    csv_path: str | Path,
    config: PipelineConfig | None = None,
) -> PreparedPatternWindowsArtifacts:
    active_config = config or PipelineConfig()
    dataset = load_weather_dataset(csv_path, active_config.dataset)
    cleaned_frame = apply_quality_masks(dataset)
    if active_config.date_start is not None:
        cleaned_frame = cleaned_frame[cleaned_frame[active_config.dataset.datetime_column] >= pd.Timestamp(active_config.date_start)]
    if active_config.date_end is not None:
        cleaned_frame = cleaned_frame[cleaned_frame[active_config.dataset.datetime_column] <= pd.Timestamp(active_config.date_end)]
    cleaned_frame = cleaned_frame.reset_index(drop=True)
    if active_config.max_rows is not None:
        cleaned_frame = cleaned_frame.iloc[: active_config.max_rows].copy()
        dataset.dataframe = cleaned_frame
    else:
        dataset.dataframe = cleaned_frame

    signal_frame = build_signal_frame(
        dataset.dataframe,
        dataset.channel_columns,
        active_config.smoothing,
    )
    extrema_events = detect_extrema(signal_frame, dataset.channel_columns)
    peak_events = detect_peaks(signal_frame, dataset.channel_columns)
    window_to_block: dict[int, int] = {}
    if active_config.window.segmentation_strategy == "predictor":
        extrema_windows = build_predictor_windows(
            signal_frame,
            dataset.channel_columns,
            extrema_events,
            peak_events,
            active_config.window,
        )
    elif active_config.window.segmentation_strategy in ("hierarchical", "new_physics"):
        extrema_windows, window_to_block = build_hierarchical_windows(
            signal_frame,
            dataset.channel_columns,
            extrema_events,
            peak_events,
            active_config.window,
        )
    else:
        extrema_windows = build_extrema_windows(
            signal_frame,
            extrema_events,
            peak_events,
            active_config.window,
        )
    raw_series, smoothed_series, diff1_series, diff2_series = build_signal_channel_arrays(
        signal_frame,
        dataset.channel_columns,
    )
    if active_config.window.segmentation_strategy == "new_physics":
        pattern_windows = [
            build_convergence_pattern_window(
                extrema_window,
                dataset.channel_columns,
                active_config,
                *slice_signal_channel_arrays(
                    raw_series,
                    smoothed_series,
                    diff1_series,
                    diff2_series,
                    dataset.channel_columns,
                    extrema_window.start_index,
                    extrema_window.end_index,
                )[:2],
            )
            for extrema_window in extrema_windows
        ]
    else:
        upper_thresholds, lower_thresholds = compute_channel_thresholds(
            dataset.dataframe,
            dataset.channel_columns,
            active_config.hazard,
        )
        pattern_windows = [
            build_pattern_window(
                signal_frame,
                extrema_window,
                dataset.channel_columns,
                active_config,
                upper_thresholds,
                lower_thresholds,
                *slice_signal_channel_arrays(
                    raw_series,
                    smoothed_series,
                    diff1_series,
                    diff2_series,
                    dataset.channel_columns,
                    extrema_window.start_index,
                    extrema_window.end_index,
                ),
            )
            for extrema_window in extrema_windows
        ]
    if window_to_block:
        for pw in pattern_windows:
            pw.parent_block_id = window_to_block.get(pw.window_id)
    return PreparedPatternWindowsArtifacts(
        dataset=dataset,
        signal_frame=signal_frame,
        extrema_events=extrema_events,
        peak_events=peak_events,
        extrema_windows=extrema_windows,
        pattern_windows=pattern_windows,
    )


def discover_patterns(
    pattern_windows: list[PatternWindow],
    config: PipelineConfig | None = None,
) -> DiscoveryArtifacts:
    active_config = config or PipelineConfig()
    feature_matrix = (
        np.vstack([window.feature_vector for window in pattern_windows])
        if pattern_windows
        else np.empty((0, 0))
    )
    if active_config.discovery.strategy == "structural":
        discovery_strategy = StructuralPatternDiscovery(active_config.discovery)
    elif active_config.discovery.strategy == "kmeans":
        discovery_strategy = NumpyKMeansDiscovery(active_config.discovery)
    else:
        raise ValueError(f"Unsupported discovery strategy: {active_config.discovery.strategy}")

    discovery = discovery_strategy.fit_predict(
        DiscoveryInput(
            window_ids=[window.window_id for window in pattern_windows],
            feature_matrix=feature_matrix,
            pattern_windows=pattern_windows,
        )
    )
    forecast_samples = build_forecast_samples(
        pattern_windows,
        discovery.labels_by_window_id,
        active_config.window,
        active_config.forecast,
    )
    return DiscoveryArtifacts(
        pattern_windows=pattern_windows,
        discovery_result=discovery,
        forecast_samples=forecast_samples,
    )


def run_pipeline(csv_path: str | Path, config: PipelineConfig | None = None) -> PipelineArtifacts:
    prepared = prepare_pattern_windows(csv_path, config)
    active_config = config or PipelineConfig()
    discovery = discover_patterns(prepared.pattern_windows, active_config)
    return PipelineArtifacts(
        dataset=prepared.dataset,
        signal_frame=prepared.signal_frame,
        extrema_events=prepared.extrema_events,
        peak_events=prepared.peak_events,
        extrema_windows=prepared.extrema_windows,
        pattern_windows=prepared.pattern_windows,
        discovery_result=discovery.discovery_result,
        forecast_samples=discovery.forecast_samples,
    )


def write_artifacts_summary(artifacts: PipelineArtifacts, output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(artifacts.summary(), indent=2),
        encoding="utf-8",
    )
    return summary_path


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _extrema_events_frame(extrema_events: list[ExtremaEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": event.timestamp,
                "channel": event.channel,
                "sign": event.sign,
                "amplitude": event.amplitude,
                "value": event.value,
                "first_diff_value": event.first_diff_value,
                "second_diff_value": event.second_diff_value,
                "index": event.index,
            }
            for event in extrema_events
        ]
    )


def _peak_events_frame(peak_events: list[PeakEvent]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": peak.timestamp,
                "channel": peak.channel,
                "sign": peak.sign,
                "peak_value": peak.peak_value,
                "prominence": peak.prominence,
                "width_steps": peak.width_steps,
                "rise_slope": peak.rise_slope,
                "fall_slope": peak.fall_slope,
                "asymmetry": peak.asymmetry,
                "left_base_value": peak.left_base_value,
                "right_base_value": peak.right_base_value,
                "index": peak.index,
                "left_index": peak.left_index,
                "right_index": peak.right_index,
            }
            for peak in peak_events
        ]
    )


def _pattern_windows_frame(artifacts: PipelineArtifacts) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in artifacts.pattern_windows:
        row: dict[str, object] = {
            "window_id": window.window_id,
            "start_time": window.start_time,
            "end_time": window.end_time,
            "start_index": window.extrema_window.start_index,
            "end_index": window.extrema_window.end_index,
            "event_count": len(window.extrema_window.events),
            "peak_count": len(window.extrema_window.peaks),
            "feature_vector_length": int(len(window.feature_vector)),
            "discovered_pattern_id": artifacts.discovery_result.labels_by_window_id.get(window.window_id),
            "time_step_hours": window.time_placeholders.time_step_hours,
            "window_length_steps": window.time_placeholders.window_length_steps,
            "forecast_horizon_steps": window.time_placeholders.forecast_horizon_steps,
            "mean_inter_event_gap_steps": window.time_placeholders.mean_inter_event_gap_steps,
            "var_inter_event_gap_steps": window.time_placeholders.var_inter_event_gap_steps,
            "mean_peak_width_steps": window.time_placeholders.mean_peak_width_steps,
            "max_peak_width_steps": window.time_placeholders.max_peak_width_steps,
        }
        for channel in window.channels:
            row[f"{channel}__duration_over_threshold"] = window.time_placeholders.duration_over_threshold_by_channel.get(
                channel,
                0.0,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _forecast_samples_frame(artifacts: PipelineArtifacts) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_window_id": sample.source_window_id,
                "history_window_ids": ",".join(str(value) for value in sample.history_window_ids),
                "target_window_ids": ",".join(str(value) for value in sample.target_window_ids),
                "history_pattern_ids": ",".join(
                    "" if value is None else str(value) for value in sample.history_pattern_ids
                ),
                "target_pattern_ids": ",".join(
                    "" if value is None else str(value) for value in sample.target_pattern_ids
                ),
                "forecast_horizon_steps": sample.forecast_horizon_steps,
                "forecast_window_count": sample.forecast_window_count,
                "history_window_count": len(sample.history_window_ids),
                "history_vector_length": int(len(sample.history_vector)),
                "feature_dim": int(sample.target_pattern_matrix.shape[1]),
            }
            for sample in artifacts.forecast_samples
        ]
    )


def _prepared_pattern_window_record(window: PatternWindow) -> dict[str, object]:
    return {
        "window_id": window.window_id,
        "start_time": window.start_time.isoformat(),
        "end_time": window.end_time.isoformat(),
        "channels": list(window.channels),
        "parent_block_id": window.parent_block_id,
        "channel_stds": {k: float(v) for k, v in window.channel_stds.items()},
        "channel_end_values": {k: float(v) for k, v in window.channel_end_values.items()},
        "intra_matrix": window.intra_matrix.astype(float).tolist(),
        "inter_matrix": window.inter_matrix.astype(float).tolist(),
        "peak_hazard_matrix": window.peak_hazard_matrix.astype(float).tolist(),
        "feature_vector": window.feature_vector.astype(float).tolist(),
        "time_placeholders": {
            "time_step_hours": float(window.time_placeholders.time_step_hours),
            "window_length_steps": int(window.time_placeholders.window_length_steps),
            "forecast_horizon_steps": int(window.time_placeholders.forecast_horizon_steps),
            "mean_inter_event_gap_steps": float(window.time_placeholders.mean_inter_event_gap_steps),
            "var_inter_event_gap_steps": float(window.time_placeholders.var_inter_event_gap_steps),
            "mean_peak_width_steps": float(window.time_placeholders.mean_peak_width_steps),
            "max_peak_width_steps": float(window.time_placeholders.max_peak_width_steps),
            "duration_over_threshold_by_channel": {
                channel: float(value)
                for channel, value in window.time_placeholders.duration_over_threshold_by_channel.items()
            },
            "normalized_event_positions": [
                float(value) for value in window.time_placeholders.normalized_event_positions
            ],
        },
        "extrema_window": {
            "window_id": int(window.extrema_window.window_id),
            "start_time": window.extrema_window.start_time.isoformat(),
            "end_time": window.extrema_window.end_time.isoformat(),
            "start_index": int(window.extrema_window.start_index),
            "end_index": int(window.extrema_window.end_index),
            "events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "channel": event.channel,
                    "sign": event.sign,
                    "amplitude": float(event.amplitude),
                    "value": float(event.value),
                    "first_diff_value": float(event.first_diff_value),
                    "second_diff_value": float(event.second_diff_value),
                    "index": int(event.index),
                }
                for event in window.extrema_window.events
            ],
            "peaks": [
                {
                    "timestamp": peak.timestamp.isoformat(),
                    "channel": peak.channel,
                    "sign": peak.sign,
                    "peak_value": float(peak.peak_value),
                    "prominence": float(peak.prominence),
                    "width_steps": float(peak.width_steps),
                    "rise_slope": float(peak.rise_slope),
                    "fall_slope": float(peak.fall_slope),
                    "asymmetry": float(peak.asymmetry),
                    "left_base_value": float(peak.left_base_value),
                    "right_base_value": float(peak.right_base_value),
                    "index": int(peak.index),
                    "left_index": int(peak.left_index),
                    "right_index": int(peak.right_index),
                }
                for peak in window.extrema_window.peaks
            ],
        },
    }


def _parse_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(str(value))


def _prepared_pattern_window_from_record(record: dict[str, object]) -> PatternWindow:
    extrema_payload = dict(record["extrema_window"])
    time_payload = dict(record["time_placeholders"])
    extrema_window = ExtremaWindow(
        window_id=int(extrema_payload["window_id"]),
        start_time=_parse_timestamp(extrema_payload["start_time"]),
        end_time=_parse_timestamp(extrema_payload["end_time"]),
        start_index=int(extrema_payload["start_index"]),
        end_index=int(extrema_payload["end_index"]),
        events=[
            ExtremaEvent(
                timestamp=_parse_timestamp(event["timestamp"]),
                channel=str(event["channel"]),
                sign=str(event["sign"]),
                amplitude=float(event["amplitude"]),
                value=float(event["value"]),
                first_diff_value=float(event["first_diff_value"]),
                second_diff_value=float(event["second_diff_value"]),
                index=int(event["index"]),
            )
            for event in extrema_payload["events"]
        ],
        peaks=[
            PeakEvent(
                timestamp=_parse_timestamp(peak["timestamp"]),
                channel=str(peak["channel"]),
                sign=str(peak["sign"]),
                peak_value=float(peak["peak_value"]),
                prominence=float(peak["prominence"]),
                width_steps=float(peak["width_steps"]),
                rise_slope=float(peak["rise_slope"]),
                fall_slope=float(peak["fall_slope"]),
                asymmetry=float(peak["asymmetry"]),
                left_base_value=float(peak["left_base_value"]),
                right_base_value=float(peak["right_base_value"]),
                index=int(peak["index"]),
                left_index=int(peak["left_index"]),
                right_index=int(peak["right_index"]),
            )
            for peak in extrema_payload["peaks"]
        ],
    )
    raw_block_id = record.get("parent_block_id")
    return PatternWindow(
        window_id=int(record["window_id"]),
        start_time=_parse_timestamp(record["start_time"]),
        end_time=_parse_timestamp(record["end_time"]),
        channels=[str(channel) for channel in record["channels"]],
        intra_matrix=np.asarray(record["intra_matrix"], dtype=float),
        inter_matrix=np.asarray(record["inter_matrix"], dtype=float),
        peak_hazard_matrix=np.asarray(record["peak_hazard_matrix"], dtype=float),
        time_placeholders=TimePlaceholders(
            time_step_hours=float(time_payload["time_step_hours"]),
            window_length_steps=int(time_payload["window_length_steps"]),
            forecast_horizon_steps=int(time_payload["forecast_horizon_steps"]),
            mean_inter_event_gap_steps=float(time_payload["mean_inter_event_gap_steps"]),
            var_inter_event_gap_steps=float(time_payload["var_inter_event_gap_steps"]),
            mean_peak_width_steps=float(time_payload["mean_peak_width_steps"]),
            max_peak_width_steps=float(time_payload["max_peak_width_steps"]),
            duration_over_threshold_by_channel={
                str(channel): float(value)
                for channel, value in dict(time_payload["duration_over_threshold_by_channel"]).items()
            },
            normalized_event_positions=[
                float(value) for value in time_payload["normalized_event_positions"]
            ],
        ),
        feature_vector=np.asarray(record["feature_vector"], dtype=float),
        extrema_window=extrema_window,
        parent_block_id=int(raw_block_id) if raw_block_id is not None else None,
        channel_stds={str(k): float(v) for k, v in dict(record.get("channel_stds", {})).items()},
        channel_end_values={str(k): float(v) for k, v in dict(record.get("channel_end_values", {})).items()},
    )


def _pattern_prototypes_records(artifacts: PipelineArtifacts) -> list[dict[str, object]]:
    return [
        {
            "pattern_id": prototype.pattern_id,
            "centroid": prototype.centroid.astype(float).tolist(),
            "member_window_ids": [int(window_id) for window_id in prototype.member_window_ids],
            "member_count": len(prototype.member_window_ids),
            "metadata": prototype.metadata,
        }
        for prototype in artifacts.discovery_result.prototypes
    ]


def _pattern_flow_records(artifacts: PipelineArtifacts) -> list[dict[str, object]]:
    return [
        {
            "window_id": window.window_id,
            "start_time": window.start_time.isoformat(),
            "end_time": window.end_time.isoformat(),
            "start_index": window.extrema_window.start_index,
            "end_index": window.extrema_window.end_index,
            "pattern_id": artifacts.discovery_result.labels_by_window_id.get(window.window_id),
        }
        for window in artifacts.pattern_windows
    ]


def _forecast_sequence_dataset_records(artifacts: PipelineArtifacts) -> list[dict[str, object]]:
    return [
        {
            "source_window_id": sample.source_window_id,
            "history_window_ids": [int(value) for value in sample.history_window_ids],
            "target_window_ids": [int(value) for value in sample.target_window_ids],
            "history_pattern_ids": sample.history_pattern_ids,
            "target_pattern_ids": sample.target_pattern_ids,
            "forecast_horizon_steps": sample.forecast_horizon_steps,
            "forecast_window_count": sample.forecast_window_count,
            "history_window_count": len(sample.history_window_ids),
            "history_vector": sample.history_vector.astype(float).tolist(),
            "history_pattern_matrix": sample.history_pattern_matrix.astype(float).tolist(),
            "target_pattern_matrix": sample.target_pattern_matrix.astype(float).tolist(),
        }
        for sample in artifacts.forecast_samples
    ]


def _prepared_pattern_window_records(
    windows: list[PatternWindow],
):
    for window in windows:
        yield _prepared_pattern_window_record(window)


def _discovery_pattern_prototype_records(
    artifacts: DiscoveryArtifacts,
):
    for prototype in artifacts.discovery_result.prototypes:
        yield {
            "pattern_id": prototype.pattern_id,
            "centroid": prototype.centroid.astype(float).tolist(),
            "member_window_ids": [int(window_id) for window_id in prototype.member_window_ids],
            "member_count": len(prototype.member_window_ids),
            "metadata": prototype.metadata,
        }


def _discovery_pattern_flow_records(
    artifacts: DiscoveryArtifacts,
):
    for window in artifacts.pattern_windows:
        yield {
            "window_id": window.window_id,
            "start_time": window.start_time.isoformat(),
            "end_time": window.end_time.isoformat(),
            "start_index": window.extrema_window.start_index,
            "end_index": window.extrema_window.end_index,
            "pattern_id": artifacts.discovery_result.labels_by_window_id.get(window.window_id),
        }


def _discovery_forecast_sequence_dataset_records(
    artifacts: DiscoveryArtifacts,
):
    for sample in artifacts.forecast_samples:
        yield {
            "source_window_id": sample.source_window_id,
            "history_window_ids": [int(value) for value in sample.history_window_ids],
            "target_window_ids": [int(value) for value in sample.target_window_ids],
            "history_pattern_ids": sample.history_pattern_ids,
            "target_pattern_ids": sample.target_pattern_ids,
            "forecast_horizon_steps": sample.forecast_horizon_steps,
            "forecast_window_count": sample.forecast_window_count,
            "history_window_count": len(sample.history_window_ids),
            "history_vector": sample.history_vector.astype(float).tolist(),
            "history_pattern_matrix": sample.history_pattern_matrix.astype(float).tolist(),
            "target_pattern_matrix": sample.target_pattern_matrix.astype(float).tolist(),
        }


def filter_windows_for_hierarchical(
    existing_pattern_windows: list[PatternWindow],
    signal_frame: pd.DataFrame,
    channels: list[str],
    config: PipelineConfig,
) -> list[PatternWindow]:
    """Filter already-built PatternWindows to those fully within a predictor regime block.

    This is the fast reuse path for the hierarchical strategy: instead of rebuilding
    all ~54 000 PatternWindow objects from scratch, we load existing ones (from a
    previous extrema run), run the predictor segmentation on the cached signal_frame,
    and keep only the windows whose start_index falls inside a regime block.

    Sets parent_block_id on each kept window in place.
    """
    regime_blocks = build_predictor_windows(
        signal_frame,
        channels,
        [],
        [],
        config.window,
    )
    length = config.window.length_steps
    stride = config.window.stride_steps
    valid_starts: dict[int, int] = {}
    for block in regime_blocks:
        block_len = block.end_index - block.start_index + 1
        if block_len < length:
            continue
        for offset in range(0, block_len - length + 1, stride):
            start_idx = block.start_index + offset
            valid_starts[start_idx] = block.window_id

    result: list[PatternWindow] = []
    for pw in existing_pattern_windows:
        block_id = valid_starts.get(pw.extrema_window.start_index)
        if block_id is not None:
            pw.parent_block_id = block_id
            result.append(pw)
    return result


def prepare_hierarchical_from_existing(
    source_prepare_dir: Path | str,
    csv_path: str | Path,
    config: PipelineConfig,
) -> list[PatternWindow]:
    """Reuse prepare artifacts from a previous extrema run for the hierarchical strategy.

    Loads signal_frame.csv and prepared_pattern_windows.jsonl.gz from
    *source_prepare_dir* (e.g. an earlier extrema run), runs predictor
    segmentation on the cached signal_frame, and returns the subset of
    PatternWindows that fall within predictor regime blocks — with
    parent_block_id already stamped on each.

    This avoids re-building ~54 000 PatternWindow objects (~15 min of CPU work).
    The signal_frame.csv already contains the smoothed_* columns needed by the
    predictor; the CSV dataset is only loaded to discover the channel list.
    """
    source = Path(source_prepare_dir)
    signal_frame = pd.read_csv(source / "signal_frame.csv")
    dataset = load_weather_dataset(csv_path, config.dataset)
    channels = dataset.channel_columns
    existing_windows = load_prepared_pattern_windows(source / "prepared_pattern_windows.jsonl.gz")
    return filter_windows_for_hierarchical(existing_windows, signal_frame, channels, config)


def write_hierarchical_prepare_artifacts(
    pattern_windows: list[PatternWindow],
    source_prepare_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, str]:
    """Write prepare artifacts for the hierarchical strategy.

    Copies signal_frame.csv, extrema_events.csv and peak_events.csv unchanged
    from *source_prepare_dir* (they are identical to any other run over the same
    CSV and date range), then serialises only the filtered hierarchical windows.
    """
    import shutil
    source = Path(source_prepare_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("signal_frame.csv", "extrema_events.csv", "peak_events.csv"):
        shutil.copy2(source / filename, destination / filename)
    pattern_windows_path = destination / "prepared_pattern_windows.jsonl.gz"
    write_prepared_pattern_windows_jsonl(
        _prepared_pattern_window_records(pattern_windows),
        pattern_windows_path,
    )
    summary = {
        "hierarchical_windows": len(pattern_windows),
        "source": str(source),
        "segmentation_strategy": "hierarchical",
    }
    summary_path = destination / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "signal_frame_path": str(destination / "signal_frame.csv"),
        "extrema_events_path": str(destination / "extrema_events.csv"),
        "peak_events_path": str(destination / "peak_events.csv"),
        "prepared_pattern_windows_path": str(pattern_windows_path),
    }


def write_prepared_artifacts(
    artifacts: PreparedPatternWindowsArtifacts,
    output_dir: str | Path,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_path = destination / "prepare_summary.json"
    summary_path.write_text(
        json.dumps(artifacts.summary(), indent=2),
        encoding="utf-8",
    )
    signal_path = destination / "signal_frame.csv"
    extrema_events_path = destination / "extrema_events.csv"
    peak_events_path = destination / "peak_events.csv"
    pattern_windows_path = destination / "prepared_pattern_windows.jsonl.gz"

    _write_frame(artifacts.signal_frame, signal_path)
    _write_frame(_extrema_events_frame(artifacts.extrema_events), extrema_events_path)
    _write_frame(_peak_events_frame(artifacts.peak_events), peak_events_path)
    write_prepared_pattern_windows_jsonl(
        _prepared_pattern_window_records(artifacts.pattern_windows),
        pattern_windows_path,
    )

    return {
        "summary_path": str(summary_path),
        "signal_frame_path": str(signal_path),
        "extrema_events_path": str(extrema_events_path),
        "peak_events_path": str(peak_events_path),
        "prepared_pattern_windows_path": str(pattern_windows_path),
    }


def load_prepared_pattern_windows(path: str | Path) -> list[PatternWindow]:
    records = read_prepared_pattern_windows_jsonl(path)
    return [_prepared_pattern_window_from_record(record) for record in records]


def load_pattern_window_end_times(path: str | Path) -> dict[int, pd.Timestamp]:
    return {
        int(record["window_id"]): _parse_timestamp(record["end_time"])
        for record in iter_jsonl(path)
    }


def load_pattern_window_new_physics_context(
    path: str | Path,
) -> dict[int, tuple[dict[str, float], dict[str, float]]]:
    """Load (channel_stds, channel_end_values) per window_id for new_physics evaluation.

    Returns an empty dict if the file contains no new_physics windows (legacy format).
    """
    result: dict[int, tuple[dict[str, float], dict[str, float]]] = {}
    for record in iter_jsonl(path):
        ch_stds = {str(k): float(v) for k, v in dict(record.get("channel_stds", {})).items()}
        if ch_stds:
            ch_end_vals = {str(k): float(v) for k, v in dict(record.get("channel_end_values", {})).items()}
            result[int(record["window_id"])] = (ch_stds, ch_end_vals)
    return result


def load_pattern_prototypes(path: str | Path) -> list[PatternPrototype]:
    records = read_pattern_prototypes_jsonl(path)
    prototypes: list[PatternPrototype] = []
    for record in records:
        prototypes.append(
            PatternPrototype(
                pattern_id=int(record["pattern_id"]),
                centroid=np.asarray(record["centroid"], dtype=float),
                member_window_ids=[int(window_id) for window_id in record["member_window_ids"]],
                metadata=dict(record.get("metadata", {})),
            )
        )
    return prototypes


def load_saved_pipeline_artifacts(
    csv_path: str | Path,
    prepared_pattern_windows_path: str | Path,
    pattern_prototypes_path: str | Path,
    config: PipelineConfig | None = None,
    sequence_dataset_path: str | Path | None = None,
    load_pattern_windows: bool = True,
    load_forecast_samples: bool = True,
) -> PipelineArtifacts:
    active_config = config or PipelineConfig()
    dataset = load_weather_dataset(csv_path, active_config.dataset)
    cleaned_frame = apply_quality_masks(dataset)
    if active_config.date_start is not None:
        cleaned_frame = cleaned_frame[cleaned_frame[active_config.dataset.datetime_column] >= pd.Timestamp(active_config.date_start)]
    if active_config.date_end is not None:
        cleaned_frame = cleaned_frame[cleaned_frame[active_config.dataset.datetime_column] <= pd.Timestamp(active_config.date_end)]
    cleaned_frame = cleaned_frame.reset_index(drop=True)
    dataset.dataframe = cleaned_frame
    if active_config.max_rows is not None:
        dataset.dataframe = dataset.dataframe.iloc[: active_config.max_rows].copy()

    pattern_windows = (
        load_prepared_pattern_windows(prepared_pattern_windows_path)
        if load_pattern_windows
        else []
    )
    prototypes = load_pattern_prototypes(pattern_prototypes_path)
    labels_by_window_id: dict[int, int] = {}
    for prototype in prototypes:
        for window_id in prototype.member_window_ids:
            labels_by_window_id[int(window_id)] = int(prototype.pattern_id)

    if load_forecast_samples:
        forecast_samples = (
            load_forecast_samples_jsonl(str(sequence_dataset_path))
            if sequence_dataset_path is not None
            else build_forecast_samples(
                pattern_windows,
                labels_by_window_id,
                active_config.window,
                active_config.forecast,
            )
        )
    else:
        forecast_samples = []

    return PipelineArtifacts(
        dataset=dataset,
        signal_frame=pd.DataFrame(),
        extrema_events=[],
        peak_events=[],
        extrema_windows=[window.extrema_window for window in pattern_windows],
        pattern_windows=pattern_windows,
        discovery_result=DiscoveryResult(
            labels_by_window_id=labels_by_window_id,
            prototypes=prototypes,
            strategy="saved_artifacts",
            selected_cluster_count=len(prototypes),
        ),
        forecast_samples=forecast_samples,
    )


def write_discovery_artifacts(
    artifacts: DiscoveryArtifacts,
    output_dir: str | Path,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "discovery_summary.json"
    summary_path.write_text(
        json.dumps(artifacts.summary(), indent=2),
        encoding="utf-8",
    )
    pattern_prototypes_path = destination / "pattern_prototypes.jsonl"
    pattern_flow_path = destination / "pattern_flow.jsonl"
    forecast_sequence_dataset_path = destination / "forecast_sequence_dataset.jsonl.gz"

    write_pattern_prototypes_jsonl(
        _discovery_pattern_prototype_records(artifacts),
        pattern_prototypes_path,
    )
    write_pattern_flow_jsonl(
        _discovery_pattern_flow_records(artifacts),
        pattern_flow_path,
    )
    write_forecast_sequence_dataset_jsonl(
        _discovery_forecast_sequence_dataset_records(artifacts),
        forecast_sequence_dataset_path,
    )
    return {
        "summary_path": str(summary_path),
        "pattern_prototypes_path": str(pattern_prototypes_path),
        "pattern_flow_path": str(pattern_flow_path),
        "forecast_sequence_dataset_path": str(forecast_sequence_dataset_path),
    }


def write_pipeline_artifacts(artifacts: PipelineArtifacts, output_dir: str | Path) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_path = write_artifacts_summary(artifacts, destination)
    signal_path = destination / "signal_frame.csv"
    extrema_events_path = destination / "extrema_events.csv"
    peak_events_path = destination / "peak_events.csv"
    pattern_windows_path = destination / "pattern_windows.csv"
    forecast_samples_path = destination / "forecast_samples.csv"
    pattern_prototypes_path = destination / "pattern_prototypes.jsonl"
    pattern_flow_path = destination / "pattern_flow.jsonl"
    forecast_sequence_dataset_path = destination / "forecast_sequence_dataset.jsonl.gz"

    _write_frame(artifacts.signal_frame, signal_path)
    _write_frame(_extrema_events_frame(artifacts.extrema_events), extrema_events_path)
    _write_frame(_peak_events_frame(artifacts.peak_events), peak_events_path)
    _write_frame(_pattern_windows_frame(artifacts), pattern_windows_path)
    _write_frame(_forecast_samples_frame(artifacts), forecast_samples_path)
    write_pattern_prototypes_jsonl(_pattern_prototypes_records(artifacts), pattern_prototypes_path)
    write_pattern_flow_jsonl(_pattern_flow_records(artifacts), pattern_flow_path)
    write_forecast_sequence_dataset_jsonl(
        _forecast_sequence_dataset_records(artifacts),
        forecast_sequence_dataset_path,
    )

    return {
        "summary_path": str(summary_path),
        "signal_frame_path": str(signal_path),
        "extrema_events_path": str(extrema_events_path),
        "peak_events_path": str(peak_events_path),
        "pattern_windows_path": str(pattern_windows_path),
        "forecast_samples_path": str(forecast_samples_path),
        "pattern_prototypes_path": str(pattern_prototypes_path),
        "pattern_flow_path": str(pattern_flow_path),
        "forecast_sequence_dataset_path": str(forecast_sequence_dataset_path),
    }
