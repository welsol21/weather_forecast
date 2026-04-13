from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from weather_patterns.config import PipelineConfig
from weather_patterns.forecasting.inference import (
    predict_future_pattern_sequence,
    summarize_forecast_result_compact,
    summarize_forecast_result,
    write_forecast_summary,
)
from weather_patterns.forecasting.evaluation import (
    evaluate_sequence_backtest,
    evaluate_sequence_backtest_from_saved_dataset,
    summarize_evaluation_payload,
    write_evaluation_summary,
)
from weather_patterns.forecasting.runtime import GpuRuntimeRequirementError
from weather_patterns.forecasting.training import (
    summarize_training_dataset,
    train_and_save_sequence_predictor,
    train_and_save_sequence_predictor_from_dataset,
    train_sequence_predictor,
    write_training_summary,
)
from weather_patterns.forecasting.torch_sequence import TorchSequencePredictor
from weather_patterns.io.artifacts import resolve_artifact_path
from weather_patterns.pipeline import (
    discover_patterns,
    load_prepared_pattern_windows,
    load_saved_pipeline_artifacts,
    prepare_hierarchical_from_existing,
    prepare_pattern_windows,
    run_pipeline,
    write_discovery_artifacts,
    write_hierarchical_prepare_artifacts,
    write_pipeline_artifacts,
    write_prepared_artifacts,
)


def _configure_workflow_logger(output_dir: Path) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_split_workflow.log"
    logger = logging.getLogger("weather_patterns.run_split_workflow")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def _configure_command_logger(output_dir: Path, logger_name: str, log_filename: str) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_filename
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def _log_stage_start(logger: logging.Logger, stage_name: str, **context: object) -> float:
    details = ", ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    message = f"stage_start stage={stage_name}"
    if details:
        message = f"{message} {details}"
    logger.info(message)
    return time.perf_counter()


def _log_stage_end(logger: logging.Logger, stage_name: str, started_at: float, **context: object) -> None:
    elapsed_seconds = time.perf_counter() - started_at
    details = ", ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    message = f"stage_end stage={stage_name} elapsed_seconds={elapsed_seconds:.3f}"
    if details:
        message = f"{message} {details}"
    logger.info(message)


def _build_model_stage_base_command() -> list[str]:
    return [sys.executable, "-m", "weather_patterns"]


def _extend_shared_model_runtime_args(command: list[str], args: argparse.Namespace) -> list[str]:
    if getattr(args, "model_device", None):
        command.extend(["--model-device", args.model_device])
    if bool(getattr(args, "allow_cpu_model", False)):
        command.append("--allow-cpu-model")
    return command


def _extend_shared_dataset_slice_args(command: list[str], args: argparse.Namespace) -> list[str]:
    if getattr(args, "date_start", None):
        command.extend(["--date-start", args.date_start])
    if getattr(args, "date_end", None):
        command.extend(["--date-end", args.date_end])
    return command


def _run_json_command(
    command: list[str],
    logger: logging.Logger,
    stage_name: str,
) -> dict[str, object]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not current_pythonpath else f"src:{current_pythonpath}"
    logger.info("stage_command stage=%s command=%s", stage_name, " ".join(command))
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stderr:
        logger.info(
            "stage_command_stderr stage=%s stderr=%s",
            stage_name,
            completed.stderr.strip(),
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage_name} subprocess failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(f"{stage_name} subprocess produced no stdout payload.")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{stage_name} subprocess returned non-JSON stdout.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{stage_name} subprocess returned {type(payload).__name__}, expected JSON object.")
    return payload


def _add_shared_model_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-device",
        default=None,
        help="Optional model runtime device override, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--allow-cpu-model",
        action="store_true",
        help="Allow CPU execution for model stages and use k-means discovery when CUDA-only discovery is unavailable.",
    )
    parser.add_argument(
        "--full-stdout",
        action="store_true",
        help="Print the full JSON payload to stdout instead of the compact summary.",
    )


def _add_evaluation_guard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=None,
        help="Abort evaluation early if the Python process RSS exceeds this many MB.",
    )


def _add_segmentation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--segmentation-strategy",
        choices=("extrema", "predictor", "hierarchical"),
        default=None,
        help="How to segment the signal into candidate pattern windows. "
             "'hierarchical' = predictor regime blocks (level 1) + sliding windows within each block (level 2).",
    )
    parser.add_argument(
        "--predictor-history-window-steps",
        type=int,
        default=None,
        help="History length used by predictor-based segmentation.",
    )
    parser.add_argument(
        "--predictor-fit-window-steps",
        type=int,
        default=None,
        help="Fit window used by local AR(2) inside predictor-based segmentation.",
    )
    parser.add_argument(
        "--predictor-min-run-steps",
        type=int,
        default=None,
        help="Suppress per-channel predictor switches shorter than this many steps.",
    )
    parser.add_argument(
        "--predictor-min-changed-channels",
        type=int,
        default=None,
        help="Emit a predictor-based boundary only when at least this many channels switch predictor class.",
    )
    parser.add_argument(
        "--predictor-min-window-steps",
        type=int,
        default=None,
        help="Merge predictor-based segments shorter than this many steps into neighbors.",
    )


def _add_dataset_slice_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--date-start",
        default=None,
        help="Inclusive lower bound for the source CSV timestamp range.",
    )
    parser.add_argument(
        "--date-end",
        default=None,
        help="Inclusive upper bound for the source CSV timestamp range.",
    )


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        max_rows=getattr(args, "max_rows", None),
        date_start=getattr(args, "date_start", None),
        date_end=getattr(args, "date_end", None),
    )
    window = config.window
    if getattr(args, "segmentation_strategy", None):
        window = replace(window, segmentation_strategy=args.segmentation_strategy)
    if getattr(args, "predictor_history_window_steps", None) is not None:
        window = replace(window, predictor_history_window_steps=args.predictor_history_window_steps)
    if getattr(args, "predictor_fit_window_steps", None) is not None:
        window = replace(window, predictor_fit_window_steps=args.predictor_fit_window_steps)
    if getattr(args, "predictor_min_run_steps", None) is not None:
        window = replace(window, predictor_min_run_steps=args.predictor_min_run_steps)
    if getattr(args, "predictor_min_changed_channels", None) is not None:
        window = replace(window, predictor_min_changed_channels=args.predictor_min_changed_channels)
    if getattr(args, "predictor_min_window_steps", None) is not None:
        window = replace(window, predictor_min_window_steps=args.predictor_min_window_steps)

    model_device = getattr(args, "model_device", None)
    allow_cpu_model = bool(getattr(args, "allow_cpu_model", False))
    compute = config.compute
    discovery = config.discovery
    if model_device is not None or allow_cpu_model:
        compute = replace(
            compute,
            model_device=model_device or ("cpu" if allow_cpu_model else compute.model_device),
            require_gpu=not allow_cpu_model,
        )
        if allow_cpu_model and discovery.strategy == "structural":
            discovery = replace(discovery, strategy="kmeans")

    return replace(
        config,
        compute=compute,
        discovery=discovery,
        window=window,
    )


def _prepare_bundle_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_path": output_dir / "prepare_summary.json",
        "signal_frame_path": output_dir / "signal_frame.csv",
        "extrema_events_path": output_dir / "extrema_events.csv",
        "peak_events_path": output_dir / "peak_events.csv",
        "prepared_pattern_windows_path": output_dir / "prepared_pattern_windows.jsonl.gz",
    }


def _has_prepare_bundle(output_dir: Path) -> bool:
    return all(resolve_artifact_path(path).exists() for path in _prepare_bundle_paths(output_dir).values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pattern-first weather forecasting MVP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_pipeline_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run the legacy end-to-end pipeline in a single command.",
    )
    run_pipeline_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    run_pipeline_parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts")
    run_pipeline_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    _add_dataset_slice_args(run_pipeline_parser)
    _add_shared_model_runtime_args(run_pipeline_parser)
    _add_segmentation_args(run_pipeline_parser)

    split_workflow_parser = subparsers.add_parser(
        "run-split-workflow",
        help="Run prepare, discovery, train, and evaluation as one staged workflow.",
    )
    split_workflow_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    split_workflow_parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts")
    split_workflow_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    _add_dataset_slice_args(split_workflow_parser)
    split_workflow_parser.add_argument(
        "--reuse-prepare",
        action="store_true",
        help="Reuse an existing prepare bundle from output-dir/prepare instead of recomputing the CPU stage.",
    )
    split_workflow_parser.add_argument(
        "--reuse-prepare-source",
        default=None,
        metavar="PATH",
        help="Path to an existing prepare/ directory (e.g. from a previous extrema run) whose "
             "PatternWindows and signal_frame will be reused. When combined with "
             "--segmentation-strategy hierarchical the windows are filtered to those within "
             "predictor regime blocks and stamped with parent_block_id — skipping the expensive "
             "PatternWindow rebuild (~54 000 objects).",
    )
    split_workflow_parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip the evaluation stage.",
    )
    split_workflow_parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip the prediction stage.",
    )
    split_workflow_parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on validation/test samples per split for faster evaluation",
    )
    _add_shared_model_runtime_args(split_workflow_parser)
    _add_evaluation_guard_args(split_workflow_parser)
    _add_segmentation_args(split_workflow_parser)

    prepare_parser = subparsers.add_parser(
        "prepare-pattern-windows",
        help="Run CPU-only preprocessing and save prepared pattern windows.",
    )
    prepare_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    prepare_parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts")
    prepare_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    _add_dataset_slice_args(prepare_parser)
    _add_segmentation_args(prepare_parser)

    discover_parser = subparsers.add_parser(
        "discover-patterns",
        help="Run pattern discovery from prepared pattern windows.",
    )
    discover_parser.add_argument(
        "--prepared-pattern-windows-path",
        required=True,
        help="Path to prepared_pattern_windows.jsonl or prepared_pattern_windows.jsonl.gz produced by prepare-pattern-windows",
    )
    discover_parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts")
    _add_shared_model_runtime_args(discover_parser)

    plot_patterns_parser = subparsers.add_parser(
        "plot-patterns",
        help="Render diagnostic plots from saved pattern artifacts.",
    )
    plot_patterns_parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing pattern_flow.jsonl and pattern_prototypes.jsonl",
    )
    plot_patterns_parser.add_argument(
        "--csv",
        required=False,
        default=None,
        help="Optional source weather CSV for overlay plots",
    )
    plot_patterns_parser.add_argument(
        "--output-dir",
        default="artifacts/plots",
        help="Directory where the generated plots will be saved",
    )

    train_sequence_parser = subparsers.add_parser(
        "train-sequence-model",
        help="Train the sequence predictor, preferably from a saved sequence dataset.",
    )
    train_sequence_parser.add_argument("--csv", required=False, help="Path to hly4935_subset.csv")
    train_sequence_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    _add_dataset_slice_args(train_sequence_parser)
    train_sequence_parser.add_argument(
        "--sequence-dataset-path",
        default=None,
        help="Optional path to forecast_sequence_dataset.jsonl or forecast_sequence_dataset.jsonl.gz for training without rerunning the pipeline",
    )
    train_sequence_parser.add_argument(
        "--checkpoint-path",
        default="artifacts/sequence_predictor.pt",
        help="Where to save the trained model checkpoint",
    )
    train_sequence_parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path for saving the training summary JSON",
    )
    _add_shared_model_runtime_args(train_sequence_parser)

    predict_sequence_parser = subparsers.add_parser(
        "predict-sequence",
        help="Run the pipeline, train the sequence predictor, and predict a future pattern sequence.",
    )
    predict_sequence_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    predict_sequence_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    _add_dataset_slice_args(predict_sequence_parser)
    predict_sequence_parser.add_argument(
        "--prepared-pattern-windows-path",
        default=None,
        help="Optional path to prepared_pattern_windows.jsonl or prepared_pattern_windows.jsonl.gz for prediction without rerunning discovery",
    )
    predict_sequence_parser.add_argument(
        "--pattern-prototypes-path",
        default=None,
        help="Optional path to pattern_prototypes.jsonl for prediction without rerunning discovery",
    )
    predict_sequence_parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint to load instead of retraining inline",
    )
    predict_sequence_parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path for saving the full prediction summary JSON",
    )
    _add_shared_model_runtime_args(predict_sequence_parser)
    _add_evaluation_guard_args(predict_sequence_parser)

    evaluate_sequence_parser = subparsers.add_parser(
        "evaluate-sequence-model",
        help="Run a chronological train/validation/test backtest and report 1..24h metrics in original channels.",
    )
    evaluate_sequence_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    evaluate_sequence_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    _add_dataset_slice_args(evaluate_sequence_parser)
    evaluate_sequence_parser.add_argument(
        "--prepared-pattern-windows-path",
        default=None,
        help="Optional path to prepared_pattern_windows.jsonl or prepared_pattern_windows.jsonl.gz for evaluation without rerunning discovery",
    )
    evaluate_sequence_parser.add_argument(
        "--pattern-prototypes-path",
        default=None,
        help="Optional path to pattern_prototypes.jsonl for evaluation without rerunning discovery",
    )
    evaluate_sequence_parser.add_argument(
        "--sequence-dataset-path",
        default=None,
        help="Optional path to forecast_sequence_dataset.jsonl or forecast_sequence_dataset.jsonl.gz for evaluation without rerunning discovery",
    )
    evaluate_sequence_parser.add_argument(
        "--output-path",
        default="artifacts/sequence_evaluation.json",
        help="Where to save the evaluation summary JSON",
    )
    evaluate_sequence_parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on validation/test samples per split for faster backtests",
    )
    _add_shared_model_runtime_args(evaluate_sequence_parser)
    _add_evaluation_guard_args(evaluate_sequence_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run-pipeline":
            config = _build_config(args)
            artifacts = run_pipeline(args.csv, config)
            output_payload = artifacts.summary()
            output_payload["artifacts"] = write_pipeline_artifacts(artifacts, args.output_dir)
            print(json.dumps(output_payload, indent=2))
            return

        if args.command == "run-split-workflow":
            config = _build_config(args)
            root_output_dir = Path(args.output_dir)
            logger, log_path = _configure_workflow_logger(root_output_dir)
            prepare_output_dir = root_output_dir / "prepare"
            discovery_output_dir = root_output_dir / "discovery"
            model_output_dir = root_output_dir / "model"
            plots_output_dir = root_output_dir / "plots"
            logger.info(
                "workflow_start csv=%s output_dir=%s reuse_prepare=%s skip_predict=%s skip_evaluate=%s sample_limit=%s max_rss_mb=%s model_device=%s require_gpu=%s discovery_strategy=%s",
                args.csv,
                root_output_dir,
                bool(args.reuse_prepare),
                bool(args.skip_predict),
                bool(args.skip_evaluate),
                args.sample_limit,
                args.max_rss_mb,
                config.compute.model_device,
                config.compute.require_gpu,
                config.discovery.strategy,
            )

            reuse_prepare = bool(args.reuse_prepare) and _has_prepare_bundle(prepare_output_dir)
            reuse_prepare_source = getattr(args, "reuse_prepare_source", None)
            prepared = None
            if reuse_prepare:
                prepare_started = _log_stage_start(logger, "prepare", mode="reuse", output_dir=prepare_output_dir)
                prepare_artifacts = {
                    name: str(path)
                    for name, path in _prepare_bundle_paths(prepare_output_dir).items()
                }
                _log_stage_end(
                    logger,
                    "prepare",
                    prepare_started,
                    reused=True,
                    prepared_pattern_windows_path=prepare_artifacts["prepared_pattern_windows_path"],
                )
            elif reuse_prepare_source and config.window.segmentation_strategy == "hierarchical":
                source_prepare_dir = Path(reuse_prepare_source)
                prepare_started = _log_stage_start(
                    logger,
                    "prepare",
                    mode="hierarchical_filter",
                    source=str(source_prepare_dir),
                    output_dir=prepare_output_dir,
                )
                hierarchical_windows = prepare_hierarchical_from_existing(
                    source_prepare_dir,
                    args.csv,
                    config,
                )
                prepare_artifacts = write_hierarchical_prepare_artifacts(
                    hierarchical_windows,
                    source_prepare_dir,
                    prepare_output_dir,
                )
                _log_stage_end(
                    logger,
                    "prepare",
                    prepare_started,
                    hierarchical_windows=len(hierarchical_windows),
                    source=str(source_prepare_dir),
                    prepared_pattern_windows_path=prepare_artifacts["prepared_pattern_windows_path"],
                )
            else:
                prepare_started = _log_stage_start(logger, "prepare", mode="compute", output_dir=prepare_output_dir)
                prepared = prepare_pattern_windows(args.csv, config)
                prepare_artifacts = write_prepared_artifacts(prepared, prepare_output_dir)
                _log_stage_end(
                    logger,
                    "prepare",
                    prepare_started,
                    rows=prepared.dataset.dataframe.shape[0],
                    pattern_windows=len(prepared.pattern_windows),
                    prepared_pattern_windows_path=prepare_artifacts["prepared_pattern_windows_path"],
                )

            pattern_windows = (
                prepared.pattern_windows
                if prepared is not None
                else load_prepared_pattern_windows(prepare_artifacts["prepared_pattern_windows_path"])
            )
            discovery_started = _log_stage_start(
                logger,
                "discovery",
                output_dir=discovery_output_dir,
                pattern_windows=len(pattern_windows),
                strategy=config.discovery.strategy,
            )
            discovery = discover_patterns(pattern_windows, config)
            discovery_artifacts = write_discovery_artifacts(discovery, discovery_output_dir)
            _log_stage_end(
                logger,
                "discovery",
                discovery_started,
                discovered_patterns=len(discovery.discovery_result.prototypes),
                forecast_samples=len(discovery.forecast_samples),
                pattern_prototypes_path=discovery_artifacts["pattern_prototypes_path"],
                forecast_sequence_dataset_path=discovery_artifacts["forecast_sequence_dataset_path"],
            )

            payload: dict[str, object] = {
                "prepare": {
                    **(
                        prepared.summary()
                        if prepared is not None
                        else {"reused": True}
                    ),
                    "artifacts": prepare_artifacts,
                },
                "discovery": {
                    **discovery.summary(),
                    "artifacts": discovery_artifacts,
                },
            }

            try:
                checkpoint_path = model_output_dir / "sequence_predictor.pt"
                training_summary_path = model_output_dir / "training_summary.json"
                training_started = _log_stage_start(
                    logger,
                    "training",
                    checkpoint_path=checkpoint_path,
                    sequence_dataset_path=discovery_artifacts["forecast_sequence_dataset_path"],
                )
                _, training_dataset, saved_checkpoint_path = train_and_save_sequence_predictor_from_dataset(
                    discovery_artifacts["forecast_sequence_dataset_path"],
                    config,
                    checkpoint_path,
                )
                training_summary = summarize_training_dataset(training_dataset)
                training_summary["checkpoint_path"] = str(saved_checkpoint_path)
                training_summary["sequence_dataset_path"] = discovery_artifacts["forecast_sequence_dataset_path"]
                training_summary["output_path"] = str(
                    write_training_summary(training_summary, training_summary_path)
                )
                payload["training"] = training_summary
                del training_dataset
                gc.collect()
                _log_stage_end(
                    logger,
                    "training",
                    training_started,
                    sample_count=training_summary["sample_count"],
                    checkpoint_path=saved_checkpoint_path,
                    output_path=training_summary["output_path"],
                )

                if not args.skip_predict:
                    prediction_started = _log_stage_start(logger, "prediction", checkpoint_path=saved_checkpoint_path)
                    prediction_output_path = model_output_dir / "prediction_summary.json"
                    prediction_command = _build_model_stage_base_command() + [
                        "predict-sequence",
                        "--csv",
                        args.csv,
                        "--prepared-pattern-windows-path",
                        prepare_artifacts["prepared_pattern_windows_path"],
                        "--pattern-prototypes-path",
                        discovery_artifacts["pattern_prototypes_path"],
                        "--checkpoint-path",
                        str(saved_checkpoint_path),
                        "--output-path",
                        str(prediction_output_path),
                    ]
                    if args.max_rows is not None:
                        prediction_command.extend(["--max-rows", str(args.max_rows)])
                    if args.max_rss_mb is not None:
                        prediction_command.extend(["--max-rss-mb", str(args.max_rss_mb)])
                    prediction_command = _extend_shared_dataset_slice_args(prediction_command, args)
                    prediction_command = _extend_shared_model_runtime_args(prediction_command, args)
                    prediction_payload = _run_json_command(prediction_command, logger, "prediction")
                    payload["prediction"] = prediction_payload
                    _log_stage_end(
                        logger,
                        "prediction",
                        prediction_started,
                        output_path=prediction_payload.get("output_path"),
                    )

                if not args.skip_evaluate:
                    evaluation_started = _log_stage_start(
                        logger,
                        "evaluation",
                        sample_limit=args.sample_limit,
                    )
                    gc.collect()
                    evaluation_output_path = model_output_dir / "sequence_evaluation.json"
                    evaluation_command = _build_model_stage_base_command() + [
                        "evaluate-sequence-model",
                        "--csv",
                        args.csv,
                        "--prepared-pattern-windows-path",
                        prepare_artifacts["prepared_pattern_windows_path"],
                        "--pattern-prototypes-path",
                        discovery_artifacts["pattern_prototypes_path"],
                        "--sequence-dataset-path",
                        discovery_artifacts["forecast_sequence_dataset_path"],
                        "--output-path",
                        str(evaluation_output_path),
                    ]
                    if args.max_rows is not None:
                        evaluation_command.extend(["--max-rows", str(args.max_rows)])
                    if args.sample_limit is not None:
                        evaluation_command.extend(["--sample-limit", str(args.sample_limit)])
                    if args.max_rss_mb is not None:
                        evaluation_command.extend(["--max-rss-mb", str(args.max_rss_mb)])
                    evaluation_command = _extend_shared_dataset_slice_args(evaluation_command, args)
                    evaluation_command = _extend_shared_model_runtime_args(evaluation_command, args)
                    evaluation_payload = _run_json_command(evaluation_command, logger, "evaluation")
                    payload["evaluation"] = evaluation_payload
                    _log_stage_end(
                        logger,
                        "evaluation",
                        evaluation_started,
                        output_path=evaluation_payload.get("output_path"),
                    )
            except GpuRuntimeRequirementError as exc:
                if not bool(getattr(args, "allow_cpu_model", False)):
                    logger.exception("workflow_failed_on_gpu_runtime")
                    raise
                payload["training"] = {
                    "skipped": True,
                    "reason": str(exc),
                    "sequence_dataset_path": discovery_artifacts["forecast_sequence_dataset_path"],
                }
                if not args.skip_predict:
                    payload["prediction"] = {"skipped": True, "reason": str(exc)}
                if not args.skip_evaluate:
                    payload["evaluation"] = {"skipped": True, "reason": str(exc)}
                logger.warning("gpu_runtime_skipped_in_cpu_mode reason=%s", exc)

            from weather_patterns.visualization.patterns import render_pattern_diagnostics

            visualization_started = _log_stage_start(logger, "visualization", output_dir=plots_output_dir)
            plot_payload = render_pattern_diagnostics(
                artifacts_dir=discovery_output_dir,
                output_dir=plots_output_dir,
                csv_path=args.csv,
            )
            payload["visualization"] = plot_payload
            _log_stage_end(
                logger,
                "visualization",
                visualization_started,
                pattern_flow_timeline_path=plot_payload.get("pattern_flow_timeline_path"),
                weather_overlay_path=plot_payload.get("weather_overlay_path"),
            )
            logger.info("workflow_complete log_path=%s", log_path)

            print(json.dumps(payload, indent=2))
            return

        if args.command == "prepare-pattern-windows":
            config = _build_config(args)
            artifacts = prepare_pattern_windows(args.csv, config)
            output_payload = artifacts.summary()
            output_payload["artifacts"] = write_prepared_artifacts(artifacts, args.output_dir)
            print(json.dumps(output_payload, indent=2))
            return

        if args.command == "discover-patterns":
            config = _build_config(args)
            pattern_windows = load_prepared_pattern_windows(args.prepared_pattern_windows_path)
            artifacts = discover_patterns(pattern_windows, config)
            output_payload = artifacts.summary()
            output_payload["prepared_pattern_windows_path"] = args.prepared_pattern_windows_path
            output_payload["artifacts"] = write_discovery_artifacts(artifacts, args.output_dir)
            print(json.dumps(output_payload, indent=2))
            return

        if args.command == "plot-patterns":
            from weather_patterns.visualization.patterns import render_pattern_diagnostics

            payload = render_pattern_diagnostics(
                artifacts_dir=args.artifacts_dir,
                output_dir=args.output_dir,
                csv_path=args.csv,
            )
            print(json.dumps(payload, indent=2))
            return

        if args.command == "train-sequence-model":
            config = _build_config(args)
            command_output_dir = Path(args.output_path).parent if args.output_path else Path(args.checkpoint_path).parent
            logger, log_path = _configure_command_logger(
                command_output_dir,
                "weather_patterns.train_sequence_model",
                "train_sequence_model.log",
            )
            logger.info(
                "training_command_start sequence_dataset_path=%s checkpoint_path=%s output_path=%s model_device=%s require_gpu=%s",
                args.sequence_dataset_path,
                args.checkpoint_path,
                args.output_path,
                config.compute.model_device,
                config.compute.require_gpu,
            )
            if args.sequence_dataset_path:
                _, training_dataset, checkpoint_path = train_and_save_sequence_predictor_from_dataset(
                    args.sequence_dataset_path,
                    config,
                    args.checkpoint_path,
                    logger=logger,
                )
            else:
                if not args.csv:
                    print("Either --csv or --sequence-dataset-path is required.", file=sys.stderr)
                    raise SystemExit(2)
                artifacts = run_pipeline(args.csv, config)
                _, training_dataset, checkpoint_path = train_and_save_sequence_predictor(
                    artifacts,
                    config,
                    args.checkpoint_path,
                )
            payload = summarize_training_dataset(training_dataset)
            payload["checkpoint_path"] = str(checkpoint_path)
            if args.sequence_dataset_path:
                payload["sequence_dataset_path"] = args.sequence_dataset_path
            if args.output_path:
                logger.info("training_summary_write_start output_path=%s", args.output_path)
                payload["output_path"] = str(write_training_summary(payload, args.output_path))
                logger.info("training_summary_write_end output_path=%s", payload["output_path"])
            logger.info(
                "training_command_complete checkpoint_path=%s sample_count=%s log_path=%s",
                checkpoint_path,
                payload["sample_count"],
                log_path,
            )
            print(json.dumps(payload, indent=2))
            return

        if args.command == "predict-sequence":
            config = _build_config(args)
            if bool(args.prepared_pattern_windows_path) != bool(args.pattern_prototypes_path):
                print(
                    "predict-sequence requires both --prepared-pattern-windows-path and --pattern-prototypes-path.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            if args.prepared_pattern_windows_path and args.pattern_prototypes_path:
                artifacts = load_saved_pipeline_artifacts(
                    csv_path=args.csv,
                    prepared_pattern_windows_path=args.prepared_pattern_windows_path,
                    pattern_prototypes_path=args.pattern_prototypes_path,
                    config=config,
                )
            else:
                artifacts = run_pipeline(args.csv, config)
            if args.checkpoint_path:
                checkpoint_path = Path(args.checkpoint_path)
                if not checkpoint_path.exists():
                    print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
                    raise SystemExit(2)
                predictor = TorchSequencePredictor.load_checkpoint(
                    checkpoint_path,
                    model_config=config.model,
                    compute_config=config.compute,
                )
            else:
                predictor, _ = train_sequence_predictor(artifacts, config)
            if args.max_rss_mb is not None:
                predictor.set_resource_limits(max_rss_mb=args.max_rss_mb)
            result = predict_future_pattern_sequence(predictor, artifacts, config)
            full_payload = summarize_forecast_result(result)
            if args.output_path:
                full_payload["output_path"] = str(write_forecast_summary(full_payload, args.output_path))
            payload = (
                full_payload
                if args.full_stdout
                else summarize_forecast_result_compact(result)
            )
            if args.output_path:
                payload["output_path"] = full_payload["output_path"]
            if args.checkpoint_path:
                payload["checkpoint_path"] = args.checkpoint_path
            print(json.dumps(payload, indent=2))
            return

        if args.command == "evaluate-sequence-model":
            config = _build_config(args)
            command_output_dir = Path(args.output_path).parent
            logger, log_path = _configure_command_logger(
                command_output_dir,
                "weather_patterns.evaluate_sequence_model",
                "evaluate_sequence_model.log",
            )
            logger.info(
                "evaluation_command_start csv=%s sequence_dataset_path=%s output_path=%s sample_limit=%s max_rss_mb=%s model_device=%s require_gpu=%s",
                args.csv,
                args.sequence_dataset_path,
                args.output_path,
                args.sample_limit,
                args.max_rss_mb,
                config.compute.model_device,
                config.compute.require_gpu,
            )
            evaluation_saved_args = [
                args.prepared_pattern_windows_path,
                args.pattern_prototypes_path,
                args.sequence_dataset_path,
            ]
            if any(evaluation_saved_args) and not all(evaluation_saved_args):
                print(
                    "evaluate-sequence-model requires --prepared-pattern-windows-path, --pattern-prototypes-path, and --sequence-dataset-path together.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            if (
                args.prepared_pattern_windows_path
                and args.pattern_prototypes_path
                and args.sequence_dataset_path
            ):
                artifacts = load_saved_pipeline_artifacts(
                    csv_path=args.csv,
                    prepared_pattern_windows_path=args.prepared_pattern_windows_path,
                    pattern_prototypes_path=args.pattern_prototypes_path,
                    config=config,
                    load_pattern_windows=False,
                    load_forecast_samples=False,
                )
                payload = evaluate_sequence_backtest_from_saved_dataset(
                    artifacts,
                    args.sequence_dataset_path,
                    config,
                    sample_limit=args.sample_limit,
                    logger=logger,
                    max_rss_mb=args.max_rss_mb,
                    prepared_pattern_windows_path=args.prepared_pattern_windows_path,
                )
            else:
                artifacts = run_pipeline(args.csv, config)
                payload = evaluate_sequence_backtest(
                    artifacts,
                    config,
                    sample_limit=args.sample_limit,
                    logger=logger,
                    max_rss_mb=args.max_rss_mb,
                )
            payload["output_path"] = str(write_evaluation_summary(payload, args.output_path))
            logger.info(
                "evaluation_command_complete output_path=%s log_path=%s",
                payload["output_path"],
                log_path,
            )
            stdout_payload = payload if args.full_stdout else summarize_evaluation_payload(payload)
            stdout_payload["output_path"] = payload["output_path"]
            print(json.dumps(stdout_payload, indent=2))
            return
    except GpuRuntimeRequirementError as exc:
        print(f"GPU runtime error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
