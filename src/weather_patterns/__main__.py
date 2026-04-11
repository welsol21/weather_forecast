from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
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
from weather_patterns.pipeline import (
    discover_patterns,
    load_prepared_pattern_windows,
    load_saved_pipeline_artifacts,
    prepare_pattern_windows,
    run_pipeline,
    write_discovery_artifacts,
    write_pipeline_artifacts,
    write_prepared_artifacts,
)


def _add_shared_model_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-device",
        default=None,
        help="Optional model runtime device override, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--allow-cpu-model",
        action="store_true",
        help="Disable the GPU-only runtime requirement for model commands.",
    )
    parser.add_argument(
        "--full-stdout",
        action="store_true",
        help="Print the full JSON payload to stdout instead of the compact summary.",
    )


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(max_rows=getattr(args, "max_rows", None))
    model_device = getattr(args, "model_device", None)
    allow_cpu_model = bool(getattr(args, "allow_cpu_model", False))
    if model_device is None and not allow_cpu_model:
        return config
    return replace(
        config,
        compute=replace(
            config.compute,
            model_device=model_device or config.compute.model_device,
            require_gpu=not allow_cpu_model,
        ),
    )


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

    prepare_parser = subparsers.add_parser(
        "prepare-pattern-windows",
        help="Run CPU-only preprocessing and save prepared pattern windows.",
    )
    prepare_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    prepare_parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts")
    prepare_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")

    discover_parser = subparsers.add_parser(
        "discover-patterns",
        help="Run GPU-only pattern discovery from prepared pattern windows.",
    )
    discover_parser.add_argument(
        "--prepared-pattern-windows-path",
        required=True,
        help="Path to prepared_pattern_windows.jsonl produced by prepare-pattern-windows",
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
        help="Train the GPU-only sequence predictor, preferably from a saved sequence dataset.",
    )
    train_sequence_parser.add_argument("--csv", required=False, help="Path to hly4935_subset.csv")
    train_sequence_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    train_sequence_parser.add_argument(
        "--sequence-dataset-path",
        default=None,
        help="Optional path to forecast_sequence_dataset.jsonl for training without rerunning the pipeline",
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
        help="Run the pipeline, train the GPU-only sequence predictor, and predict a future pattern sequence.",
    )
    predict_sequence_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    predict_sequence_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    predict_sequence_parser.add_argument(
        "--prepared-pattern-windows-path",
        default=None,
        help="Optional path to prepared_pattern_windows.jsonl for prediction without rerunning discovery",
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

    evaluate_sequence_parser = subparsers.add_parser(
        "evaluate-sequence-model",
        help="Run a chronological train/validation/test backtest and report 1..24h metrics in original channels.",
    )
    evaluate_sequence_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    evaluate_sequence_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    evaluate_sequence_parser.add_argument(
        "--prepared-pattern-windows-path",
        default=None,
        help="Optional path to prepared_pattern_windows.jsonl for evaluation without rerunning discovery",
    )
    evaluate_sequence_parser.add_argument(
        "--pattern-prototypes-path",
        default=None,
        help="Optional path to pattern_prototypes.jsonl for evaluation without rerunning discovery",
    )
    evaluate_sequence_parser.add_argument(
        "--sequence-dataset-path",
        default=None,
        help="Optional path to forecast_sequence_dataset.jsonl for evaluation without rerunning discovery",
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
            if args.sequence_dataset_path:
                _, training_dataset, checkpoint_path = train_and_save_sequence_predictor_from_dataset(
                    args.sequence_dataset_path,
                    config,
                    args.checkpoint_path,
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
                payload["output_path"] = str(write_training_summary(payload, args.output_path))
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
                    sequence_dataset_path=args.sequence_dataset_path,
                    config=config,
                )
            else:
                artifacts = run_pipeline(args.csv, config)
            payload = evaluate_sequence_backtest(
                artifacts,
                config,
                sample_limit=args.sample_limit,
            )
            payload["output_path"] = str(write_evaluation_summary(payload, args.output_path))
            stdout_payload = payload if args.full_stdout else summarize_evaluation_payload(payload)
            stdout_payload["output_path"] = payload["output_path"]
            print(json.dumps(stdout_payload, indent=2))
            return
    except GpuRuntimeRequirementError as exc:
        print(f"GPU runtime error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
