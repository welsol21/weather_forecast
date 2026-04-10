from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weather_patterns.config import PipelineConfig
from weather_patterns.forecasting.inference import (
    predict_future_pattern_sequence,
    summarize_forecast_result,
)
from weather_patterns.forecasting.runtime import GpuRuntimeRequirementError
from weather_patterns.forecasting.training import (
    summarize_training_dataset,
    train_and_save_sequence_predictor,
    train_sequence_predictor,
)
from weather_patterns.forecasting.torch_sequence import TorchSequencePredictor
from weather_patterns.pipeline import run_pipeline, write_pipeline_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pattern-first weather forecasting MVP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_pipeline_parser = subparsers.add_parser("run-pipeline", help="Run the end-to-end pipeline.")
    run_pipeline_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    run_pipeline_parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts")
    run_pipeline_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")

    train_sequence_parser = subparsers.add_parser(
        "train-sequence-model",
        help="Run the pipeline and train the GPU-only sequence predictor.",
    )
    train_sequence_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    train_sequence_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    train_sequence_parser.add_argument(
        "--checkpoint-path",
        default="artifacts/sequence_predictor.pt",
        help="Where to save the trained model checkpoint",
    )

    predict_sequence_parser = subparsers.add_parser(
        "predict-sequence",
        help="Run the pipeline, train the GPU-only sequence predictor, and predict a future pattern sequence.",
    )
    predict_sequence_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    predict_sequence_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    predict_sequence_parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint to load instead of retraining inline",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run-pipeline":
            config = PipelineConfig(max_rows=args.max_rows)
            artifacts = run_pipeline(args.csv, config)
            output_payload = artifacts.summary()
            output_payload["artifacts"] = write_pipeline_artifacts(artifacts, args.output_dir)
            print(json.dumps(output_payload, indent=2))
            return

        if args.command == "train-sequence-model":
            config = PipelineConfig(max_rows=args.max_rows)
            artifacts = run_pipeline(args.csv, config)
            _, training_dataset, checkpoint_path = train_and_save_sequence_predictor(
                artifacts,
                config,
                args.checkpoint_path,
            )
            payload = summarize_training_dataset(training_dataset)
            payload["checkpoint_path"] = str(checkpoint_path)
            print(json.dumps(payload, indent=2))
            return

        if args.command == "predict-sequence":
            config = PipelineConfig(max_rows=args.max_rows)
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
            payload = summarize_forecast_result(result)
            if args.checkpoint_path:
                payload["checkpoint_path"] = args.checkpoint_path
            print(json.dumps(payload, indent=2))
            return
    except GpuRuntimeRequirementError as exc:
        print(f"GPU runtime error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
