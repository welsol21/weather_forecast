from __future__ import annotations

import argparse
import json

from weather_patterns.config import PipelineConfig
from weather_patterns.pipeline import run_pipeline, write_artifacts_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pattern-first weather forecasting MVP.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_pipeline_parser = subparsers.add_parser("run-pipeline", help="Run the end-to-end pipeline.")
    run_pipeline_parser.add_argument("--csv", required=True, help="Path to hly4935_subset.csv")
    run_pipeline_parser.add_argument("--output-dir", default="artifacts", help="Directory for generated artifacts")
    run_pipeline_parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick runs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-pipeline":
        config = PipelineConfig(max_rows=args.max_rows)
        artifacts = run_pipeline(args.csv, config)
        summary_path = write_artifacts_summary(artifacts, args.output_dir)
        print(json.dumps(artifacts.summary(), indent=2))
        print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
