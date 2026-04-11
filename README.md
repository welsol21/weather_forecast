# Weather Patterns MVP

Docker-first MVP for the pattern-first weather forecasting approach described in `TZ.docx`.

## Documentation

- `TZ.docx`: source concept and mathematical direction.
- `prompt.txt`: implementation-oriented MVP scope.
- `docs/project_formulation.md`: consolidated project formulation that develops the ideas from `TZ.docx` and `prompt.txt`, including the sequence-pattern inference target.

## Run with Docker

```bash
docker compose up --build
```

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m weather_patterns prepare-pattern-windows --csv hly4935_subset.csv --output-dir artifacts/prepare
python -m weather_patterns discover-patterns --prepared-pattern-windows-path artifacts/prepare/prepared_pattern_windows.jsonl --output-dir artifacts/discovery
python -m weather_patterns train-sequence-model --sequence-dataset-path artifacts/discovery/forecast_sequence_dataset.jsonl --checkpoint-path artifacts/sequence_predictor.pt
```

If you prefer not to install the package in editable mode during local development, you can run the commands with `PYTHONPATH=src`.

`run-pipeline` still exists as a legacy single-command path, but the intended workflow is now split into `prepare-pattern-windows` and `discover-patterns`.

## GPU-Only Discovery And Model Commands

These commands are intended for environments with CUDA-enabled PyTorch. The repository does not pin `torch` in the base requirements because the exact wheel depends on the target CUDA stack.

```bash
python -m weather_patterns discover-patterns --prepared-pattern-windows-path artifacts/prepare/prepared_pattern_windows.jsonl --output-dir artifacts/discovery
python -m weather_patterns train-sequence-model --sequence-dataset-path artifacts/discovery/forecast_sequence_dataset.jsonl --checkpoint-path artifacts/sequence_predictor.pt
python -m weather_patterns predict-sequence --csv hly4935_subset.csv --prepared-pattern-windows-path artifacts/prepare/prepared_pattern_windows.jsonl --pattern-prototypes-path artifacts/discovery/pattern_prototypes.jsonl --checkpoint-path artifacts/sequence_predictor.pt
python -m weather_patterns evaluate-sequence-model --csv hly4935_subset.csv --prepared-pattern-windows-path artifacts/prepare/prepared_pattern_windows.jsonl --pattern-prototypes-path artifacts/discovery/pattern_prototypes.jsonl --sequence-dataset-path artifacts/discovery/forecast_sequence_dataset.jsonl --output-path artifacts/sequence_evaluation.json
```

`discover-patterns`, `train-sequence-model`, `predict-sequence`, and `evaluate-sequence-model` should be treated as CUDA stages in the target workflow.

`predict-sequence` and `evaluate-sequence-model` print compact JSON summaries by default. Add `--full-stdout` when you want the full JSON payload in the terminal as well.

`train-sequence-model` also accepts `--output-path` for saving the training summary JSON alongside the checkpoint metadata in stdout.

`predict-sequence` also accepts `--output-path` for saving the full prediction summary JSON without forcing the full payload into stdout.

`predict-sequence` can consume saved discovery artifacts via `--prepared-pattern-windows-path` and `--pattern-prototypes-path`. `evaluate-sequence-model` can consume the same discovery artifacts plus `--sequence-dataset-path`.

## What the pipeline does

1. Loads and cleans the hourly weather CSV.
2. Maps source columns into placeholder channels.
3. Smooths the signals and computes first/second differences.
4. Detects extrema and peaks.
5. Builds extrema windows and pattern representations.
6. Runs a baseline pattern discovery stage.
7. Builds supervised forecast samples for future pattern sequence prediction.

## Prepare Artifacts

`prepare-pattern-windows` writes the CPU-side preparation bundle:

- `prepare_summary.json`
- `signal_frame.csv`
- `extrema_events.csv`
- `peak_events.csv`
- `prepared_pattern_windows.jsonl`

## Discovery Artifacts

`discover-patterns` writes the GPU-side discovery bundle:

- `discovery_summary.json`
- `pattern_prototypes.jsonl`
- `pattern_flow.jsonl`
- `forecast_sequence_dataset.jsonl`

## Runtime Note

Signal processing, event extraction, and pattern window construction belong to the preparation stage.

Pattern discovery and all later model stages must be treated as GPU-only. The project config now encodes this requirement through `PipelineConfig.compute`, and CUDA availability is validated through `weather_patterns.forecasting.runtime.resolve_model_device`.

The repository now also contains a GPU-only `torch` sequence predictor skeleton that consumes sequence-shaped forecast samples and predicts a future pattern matrix. It is intentionally isolated from the CPU preprocessing pipeline and requires CUDA-aware PyTorch at runtime.
