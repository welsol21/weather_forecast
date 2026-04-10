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
python -m weather_patterns run-pipeline --csv hly4935_subset.csv --output-dir artifacts
```

## GPU-Only Model Commands

These commands are intended for environments with CUDA-enabled PyTorch. The repository does not pin `torch` in the base requirements because the exact wheel depends on the target CUDA stack.

```bash
python -m weather_patterns train-sequence-model --csv hly4935_subset.csv --max-rows 240 --checkpoint-path artifacts/sequence_predictor.pt
python -m weather_patterns predict-sequence --csv hly4935_subset.csv --max-rows 240 --checkpoint-path artifacts/sequence_predictor.pt
```

## What the pipeline does

1. Loads and cleans the hourly weather CSV.
2. Maps source columns into placeholder channels.
3. Smooths the signals and computes first/second differences.
4. Detects extrema and peaks.
5. Builds extrema windows and pattern representations.
6. Runs a baseline pattern discovery stage.
7. Builds supervised forecast samples for future pattern sequence prediction.

## Runtime Note

Signal processing, event extraction, and pattern discovery in the current MVP can run in the regular pipeline environment.

All model stages added on top of this MVP, including training and inference, must be treated as GPU-only. The project config now encodes this requirement through `PipelineConfig.compute`, and future model entry points should validate CUDA availability through `weather_patterns.forecasting.runtime.resolve_model_device`.

The repository now also contains a GPU-only `torch` sequence predictor skeleton that consumes sequence-shaped forecast samples and predicts a future pattern matrix. It is intentionally isolated from the CPU preprocessing pipeline and requires CUDA-aware PyTorch at runtime.
