# Weather Patterns MVP

Docker-first MVP for the pattern-first weather forecasting approach described in `TZ.docx`.

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

## What the pipeline does

1. Loads and cleans the hourly weather CSV.
2. Maps source columns into placeholder channels.
3. Smooths the signals and computes first/second differences.
4. Detects extrema and peaks.
5. Builds extrema windows and pattern representations.
6. Runs a baseline pattern discovery stage.
7. Builds supervised forecast samples for future pattern prediction.
