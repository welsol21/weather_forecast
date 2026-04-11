from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_jsonl(records: list[dict[str, object]], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object records in {source}, got {type(payload).__name__}.")
            records.append(payload)
    return records


def write_pattern_prototypes_jsonl(records: list[dict[str, object]], path: str | Path) -> Path:
    return write_jsonl(records, path)


def read_pattern_prototypes_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_pattern_flow_jsonl(records: list[dict[str, object]], path: str | Path) -> Path:
    return write_jsonl(records, path)


def read_pattern_flow_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_forecast_sequence_dataset_jsonl(records: list[dict[str, object]], path: str | Path) -> Path:
    return write_jsonl(records, path)


def read_forecast_sequence_dataset_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)

