from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import IO
from typing import Any, Iterable, Iterator


def _open_text(path: Path, mode: str) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def resolve_artifact_path(path: str | Path) -> Path:
    source = Path(path)
    if source.exists():
        return source
    if source.suffix != ".gz":
        gzip_variant = source.with_name(f"{source.name}.gz")
        if gzip_variant.exists():
            return gzip_variant
    return source


def write_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(destination, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = resolve_artifact_path(path)
    with _open_text(source, "rt") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object records in {source}, got {type(payload).__name__}.")
            yield payload


def write_pattern_prototypes_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> Path:
    return write_jsonl(records, path)


def write_prepared_pattern_windows_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> Path:
    return write_jsonl(records, path)


def read_prepared_pattern_windows_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def read_pattern_prototypes_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_pattern_flow_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> Path:
    return write_jsonl(records, path)


def read_pattern_flow_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_forecast_sequence_dataset_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> Path:
    return write_jsonl(records, path)


def read_forecast_sequence_dataset_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
