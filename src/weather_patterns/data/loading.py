from __future__ import annotations

from pathlib import Path

import pandas as pd

from weather_patterns.config import DatasetConfig
from weather_patterns.models import LoadedWeatherDataset


NORMALIZED_SOURCE_COLUMNS = [
    "date",
    "rain_quality",
    "rain",
    "temp_quality",
    "temp",
    "wetb_quality",
    "wetb",
    "dewpt",
    "vappr",
    "rhum",
    "msl",
    "pressure_quality",
    "wdsp",
    "wind_speed_quality",
    "wddir",
    "ww",
    "w",
    "sun",
    "vis",
    "clht",
    "clamt",
]


def _find_table_start(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.lower().startswith("date,"):
            return index
    raise ValueError("CSV header line starting with 'date,' was not found.")


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def load_weather_dataset(path: str | Path, config: DatasetConfig) -> LoadedWeatherDataset:
    source_path = Path(path)
    lines = _read_lines(source_path)
    header_index = _find_table_start(lines)
    metadata_lines = lines[:header_index]
    dataframe = pd.read_csv(
        source_path,
        skiprows=header_index + 1,
        names=NORMALIZED_SOURCE_COLUMNS,
        na_values=list(config.missing_tokens),
    )
    dataframe[config.datetime_column] = pd.to_datetime(
        dataframe[config.datetime_column],
        format=config.datetime_format,
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=[config.datetime_column]).sort_values(config.datetime_column)
    dataframe = dataframe.drop_duplicates(subset=[config.datetime_column]).reset_index(drop=True)

    quality_columns = [spec.quality_column for spec in config.channels if spec.quality_column]
    numeric_columns = [column for column in dataframe.columns if column != config.datetime_column]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    renamed = {spec.source_column: spec.name for spec in config.channels}
    dataframe = dataframe.rename(columns=renamed)

    for spec in config.channels:
        if spec.quality_column and spec.quality_column in dataframe.columns:
            dataframe[spec.quality_column] = dataframe[spec.quality_column].astype("Int64")

    channel_columns = [spec.name for spec in config.channels]
    expected_columns = [config.datetime_column, *channel_columns, *quality_columns]
    for column in expected_columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA

    return LoadedWeatherDataset(
        dataframe=dataframe,
        channel_columns=channel_columns,
        quality_columns=quality_columns,
        metadata_lines=metadata_lines,
        source_path=source_path,
    )


def apply_quality_masks(dataset: LoadedWeatherDataset) -> pd.DataFrame:
    frame = dataset.dataframe.copy()
    quality_map = {
        "rainfall": "rain_quality",
        "temperature": "temp_quality",
        "wet_bulb": "wetb_quality",
        "pressure": "pressure_quality",
        "wind_speed": "wind_speed_quality",
    }
    for channel, quality_column in quality_map.items():
        if quality_column not in frame.columns or channel not in frame.columns:
            continue
        invalid_mask = frame[quality_column].astype("Int64").fillna(9) > 2
        frame.loc[invalid_mask, channel] = pd.NA
    return frame
