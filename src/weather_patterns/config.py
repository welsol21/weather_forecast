from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    source_column: str
    quality_column: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    datetime_column: str = "date"
    datetime_format: str = "%d-%b-%Y %H:%M"
    missing_tokens: tuple[str, ...] = (" ", "")
    channels: tuple[ChannelSpec, ...] = (
        ChannelSpec("temperature", "temp", "temp_quality"),
        ChannelSpec("relative_humidity", "rhum", None),
        ChannelSpec("pressure", "msl", "pressure_quality"),
        ChannelSpec("wind_speed", "wdsp", "wind_speed_quality"),
        ChannelSpec("rainfall", "rain", "rain_quality"),
        ChannelSpec("dew_point", "dewpt", None),
        ChannelSpec("wet_bulb", "wetb", "wetb_quality"),
        ChannelSpec("vapour_pressure", "vappr", None),
        ChannelSpec("wind_direction", "wddir", None),
    )


@dataclass(frozen=True)
class SmoothingConfig:
    method: str = "rolling_mean"
    window: int = 5
    min_periods: int = 1
    center: bool = True
    savgol_window: int = 7
    savgol_polyorder: int = 2
    fallback_to_rolling: bool = True


@dataclass(frozen=True)
class WindowConfig:
    length_steps: int = 24
    stride_steps: int = 1
    forecast_horizon_steps: int = 24
    correlation_lag_steps: int = 6
    event_match_tolerance_steps: int = 1


@dataclass(frozen=True)
class HazardConfig:
    upper_quantile: float = 0.9
    lower_quantile: float = 0.1
    rainfall_drought_threshold: float = 0.1
    compound_rain_threshold_quantile: float = 0.8
    compound_wind_threshold_quantile: float = 0.8


@dataclass(frozen=True)
class DiscoveryConfig:
    strategy: str = "kmeans"
    n_clusters: int = 8
    max_iterations: int = 50
    random_seed: int = 42


@dataclass(frozen=True)
class ForecastConfig:
    history_window_count: int = 4
    target_window_count: int | None = None


@dataclass(frozen=True)
class ComputeConfig:
    model_device: str = "cuda"
    require_gpu: bool = True


@dataclass(frozen=True)
class SequenceModelConfig:
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    random_seed: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    model: SequenceModelConfig = field(default_factory=SequenceModelConfig)
    time_step_hours: float = 1.0
    max_rows: int | None = None
