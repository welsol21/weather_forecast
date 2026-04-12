from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable
import logging
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from weather_patterns.config import ComputeConfig, SequenceModelConfig
from weather_patterns.forecasting.base import SequencePredictor
from weather_patterns.forecasting.runtime import GpuRuntimeRequirementError, resolve_model_device
from weather_patterns.models import ForecastResult, ForecastSample, ForecastTrainingDataset, PatternPrototype


def _log(logger: logging.Logger | None, message: str, **context: object) -> None:
    if logger is None:
        return
    details = ", ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    logger.info(f"{message}{' ' + details if details else ''}")


def _read_current_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    return round(int(parts[1]) / 1024.0, 2)
    except OSError:
        pass
    return None


def _read_peak_rss_mb() -> float | None:
    try:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except OSError:
        return None
    return round(float(peak_kb) / 1024.0, 2)


def _nearest_pattern_ids(
    predicted_pattern_matrix: np.ndarray,
    prototypes: list[PatternPrototype] | None,
) -> list[int | None]:
    if not prototypes:
        return [None for _ in range(len(predicted_pattern_matrix))]
    prototype_ids = [prototype.pattern_id for prototype in prototypes]
    prototype_matrix = np.stack([prototype.centroid for prototype in prototypes], axis=0)
    distances = np.linalg.norm(
        predicted_pattern_matrix[:, None, :] - prototype_matrix[None, :, :],
        axis=2,
    )
    nearest_indices = distances.argmin(axis=1)
    return [prototype_ids[index] for index in nearest_indices]


@dataclass(slots=True)
class TorchSequencePredictor(SequencePredictor):
    model_config: SequenceModelConfig
    compute_config: ComputeConfig

    def __post_init__(self) -> None:
        self.device = resolve_model_device(
            device=self.compute_config.model_device,
            require_gpu=self.compute_config.require_gpu,
        )
        self._torch = None
        self._model = None
        self._feature_dim: int | None = None
        self._forecast_window_count: int | None = None
        self._history_window_count: int | None = None
        self._input_mean: np.ndarray | None = None
        self._input_std: np.ndarray | None = None
        self._context_mean: np.ndarray | None = None
        self._context_std: np.ndarray | None = None
        self._target_mean: np.ndarray | None = None
        self._target_std: np.ndarray | None = None
        self._logger: logging.Logger | None = None
        self._predict_call_count = 0
        self._predict_log_first_n = 3
        self._predict_log_every_n = 100
        self._max_rss_mb: float | None = None

    def set_logger(self, logger: logging.Logger | None) -> None:
        self._logger = logger

    def set_resource_limits(self, max_rss_mb: float | None = None) -> None:
        self._max_rss_mb = max_rss_mb

    def _enforce_rss_limit(self, stage: str, **context: object) -> None:
        if self._max_rss_mb is None:
            return
        current_rss_mb = _read_current_rss_mb()
        if current_rss_mb is None or current_rss_mb <= self._max_rss_mb:
            return
        payload = {
            "stage": stage,
            "rss_mb": current_rss_mb,
            "peak_rss_mb": _read_peak_rss_mb(),
            "rss_limit_mb": self._max_rss_mb,
            **context,
        }
        _log(self._logger, "sequence_rss_limit_exceeded", **payload)
        details = ", ".join(f"{key}={value}" for key, value in payload.items() if value is not None)
        raise MemoryError(f"RSS limit exceeded: {details}")

    def _lazy_import_torch(self) -> None:
        if self._torch is None:
            try:
                import torch
                import torch.nn as nn
            except ModuleNotFoundError as exc:
                if exc.name == "torch":
                    raise GpuRuntimeRequirementError(
                        "PyTorch is required for sequence model stages. Install a CPU or CUDA build of torch first."
                    ) from exc
                raise

            self._torch = torch
            self._nn = nn

    def _build_model(
        self,
        history_window_count: int,
        feature_dim: int,
        forecast_window_count: int,
        context_dim: int,
    ) -> None:
        self._lazy_import_torch()
        nn = self._nn
        torch = self._torch

        class SequenceRegressor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                dropout = self_model_config.dropout if self_model_config.num_layers > 1 else 0.0
                self.encoder = nn.GRU(
                    input_size=feature_dim,
                    hidden_size=self_model_config.hidden_size,
                    num_layers=self_model_config.num_layers,
                    batch_first=True,
                    dropout=dropout,
                )
                self.context_mlp = nn.Sequential(
                    nn.LayerNorm(context_dim),
                    nn.Linear(context_dim, self_model_config.hidden_size),
                    nn.GELU(),
                    nn.Dropout(self_model_config.dropout),
                    nn.Linear(self_model_config.hidden_size, self_model_config.hidden_size),
                )
                self.fusion = nn.Sequential(
                    nn.LayerNorm(self_model_config.hidden_size * 2),
                    nn.Linear(self_model_config.hidden_size * 2, self_model_config.hidden_size),
                    nn.GELU(),
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(self_model_config.hidden_size),
                    nn.Linear(
                        self_model_config.hidden_size,
                        forecast_window_count * feature_dim,
                    ),
                )

            def forward(self, history_pattern_tensor, history_context, baseline_target):  # type: ignore[no-untyped-def]
                _, hidden = self.encoder(history_pattern_tensor)
                encoded = hidden[-1]
                context_encoded = self.context_mlp(history_context)
                fused = self.fusion(torch.cat([encoded, context_encoded], dim=1))
                output = self.head(fused).view(-1, forecast_window_count, feature_dim)
                return baseline_target + output

        self_model_config = self.model_config
        self._model = SequenceRegressor().to(self.device)
        self._feature_dim = feature_dim
        self._forecast_window_count = forecast_window_count

    def fit(
        self,
        dataset: ForecastTrainingDataset,
        logger: logging.Logger | None = None,
        log_interval_batches: int = 50,
    ) -> None:
        self.set_logger(logger)
        self._lazy_import_torch()
        torch = self._torch
        torch.manual_seed(self.model_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.model_config.random_seed)

        self._build_model(
            history_window_count=dataset.history_window_count,
            feature_dim=dataset.feature_dim,
            forecast_window_count=dataset.forecast_window_count,
            context_dim=dataset.history_vector_matrix.shape[1],
        )
        self._history_window_count = dataset.history_window_count
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )
        loss_fn = self._nn.MSELoss()

        self._input_mean = dataset.history_pattern_tensor.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self._input_std = dataset.history_pattern_tensor.std(axis=(0, 1), keepdims=True).astype(np.float32)
        self._input_std = np.where(self._input_std > 1e-6, self._input_std, 1.0).astype(np.float32)
        self._context_mean = dataset.history_vector_matrix.mean(axis=0, keepdims=True).astype(np.float32)
        self._context_std = dataset.history_vector_matrix.std(axis=0, keepdims=True).astype(np.float32)
        self._context_std = np.where(self._context_std > 1e-6, self._context_std, 1.0).astype(np.float32)
        self._target_mean = dataset.target_pattern_tensor.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self._target_std = dataset.target_pattern_tensor.std(axis=(0, 1), keepdims=True).astype(np.float32)
        self._target_std = np.where(self._target_std > 1e-6, self._target_std, 1.0).astype(np.float32)

        sample_count = dataset.history_pattern_tensor.shape[0]
        batch_size = min(self.model_config.batch_size, sample_count)
        batch_count = (sample_count + batch_size - 1) // batch_size
        history_tensor = dataset.history_pattern_tensor
        context_tensor = dataset.history_vector_matrix
        target_tensor = dataset.target_pattern_tensor

        for epoch_index in range(self.model_config.epochs):
            epoch_started_at = time.perf_counter()
            permutation = np.random.permutation(sample_count)
            for batch_number, start in enumerate(range(0, sample_count, batch_size), start=1):
                batch_started_at = time.perf_counter()
                batch_indices = permutation[start : start + batch_size]
                batch_history_np = history_tensor[batch_indices]
                batch_context_np = context_tensor[batch_indices]
                batch_target_np = target_tensor[batch_indices]
                batch_baseline_np = (
                    batch_history_np[:, -1:, :] - self._target_mean
                ) / self._target_std
                batch_baseline_np = np.repeat(
                    batch_baseline_np,
                    dataset.forecast_window_count,
                    axis=1,
                )

                batch_history = torch.as_tensor(
                    (batch_history_np - self._input_mean) / self._input_std,
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_context = torch.as_tensor(
                    (batch_context_np - self._context_mean) / self._context_std,
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_target = torch.as_tensor(
                    (batch_target_np - self._target_mean) / self._target_std,
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_baseline = torch.as_tensor(
                    batch_baseline_np,
                    dtype=torch.float32,
                    device=self.device,
                )

                optimizer.zero_grad(set_to_none=True)
                predicted = self._model(batch_history, batch_context, batch_baseline)
                loss = loss_fn(predicted, batch_target)
                loss.backward()
                optimizer.step()
                batch_elapsed_ms = round((time.perf_counter() - batch_started_at) * 1000.0, 2)
                if (
                    batch_number == 1
                    or batch_number == batch_count
                    or batch_number % log_interval_batches == 0
                ):
                    _log(
                        self._logger,
                        "sequence_fit_batch",
                        mode="in_memory",
                        epoch=epoch_index + 1,
                        total_epochs=self.model_config.epochs,
                        batch=batch_number,
                        total_batches=batch_count,
                        batch_size=len(batch_indices),
                        loss=round(float(loss.detach().item()), 6),
                        elapsed_ms=batch_elapsed_ms,
                    )
                self._enforce_rss_limit(
                    "fit_batch",
                    mode="in_memory",
                    epoch=epoch_index + 1,
                    batch=batch_number,
                    total_batches=batch_count,
                )
            _log(
                self._logger,
                "sequence_fit_epoch_end",
                mode="in_memory",
                epoch=epoch_index + 1,
                total_epochs=self.model_config.epochs,
                elapsed_seconds=round(time.perf_counter() - epoch_started_at, 3),
            )

    def fit_samples(
        self,
        samples: list[ForecastSample],
        logger: logging.Logger | None = None,
        log_interval_batches: int = 50,
    ) -> None:
        if not samples:
            raise ValueError("At least one forecast sample is required to fit the predictor.")
        self.set_logger(logger)

        self._lazy_import_torch()
        torch = self._torch
        torch.manual_seed(self.model_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.model_config.random_seed)

        first_sample = samples[0]
        history_window_count = first_sample.history_pattern_matrix.shape[0]
        feature_dim = first_sample.history_pattern_matrix.shape[1]
        forecast_window_count = first_sample.target_pattern_matrix.shape[0]
        context_dim = first_sample.history_vector.shape[0]

        for sample in samples:
            if sample.history_pattern_matrix.shape != (history_window_count, feature_dim):
                raise ValueError("All history pattern matrices must share the same shape.")
            if sample.target_pattern_matrix.shape != (forecast_window_count, feature_dim):
                raise ValueError("All target pattern matrices must share the same shape.")
            if sample.history_vector.shape != (context_dim,):
                raise ValueError("All history vectors must share the same shape.")

        self._build_model(
            history_window_count=history_window_count,
            feature_dim=feature_dim,
            forecast_window_count=forecast_window_count,
            context_dim=context_dim,
        )
        self._history_window_count = history_window_count
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )
        loss_fn = self._nn.MSELoss()

        sample_count = len(samples)
        history_sum = np.zeros((feature_dim,), dtype=np.float64)
        history_sum_sq = np.zeros((feature_dim,), dtype=np.float64)
        context_sum = np.zeros((context_dim,), dtype=np.float64)
        context_sum_sq = np.zeros((context_dim,), dtype=np.float64)
        target_sum = np.zeros((feature_dim,), dtype=np.float64)
        target_sum_sq = np.zeros((feature_dim,), dtype=np.float64)

        for sample in samples:
            history = sample.history_pattern_matrix.astype(np.float64, copy=False)
            context = sample.history_vector.astype(np.float64, copy=False)
            target = sample.target_pattern_matrix.astype(np.float64, copy=False)
            history_sum += history.sum(axis=0)
            history_sum_sq += np.square(history).sum(axis=0)
            context_sum += context
            context_sum_sq += np.square(context)
            target_sum += target.sum(axis=0)
            target_sum_sq += np.square(target).sum(axis=0)

        history_count = sample_count * history_window_count
        target_count = sample_count * forecast_window_count
        self._input_mean = (history_sum / history_count).reshape(1, 1, feature_dim).astype(np.float32)
        input_var = (history_sum_sq / history_count) - np.square(history_sum / history_count)
        self._input_std = np.sqrt(np.maximum(input_var, 1e-12)).reshape(1, 1, feature_dim).astype(np.float32)
        self._input_std = np.where(self._input_std > 1e-6, self._input_std, 1.0).astype(np.float32)

        self._context_mean = (context_sum / sample_count).reshape(1, context_dim).astype(np.float32)
        context_var = (context_sum_sq / sample_count) - np.square(context_sum / sample_count)
        self._context_std = np.sqrt(np.maximum(context_var, 1e-12)).reshape(1, context_dim).astype(np.float32)
        self._context_std = np.where(self._context_std > 1e-6, self._context_std, 1.0).astype(np.float32)

        self._target_mean = (target_sum / target_count).reshape(1, 1, feature_dim).astype(np.float32)
        target_var = (target_sum_sq / target_count) - np.square(target_sum / target_count)
        self._target_std = np.sqrt(np.maximum(target_var, 1e-12)).reshape(1, 1, feature_dim).astype(np.float32)
        self._target_std = np.where(self._target_std > 1e-6, self._target_std, 1.0).astype(np.float32)

        batch_size = min(self.model_config.batch_size, sample_count)
        batch_count = (sample_count + batch_size - 1) // batch_size
        for epoch_index in range(self.model_config.epochs):
            epoch_started_at = time.perf_counter()
            permutation = np.random.permutation(sample_count)
            for batch_number, start in enumerate(range(0, sample_count, batch_size), start=1):
                batch_samples = [samples[index] for index in permutation[start : start + batch_size]]
                self._fit_sample_batch(
                    batch_samples=batch_samples,
                    forecast_window_count=forecast_window_count,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    mode="in_memory_samples",
                    epoch_index=epoch_index + 1,
                    total_epochs=self.model_config.epochs,
                    batch_number=batch_number,
                    total_batches=batch_count,
                    log_interval_batches=log_interval_batches,
                )
            _log(
                self._logger,
                "sequence_fit_epoch_end",
                mode="in_memory_samples",
                epoch=epoch_index + 1,
                total_epochs=self.model_config.epochs,
                elapsed_seconds=round(time.perf_counter() - epoch_started_at, 3),
            )

    def fit_sample_iterator(
        self,
        sample_iter_factory: Callable[[], Iterable[ForecastSample]],
        sample_count: int,
        logger: logging.Logger | None = None,
        log_interval_batches: int = 50,
    ) -> None:
        if sample_count <= 0:
            raise ValueError("At least one forecast sample is required to fit the predictor.")
        self.set_logger(logger)

        self._lazy_import_torch()
        torch = self._torch
        torch.manual_seed(self.model_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.model_config.random_seed)

        first_sample: ForecastSample | None = None
        history_sum: np.ndarray | None = None
        history_sum_sq: np.ndarray | None = None
        context_sum: np.ndarray | None = None
        context_sum_sq: np.ndarray | None = None
        target_sum: np.ndarray | None = None
        target_sum_sq: np.ndarray | None = None

        for sample in sample_iter_factory():
            if first_sample is None:
                first_sample = sample
                history_window_count = sample.history_pattern_matrix.shape[0]
                feature_dim = sample.history_pattern_matrix.shape[1]
                forecast_window_count = sample.target_pattern_matrix.shape[0]
                context_dim = sample.history_vector.shape[0]
                history_sum = np.zeros((feature_dim,), dtype=np.float64)
                history_sum_sq = np.zeros((feature_dim,), dtype=np.float64)
                context_sum = np.zeros((context_dim,), dtype=np.float64)
                context_sum_sq = np.zeros((context_dim,), dtype=np.float64)
                target_sum = np.zeros((feature_dim,), dtype=np.float64)
                target_sum_sq = np.zeros((feature_dim,), dtype=np.float64)
            if sample.history_pattern_matrix.shape != (history_window_count, feature_dim):
                raise ValueError("All history pattern matrices must share the same shape.")
            if sample.target_pattern_matrix.shape != (forecast_window_count, feature_dim):
                raise ValueError("All target pattern matrices must share the same shape.")
            if sample.history_vector.shape != (context_dim,):
                raise ValueError("All history vectors must share the same shape.")

            history = sample.history_pattern_matrix.astype(np.float64, copy=False)
            context = sample.history_vector.astype(np.float64, copy=False)
            target = sample.target_pattern_matrix.astype(np.float64, copy=False)
            history_sum += history.sum(axis=0)
            history_sum_sq += np.square(history).sum(axis=0)
            context_sum += context
            context_sum_sq += np.square(context)
            target_sum += target.sum(axis=0)
            target_sum_sq += np.square(target).sum(axis=0)

        if first_sample is None:
            raise ValueError("At least one forecast sample is required to fit the predictor.")

        self._build_model(
            history_window_count=history_window_count,
            feature_dim=feature_dim,
            forecast_window_count=forecast_window_count,
            context_dim=context_dim,
        )
        self._history_window_count = history_window_count
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )
        loss_fn = self._nn.MSELoss()

        history_count = sample_count * history_window_count
        target_count = sample_count * forecast_window_count
        self._input_mean = (history_sum / history_count).reshape(1, 1, feature_dim).astype(np.float32)
        input_var = (history_sum_sq / history_count) - np.square(history_sum / history_count)
        self._input_std = np.sqrt(np.maximum(input_var, 1e-12)).reshape(1, 1, feature_dim).astype(np.float32)
        self._input_std = np.where(self._input_std > 1e-6, self._input_std, 1.0).astype(np.float32)

        self._context_mean = (context_sum / sample_count).reshape(1, context_dim).astype(np.float32)
        context_var = (context_sum_sq / sample_count) - np.square(context_sum / sample_count)
        self._context_std = np.sqrt(np.maximum(context_var, 1e-12)).reshape(1, context_dim).astype(np.float32)
        self._context_std = np.where(self._context_std > 1e-6, self._context_std, 1.0).astype(np.float32)

        self._target_mean = (target_sum / target_count).reshape(1, 1, feature_dim).astype(np.float32)
        target_var = (target_sum_sq / target_count) - np.square(target_sum / target_count)
        self._target_std = np.sqrt(np.maximum(target_var, 1e-12)).reshape(1, 1, feature_dim).astype(np.float32)
        self._target_std = np.where(self._target_std > 1e-6, self._target_std, 1.0).astype(np.float32)

        batch_size = min(self.model_config.batch_size, sample_count)
        batch_count = (sample_count + batch_size - 1) // batch_size
        for epoch_index in range(self.model_config.epochs):
            epoch_started_at = time.perf_counter()
            batch_samples: list[ForecastSample] = []
            batch_number = 0
            for sample in sample_iter_factory():
                batch_samples.append(sample)
                if len(batch_samples) < batch_size:
                    continue
                batch_number += 1
                self._fit_sample_batch(
                    batch_samples=batch_samples,
                    forecast_window_count=forecast_window_count,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    mode="streaming",
                    epoch_index=epoch_index + 1,
                    total_epochs=self.model_config.epochs,
                    batch_number=batch_number,
                    total_batches=batch_count,
                    log_interval_batches=log_interval_batches,
                )
                batch_samples = []
            if batch_samples:
                batch_number += 1
                self._fit_sample_batch(
                    batch_samples=batch_samples,
                    forecast_window_count=forecast_window_count,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    mode="streaming",
                    epoch_index=epoch_index + 1,
                    total_epochs=self.model_config.epochs,
                    batch_number=batch_number,
                    total_batches=batch_count,
                    log_interval_batches=log_interval_batches,
                )
            _log(
                self._logger,
                "sequence_fit_epoch_end",
                mode="streaming",
                epoch=epoch_index + 1,
                total_epochs=self.model_config.epochs,
                elapsed_seconds=round(time.perf_counter() - epoch_started_at, 3),
            )

    def _fit_sample_batch(
        self,
        batch_samples: list[ForecastSample],
        forecast_window_count: int,
        optimizer,
        loss_fn,
        mode: str,
        epoch_index: int,
        total_epochs: int,
        batch_number: int,
        total_batches: int,
        log_interval_batches: int,
    ) -> None:
        batch_started_at = time.perf_counter()
        torch = self._torch
        batch_history_np = np.stack(
            [sample.history_pattern_matrix for sample in batch_samples],
            axis=0,
        ).astype(np.float32, copy=False)
        batch_context_np = np.stack(
            [sample.history_vector for sample in batch_samples],
            axis=0,
        ).astype(np.float32, copy=False)
        batch_target_np = np.stack(
            [sample.target_pattern_matrix for sample in batch_samples],
            axis=0,
        ).astype(np.float32, copy=False)
        batch_baseline_np = (
            batch_history_np[:, -1:, :] - self._target_mean
        ) / self._target_std
        batch_baseline_np = np.repeat(
            batch_baseline_np,
            forecast_window_count,
            axis=1,
        )

        batch_history = torch.as_tensor(
            (batch_history_np - self._input_mean) / self._input_std,
            dtype=torch.float32,
            device=self.device,
        )
        batch_context = torch.as_tensor(
            (batch_context_np - self._context_mean) / self._context_std,
            dtype=torch.float32,
            device=self.device,
        )
        batch_target = torch.as_tensor(
            (batch_target_np - self._target_mean) / self._target_std,
            dtype=torch.float32,
            device=self.device,
        )
        batch_baseline = torch.as_tensor(
            batch_baseline_np,
            dtype=torch.float32,
            device=self.device,
        )

        optimizer.zero_grad(set_to_none=True)
        predicted = self._model(batch_history, batch_context, batch_baseline)
        loss = loss_fn(predicted, batch_target)
        loss.backward()
        optimizer.step()
        batch_elapsed_ms = round((time.perf_counter() - batch_started_at) * 1000.0, 2)
        if (
            batch_number == 1
            or batch_number == total_batches
            or batch_number % log_interval_batches == 0
        ):
            _log(
                self._logger,
                "sequence_fit_batch",
                mode=mode,
                epoch=epoch_index,
                total_epochs=total_epochs,
                batch=batch_number,
                total_batches=total_batches,
                batch_size=len(batch_samples),
                loss=round(float(loss.detach().item()), 6),
                elapsed_ms=batch_elapsed_ms,
            )
        self._enforce_rss_limit(
            "fit_batch",
            mode=mode,
            epoch=epoch_index,
            batch=batch_number,
            total_batches=total_batches,
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        if self._model is None or self._feature_dim is None or self._forecast_window_count is None:
            raise RuntimeError("The predictor must be trained before it can be saved.")
        if self._history_window_count is None:
            raise RuntimeError("Missing history window count for checkpoint export.")
        if (
            self._input_mean is None
            or self._input_std is None
            or self._context_mean is None
            or self._context_std is None
            or self._target_mean is None
            or self._target_std is None
        ):
            raise RuntimeError("Missing normalization statistics for checkpoint export.")

        self._lazy_import_torch()
        torch = self._torch
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "feature_dim": self._feature_dim,
                "forecast_window_count": self._forecast_window_count,
                "history_window_count": self._history_window_count,
                "input_mean": self._input_mean,
                "input_std": self._input_std,
                "context_mean": self._context_mean,
                "context_std": self._context_std,
                "target_mean": self._target_mean,
                "target_std": self._target_std,
                "context_dim": int(self._context_mean.shape[1]),
                "model_config": {
                    "hidden_size": self.model_config.hidden_size,
                    "num_layers": self.model_config.num_layers,
                    "dropout": self.model_config.dropout,
                    "learning_rate": self.model_config.learning_rate,
                    "weight_decay": self.model_config.weight_decay,
                    "batch_size": self.model_config.batch_size,
                    "epochs": self.model_config.epochs,
                    "random_seed": self.model_config.random_seed,
                },
            },
            destination,
        )
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model_config: SequenceModelConfig,
        compute_config: ComputeConfig,
    ) -> "TorchSequencePredictor":
        predictor = cls(model_config=model_config, compute_config=compute_config)
        predictor._lazy_import_torch()
        torch = predictor._torch
        checkpoint = torch.load(
            Path(path),
            map_location=predictor.device,
            weights_only=False,
        )
        predictor._build_model(
            history_window_count=int(checkpoint["history_window_count"]),
            feature_dim=int(checkpoint["feature_dim"]),
            forecast_window_count=int(checkpoint["forecast_window_count"]),
            context_dim=int(checkpoint["context_dim"]),
        )
        predictor._model.load_state_dict(checkpoint["state_dict"])
        predictor._history_window_count = int(checkpoint["history_window_count"])
        predictor._input_mean = np.asarray(checkpoint["input_mean"], dtype=np.float32)
        predictor._input_std = np.asarray(checkpoint["input_std"], dtype=np.float32)
        predictor._context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float32)
        predictor._context_std = np.asarray(checkpoint["context_std"], dtype=np.float32)
        predictor._target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        predictor._target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
        predictor._model.eval()
        return predictor

    def predict(
        self,
        history_pattern_matrix: np.ndarray,
        history_vector: np.ndarray,
        forecast_time: pd.Timestamp,
        horizon_steps: int,
        prototypes: list[PatternPrototype] | None = None,
    ) -> ForecastResult:
        predict_started_at = time.perf_counter()
        if self._model is None or self._feature_dim is None or self._forecast_window_count is None:
            raise RuntimeError("The predictor must be trained before inference.")
        if self._history_window_count is None:
            raise RuntimeError("Missing history window count for inference.")
        if (
            self._input_mean is None
            or self._input_std is None
            or self._context_mean is None
            or self._context_std is None
            or self._target_mean is None
            or self._target_std is None
        ):
            raise RuntimeError("The predictor must be trained before inference normalization is available.")
        if history_pattern_matrix.shape[1] != self._feature_dim:
            raise ValueError("Unexpected history pattern feature dimension.")
        if history_pattern_matrix.shape[0] != self._history_window_count:
            raise ValueError("Unexpected history sequence length.")

        self._lazy_import_torch()
        torch = self._torch
        normalized_history = (history_pattern_matrix.astype(np.float32) - self._input_mean[0]) / self._input_std[0]
        normalized_context = (history_vector.astype(np.float32)[None, :] - self._context_mean) / self._context_std
        baseline_target = ((history_pattern_matrix[-1:, :] - self._target_mean[0]) / self._target_std[0]).astype(np.float32)
        baseline_target = np.repeat(baseline_target[None, :, :], self._forecast_window_count, axis=1)
        history_tensor = torch.as_tensor(
            normalized_history[None, :, :],
            dtype=torch.float32,
            device=self.device,
        )
        context_tensor = torch.as_tensor(
            normalized_context,
            dtype=torch.float32,
            device=self.device,
        )
        baseline_tensor = torch.as_tensor(
            baseline_target,
            dtype=torch.float32,
            device=self.device,
        )
        self._model.eval()
        with torch.no_grad():
            predicted_pattern_tensor = self._model(history_tensor, context_tensor, baseline_tensor)
        predicted_pattern_matrix = predicted_pattern_tensor[0].detach().cpu().numpy()
        predicted_pattern_matrix = predicted_pattern_matrix * self._target_std[0] + self._target_mean[0]
        predicted_pattern_ids = _nearest_pattern_ids(predicted_pattern_matrix, prototypes)
        self._predict_call_count += 1
        predict_elapsed_ms = round((time.perf_counter() - predict_started_at) * 1000.0, 2)
        if (
            self._predict_call_count <= self._predict_log_first_n
            or self._predict_call_count % self._predict_log_every_n == 0
        ):
            _log(
                self._logger,
                "sequence_predict_call",
                predict_call=self._predict_call_count,
                forecast_time=forecast_time.isoformat(),
                horizon_steps=horizon_steps,
                elapsed_ms=predict_elapsed_ms,
                device=self.device,
            )
        self._enforce_rss_limit(
            "predict_call",
            predict_call=self._predict_call_count,
            forecast_time=forecast_time.isoformat(),
            elapsed_ms=predict_elapsed_ms,
            device=self.device,
        )

        return ForecastResult(
            forecast_time=forecast_time,
            horizon_steps=horizon_steps,
            predicted_window_count=self._forecast_window_count,
            predicted_pattern_ids=predicted_pattern_ids,
            predicted_pattern_matrix=predicted_pattern_matrix,
            predicted_timestamps=[],
            predicted_values={},
            predicted_interval_timestamps=[],
            predicted_interval_values={},
            predicted_time_placeholders=[],
            predicted_peak_hazard=[],
            predicted_interval_time_placeholders=[],
            predicted_interval_peak_hazard=[],
        )
