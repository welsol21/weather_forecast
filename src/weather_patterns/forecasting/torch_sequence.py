from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from weather_patterns.config import ComputeConfig, SequenceModelConfig
from weather_patterns.forecasting.base import SequencePredictor
from weather_patterns.forecasting.runtime import resolve_model_device
from weather_patterns.models import ForecastResult, ForecastTrainingDataset, PatternPrototype


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

    def _lazy_import_torch(self) -> None:
        if self._torch is None:
            import torch
            import torch.nn as nn

            self._torch = torch
            self._nn = nn

    def _build_model(self, history_window_count: int, feature_dim: int, forecast_window_count: int) -> None:
        self._lazy_import_torch()
        nn = self._nn

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
                self.head = nn.Sequential(
                    nn.LayerNorm(self_model_config.hidden_size),
                    nn.Linear(
                        self_model_config.hidden_size,
                        forecast_window_count * feature_dim,
                    ),
                )

            def forward(self, history_pattern_tensor):  # type: ignore[no-untyped-def]
                _, hidden = self.encoder(history_pattern_tensor)
                encoded = hidden[-1]
                output = self.head(encoded)
                return output.view(-1, forecast_window_count, feature_dim)

        self_model_config = self.model_config
        self._model = SequenceRegressor().to(self.device)
        self._feature_dim = feature_dim
        self._forecast_window_count = forecast_window_count

    def fit(self, dataset: ForecastTrainingDataset) -> None:
        self._lazy_import_torch()
        torch = self._torch
        torch.manual_seed(self.model_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.model_config.random_seed)

        self._build_model(
            history_window_count=dataset.history_window_count,
            feature_dim=dataset.feature_dim,
            forecast_window_count=dataset.forecast_window_count,
        )
        self._history_window_count = dataset.history_window_count
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )
        loss_fn = self._nn.MSELoss()

        history_pattern_tensor = torch.as_tensor(
            dataset.history_pattern_tensor,
            dtype=torch.float32,
            device=self.device,
        )
        target_pattern_tensor = torch.as_tensor(
            dataset.target_pattern_tensor,
            dtype=torch.float32,
            device=self.device,
        )

        sample_count = history_pattern_tensor.shape[0]
        batch_size = min(self.model_config.batch_size, sample_count)
        for _ in range(self.model_config.epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_history = history_pattern_tensor[batch_indices]
                batch_target = target_pattern_tensor[batch_indices]

                optimizer.zero_grad(set_to_none=True)
                predicted = self._model(batch_history)
                loss = loss_fn(predicted, batch_target)
                loss.backward()
                optimizer.step()

    def save_checkpoint(self, path: str | Path) -> Path:
        if self._model is None or self._feature_dim is None or self._forecast_window_count is None:
            raise RuntimeError("The predictor must be trained before it can be saved.")
        if self._history_window_count is None:
            raise RuntimeError("Missing history window count for checkpoint export.")

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
        checkpoint = torch.load(Path(path), map_location=predictor.device)
        predictor._build_model(
            history_window_count=int(checkpoint["history_window_count"]),
            feature_dim=int(checkpoint["feature_dim"]),
            forecast_window_count=int(checkpoint["forecast_window_count"]),
        )
        predictor._model.load_state_dict(checkpoint["state_dict"])
        predictor._history_window_count = int(checkpoint["history_window_count"])
        predictor._model.eval()
        return predictor

    def predict(
        self,
        history_pattern_matrix: np.ndarray,
        forecast_time: pd.Timestamp,
        horizon_steps: int,
        prototypes: list[PatternPrototype] | None = None,
    ) -> ForecastResult:
        if self._model is None or self._feature_dim is None or self._forecast_window_count is None:
            raise RuntimeError("The predictor must be trained before inference.")
        if self._history_window_count is None:
            raise RuntimeError("Missing history window count for inference.")
        if history_pattern_matrix.shape[1] != self._feature_dim:
            raise ValueError("Unexpected history pattern feature dimension.")
        if history_pattern_matrix.shape[0] != self._history_window_count:
            raise ValueError("Unexpected history sequence length.")

        self._lazy_import_torch()
        torch = self._torch
        history_tensor = torch.as_tensor(
            history_pattern_matrix[None, :, :],
            dtype=torch.float32,
            device=self.device,
        )
        self._model.eval()
        with torch.no_grad():
            predicted_pattern_tensor = self._model(history_tensor)
        predicted_pattern_matrix = predicted_pattern_tensor[0].detach().cpu().numpy()
        predicted_pattern_ids = _nearest_pattern_ids(predicted_pattern_matrix, prototypes)

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
