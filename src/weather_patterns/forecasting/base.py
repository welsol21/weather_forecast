from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from weather_patterns.models import ForecastResult, ForecastTrainingDataset, PatternPrototype


class SequencePredictor(ABC):
    @abstractmethod
    def fit(self, dataset: ForecastTrainingDataset) -> None:
        """Train the predictor on sequence-structured pattern data."""

    @abstractmethod
    def predict(
        self,
        history_pattern_matrix: np.ndarray,
        forecast_time: pd.Timestamp,
        horizon_steps: int,
        prototypes: list[PatternPrototype] | None = None,
    ) -> ForecastResult:
        """Predict a future pattern sequence for the requested horizon."""
