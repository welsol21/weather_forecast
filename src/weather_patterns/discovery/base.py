from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from weather_patterns.models import DiscoveryResult


@dataclass(slots=True)
class DiscoveryInput:
    window_ids: list[int]
    feature_matrix: np.ndarray


class PatternDiscoveryStrategy(Protocol):
    def fit_predict(self, data: DiscoveryInput) -> DiscoveryResult:
        ...
