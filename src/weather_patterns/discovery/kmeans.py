from __future__ import annotations

import numpy as np

from weather_patterns.config import DiscoveryConfig
from weather_patterns.discovery.base import DiscoveryInput, PatternDiscoveryStrategy
from weather_patterns.models import DiscoveryResult, PatternPrototype


class NumpyKMeansDiscovery(PatternDiscoveryStrategy):
    def __init__(self, config: DiscoveryConfig) -> None:
        self.config = config

    def fit_predict(self, data: DiscoveryInput) -> DiscoveryResult:
        matrix = data.feature_matrix
        if len(matrix) == 0:
            return DiscoveryResult(labels_by_window_id={}, prototypes=[])

        k = min(self.config.n_clusters, len(matrix))
        rng = np.random.default_rng(self.config.random_seed)
        centroids = matrix[rng.choice(len(matrix), size=k, replace=False)].copy()
        labels = np.zeros(len(matrix), dtype=int)

        for _ in range(self.config.max_iterations):
            distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster_id in range(k):
                members = matrix[labels == cluster_id]
                if len(members) == 0:
                    centroids[cluster_id] = matrix[rng.integers(0, len(matrix))]
                else:
                    centroids[cluster_id] = members.mean(axis=0)

        labels_by_window_id = {
            window_id: int(label)
            for window_id, label in zip(data.window_ids, labels, strict=False)
        }
        prototypes = [
            PatternPrototype(
                pattern_id=cluster_id,
                centroid=centroids[cluster_id],
                member_window_ids=[
                    window_id
                    for window_id, label in labels_by_window_id.items()
                    if label == cluster_id
                ],
            )
            for cluster_id in range(k)
        ]
        return DiscoveryResult(labels_by_window_id=labels_by_window_id, prototypes=prototypes)
