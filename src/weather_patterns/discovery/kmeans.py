from __future__ import annotations

import numpy as np

from weather_patterns.config import DiscoveryConfig
from weather_patterns.discovery.base import DiscoveryInput, PatternDiscoveryStrategy
from weather_patterns.models import DiscoveryResult, PatternPrototype


class NumpyKMeansDiscovery(PatternDiscoveryStrategy):
    def __init__(self, config: DiscoveryConfig) -> None:
        self.config = config

    def _fit_kmeans(self, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.random_seed)
        centroids = matrix[rng.choice(len(matrix), size=k, replace=False)].copy()
        labels = np.full(len(matrix), -1, dtype=int)

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
        return labels, centroids

    def _candidate_cluster_counts(self, row_count: int) -> list[int]:
        if not self.config.auto_select_k:
            return [min(self.config.n_clusters, row_count)]

        lower = max(2, min(self.config.min_clusters, row_count))
        upper = max(lower, min(self.config.max_clusters, row_count))
        step = max(1, self.config.cluster_step)
        candidates = list(range(lower, upper + 1, step))
        if upper not in candidates:
            candidates.append(upper)
        return sorted(set(candidates))

    def _cluster_quality_score(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> float:
        if len(matrix) <= 1 or len(centroids) <= 1:
            return 0.0

        sample_size = min(len(matrix), max(64, self.config.quality_sample_size))
        rng = np.random.default_rng(self.config.random_seed)
        sample_indices = (
            rng.choice(len(matrix), size=sample_size, replace=False)
            if sample_size < len(matrix)
            else np.arange(len(matrix))
        )
        sample_points = matrix[sample_indices]
        sample_labels = labels[sample_indices]

        distances = np.linalg.norm(
            sample_points[:, None, :] - centroids[None, :, :],
            axis=2,
        )
        own_distance = distances[np.arange(len(sample_points)), sample_labels]
        masked = distances.copy()
        masked[np.arange(len(sample_points)), sample_labels] = np.inf
        nearest_other_distance = masked.min(axis=1)
        silhouette_like = (nearest_other_distance - own_distance) / np.maximum(
            np.maximum(nearest_other_distance, own_distance),
            1e-9,
        )

        cluster_sizes = np.bincount(labels, minlength=len(centroids)).astype(float)
        nonzero_sizes = cluster_sizes[cluster_sizes > 0]
        if len(nonzero_sizes) == 0:
            return float("-inf")
        min_cluster_share = nonzero_sizes.min() / len(labels)
        balance_penalty = min_cluster_share / (1.0 / len(centroids))
        return float(np.mean(silhouette_like) * balance_penalty)

    def fit_predict(self, data: DiscoveryInput) -> DiscoveryResult:
        matrix = data.feature_matrix
        if len(matrix) == 0:
            return DiscoveryResult(
                labels_by_window_id={},
                prototypes=[],
                strategy="kmeans",
                selected_cluster_count=0,
                selected_quality_score=0.0,
                selection_metric_name="cluster_quality_by_k",
                candidate_quality={},
            )

        candidate_quality_by_k: dict[int, float] = {}
        best_labels: np.ndarray | None = None
        best_centroids: np.ndarray | None = None
        best_quality = float("-inf")
        best_k = 0

        for k in self._candidate_cluster_counts(len(matrix)):
            labels, centroids = self._fit_kmeans(matrix, k)
            quality = self._cluster_quality_score(matrix, labels, centroids)
            candidate_quality_by_k[int(k)] = float(quality)
            if quality > best_quality:
                best_quality = quality
                best_labels = labels
                best_centroids = centroids
                best_k = k

        if best_labels is None or best_centroids is None:
            raise RuntimeError("Pattern discovery failed to select a valid cluster configuration.")

        labels_by_window_id = {
            window_id: int(label)
            for window_id, label in zip(data.window_ids, best_labels, strict=False)
        }
        prototypes = [
            PatternPrototype(
                pattern_id=cluster_id,
                centroid=best_centroids[cluster_id],
                member_window_ids=[
                    window_id
                    for window_id, label in labels_by_window_id.items()
                    if label == cluster_id
                ],
            )
            for cluster_id in range(best_k)
        ]
        return DiscoveryResult(
            labels_by_window_id=labels_by_window_id,
            prototypes=prototypes,
            strategy="kmeans",
            selected_cluster_count=int(best_k),
            selected_quality_score=float(best_quality),
            selection_metric_name="cluster_quality_by_k",
            candidate_quality={str(key): value for key, value in candidate_quality_by_k.items()},
        )
