"""K-medoids clustering with functional integral distance for Run 8.

Distance between two pattern windows:
    d(p1, p2) = Σ_channels ∫₀¹ (f1_c(t) - f2_c(t))² dt

where each f_c is the ODE solution for that channel, evaluated on [0, 1].

The full pairwise distance matrix is computed once, then k-medoids iterates
on the precomputed matrix.  This is O(N²) in memory but O(k·N) per iteration.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from weather_patterns.config import DiscoveryConfig
from weather_patterns.discovery.base import DiscoveryInput
from weather_patterns.models import DiscoveryResult, PatternPrototype
from weather_patterns.pattern.convergence import (
    DIMS_PER_CHANNEL,
    DIMS_STRUCTURAL,
    pattern_distance,
)

logger = logging.getLogger(__name__)


def _build_distance_matrix(
    feature_matrix: np.ndarray,
    n_channels: int,
) -> np.ndarray:
    """Compute symmetric N×N pairwise distance matrix."""
    n = len(feature_matrix)
    dist = np.zeros((n, n), dtype=float)
    t0 = time.perf_counter()
    total_pairs = n * (n - 1) // 2
    done = 0
    log_every = max(1, total_pairs // 20)

    for i in range(n):
        for j in range(i + 1, n):
            d = pattern_distance(feature_matrix[i], feature_matrix[j], n_channels)
            dist[i, j] = d
            dist[j, i] = d
            done += 1
            if done % log_every == 0:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total_pairs - done) / rate if rate > 0 else 0
                logger.info(
                    "distance_matrix progress=%d/%d (%.1f%%) elapsed_s=%.0f remaining_s=%.0f",
                    done, total_pairs, 100 * done / total_pairs, elapsed, remaining,
                )

    logger.info(
        "distance_matrix_complete n=%d elapsed_s=%.1f",
        n, time.perf_counter() - t0,
    )
    return dist


def _fit_kmedoids(
    dist: np.ndarray,
    k: int,
    max_iterations: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """K-medoids on precomputed distance matrix.

    Returns (labels, medoid_indices) both of length N and k respectively.
    """
    n = len(dist)
    rng = np.random.default_rng(random_seed)
    medoid_indices = rng.choice(n, size=k, replace=False)
    labels = np.full(n, -1, dtype=int)

    for iteration in range(max_iterations):
        # Assignment step
        dists_to_medoids = dist[:, medoid_indices]   # N × k
        new_labels = dists_to_medoids.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            logger.info("kmedoids_converged k=%d iteration=%d", k, iteration)
            break
        labels = new_labels

        # Update step: for each cluster, pick the point minimising total distance
        new_medoids = medoid_indices.copy()
        for cluster_id in range(k):
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                new_medoids[cluster_id] = rng.integers(0, n)
                continue
            # Total distance from each member to all other members
            sub_dist = dist[np.ix_(members, members)]
            total_dist = sub_dist.sum(axis=1)
            new_medoids[cluster_id] = members[total_dist.argmin()]
        medoid_indices = new_medoids

    return labels, medoid_indices


def _silhouette_quality(
    dist: np.ndarray,
    labels: np.ndarray,
    medoid_indices: np.ndarray,
    sample_size: int,
    random_seed: int,
) -> float:
    """Silhouette-like quality score on the precomputed distance matrix."""
    n = len(dist)
    k = len(medoid_indices)
    if n <= 1 or k <= 1:
        return 0.0

    rng = np.random.default_rng(random_seed)
    idx = (
        rng.choice(n, size=min(sample_size, n), replace=False)
        if sample_size < n
        else np.arange(n)
    )

    scores = []
    for i in idx:
        own_cluster = labels[i]
        own_members = np.where(labels == own_cluster)[0]
        a = float(dist[i, own_members].mean()) if len(own_members) > 1 else 0.0

        b = np.inf
        for c in range(k):
            if c == own_cluster:
                continue
            other_members = np.where(labels == c)[0]
            if len(other_members) == 0:
                continue
            b = min(b, float(dist[i, other_members].mean()))

        if b == np.inf:
            scores.append(0.0)
        else:
            denom = max(a, b, 1e-9)
            scores.append((b - a) / denom)

    if not scores:
        return 0.0

    cluster_sizes = np.bincount(labels, minlength=k).astype(float)
    nonzero = cluster_sizes[cluster_sizes > 0]
    balance = (nonzero.min() / len(labels)) / (1.0 / k) if len(nonzero) > 0 else 0.0
    return float(np.mean(scores) * balance)


class KMedoidsDiscovery:
    """K-medoids clustering using functional integral distance."""

    def __init__(self, config: DiscoveryConfig, n_channels: int) -> None:
        self.config = config
        self.n_channels = n_channels

    def _candidate_ks(self, n: int) -> list[int]:
        if not self.config.auto_select_k:
            return [min(self.config.n_clusters, n)]
        lo = max(2, min(self.config.min_clusters, n))
        hi = max(lo, min(self.config.max_clusters, n))
        step = max(1, self.config.cluster_step)
        candidates = list(range(lo, hi + 1, step))
        if hi not in candidates:
            candidates.append(hi)
        return sorted(set(candidates))

    def fit_predict(self, data: DiscoveryInput) -> DiscoveryResult:
        matrix = data.feature_matrix
        n = len(matrix)

        if n == 0:
            return DiscoveryResult(
                labels_by_window_id={},
                prototypes=[],
                strategy="kmedoids",
                selected_cluster_count=0,
                selected_quality_score=0.0,
                selection_metric_name="silhouette_kmedoids",
                candidate_quality={},
            )

        logger.info("kmedoids_start n=%d n_channels=%d", n, self.n_channels)
        dist = _build_distance_matrix(matrix, self.n_channels)

        best_labels: np.ndarray | None = None
        best_medoids: np.ndarray | None = None
        best_quality = float("-inf")
        best_k = 0
        candidate_quality: dict[int, float] = {}

        for k in self._candidate_ks(n):
            t0 = time.perf_counter()
            labels, medoids = _fit_kmedoids(
                dist, k,
                max_iterations=self.config.max_iterations,
                random_seed=self.config.random_seed,
            )
            quality = _silhouette_quality(
                dist, labels, medoids,
                sample_size=self.config.quality_sample_size,
                random_seed=self.config.random_seed,
            )
            candidate_quality[k] = quality
            logger.info(
                "kmedoids_k k=%d quality=%.4f elapsed_ms=%.0f",
                k, quality, (time.perf_counter() - t0) * 1000,
            )
            if quality > best_quality:
                best_quality = quality
                best_labels = labels
                best_medoids = medoids
                best_k = k

        if best_labels is None or best_medoids is None:
            raise RuntimeError("K-medoids discovery failed to select a valid k.")

        labels_by_window_id = {
            wid: int(lbl)
            for wid, lbl in zip(data.window_ids, best_labels, strict=False)
        }
        prototypes = []
        for cluster_id in range(best_k):
            medoid_idx = int(best_medoids[cluster_id])
            member_ids = [
                wid for wid, lbl in labels_by_window_id.items()
                if lbl == cluster_id
            ]
            prototypes.append(PatternPrototype(
                pattern_id=cluster_id,
                centroid=matrix[medoid_idx].copy(),
                member_window_ids=member_ids,
                metadata={"medoid_window_id": int(data.window_ids[medoid_idx])},
            ))

        logger.info(
            "kmedoids_complete best_k=%d best_quality=%.4f",
            best_k, best_quality,
        )
        return DiscoveryResult(
            labels_by_window_id=labels_by_window_id,
            prototypes=prototypes,
            strategy="kmedoids",
            selected_cluster_count=best_k,
            selected_quality_score=float(best_quality),
            selection_metric_name="silhouette_kmedoids",
            candidate_quality={str(k): v for k, v in candidate_quality.items()},
        )
