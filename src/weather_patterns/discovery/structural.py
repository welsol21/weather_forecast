from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weather_patterns.config import DiscoveryConfig
from weather_patterns.discovery.base import DiscoveryInput, PatternDiscoveryStrategy
from weather_patterns.forecasting.runtime import resolve_model_device
from weather_patterns.models import DiscoveryResult, PatternPrototype, PatternWindow
from weather_patterns.pattern.representation import (
    INTER_FEATURES,
    INTRA_FEATURES,
    PEAK_HAZARD_FEATURES,
)


def _resample_positions(values: list[float], target_size: int = 8) -> np.ndarray:
    if target_size <= 0:
        return np.empty(0, dtype=float)
    if not values:
        return np.zeros(target_size, dtype=float)
    if len(values) == 1:
        return np.full(target_size, float(values[0]), dtype=float)
    source_x = np.linspace(0.0, 1.0, num=len(values))
    target_x = np.linspace(0.0, 1.0, num=target_size)
    return np.interp(target_x, source_x, np.asarray(values, dtype=float)).astype(float)


def _safe_divide(value: float, scale: float) -> float:
    if np.isclose(scale, 0.0):
        return 0.0
    return float(value / scale)


def _structure_vector(window: PatternWindow) -> np.ndarray:
    feature_index = {name: index for index, name in enumerate(INTRA_FEATURES)}
    inter_index = {name: index for index, name in enumerate(INTER_FEATURES)}
    peak_index = {name: index for index, name in enumerate(PEAK_HAZARD_FEATURES)}
    window_length = max(float(window.time_placeholders.window_length_steps), 1.0)
    event_count = max(len(window.extrema_window.events), 1)

    parts: list[np.ndarray] = []
    channel_ranges = np.abs(window.intra_matrix[feature_index["range_on_window"], :])
    channel_scales = np.where(channel_ranges > 1e-6, channel_ranges, 1.0)

    intra_rows = [
        "delta_1",
        "delta_6",
        "delta_24",
        "second_diff",
        "variance_on_window",
        "extrema_count",
        "extrema_amplitude_sum",
    ]
    intra_block: list[float] = []
    for channel_index, scale in enumerate(channel_scales):
        intra_block.extend(
            [
                _safe_divide(window.intra_matrix[feature_index["delta_1"], channel_index], scale),
                _safe_divide(window.intra_matrix[feature_index["delta_6"], channel_index], scale),
                _safe_divide(window.intra_matrix[feature_index["delta_24"], channel_index], scale),
                _safe_divide(window.intra_matrix[feature_index["second_diff"], channel_index], scale),
                _safe_divide(window.intra_matrix[feature_index["variance_on_window"], channel_index], scale * scale),
                _safe_divide(window.intra_matrix[feature_index["extrema_count"], channel_index], window_length),
                _safe_divide(window.intra_matrix[feature_index["extrema_amplitude_sum"], channel_index], scale * event_count),
            ]
        )
    parts.append(np.asarray(intra_block, dtype=float))

    inter_block = window.inter_matrix.copy()
    inter_block[inter_index["synchronous_extrema_count"], :] /= window_length
    inter_block[inter_index["mean_event_lag"], :] /= window_length
    inter_block[inter_index["slope_ratio"], :] = np.tanh(inter_block[inter_index["slope_ratio"], :])
    parts.append(inter_block.ravel(order="C"))

    peak_block = window.peak_hazard_matrix.copy()
    for channel_index, scale in enumerate(channel_scales):
        peak_block[peak_index["number_of_peaks"], channel_index] /= window_length
        peak_block[peak_index["mean_prominence"], channel_index] = _safe_divide(
            peak_block[peak_index["mean_prominence"], channel_index],
            scale,
        )
        peak_block[peak_index["max_prominence"], channel_index] = _safe_divide(
            peak_block[peak_index["max_prominence"], channel_index],
            scale,
        )
        peak_block[peak_index["mean_peak_width"], channel_index] /= window_length
        peak_block[peak_index["max_peak_width"], channel_index] /= window_length
        peak_block[peak_index["max_rise_slope"], channel_index] = _safe_divide(
            peak_block[peak_index["max_rise_slope"], channel_index],
            scale,
        )
        peak_block[peak_index["duration_over_upper_threshold"], channel_index] /= window_length
        peak_block[peak_index["upper_tail_excess"], channel_index] = _safe_divide(
            peak_block[peak_index["upper_tail_excess"], channel_index],
            scale * window_length,
        )
        peak_block[peak_index["lower_tail_excess"], channel_index] = _safe_divide(
            peak_block[peak_index["lower_tail_excess"], channel_index],
            scale * window_length,
        )
        peak_block[peak_index["cumulative_risk"], channel_index] = _safe_divide(
            peak_block[peak_index["cumulative_risk"], channel_index],
            scale * window_length,
        )
        peak_block[peak_index["max_peak_value"], channel_index] = _safe_divide(
            peak_block[peak_index["max_peak_value"], channel_index],
            scale,
        )
    parts.append(peak_block.ravel(order="C"))

    by_channel_sign_counts: list[float] = []
    for channel in window.channels:
        channel_events = [event for event in window.extrema_window.events if event.channel == channel]
        positive = sum(1 for event in channel_events if event.sign == "max")
        negative = sum(1 for event in channel_events if event.sign != "max")
        by_channel_sign_counts.extend(
            [
                positive / window_length,
                negative / window_length,
            ]
        )
    parts.append(np.asarray(by_channel_sign_counts, dtype=float))

    time_block = np.asarray(
        [
            _safe_divide(window.time_placeholders.mean_inter_event_gap_steps, window_length),
            _safe_divide(window.time_placeholders.var_inter_event_gap_steps, window_length * window_length),
            _safe_divide(window.time_placeholders.mean_peak_width_steps, window_length),
            _safe_divide(window.time_placeholders.max_peak_width_steps, window_length),
            len(window.time_placeholders.normalized_event_positions) / window_length,
        ],
        dtype=float,
    )
    duration_block = np.asarray(
        [
            _safe_divide(window.time_placeholders.duration_over_threshold_by_channel.get(channel, 0.0), window_length)
            for channel in window.channels
        ],
        dtype=float,
    )
    position_block = _resample_positions(window.time_placeholders.normalized_event_positions, target_size=8)
    parts.extend([time_block, duration_block, position_block])

    return np.concatenate(parts).astype(float)


def _robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(matrix, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-6, scale, 1.0)
    scaled = (matrix - center) / scale
    return scaled, center, scale


@dataclass(slots=True)
class _ClusteringState:
    labels: np.ndarray
    centroids: np.ndarray
    member_ids: list[list[int]]
    mean_distance: float


class StructuralPatternDiscovery(PatternDiscoveryStrategy):
    def __init__(self, config: DiscoveryConfig) -> None:
        self.config = config
        self.device = resolve_model_device(device="cuda", require_gpu=True, stage_name="pattern discovery")
        self._torch = None

    def _lazy_import_torch(self) -> None:
        if self._torch is None:
            import torch

            self._torch = torch

    def _effective_pattern_bounds(self, row_count: int) -> tuple[int, int]:
        max_by_size = max(2, row_count // max(self.config.min_cluster_size, 1))
        upper = max(2, min(self.config.max_pattern_count, max_by_size))
        lower = min(max(2, self.config.min_pattern_count), upper)
        return lower, upper

    def _candidate_thresholds(self, matrix: np.ndarray) -> list[float]:
        self._lazy_import_torch()
        torch = self._torch
        matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        sample_size = min(len(matrix), max(128, self.config.quality_sample_size))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.random_seed)
        sample_indices = (
            torch.randperm(len(matrix_tensor), generator=generator, device=self.device)[:sample_size]
            if sample_size < len(matrix_tensor)
            else torch.arange(len(matrix_tensor), device=self.device)
        )
        sample = matrix_tensor[sample_indices]
        distances = torch.cdist(sample, sample)
        diagonal = torch.arange(len(sample), device=self.device)
        distances[diagonal, diagonal] = float("inf")
        nearest = distances.min(dim=1).values
        return sorted(
            {
                float(torch.quantile(nearest, quantile).item())
                for quantile in self.config.candidate_distance_quantiles
            }
        )

    def _assign_online(self, matrix: np.ndarray, threshold: float) -> _ClusteringState:
        self._lazy_import_torch()
        torch = self._torch
        matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        labels = np.full(len(matrix), -1, dtype=int)
        centroids: list[object] = []
        counts: list[int] = []
        member_ids: list[list[int]] = []
        total_distance = 0.0

        for row_index in range(len(matrix_tensor)):
            row = matrix_tensor[row_index]
            if not centroids:
                centroids.append(row.clone())
                counts.append(1)
                member_ids.append([row_index])
                labels[row_index] = 0
                continue

            centroid_matrix = torch.stack(centroids, dim=0)
            distances = torch.norm(centroid_matrix - row.unsqueeze(0), dim=1)
            best_cluster = int(torch.argmin(distances).item())
            best_distance = float(distances[best_cluster].item())
            if best_distance <= threshold:
                old_count = counts[best_cluster]
                counts[best_cluster] += 1
                centroids[best_cluster] = (
                    centroids[best_cluster] * old_count + row
                ) / counts[best_cluster]
                member_ids[best_cluster].append(row_index)
                labels[row_index] = best_cluster
                total_distance += best_distance
            else:
                centroids.append(row.clone())
                counts.append(1)
                member_ids.append([row_index])
                labels[row_index] = len(centroids) - 1

        return _ClusteringState(
            labels=labels,
            centroids=torch.stack(centroids, dim=0).detach().cpu().numpy(),
            member_ids=member_ids,
            mean_distance=total_distance / max(len(matrix), 1),
        )

    def _reassign_small_clusters(self, state: _ClusteringState, matrix: np.ndarray) -> _ClusteringState:
        self._lazy_import_torch()
        torch = self._torch
        large_clusters = [
            cluster_id
            for cluster_id, members in enumerate(state.member_ids)
            if len(members) >= self.config.min_cluster_size
        ]
        if not large_clusters:
            return state

        labels = state.labels.copy()
        centroids = state.centroids.copy()
        matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        for cluster_id, members in enumerate(state.member_ids):
            if len(members) >= self.config.min_cluster_size:
                continue
            target_centroids = torch.as_tensor(centroids[large_clusters], dtype=torch.float32, device=self.device)
            for row_index in members:
                distances = torch.norm(target_centroids - matrix_tensor[row_index].unsqueeze(0), dim=1)
                labels[row_index] = large_clusters[int(torch.argmin(distances).item())]

        unique_labels = sorted(set(int(label) for label in labels))
        relabel = {old: new for new, old in enumerate(unique_labels)}
        remapped_labels = np.asarray([relabel[int(label)] for label in labels], dtype=int)
        member_ids = [[] for _ in unique_labels]
        for row_index, label in enumerate(remapped_labels):
            member_ids[int(label)].append(row_index)
        centroids = np.vstack([matrix[members].mean(axis=0) for members in member_ids])
        distances = np.linalg.norm(matrix - centroids[remapped_labels], axis=1)
        return _ClusteringState(
            labels=remapped_labels,
            centroids=centroids,
            member_ids=member_ids,
            mean_distance=float(np.mean(distances)),
        )

    def _merge_to_upper_bound(self, state: _ClusteringState, matrix: np.ndarray) -> _ClusteringState:
        self._lazy_import_torch()
        torch = self._torch
        _, upper_bound = self._effective_pattern_bounds(len(matrix))
        labels = state.labels.copy()
        member_ids = [members[:] for members in state.member_ids]
        centroids = state.centroids.copy()

        while len(member_ids) > upper_bound:
            cluster_sizes = np.asarray([len(members) for members in member_ids], dtype=int)
            source_cluster = int(cluster_sizes.argmin())
            centroids_tensor = torch.as_tensor(centroids, dtype=torch.float32, device=self.device)
            centroid_distances = torch.norm(
                centroids_tensor - centroids_tensor[source_cluster].unsqueeze(0),
                dim=1,
            )
            centroid_distances[source_cluster] = float("inf")
            target_cluster = int(torch.argmin(centroid_distances).item())
            member_ids[target_cluster].extend(member_ids[source_cluster])
            del member_ids[source_cluster]

            remapped_labels = np.full(len(matrix), -1, dtype=int)
            for new_label, members in enumerate(member_ids):
                for row_index in members:
                    remapped_labels[row_index] = new_label
            labels = remapped_labels
            centroids = np.vstack([matrix[members].mean(axis=0) for members in member_ids])

        distances = np.linalg.norm(matrix - centroids[labels], axis=1)
        return _ClusteringState(
            labels=labels,
            centroids=centroids,
            member_ids=member_ids,
            mean_distance=float(np.mean(distances)),
        )

    def _quality_score(self, state: _ClusteringState) -> float:
        self._lazy_import_torch()
        torch = self._torch
        cluster_count = len(state.member_ids)
        if cluster_count <= 1:
            return float("-inf")
        lower_bound, upper_bound = self._effective_pattern_bounds(sum(len(members) for members in state.member_ids))
        centroids_tensor = torch.as_tensor(state.centroids, dtype=torch.float32, device=self.device)
        centroid_distances = torch.norm(
            centroids_tensor[:, None, :] - centroids_tensor[None, :, :],
            dim=2,
        )
        diagonal = torch.arange(len(centroids_tensor), device=self.device)
        centroid_distances[diagonal, diagonal] = float("inf")
        mean_separation = float(centroid_distances.min(dim=1).values.mean().item())
        cluster_sizes = np.asarray([len(members) for members in state.member_ids], dtype=float)
        balance = float(cluster_sizes.min() / cluster_sizes.mean()) if cluster_sizes.size else 0.0
        count_penalty = 1.0
        if cluster_count < lower_bound:
            count_penalty *= cluster_count / max(lower_bound, 1)
        if cluster_count > upper_bound:
            count_penalty *= upper_bound / cluster_count
        return float((mean_separation / max(state.mean_distance, 1e-6)) * (0.5 + 0.5 * balance) * count_penalty)

    def fit_predict(self, data: DiscoveryInput) -> DiscoveryResult:
        if not data.pattern_windows:
            return DiscoveryResult(
                labels_by_window_id={},
                prototypes=[],
                strategy="structural",
                selected_cluster_count=0,
                selected_quality_score=0.0,
                selection_metric_name="threshold_score",
                candidate_quality={},
            )

        structure_matrix = np.vstack([_structure_vector(window) for window in data.pattern_windows]).astype(float)
        scaled_matrix, _, _ = _robust_scale(structure_matrix)

        candidate_quality: dict[str, float] = {}
        best_threshold = 0.0
        best_state: _ClusteringState | None = None
        best_score = float("-inf")

        for threshold in self._candidate_thresholds(scaled_matrix):
            state = self._assign_online(scaled_matrix, threshold)
            state = self._reassign_small_clusters(state, scaled_matrix)
            state = self._merge_to_upper_bound(state, scaled_matrix)
            score = self._quality_score(state)
            candidate_quality[f"{threshold:.6f}"] = float(score)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_state = state

        if best_state is None:
            raise RuntimeError("Structural pattern discovery failed to produce a valid clustering.")

        labels_by_window_id = {
            window.window_id: int(best_state.labels[row_index])
            for row_index, window in enumerate(data.pattern_windows)
        }
        prototypes: list[PatternPrototype] = []
        for pattern_id, members in enumerate(best_state.member_ids):
            member_vectors = np.vstack([data.pattern_windows[row_index].feature_vector for row_index in members])
            member_structure_vectors = scaled_matrix[members]
            prototypes.append(
                PatternPrototype(
                    pattern_id=pattern_id,
                    centroid=member_vectors.mean(axis=0),
                    member_window_ids=[data.pattern_windows[row_index].window_id for row_index in members],
                    metadata={
                        "member_count": len(members),
                        "structure_threshold": best_threshold,
                        "structure_centroid": member_structure_vectors.mean(axis=0).astype(float).tolist(),
                    },
                )
            )

        return DiscoveryResult(
            labels_by_window_id=labels_by_window_id,
            prototypes=prototypes,
            strategy="structural",
            selected_cluster_count=len(prototypes),
            selected_quality_score=float(best_score),
            selection_metric_name="threshold_score",
            candidate_quality=candidate_quality,
        )
