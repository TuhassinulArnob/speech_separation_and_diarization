"""Spectral clustering stage for normalized speaker embeddings.

This module consumes ``NormalizedEmbedding`` objects and assigns speaker cluster
IDs. It does not reconstruct timelines, write RTTM files, compute DER/JER, or
implement a baseline system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.embedding_normalization import NormalizedEmbedding

LOGGER = logging.getLogger(__name__)

SUPPORTED_CLUSTERING_ALGORITHMS = {"spectral"}
SUPPORTED_AFFINITY_METRICS = {"cosine"}
SUPPORTED_SPEAKER_COUNT_METHODS = {"eigengap"}


@dataclass(frozen=True)
class ClusteredEmbedding:
    """A normalized embedding with an assigned speaker cluster ID."""

    segment_id: int
    start: float
    end: float
    cluster_id: int
    vector: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        """Return serializable metadata without the embedding vector."""

        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "cluster_id": self.cluster_id,
            **self.metadata,
        }


class SpectralEmbeddingClusterer:
    """Assign speaker labels with spectral clustering."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        algorithm: str | None = None,
        affinity_metric: str | None = None,
        affinity_threshold: float | None = None,
        num_speakers: int | None = None,
        oracle_num_speakers: bool | None = None,
        speaker_count_method: str | None = None,
        max_estimated_speakers: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.config = config or load_config()
        self.algorithm = (algorithm or self.config.clustering_algorithm).lower()
        self.affinity_metric = (
            affinity_metric or self.config.clustering_affinity_metric
        ).lower()
        self.affinity_threshold = (
            self.config.clustering_affinity_threshold
            if affinity_threshold is None
            else float(affinity_threshold)
        )
        self.num_speakers = (
            self.config.clustering_num_speakers
            if num_speakers is None
            else int(num_speakers)
        )
        self.oracle_num_speakers = (
            self.config.clustering_oracle_num_speakers
            if oracle_num_speakers is None
            else bool(oracle_num_speakers)
        )
        self.speaker_count_method = (
            speaker_count_method or self.config.clustering_speaker_count_method
        ).lower()
        self.max_estimated_speakers = (
            self.config.clustering_max_estimated_speakers
            if max_estimated_speakers is None
            else int(max_estimated_speakers)
        )
        self.random_seed = (
            self.config.clustering_random_seed
            if random_seed is None
            else int(random_seed)
        )
        _validate_clustering_config(
            self.algorithm,
            self.affinity_metric,
            self.affinity_threshold,
            self.num_speakers,
            self.oracle_num_speakers,
            self.speaker_count_method,
            self.max_estimated_speakers,
        )

    def fit_predict(
        self,
        embeddings: Iterable[NormalizedEmbedding],
    ) -> list[ClusteredEmbedding]:
        """Cluster normalized embeddings and return speaker labels."""

        embedding_list = list(embeddings)
        if not embedding_list:
            return []

        vectors = _stack_vectors(embedding_list)
        similarity = compute_cosine_similarity_matrix(vectors)
        affinity = build_affinity_matrix(
            similarity,
            threshold=self.affinity_threshold,
        )
        speaker_count = self._resolve_speaker_count(affinity)
        labels = self._cluster_affinity(affinity, speaker_count)

        clustered = [
            _build_clustered_embedding(embedding, int(label), self)
            for embedding, label in zip(embedding_list, labels)
        ]
        LOGGER.info("Clustered %d embeddings into %d speakers", len(clustered), speaker_count)
        return clustered

    def _resolve_speaker_count(self, affinity: np.ndarray) -> int:
        """Return oracle or estimated speaker count for the current batch."""

        n_samples = affinity.shape[0]
        if n_samples == 1:
            return 1
        if self.oracle_num_speakers:
            return min(self.num_speakers, n_samples)
        if self.speaker_count_method == "eigengap":
            return estimate_speaker_count_eigengap(
                affinity,
                max_speakers=min(self.max_estimated_speakers, n_samples),
            )
        raise ValueError(f"Unsupported speaker count method: {self.speaker_count_method}")

    def _cluster_affinity(self, affinity: np.ndarray, speaker_count: int) -> np.ndarray:
        """Run spectral clustering or return trivial labels for small cases."""

        n_samples = affinity.shape[0]
        if speaker_count <= 1:
            return np.zeros(n_samples, dtype=np.int64)
        if speaker_count >= n_samples:
            return np.arange(n_samples, dtype=np.int64)

        component_labels = _connected_component_labels(affinity)
        if len(set(component_labels.tolist())) == speaker_count:
            return _renumber_labels_by_first_seen(component_labels)

        model = SpectralClustering(
            n_clusters=speaker_count,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=self.random_seed,
        )
        labels = model.fit_predict(affinity)
        return _renumber_labels_by_first_seen(labels)


def cluster_embeddings(
    embeddings: Iterable[NormalizedEmbedding],
    *,
    config_path: str | Path | None = None,
    algorithm: str | None = None,
    affinity_metric: str | None = None,
    affinity_threshold: float | None = None,
    num_speakers: int | None = None,
    oracle_num_speakers: bool | None = None,
    speaker_count_method: str | None = None,
    max_estimated_speakers: int | None = None,
    random_seed: int | None = None,
) -> list[ClusteredEmbedding]:
    """Convenience function for spectral clustering."""

    config = load_config(config_path) if config_path else load_config()
    clusterer = SpectralEmbeddingClusterer(
        config,
        algorithm=algorithm,
        affinity_metric=affinity_metric,
        affinity_threshold=affinity_threshold,
        num_speakers=num_speakers,
        oracle_num_speakers=oracle_num_speakers,
        speaker_count_method=speaker_count_method,
        max_estimated_speakers=max_estimated_speakers,
        random_seed=random_seed,
    )
    return clusterer.fit_predict(embeddings)


def compute_cosine_similarity_matrix(vectors: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Compute pairwise cosine similarity.

    Formula:
    cos(x_i, x_j) = (x_i dot x_j) / (||x_i||_2 ||x_j||_2)
    """

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.maximum(norms, epsilon)
    normalized = matrix / safe
    similarity = normalized @ normalized.T
    return np.clip(similarity, -1.0, 1.0).astype(np.float32)


def build_affinity_matrix(
    similarity_matrix: np.ndarray,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Build a non-negative thresholded affinity matrix from cosine similarity."""

    similarity = np.asarray(similarity_matrix, dtype=np.float32)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity_matrix must be square.")

    affinity = np.clip(similarity, 0.0, 1.0)
    affinity[affinity < threshold] = 0.0
    np.fill_diagonal(affinity, 1.0)
    return ((affinity + affinity.T) / 2.0).astype(np.float32)


def estimate_speaker_count_eigengap(
    affinity_matrix: np.ndarray,
    *,
    max_speakers: int,
) -> int:
    """Estimate speaker count with the eigengap heuristic.

    Graph Laplacian:
    L_sym = I - D^(-1/2) A D^(-1/2)

    Eigen-decomposition:
    L_sym u = lambda u

    The estimated speaker count is the index of the largest gap between
    consecutive small eigenvalues.
    """

    affinity = np.asarray(affinity_matrix, dtype=np.float64)
    n_samples = affinity.shape[0]
    if n_samples == 0:
        return 0
    if n_samples == 1:
        return 1

    limit = max(1, min(max_speakers, n_samples))
    laplacian = compute_normalized_graph_laplacian(affinity)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(np.real(eigenvalues))

    if limit == 1:
        return 1
    gaps = np.diff(eigenvalues[: limit + 1])
    if gaps.size == 0:
        return 1
    return int(np.argmax(gaps) + 1)


def compute_normalized_graph_laplacian(affinity_matrix: np.ndarray) -> np.ndarray:
    """Compute the symmetric normalized graph Laplacian."""

    affinity = np.asarray(affinity_matrix, dtype=np.float64)
    degrees = affinity.sum(axis=1)
    inv_sqrt_degrees = np.zeros_like(degrees)
    positive = degrees > 0
    inv_sqrt_degrees[positive] = 1.0 / np.sqrt(degrees[positive])
    degree_matrix = np.diag(inv_sqrt_degrees)
    identity = np.eye(affinity.shape[0], dtype=np.float64)
    return identity - degree_matrix @ affinity @ degree_matrix


def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a cosine similarity matrix heatmap."""

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(similarity_matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_title("Cosine Similarity Matrix")
    axis.set_xlabel("Segment index")
    axis.set_ylabel("Segment index")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_affinity_matrix(
    affinity_matrix: np.ndarray,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a thresholded affinity matrix heatmap."""

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(affinity_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_title("Affinity Matrix")
    axis.set_xlabel("Segment index")
    axis.set_ylabel("Segment index")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_cluster_distribution(
    clustered_embeddings: Iterable[ClusteredEmbedding],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the number of segments assigned to each cluster."""

    clustered = list(clustered_embeddings)
    labels = [embedding.cluster_id for embedding in clustered]
    unique_labels, counts = (
        np.unique(labels, return_counts=True)
        if labels
        else (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    )

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([str(label) for label in unique_labels], counts)
    axis.set_title("Cluster Distribution")
    axis.set_xlabel("Cluster ID")
    axis.set_ylabel("Segment count")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def _stack_vectors(embeddings: Iterable[NormalizedEmbedding]) -> np.ndarray:
    """Stack normalized vectors and validate dimensions."""

    embedding_list = list(embeddings)
    vectors = [np.asarray(embedding.vector, dtype=np.float32) for embedding in embedding_list]
    dimensions = {vector.shape[0] for vector in vectors}
    if len(dimensions) > 1:
        raise ValueError("All embeddings must have the same vector dimension.")
    return np.vstack(vectors).astype(np.float32)


def _build_clustered_embedding(
    embedding: NormalizedEmbedding,
    cluster_id: int,
    clusterer: SpectralEmbeddingClusterer,
) -> ClusteredEmbedding:
    """Create a clustered embedding while preserving source metadata."""

    metadata = embedding.to_metadata()
    metadata.update(
        {
            "clustering_algorithm": clusterer.algorithm,
            "affinity_metric": clusterer.affinity_metric,
            "affinity_threshold": clusterer.affinity_threshold,
            "oracle_num_speakers": clusterer.oracle_num_speakers,
            "configured_num_speakers": clusterer.num_speakers,
            "speaker_count_method": clusterer.speaker_count_method,
            "random_seed": clusterer.random_seed,
        }
    )
    return ClusteredEmbedding(
        segment_id=embedding.segment_id,
        start=embedding.start,
        end=embedding.end,
        cluster_id=cluster_id,
        vector=embedding.vector.astype(np.float32),
        metadata=metadata,
    )


def _renumber_labels_by_first_seen(labels: np.ndarray) -> np.ndarray:
    """Make cluster IDs deterministic by first appearance in segment order."""

    mapping: dict[int, int] = {}
    renumbered: list[int] = []
    for label in labels:
        original = int(label)
        if original not in mapping:
            mapping[original] = len(mapping)
        renumbered.append(mapping[original])
    return np.array(renumbered, dtype=np.int64)


def _connected_component_labels(affinity: np.ndarray) -> np.ndarray:
    """Return graph component labels for positive-affinity edges."""

    n_samples = affinity.shape[0]
    labels = np.full(n_samples, -1, dtype=np.int64)
    current_label = 0
    adjacency = affinity > 0.0
    np.fill_diagonal(adjacency, False)

    for start in range(n_samples):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = current_label
        while stack:
            node = stack.pop()
            neighbors = np.flatnonzero(adjacency[node])
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = current_label
                    stack.append(int(neighbor))
        current_label += 1

    return labels


def _validate_clustering_config(
    algorithm: str,
    affinity_metric: str,
    affinity_threshold: float,
    num_speakers: int,
    oracle_num_speakers: bool,
    speaker_count_method: str,
    max_estimated_speakers: int,
) -> None:
    """Validate spectral clustering configuration."""

    if algorithm not in SUPPORTED_CLUSTERING_ALGORITHMS:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    if affinity_metric not in SUPPORTED_AFFINITY_METRICS:
        raise ValueError(f"Unsupported affinity metric: {affinity_metric}")
    if not 0.0 <= affinity_threshold <= 1.0:
        raise ValueError("clustering_affinity_threshold must be between 0 and 1.")
    if oracle_num_speakers and num_speakers <= 0:
        raise ValueError("clustering_num_speakers must be positive in oracle mode.")
    if speaker_count_method not in SUPPORTED_SPEAKER_COUNT_METHODS:
        raise ValueError(f"Unsupported speaker count method: {speaker_count_method}")
    if max_estimated_speakers <= 0:
        raise ValueError("clustering_max_estimated_speakers must be positive.")


def _save_plot_if_requested(
    figure: plt.Figure,
    output_path: str | Path | None,
) -> None:
    """Save a Matplotlib figure if requested."""

    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    LOGGER.info("Saved clustering plot to %s", path)
