"""Embedding normalization stage before spectral clustering.

This module consumes raw ``SegmentEmbedding`` objects from the WavLM stage and
returns ``NormalizedEmbedding`` objects. It does not cluster speakers, evaluate
diarization quality, reconstruct timelines, or write RTTM files.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.ssl_embedding import SegmentEmbedding, load_embedding_cache

LOGGER = logging.getLogger(__name__)

SUPPORTED_NORMALIZATION_METHODS = {"l2", "standard", "none"}


@dataclass(frozen=True)
class NormalizedEmbedding:
    """A normalized embedding aligned to the original speech segment."""

    segment_id: int
    start: float
    end: float
    vector: np.ndarray
    dimension: int
    model_name: str
    normalization_method: str
    metadata: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        """Return serializable metadata without the embedding vector."""

        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "normalization_method": self.normalization_method,
            **self.metadata,
        }


class EmbeddingNormalizer:
    """Fit and apply configurable embedding normalization."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        normalization_method: str | None = None,
        pca_enabled: bool | None = None,
        pca_components: int | None = None,
    ) -> None:
        self.config = config or load_config()
        self.normalization_method = (
            normalization_method or self.config.embedding_normalization_method
        ).lower()
        self.pca_enabled = (
            self.config.embedding_pca_enabled
            if pca_enabled is None
            else bool(pca_enabled)
        )
        self.pca_components = (
            self.config.embedding_pca_components
            if pca_components is None
            else pca_components
        )
        self.standard_scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        _validate_normalization_config(
            self.normalization_method,
            self.pca_enabled,
            self.pca_components,
        )

    def fit_transform(
        self,
        embeddings: Iterable[SegmentEmbedding],
    ) -> list[NormalizedEmbedding]:
        """Normalize segment embeddings, fitting scaler/PCA when needed."""

        embedding_list = list(embeddings)
        if not embedding_list:
            return []

        vectors = _stack_vectors(embedding_list)
        normalized = self._normalize(vectors)

        if self.pca_enabled:
            normalized = self._fit_transform_pca(normalized)

        results = [
            _build_normalized_embedding(
                embedding,
                vector,
                self.normalization_method,
                self.pca_enabled,
                self.pca_components,
            )
            for embedding, vector in zip(embedding_list, normalized)
        ]
        LOGGER.info("Normalized %d embeddings", len(results))
        return results

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the configured normalization transform."""

        if self.normalization_method == "l2":
            return l2_normalize(vectors)
        if self.normalization_method == "standard":
            self.standard_scaler = StandardScaler()
            return self.standard_scaler.fit_transform(vectors).astype(np.float32)
        if self.normalization_method == "none":
            return vectors.astype(np.float32, copy=True)
        raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

    def _fit_transform_pca(self, vectors: np.ndarray) -> np.ndarray:
        """Fit PCA and transform normalized embeddings."""

        if self.pca_components is None:
            raise ValueError("embedding_pca_components is required when PCA is enabled.")

        self.pca = PCA(n_components=self.pca_components)
        return self.pca.fit_transform(vectors).astype(np.float32)


def normalize_embeddings(
    embeddings: Iterable[SegmentEmbedding],
    *,
    config_path: str | Path | None = None,
    normalization_method: str | None = None,
    pca_enabled: bool | None = None,
    pca_components: int | None = None,
) -> list[NormalizedEmbedding]:
    """Convenience function for the embedding normalization stage."""

    config = load_config(config_path) if config_path else load_config()
    normalizer = EmbeddingNormalizer(
        config,
        normalization_method=normalization_method,
        pca_enabled=pca_enabled,
        pca_components=pca_components,
    )
    return normalizer.fit_transform(embeddings)


def l2_normalize(vectors: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Normalize each row vector to unit L2 norm."""

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, epsilon)
    return (matrix / safe_norms).astype(np.float32)


def plot_norm_distribution(
    embeddings: Iterable[NormalizedEmbedding],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the distribution of embedding vector L2 norms."""

    embedding_list = list(embeddings)
    norms = [float(np.linalg.norm(embedding.vector)) for embedding in embedding_list]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(norms, bins=min(20, max(1, len(norms))))
    axis.set_title("Normalized Embedding Norms")
    axis.set_xlabel("L2 norm")
    axis.set_ylabel("Count")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_vector_statistics(
    embeddings: Iterable[NormalizedEmbedding],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-dimension vector mean and standard deviation."""

    embedding_list = list(embeddings)
    vectors = _stack_vectors(embedding_list) if embedding_list else np.empty((0, 0))
    means = vectors.mean(axis=0) if vectors.size else np.array([], dtype=np.float32)
    stds = vectors.std(axis=0) if vectors.size else np.array([], dtype=np.float32)
    dimensions = np.arange(len(means))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(dimensions, means)
    axes[0].set_title("Per-Dimension Mean")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Mean")

    axes[1].plot(dimensions, stds)
    axes[1].set_title("Per-Dimension Std")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Std")

    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def _stack_vectors(
    embeddings: Iterable[SegmentEmbedding | NormalizedEmbedding],
) -> np.ndarray:
    """Stack embedding vectors and validate dimensions."""

    embedding_list = list(embeddings)
    vectors = [np.asarray(embedding.vector, dtype=np.float32) for embedding in embedding_list]
    dimensions = {vector.shape[0] for vector in vectors}
    if len(dimensions) > 1:
        raise ValueError("All embeddings must have the same vector dimension.")
    return np.vstack(vectors).astype(np.float32)


def _build_normalized_embedding(
    embedding: SegmentEmbedding,
    vector: np.ndarray,
    normalization_method: str,
    pca_enabled: bool,
    pca_components: int | None,
) -> NormalizedEmbedding:
    """Create a normalized embedding while preserving source metadata."""

    metadata = embedding.to_metadata()
    metadata.update(
        {
            "source_dimension": embedding.dimension,
            "pca_enabled": pca_enabled,
            "pca_components": pca_components,
        }
    )
    return NormalizedEmbedding(
        segment_id=embedding.segment_id,
        start=embedding.start,
        end=embedding.end,
        vector=vector.astype(np.float32),
        dimension=int(vector.shape[0]),
        model_name=embedding.model_name,
        normalization_method=normalization_method,
        metadata=metadata,
    )


def _validate_normalization_config(
    normalization_method: str,
    pca_enabled: bool,
    pca_components: int | None,
) -> None:
    """Validate normalization and PCA configuration."""

    if normalization_method not in SUPPORTED_NORMALIZATION_METHODS:
        raise ValueError(f"Unsupported normalization method: {normalization_method}")
    if pca_enabled and (pca_components is None or pca_components <= 0):
        raise ValueError("embedding_pca_components must be positive when PCA is enabled.")


def _save_plot_if_requested(
    figure: plt.Figure,
    output_path: str | Path | None,
) -> None:
    """Save a Matplotlib figure if an output path was supplied."""

    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    LOGGER.info("Saved normalization plot to %s", path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for embedding normalization."""

    parser = argparse.ArgumentParser(description="Normalize WavLM segment embeddings.")
    parser.add_argument("embedding_cache", type=Path, help="Input raw embedding .npz cache.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config file.")
    parser.add_argument(
        "--method",
        choices=sorted(SUPPORTED_NORMALIZATION_METHODS),
        default=None,
        help="Normalization method. Defaults to config value.",
    )
    parser.add_argument("--pca", action="store_true", help="Enable PCA.")
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--plot-norms", type=Path, default=None)
    parser.add_argument("--plot-stats", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run normalization from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    embeddings = load_embedding_cache(args.embedding_cache)
    normalized = normalize_embeddings(
        embeddings,
        config_path=args.config,
        normalization_method=args.method,
        pca_enabled=args.pca,
        pca_components=args.pca_components,
    )

    if args.plot_norms is not None:
        plot_norm_distribution(normalized, args.plot_norms)
    if args.plot_stats is not None:
        plot_vector_statistics(normalized, args.plot_stats)

    for embedding in normalized:
        print(
            f"{embedding.segment_id}\t{embedding.start:.3f}\t"
            f"{embedding.end:.3f}\t{embedding.dimension}\t"
            f"{embedding.normalization_method}"
        )


if __name__ == "__main__":
    main()
