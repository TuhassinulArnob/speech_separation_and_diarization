"""Tests for embedding normalization."""

from __future__ import annotations

import unittest

import numpy as np

from Project.ssl_pipeline.embedding_normalization import (
    EmbeddingNormalizer,
    l2_normalize,
    normalize_embeddings,
    plot_norm_distribution,
    plot_vector_statistics,
)
from Project.ssl_pipeline.ssl_embedding import SegmentEmbedding


class EmbeddingNormalizationTests(unittest.TestCase):
    """Validate normalization without clustering or evaluation."""

    def test_l2_normalization_returns_unit_vectors(self) -> None:
        vectors = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

        normalized = l2_normalize(vectors)

        self.assertTrue(np.allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0]))
        self.assertTrue(np.allclose(normalized[0], [0.6, 0.8]))

    def test_l2_zero_vector_stays_zero(self) -> None:
        normalized = l2_normalize(np.array([[0.0, 0.0]], dtype=np.float32))

        self.assertTrue(np.array_equal(normalized, np.array([[0.0, 0.0]], dtype=np.float32)))

    def test_standard_scaler_zero_centers_dimensions(self) -> None:
        normalizer = EmbeddingNormalizer(
            normalization_method="standard",
            pca_enabled=False,
        )

        normalized = normalizer.fit_transform(_embeddings())
        vectors = np.vstack([item.vector for item in normalized])

        self.assertTrue(np.allclose(vectors.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-6))
        self.assertTrue(np.allclose(vectors.std(axis=0), [1.0, 1.0, 1.0], atol=1e-6))

    def test_pca_reduces_vector_dimensions(self) -> None:
        normalizer = EmbeddingNormalizer(
            normalization_method="standard",
            pca_enabled=True,
            pca_components=2,
        )

        normalized = normalizer.fit_transform(_embeddings())

        self.assertEqual([item.dimension for item in normalized], [2, 2, 2])
        self.assertEqual(normalized[0].vector.shape, (2,))

    def test_normalized_embeddings_preserve_segment_metadata(self) -> None:
        source = _embeddings()[1]

        normalized = normalize_embeddings(
            [source],
            normalization_method="l2",
            pca_enabled=False,
        )

        self.assertEqual(normalized[0].segment_id, source.segment_id)
        self.assertEqual(normalized[0].start, source.start)
        self.assertEqual(normalized[0].end, source.end)
        self.assertEqual(normalized[0].model_name, source.model_name)
        self.assertEqual(normalized[0].metadata["source_dimension"], source.dimension)

    def test_mixed_vector_dimensions_raise(self) -> None:
        embeddings = [
            SegmentEmbedding(0, 0.0, 1.0, np.ones(2, dtype=np.float32), 2, "m"),
            SegmentEmbedding(1, 1.0, 2.0, np.ones(3, dtype=np.float32), 3, "m"),
        ]

        with self.assertRaisesRegex(ValueError, "same vector dimension"):
            normalize_embeddings(embeddings, normalization_method="l2", pca_enabled=False)

    def test_plots_return_figures(self) -> None:
        normalized = normalize_embeddings(
            _embeddings(),
            normalization_method="l2",
            pca_enabled=False,
        )

        norm_figure = plot_norm_distribution(normalized)
        stats_figure = plot_vector_statistics(normalized)

        self.assertEqual(len(norm_figure.axes), 1)
        self.assertEqual(len(stats_figure.axes), 2)


def _embeddings() -> list[SegmentEmbedding]:
    """Return deterministic segment embeddings for tests."""

    return [
        SegmentEmbedding(
            0,
            0.0,
            1.0,
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            3,
            "fake/wavlm",
        ),
        SegmentEmbedding(
            1,
            1.0,
            2.0,
            np.array([2.0, 4.0, 6.0], dtype=np.float32),
            3,
            "fake/wavlm",
        ),
        SegmentEmbedding(
            2,
            2.0,
            3.0,
            np.array([3.0, 6.0, 9.0], dtype=np.float32),
            3,
            "fake/wavlm",
        ),
    ]


if __name__ == "__main__":
    unittest.main()
