"""Tests for spectral clustering over normalized embeddings."""

from __future__ import annotations

import unittest

import numpy as np

from Project.ssl_pipeline.embedding_normalization import NormalizedEmbedding
from Project.ssl_pipeline.spectral_clustering import (
    SpectralEmbeddingClusterer,
    build_affinity_matrix,
    cluster_embeddings,
    compute_cosine_similarity_matrix,
    estimate_speaker_count_eigengap,
    plot_affinity_matrix,
    plot_cluster_distribution,
    plot_similarity_matrix,
)


class SpectralClusteringTests(unittest.TestCase):
    """Validate clustering labels without timeline reconstruction or RTTM."""

    def test_cosine_similarity_matrix(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        similarity = compute_cosine_similarity_matrix(vectors)

        self.assertTrue(np.allclose(np.diag(similarity), [1.0, 1.0, 1.0]))
        self.assertAlmostEqual(float(similarity[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(similarity[0, 2]), 1.0 / np.sqrt(2.0), places=6)

    def test_affinity_matrix_thresholds_weak_edges(self) -> None:
        similarity = np.array(
            [[1.0, 0.8, -0.2], [0.8, 1.0, 0.3], [-0.2, 0.3, 1.0]],
            dtype=np.float32,
        )

        affinity = build_affinity_matrix(similarity, threshold=0.5)

        self.assertAlmostEqual(float(affinity[0, 1]), 0.8, places=6)
        self.assertEqual(float(affinity[1, 2]), 0.0)
        self.assertEqual(float(affinity[0, 2]), 0.0)
        self.assertTrue(np.allclose(np.diag(affinity), [1.0, 1.0, 1.0]))

    def test_single_speaker_oracle_assigns_one_cluster(self) -> None:
        clustered = cluster_embeddings(
            _two_speaker_embeddings()[:3],
            num_speakers=1,
            oracle_num_speakers=True,
            random_seed=7,
        )

        self.assertEqual({item.cluster_id for item in clustered}, {0})

    def test_two_speakers_oracle_groups_embeddings(self) -> None:
        clustered = cluster_embeddings(
            _two_speaker_embeddings(),
            num_speakers=2,
            oracle_num_speakers=True,
            random_seed=7,
        )
        labels = [item.cluster_id for item in clustered]

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_three_speakers_oracle_groups_embeddings(self) -> None:
        clustered = cluster_embeddings(
            _three_speaker_embeddings(),
            num_speakers=3,
            oracle_num_speakers=True,
            random_seed=7,
        )
        labels = [item.cluster_id for item in clustered]

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertEqual(labels[4], labels[5])
        self.assertEqual(len(set(labels)), 3)

    def test_estimated_speaker_count_uses_eigengap(self) -> None:
        embeddings = _three_speaker_embeddings()
        vectors = np.vstack([item.vector for item in embeddings])
        affinity = build_affinity_matrix(compute_cosine_similarity_matrix(vectors))

        speaker_count = estimate_speaker_count_eigengap(affinity, max_speakers=5)

        self.assertEqual(speaker_count, 3)

    def test_estimated_speaker_count_clusters_embeddings(self) -> None:
        clustered = cluster_embeddings(
            _three_speaker_embeddings(),
            oracle_num_speakers=False,
            max_estimated_speakers=5,
            random_seed=7,
        )

        self.assertEqual(len({item.cluster_id for item in clustered}), 3)

    def test_small_dataset_one_embedding(self) -> None:
        clustered = cluster_embeddings(
            _two_speaker_embeddings()[:1],
            num_speakers=3,
            oracle_num_speakers=True,
        )

        self.assertEqual(len(clustered), 1)
        self.assertEqual(clustered[0].cluster_id, 0)

    def test_small_dataset_two_embeddings_with_large_oracle_count(self) -> None:
        clustered = cluster_embeddings(
            _two_speaker_embeddings()[::2],
            num_speakers=5,
            oracle_num_speakers=True,
        )

        self.assertEqual([item.cluster_id for item in clustered], [0, 1])

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(cluster_embeddings([]), [])

    def test_random_seed_makes_output_deterministic(self) -> None:
        kwargs = {
            "num_speakers": 2,
            "oracle_num_speakers": True,
            "random_seed": 123,
        }

        first = cluster_embeddings(_two_speaker_embeddings(), **kwargs)
        second = cluster_embeddings(_two_speaker_embeddings(), **kwargs)

        self.assertEqual(
            [item.cluster_id for item in first],
            [item.cluster_id for item in second],
        )

    def test_clustered_embedding_preserves_metadata(self) -> None:
        source = _two_speaker_embeddings()[0]

        clustered = cluster_embeddings(
            [source],
            num_speakers=1,
            oracle_num_speakers=True,
        )

        self.assertEqual(clustered[0].segment_id, source.segment_id)
        self.assertEqual(clustered[0].start, source.start)
        self.assertEqual(clustered[0].end, source.end)
        self.assertEqual(clustered[0].metadata["normalization_method"], "l2")
        self.assertEqual(clustered[0].metadata["source"], "unit-test")

    def test_plots_return_figures(self) -> None:
        embeddings = _two_speaker_embeddings()
        vectors = np.vstack([item.vector for item in embeddings])
        similarity = compute_cosine_similarity_matrix(vectors)
        affinity = build_affinity_matrix(similarity)
        clustered = cluster_embeddings(
            embeddings,
            num_speakers=2,
            oracle_num_speakers=True,
            random_seed=7,
        )

        self.assertEqual(len(plot_similarity_matrix(similarity).axes), 2)
        self.assertEqual(len(plot_affinity_matrix(affinity).axes), 2)
        self.assertEqual(len(plot_cluster_distribution(clustered).axes), 1)

    def test_invalid_configuration_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported clustering algorithm"):
            SpectralEmbeddingClusterer(algorithm="kmeans")


def _two_speaker_embeddings() -> list[NormalizedEmbedding]:
    """Return two clearly separated synthetic speaker groups."""

    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.99, 0.01], dtype=np.float32),
        np.array([-1.0, 0.0], dtype=np.float32),
        np.array([-0.99, -0.01], dtype=np.float32),
    ]
    return [_embedding(index, vector) for index, vector in enumerate(vectors)]


def _three_speaker_embeddings() -> list[NormalizedEmbedding]:
    """Return three disconnected synthetic speaker groups."""

    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.99, 0.01], dtype=np.float32),
        np.array([-0.5, 0.866], dtype=np.float32),
        np.array([-0.49, 0.872], dtype=np.float32),
        np.array([-0.5, -0.866], dtype=np.float32),
        np.array([-0.49, -0.872], dtype=np.float32),
    ]
    return [_embedding(index, vector) for index, vector in enumerate(vectors)]


def _embedding(index: int, vector: np.ndarray) -> NormalizedEmbedding:
    """Create a normalized embedding test fixture."""

    normalized = vector / np.linalg.norm(vector)
    return NormalizedEmbedding(
        segment_id=index,
        start=float(index),
        end=float(index + 1),
        vector=normalized.astype(np.float32),
        dimension=int(vector.shape[0]),
        model_name="fake/wavlm",
        normalization_method="l2",
        metadata={"source": "unit-test"},
    )


if __name__ == "__main__":
    unittest.main()
