"""Tests for repository cleanup configuration behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from Project.ssl_pipeline.config import get_huggingface_token, load_config


class ConfigTests(unittest.TestCase):
    """Validate safe configuration and secret handling."""

    def test_load_config_reads_default_values(self) -> None:
        config = load_config()

        self.assertEqual(config.sample_rate, 16000)
        self.assertTrue(config.mono)
        self.assertEqual(config.pyannote_model, "pyannote/speaker-diarization")
        self.assertEqual(config.segment_window_seconds, 2.0)
        self.assertEqual(config.segment_hop_seconds, 1.5)
        self.assertEqual(config.segment_min_duration_seconds, 0.5)
        self.assertEqual(config.segment_max_duration_seconds, 2.0)
        self.assertEqual(config.vad_backend, "auto")
        self.assertEqual(config.vad_pyannote_model, "pyannote/voice-activity-detection")
        self.assertEqual(config.embedding_model_name, "microsoft/wavlm-base-plus")
        self.assertEqual(config.embedding_pooling_method, "mean")
        self.assertEqual(config.embedding_hidden_layer, -1)
        self.assertEqual(config.embedding_batch_size, 4)
        self.assertEqual(config.embedding_normalization_method, "l2")
        self.assertFalse(config.embedding_pca_enabled)
        self.assertIsNone(config.embedding_pca_components)
        self.assertEqual(config.clustering_algorithm, "spectral")
        self.assertEqual(config.clustering_affinity_metric, "cosine")
        self.assertEqual(config.clustering_affinity_threshold, 0.0)
        self.assertEqual(config.clustering_num_speakers, 2)
        self.assertTrue(config.clustering_oracle_num_speakers)
        self.assertEqual(config.clustering_speaker_count_method, "eigengap")
        self.assertEqual(config.clustering_max_estimated_speakers, 8)
        self.assertEqual(config.clustering_random_seed, 42)
        self.assertTrue(config.timeline_merge_adjacent)
        self.assertEqual(config.timeline_merge_silence_threshold_seconds, 0.2)
        self.assertEqual(config.timeline_min_duration_seconds, 0.1)
        self.assertFalse(config.timeline_label_smoothing_enabled)
        self.assertEqual(config.rttm_recording_id, "recording1")
        self.assertEqual(config.rttm_channel_id, "1")
        self.assertEqual(config.rttm_decimal_places, 2)
        self.assertEqual(config.evaluation_collar_seconds, 0.0)
        self.assertTrue(config.evaluation_use_pyannote)
        self.assertEqual(config.baseline_mfcc_coefficients, 13)
        self.assertEqual(config.baseline_frame_length_seconds, 0.025)
        self.assertEqual(config.baseline_frame_hop_seconds, 0.01)
        self.assertEqual(config.baseline_kmeans_random_seed, 42)
        self.assertEqual(config.baseline_num_speakers, 2)
        self.assertTrue(config.baseline_oracle_num_speakers)
        self.assertEqual(config.separation_model_backend, "speechbrain")
        self.assertEqual(config.separation_model_name, "speechbrain/sepformer-whamr")
        self.assertEqual(config.separation_device, "auto")
        self.assertEqual(config.separation_output_dir, "outputs/separated_audio")
        self.assertEqual(config.separation_sample_rate, 8000)
        self.assertEqual(config.separation_num_speakers, 2)
        self.assertEqual(config.separation_chunk_duration_seconds, 30.0)
        self.assertEqual(config.separation_chunk_overlap_seconds, 1.0)

    def test_missing_token_raises_clear_error(self) -> None:
        env_var = "TEST_HUGGINGFACE_TOKEN"
        os.environ.pop(env_var, None)

        with self.assertRaisesRegex(OSError, env_var):
            get_huggingface_token(env_var)

    def test_optional_token_can_be_none(self) -> None:
        env_var = "TEST_HUGGINGFACE_TOKEN"
        os.environ.pop(env_var, None)

        self.assertIsNone(get_huggingface_token(env_var, required=False))

    def test_missing_config_file_raises(self) -> None:
        missing_path = Path(tempfile.gettempdir()) / "missing-diarization-config.json"

        with self.assertRaises(FileNotFoundError):
            load_config(missing_path)


if __name__ == "__main__":
    unittest.main()
