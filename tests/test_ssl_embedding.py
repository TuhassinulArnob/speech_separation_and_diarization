"""Tests for WavLM embedding extraction."""

from __future__ import annotations

import json
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from Project.ssl_pipeline.config import load_config
from Project.ssl_pipeline.segmentation import SpeechSegment
from Project.ssl_pipeline.ssl_embedding import (
    SegmentEmbedding,
    WavLMEmbeddingExtractor,
    extract_segment_embeddings,
    get_huggingface_token,
    load_embedding_cache,
    resolve_device,
    save_embedding_cache,
    _pool_hidden_state,
)


SAMPLE_RATE = 16_000
TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class WavLMEmbeddingTests(unittest.TestCase):
    """Validate WavLM embedding extraction without downloading real models."""

    def test_model_loading_uses_configured_model_name(self) -> None:
        config = load_config(_write_config({"embedding_device": "cpu"}))
        fake_model = FakeModel()

        with patch(
            "Project.ssl_pipeline.ssl_embedding.AutoFeatureExtractor.from_pretrained",
            return_value=FakeFeatureExtractor(),
        ) as feature_loader, patch(
            "Project.ssl_pipeline.ssl_embedding.WavLMModel.from_pretrained",
            return_value=fake_model,
        ) as model_loader, patch(
            "Project.ssl_pipeline.ssl_embedding.get_huggingface_token",
            return_value="file-token",
        ):
            extractor = WavLMEmbeddingExtractor(config)
            extractor.load_model()

        feature_loader.assert_called_once_with("fake/wavlm", token="file-token")
        model_loader.assert_called_once_with("fake/wavlm", token="file-token")
        self.assertEqual(fake_model.device_seen, torch.device("cpu"))
        self.assertTrue(fake_model.eval_called)

    def test_cpu_mode_resolves_to_cpu(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_gpu_mode_resolves_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this machine.")

        self.assertEqual(resolve_device("cuda"), torch.device("cuda"))

    def test_gpu_mode_raises_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available on this machine.")

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            resolve_device("cuda")

    def test_batch_inference_returns_one_embedding_per_segment(self) -> None:
        audio_path = _fixture_path("embedding_audio.wav")
        _write_wav(audio_path, _tone(3.0))
        segments = [
            SpeechSegment(0, 0.0, 1.0),
            SpeechSegment(1, 1.0, 2.0),
            SpeechSegment(2, 2.0, 3.0),
        ]
        config = load_config(
            _write_config({"embedding_batch_size": 2, "embedding_device": "cpu"})
        )
        extractor = WavLMEmbeddingExtractor(
            config,
            feature_extractor=FakeFeatureExtractor(),
            model=FakeModel(hidden_size=6),
        )

        embeddings = extractor.extract_embeddings(audio_path, segments, use_cache=False)

        self.assertEqual(len(embeddings), 3)
        self.assertEqual([embedding.dimension for embedding in embeddings], [6, 6, 6])
        self.assertEqual([embedding.segment_id for embedding in embeddings], [0, 1, 2])

    def test_pooling_methods_return_expected_vectors(self) -> None:
        hidden_state = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]],
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]],
            ]
        )

        mean = _pool_hidden_state(hidden_state, "mean")
        maximum = _pool_hidden_state(hidden_state, "max")
        cls = _pool_hidden_state(hidden_state, "cls")
        statistics = _pool_hidden_state(hidden_state, "statistics")

        self.assertTrue(torch.allclose(mean[0], torch.tensor([3.0, 14.0 / 3.0])))
        self.assertTrue(torch.equal(maximum[0], torch.tensor([5.0, 8.0])))
        self.assertTrue(torch.equal(cls[0], torch.tensor([1.0, 2.0])))
        self.assertEqual(tuple(statistics.shape), (2, 4))

    def test_statistics_pooling_doubles_vector_dimension(self) -> None:
        config = load_config(
            _write_config(
                {
                    "embedding_device": "cpu",
                    "embedding_pooling_method": "statistics",
                }
            )
        )
        extractor = WavLMEmbeddingExtractor(
            config,
            feature_extractor=FakeFeatureExtractor(),
            model=FakeModel(hidden_size=6),
        )

        vectors = extractor._infer_batch([np.zeros(16, dtype=np.float32)], SAMPLE_RATE)

        self.assertEqual(vectors[0].shape, (12,))

    def test_cache_loading_skips_model_inference(self) -> None:
        cache_path = _fixture_path("embedding_cache.npz")
        cached = [
            SegmentEmbedding(
                segment_id=0,
                start=0.0,
                end=1.0,
                vector=np.ones(4, dtype=np.float32),
                dimension=4,
                model_name="fake/wavlm",
            )
        ]
        save_embedding_cache(cached, cache_path)

        embeddings = extract_segment_embeddings(
            _fixture_path("unused.wav"),
            [SpeechSegment(0, 0.0, 1.0)],
            config_path=_write_config(),
            cache_path=cache_path,
            use_cache=True,
        )

        self.assertEqual(len(embeddings), 1)
        self.assertTrue(np.array_equal(embeddings[0].vector, cached[0].vector))

    def test_cache_round_trip_preserves_metadata(self) -> None:
        cache_path = _fixture_path("embedding_roundtrip.npz")
        embeddings = [
            SegmentEmbedding(0, 0.0, 1.0, np.array([1.0, 2.0], dtype=np.float32), 2, "m"),
            SegmentEmbedding(1, 1.0, 2.0, np.array([3.0, 4.0], dtype=np.float32), 2, "m"),
        ]

        save_embedding_cache(embeddings, cache_path)
        loaded = load_embedding_cache(cache_path)

        self.assertEqual([item.to_metadata() for item in loaded], [item.to_metadata() for item in embeddings])
        self.assertTrue(np.array_equal(loaded[1].vector, embeddings[1].vector))

    def test_env_file_token_is_used_when_environment_is_missing(self) -> None:
        env_path = _fixture_path("embedding.env")
        env_path.write_text("HUGGINGFACE_TOKEN=from-env-file\n", encoding="utf-8")

        with patch.dict("os.environ", {}, clear=True):
            token = get_huggingface_token(env_path=env_path)

        self.assertEqual(token, "from-env-file")


class FakeFeatureExtractor:
    """Small feature extractor replacement for tests."""

    def __call__(
        self,
        audio_batch: list[np.ndarray],
        *,
        sampling_rate: int,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del sampling_rate, padding, return_tensors
        batch_size = len(audio_batch)
        return {"input_values": torch.zeros((batch_size, 16), dtype=torch.float32)}


class FakeModel:
    """Small WavLM replacement that returns deterministic hidden states."""

    def __init__(self, hidden_size: int = 4) -> None:
        self.hidden_size = hidden_size
        self.device_seen: torch.device | None = None
        self.eval_called = False

    def to(self, device: torch.device) -> "FakeModel":
        self.device_seen = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def __call__(
        self,
        input_values: torch.Tensor,
        *,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del output_hidden_states
        batch_size = input_values.shape[0]
        hidden = torch.ones((batch_size, 5, self.hidden_size), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=(hidden, hidden * 2))


def _write_config(overrides: dict[str, object] | None = None) -> Path:
    """Write an embedding config for tests."""

    config = {
        "sample_rate": 16000,
        "mono": True,
        "normalize_audio": True,
        "segment_duration_seconds": 2.0,
        "segment_window_seconds": 2.0,
        "segment_hop_seconds": 1.5,
        "segment_min_duration_seconds": 0.5,
        "segment_max_duration_seconds": 2.0,
        "segment_include_short_final": True,
        "pyannote_model": "pyannote/speaker-diarization",
        "vad_backend": "energy",
        "vad_pyannote_model": "pyannote/voice-activity-detection",
        "vad_min_speech_duration_seconds": 0.25,
        "vad_min_silence_duration_seconds": 0.15,
        "vad_merge_threshold_seconds": 0.2,
        "vad_frame_duration_ms": 30,
        "vad_webrtc_aggressiveness": 2,
        "vad_energy_threshold": 0.01,
        "embedding_model_name": "fake/wavlm",
        "embedding_pooling_method": "mean",
        "embedding_hidden_layer": -1,
        "embedding_batch_size": 2,
        "embedding_device": "cpu",
        "embedding_cache_enabled": True,
        "embedding_cache_dir": "tmp_tests",
    }
    if overrides:
        config.update(overrides)

    path = _fixture_path("embedding_config.json")
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _tone(duration_seconds: float) -> np.ndarray:
    """Create a synthetic audio fixture."""

    samples = int(duration_seconds * SAMPLE_RATE)
    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Write a mono PCM WAV file for tests."""

    pcm_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_audio.tobytes())


def _fixture_path(file_name: str) -> Path:
    """Return a writable path for generated test fixtures."""

    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return TEST_TMP_ROOT / file_name


if __name__ == "__main__":
    unittest.main()
