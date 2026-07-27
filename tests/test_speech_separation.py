"""Tests for true speech separation module."""

from __future__ import annotations

import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from Project.ssl_pipeline.config import DiarizationConfig
from Project.ssl_pipeline.speech_separation import (
    SpeechSeparator,
    _format_sources,
    load_audio,
    plot_spectrogram_comparison,
    plot_waveform_comparison,
    resolve_separation_device,
    save_separated_audio,
)


SAMPLE_RATE = 16_000
TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class SpeechSeparationTests(unittest.TestCase):
    """Validate speech separation without downloading pretrained weights."""

    def test_two_speaker_mixture_outputs_two_waveforms(self) -> None:
        mixture = _tone(1.0, 220.0) + _tone(1.0, 440.0)
        separator = SpeechSeparator(model=FakeSeparationModel())

        sources = separator.separate_waveform(mixture.astype(np.float32), SAMPLE_RATE)

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].shape, mixture.shape)
        self.assertEqual(sources[1].shape, mixture.shape)
        self.assertTrue(np.any(sources[0] != 0.0))
        self.assertFalse(np.allclose(sources[0], sources[1]))

    def test_speechbrain_batch_time_speaker_shape_keeps_distinct_sources(self) -> None:
        samples = 8
        source_one = np.linspace(0.0, 0.7, samples, dtype=np.float32)
        source_two = -source_one
        speechbrain_output = np.stack([source_one, source_two], axis=1)[np.newaxis, :, :]

        sources = _format_sources(
            speechbrain_output,
            expected_length=samples,
            expected_num_speakers=2,
        )

        self.assertEqual(len(sources), 2)
        np.testing.assert_allclose(sources[0], source_one)
        np.testing.assert_allclose(sources[1], source_two)

    def test_source_first_batch_shape_keeps_distinct_sources(self) -> None:
        samples = 8
        source_one = np.linspace(0.0, 0.7, samples, dtype=np.float32)
        source_two = np.ones(samples, dtype=np.float32) * 0.25
        source_first_output = np.stack([source_one, source_two], axis=0)[np.newaxis, :, :]

        sources = _format_sources(
            source_first_output,
            expected_length=samples,
            expected_num_speakers=2,
        )

        self.assertEqual(len(sources), 2)
        np.testing.assert_allclose(sources[0], source_one)
        np.testing.assert_allclose(sources[1], source_two)

    def test_long_audio_uses_chunked_inference_without_length_drift(self) -> None:
        audio = _tone(0.2, 220.0)
        model = CountingSeparationModel()
        config = DiarizationConfig(
            separation_chunk_duration_seconds=0.05,
            separation_chunk_overlap_seconds=0.01,
        )
        separator = SpeechSeparator(config=config, model=model)

        sources = separator.separate_waveform(audio, SAMPLE_RATE)

        self.assertGreater(model.calls, 1)
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(sources[0]), len(audio))
        self.assertEqual(len(sources[1]), len(audio))
        self.assertFalse(np.allclose(sources[0], sources[1]))

    def test_silent_input_returns_silent_sources(self) -> None:
        separator = SpeechSeparator(model=FakeSeparationModel())
        audio = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)

        sources = separator.separate_waveform(audio, SAMPLE_RATE)

        self.assertEqual(len(sources), 2)
        self.assertTrue(np.allclose(sources[0], 0.0))
        self.assertTrue(np.allclose(sources[1], 0.0))

    def test_short_audio_is_supported(self) -> None:
        separator = SpeechSeparator(model=FakeSeparationModel())
        audio = _tone(0.02, 330.0)

        sources = separator.separate_waveform(audio, SAMPLE_RATE)

        self.assertEqual(len(sources), 2)
        self.assertEqual(len(sources[0]), len(audio))

    def test_invalid_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_audio(_fixture_path("missing.wav"), target_sample_rate=SAMPLE_RATE)

    def test_cpu_mode_resolves_to_cpu(self) -> None:
        self.assertEqual(resolve_separation_device("cpu"), torch.device("cpu"))

    def test_gpu_mode_resolves_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this machine.")

        self.assertEqual(resolve_separation_device("cuda"), torch.device("cuda"))

    def test_gpu_mode_raises_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available on this machine.")

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            resolve_separation_device("cuda")

    def test_separate_file_saves_speaker_wavs(self) -> None:
        audio_path = _fixture_path("mixture.wav")
        output_dir = TEST_TMP_ROOT / "separated_audio_test"
        _write_wav(audio_path, _tone(0.2, 220.0) + _tone(0.2, 440.0))
        separator = SpeechSeparator(model=FakeSeparationModel())

        result = separator.separate_file(audio_path, output_dir=output_dir)

        self.assertEqual(len(result.output_paths), 2)
        self.assertTrue((output_dir / "speaker1.wav").exists())
        self.assertTrue((output_dir / "speaker2.wav").exists())
        self.assertEqual(result.sample_rate, 8000)

    def test_save_separated_audio_uses_expected_names(self) -> None:
        output_dir = TEST_TMP_ROOT / "manual_separated"
        paths = save_separated_audio(
            [np.zeros(100, dtype=np.float32), np.ones(100, dtype=np.float32) * 0.1],
            SAMPLE_RATE,
            output_dir,
        )

        self.assertEqual([path.name for path in paths], ["speaker1.wav", "speaker2.wav"])

    def test_plots_return_figures(self) -> None:
        mixture = _tone(0.2, 220.0)
        sources = [mixture * 0.5, mixture * -0.5]

        self.assertEqual(len(plot_waveform_comparison(mixture, sources, SAMPLE_RATE).axes), 3)
        self.assertEqual(len(plot_spectrogram_comparison(mixture, sources, SAMPLE_RATE).axes), 3)


class FakeSeparationModel:
    """Small deterministic separator replacement for tests."""

    def separate_batch(self, waveform: torch.Tensor) -> torch.Tensor:
        mixture = waveform[0]
        source_one = mixture * 0.6
        source_two = mixture * 0.4
        return torch.stack([source_one, source_two], dim=0).unsqueeze(0)


class CountingSeparationModel:
    """SpeechBrain-shaped fake model that records chunked inference calls."""

    def __init__(self) -> None:
        self.calls = 0

    def separate_batch(self, waveform: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        mixture = waveform[0]
        source_one = mixture * 0.7
        source_two = mixture * -0.3
        return torch.stack([source_one, source_two], dim=1).unsqueeze(0)


def _tone(duration_seconds: float, frequency: float) -> np.ndarray:
    """Create a synthetic tone."""

    samples = int(duration_seconds * SAMPLE_RATE)
    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Write a mono PCM WAV fixture."""

    pcm_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(exist_ok=True)
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
