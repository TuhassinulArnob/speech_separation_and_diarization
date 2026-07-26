"""Tests for the Voice Activity Detection module."""

from __future__ import annotations

import json
import unittest
import wave
from pathlib import Path

import numpy as np

from Project.ssl_pipeline.vad import detect_speech_regions


SAMPLE_RATE = 16_000
TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class VadTests(unittest.TestCase):
    """Validate VAD behavior on synthetic audio."""

    def test_detects_speech_audio(self) -> None:
        audio_path = _fixture_path("speech.wav")
        _write_wav(audio_path, _tone(1.0))

        regions = detect_speech_regions(audio_path, backend="energy")

        self.assertEqual(len(regions), 1)
        self.assertLessEqual(regions[0][0], 0.03)
        self.assertGreaterEqual(regions[0][1], 0.9)

    def test_returns_empty_for_silent_audio(self) -> None:
        audio_path = _fixture_path("silence.wav")
        _write_wav(audio_path, np.zeros(SAMPLE_RATE, dtype=np.float32))

        regions = detect_speech_regions(audio_path, backend="energy")

        self.assertEqual(regions, [])

    def test_detects_mixed_audio_and_merges_short_silence(self) -> None:
        audio_path = _fixture_path("mixed.wav")
        audio = np.concatenate(
            [
                np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32),
                _tone(0.5),
                np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32),
                _tone(0.5),
                np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32),
            ]
        )
        _write_wav(audio_path, audio)

        regions = detect_speech_regions(audio_path, backend="energy")

        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0][0], 0.3, delta=0.05)
        self.assertAlmostEqual(regions[0][1], 1.4, delta=0.08)

    def test_filters_very_short_speech(self) -> None:
        audio_path = _fixture_path("short.wav")
        audio = np.concatenate(
            [
                np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32),
                _tone(0.1),
                np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32),
            ]
        )
        _write_wav(audio_path, audio)

        regions = detect_speech_regions(audio_path, backend="energy")

        self.assertEqual(regions, [])

    def test_exports_regions_to_json(self) -> None:
        audio_path = _fixture_path("export_speech.wav")
        output_path = _fixture_path("regions.json")
        _write_wav(audio_path, _tone(0.5))

        regions = detect_speech_regions(
            audio_path,
            backend="energy",
            export_json_path=output_path,
        )

        exported = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(len(regions), 1)
        self.assertEqual(len(exported), 1)
        self.assertIn("start_time", exported[0])
        self.assertIn("end_time", exported[0])


def _tone(duration_seconds: float) -> np.ndarray:
    """Create a synthetic speech-like sine wave."""

    samples = int(duration_seconds * SAMPLE_RATE)
    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Write a mono WAV file for VAD tests."""

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
