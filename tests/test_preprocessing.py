"""Unit tests for preprocessing audio loading and resampling."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from Project.ssl_pipeline.preprocessing import preprocess_audio


def _write_wav(path: Path, sample_rate: int, channels: int, duration: float = 0.1) -> None:
    samples = np.linspace(-0.5, 0.5, int(sample_rate * duration), dtype=np.float32)
    if channels == 2:
        samples = np.stack([samples, -samples], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate)


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("tests/tmp_preprocessing")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for child in self.temp_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                for sub in child.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                child.rmdir()

    def test_preprocess_wav_mono_output(self) -> None:
        input_path = self.temp_dir / "input_mono.wav"
        output_path = self.temp_dir / "processed.wav"
        _write_wav(input_path, sample_rate=22050, channels=1)

        preprocess_audio(input_path, output_path)

        self.assertTrue(output_path.exists())
        audio, sample_rate = sf.read(output_path, always_2d=False)
        self.assertEqual(sample_rate, 16000)
        self.assertTrue(audio.ndim == 1 or audio.shape[1] == 1)

    def test_preprocess_wav_stereo_output(self) -> None:
        input_path = self.temp_dir / "input_stereo.wav"
        output_path = self.temp_dir / "processed.wav"
        _write_wav(input_path, sample_rate=22050, channels=2)

        preprocess_audio(input_path, output_path)

        self.assertTrue(output_path.exists())
        audio, sample_rate = sf.read(output_path, always_2d=False)
        self.assertEqual(sample_rate, 16000)
        self.assertTrue(audio.ndim == 1 or audio.shape[1] == 1)


if __name__ == "__main__":
    unittest.main()
