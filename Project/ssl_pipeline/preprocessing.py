"""Audio preprocessing utilities."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import audioread
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from Project.ssl_pipeline.config import load_config

LOGGER = logging.getLogger(__name__)


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak <= 0.0:
        return audio
    return audio / peak


def _load_audio_file(file_path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(str(file_path), always_2d=False)
        return audio.astype(np.float32), sample_rate
    except RuntimeError:
        with audioread.audio_open(str(file_path)) as reader:
            channels = reader.channels
            sample_rate = reader.samplerate
            frames: list[np.ndarray] = []
            for frame in reader:
                array = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    array = array.reshape(-1, channels)
                frames.append(array)
            if not frames:
                return np.empty((0,), dtype=np.float32), sample_rate
            audio = np.concatenate(frames, axis=0)
            return audio, sample_rate


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio

    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor

    if audio.ndim == 1:
        return resample_poly(audio, up, down).astype(np.float32)

    channels = []
    for ch in range(audio.shape[1]):
        channels.append(resample_poly(audio[:, ch], up, down).astype(np.float32))
    return np.stack(channels, axis=1)


def preprocess_audio(input_path: str | Path, output_path: str | Path) -> None:
    """Resample, convert to mono, normalize, and save an audio file."""

    config = load_config()
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_file}")

    audio, source_sample_rate = _load_audio_file(input_file)

    if audio.ndim > 1 and config.mono:
        audio = audio.mean(axis=1)

    if source_sample_rate != config.sample_rate:
        audio = _resample_audio(audio, source_sample_rate, config.sample_rate)

    if config.normalize_audio:
        audio = _normalize_audio(audio)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, audio, config.sample_rate)
    LOGGER.info("Preprocessing complete: %s", output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline_dir = Path(__file__).resolve().parent

    preprocess_audio(
        pipeline_dir / "test.wav",
        pipeline_dir / "processed.wav",
    )
