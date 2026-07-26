"""Audio preprocessing utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import soundfile as sf

from Project.ssl_pipeline.config import load_config

LOGGER = logging.getLogger(__name__)


def preprocess_audio(input_path: str | Path, output_path: str | Path) -> None:
    """Resample, convert to mono, normalize, and save an audio file."""

    config = load_config()
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_file}")

    audio, _ = librosa.load(
        input_file,
        sr=config.sample_rate,
        mono=config.mono,
    )

    if config.normalize_audio:
        audio = librosa.util.normalize(audio)

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
