"""Smoke-test pyannote diarization on a local audio file.

Set ``HUGGINGFACE_TOKEN`` in the environment before running this script.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyannote.audio import Pipeline

from Project.ssl_pipeline.config import get_huggingface_token, load_config

LOGGER = logging.getLogger(__name__)


def run_diarization(audio_path: str | Path = "audio.mp3") -> None:
    """Run the configured pyannote pipeline and log speaker turns."""

    config = load_config()
    token = get_huggingface_token(config.token_env_var)
    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    pipeline = Pipeline.from_pretrained(
        config.pyannote_model,
        use_auth_token=token,
    )
    diarization = pipeline(str(audio_file))

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        LOGGER.info("%.1fs --> %.1fs : %s", turn.start, turn.end, speaker)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_diarization()
