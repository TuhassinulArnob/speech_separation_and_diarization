"""Speaker-attributed transcription helpers."""

from .whisper_transcription import (
    TranscriptSegment,
    load_whisper_model,
    transcribe_speaker_timeline,
)

__all__ = [
    "TranscriptSegment",
    "load_whisper_model",
    "transcribe_speaker_timeline",
]
