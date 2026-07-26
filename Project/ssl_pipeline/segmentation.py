"""Speech segmentation based on VAD timestamps.

This module is the official speech segmentation stage of the diarization
pipeline. It does not segment the whole raw recording. Instead, it consumes
speech regions produced by VAD and creates shorter windows suitable for WavLM
embedding extraction.

Example:
    python -m Project.ssl_pipeline.segmentation Project/ssl_pipeline/processed.wav \
        --vad-backend energy --output outputs/segments.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.vad import detect_speech_regions

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXPORT_FORMATS = {".csv", ".json"}


@dataclass(frozen=True)
class SpeechSegment:
    """A speech-only segment prepared for embedding extraction."""

    segment_id: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Return segment duration in seconds."""

        return self.end - self.start

    def to_dict(self) -> dict[str, float | int]:
        """Serialize the segment into a dictionary."""

        return asdict(self)


def segment_speech(
    speech_regions: Iterable[tuple[float, float]] | None = None,
    *,
    audio_path: str | Path | None = None,
    config_path: str | Path | None = None,
    vad_backend: str | None = None,
) -> list[SpeechSegment]:
    """Create fixed-length speech segments from VAD regions.

    Args:
        speech_regions: Speech timestamps from VAD as ``(start, end)`` tuples.
        audio_path: Optional audio file. If ``speech_regions`` is not provided,
            VAD is run on this file using ``detect_speech_regions``.
        config_path: Optional project configuration path.
        vad_backend: Optional backend override passed to VAD when ``audio_path``
            is used.

    Returns:
        Ordered speech segments with stable integer IDs.

    Raises:
        ValueError: If neither ``speech_regions`` nor ``audio_path`` is given,
            or if segmentation configuration is invalid.
    """

    config = load_config(config_path) if config_path else load_config()
    _validate_segmentation_config(config)

    if speech_regions is None:
        if audio_path is None:
            raise ValueError("Provide speech_regions or audio_path.")
        speech_regions = detect_speech_regions(
            audio_path,
            config_path=config_path,
            backend=vad_backend,
        )

    normalized_regions = _normalize_regions(speech_regions)
    segments = _create_segments(normalized_regions, config)

    LOGGER.info(
        "Created %d speech segments from %d VAD regions",
        len(segments),
        len(normalized_regions),
    )
    return segments


def save_segments(
    segments: Iterable[SpeechSegment],
    output_path: str | Path,
) -> None:
    """Save speech segments as JSON or CSV based on file extension."""

    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError("Segments can only be saved as .json or .csv.")

    path.parent.mkdir(parents=True, exist_ok=True)
    segment_rows = [segment.to_dict() for segment in segments]

    if suffix == ".json":
        with path.open("w", encoding="utf-8") as file:
            json.dump(segment_rows, file, indent=2)
    else:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["segment_id", "start", "end"])
            writer.writeheader()
            writer.writerows(segment_rows)

    LOGGER.info("Saved %d speech segments to %s", len(segment_rows), path)


def load_segments(input_path: str | Path) -> list[SpeechSegment]:
    """Load speech segments from JSON or CSV."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Segments file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        raise ValueError("Segments can only be loaded from .json or .csv.")

    segments = [
        SpeechSegment(
            segment_id=int(row["segment_id"]),
            start=float(row["start"]),
            end=float(row["end"]),
        )
        for row in rows
    ]
    LOGGER.info("Loaded %d speech segments from %s", len(segments), path)
    return segments


def segment_audio(
    audio_path: str | Path,
    segment_duration: float | None = None,
) -> list[SpeechSegment]:
    """Compatibility wrapper that now segments only detected speech.

    New code should call ``segment_speech`` directly. ``segment_duration`` is
    accepted for older callers and temporarily overrides the configured window
    and maximum segment duration.
    """

    if segment_duration is None:
        return segment_speech(audio_path=audio_path)

    config = load_config()
    override_config = DiarizationConfig(
        sample_rate=config.sample_rate,
        mono=config.mono,
        normalize_audio=config.normalize_audio,
        segment_duration_seconds=segment_duration,
        segment_window_seconds=segment_duration,
        segment_hop_seconds=segment_duration,
        segment_min_duration_seconds=config.segment_min_duration_seconds,
        segment_max_duration_seconds=segment_duration,
        segment_include_short_final=config.segment_include_short_final,
        pyannote_model=config.pyannote_model,
        token_env_var=config.token_env_var,
        vad_backend=config.vad_backend,
        vad_pyannote_model=config.vad_pyannote_model,
        vad_min_speech_duration_seconds=config.vad_min_speech_duration_seconds,
        vad_min_silence_duration_seconds=config.vad_min_silence_duration_seconds,
        vad_merge_threshold_seconds=config.vad_merge_threshold_seconds,
        vad_frame_duration_ms=config.vad_frame_duration_ms,
        vad_webrtc_aggressiveness=config.vad_webrtc_aggressiveness,
        vad_energy_threshold=config.vad_energy_threshold,
    )
    speech_regions = detect_speech_regions(audio_path)
    _validate_segmentation_config(override_config)
    return _create_segments(_normalize_regions(speech_regions), override_config)


def _create_segments(
    speech_regions: list[tuple[float, float]],
    config: DiarizationConfig,
) -> list[SpeechSegment]:
    """Create windowed segments inside VAD speech regions."""

    segments: list[SpeechSegment] = []
    next_segment_id = 0

    for region_start, region_end in speech_regions:
        region_duration = region_end - region_start
        if region_duration < config.segment_min_duration_seconds:
            continue

        cursor = region_start
        while cursor < region_end:
            segment_end = min(cursor + config.segment_window_seconds, region_end)
            duration = segment_end - cursor

            if duration > config.segment_max_duration_seconds:
                segment_end = cursor + config.segment_max_duration_seconds
                duration = config.segment_max_duration_seconds

            if duration >= config.segment_min_duration_seconds:
                segments.append(
                    SpeechSegment(
                        segment_id=next_segment_id,
                        start=round(cursor, 3),
                        end=round(segment_end, 3),
                    )
                )
                next_segment_id += 1
            elif config.segment_include_short_final and not segments:
                segments.append(
                    SpeechSegment(
                        segment_id=next_segment_id,
                        start=round(cursor, 3),
                        end=round(segment_end, 3),
                    )
                )
                next_segment_id += 1

            if segment_end >= region_end:
                break
            cursor += config.segment_hop_seconds

    return segments


def _normalize_regions(
    speech_regions: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Validate and sort VAD timestamp tuples."""

    normalized: list[tuple[float, float]] = []
    for start, end in speech_regions:
        start_time = float(start)
        end_time = float(end)
        if start_time < 0:
            raise ValueError("Speech region start time cannot be negative.")
        if end_time <= start_time:
            raise ValueError("Speech region end time must be greater than start time.")
        normalized.append((start_time, end_time))

    return sorted(normalized, key=lambda region: region[0])


def _validate_segmentation_config(config: DiarizationConfig) -> None:
    """Validate segmentation parameters before creating windows."""

    if config.segment_window_seconds <= 0:
        raise ValueError("segment_window_seconds must be greater than zero.")
    if config.segment_hop_seconds <= 0:
        raise ValueError("segment_hop_seconds must be greater than zero.")
    if config.segment_min_duration_seconds <= 0:
        raise ValueError("segment_min_duration_seconds must be greater than zero.")
    if config.segment_max_duration_seconds <= 0:
        raise ValueError("segment_max_duration_seconds must be greater than zero.")
    if config.segment_min_duration_seconds > config.segment_max_duration_seconds:
        raise ValueError(
            "segment_min_duration_seconds cannot exceed "
            "segment_max_duration_seconds."
        )
    if config.segment_window_seconds > config.segment_max_duration_seconds:
        raise ValueError(
            "segment_window_seconds cannot exceed segment_max_duration_seconds."
        )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone speech segmentation."""

    parser = argparse.ArgumentParser(
        description="Create embedding-ready speech segments from VAD regions."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to a 16 kHz mono WAV file. VAD is run before segmentation.",
    )
    parser.add_argument(
        "--vad-backend",
        default=None,
        help="Optional VAD backend override: auto, pyannote, webrtc, or energy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .json or .csv output path for segments.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON configuration path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run speech segmentation from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    segments = segment_speech(
        audio_path=args.audio_path,
        config_path=args.config,
        vad_backend=args.vad_backend,
    )
    if args.output is not None:
        save_segments(segments, args.output)

    for segment in segments:
        print(f"{segment.segment_id}\t{segment.start:.3f}\t{segment.end:.3f}")


if __name__ == "__main__":
    main()
