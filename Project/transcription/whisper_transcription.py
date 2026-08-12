"""Whisper-based speaker-attributed transcription."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    import torch
    import whisper
    from whisper import Whisper
    from whisper.audio import SAMPLE_RATE
except ImportError:  # pragma: no cover
    whisper = None  # type: ignore
    torch = None  # type: ignore
    SAMPLE_RATE = 16000

LOGGER = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "base"
DEFAULT_WHISPER_LANGUAGE = "auto"
DEFAULT_OUTPUT_FORMATS = {"json", "txt", "srt"}


@dataclass(frozen=True)
class TranscriptSegment:
    speaker: str
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


class WhisperUnavailableError(RuntimeError):
    pass


def load_whisper_model(model_name: str = DEFAULT_WHISPER_MODEL, device: str | None = None) -> Whisper:
    if whisper is None:
        raise WhisperUnavailableError(
            "Whisper transcription is unavailable. Install openai-whisper in the Python environment."
        )
    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    LOGGER.info("Loading Whisper model '%s' on device %s", model_name, device)
    return whisper.load_model(model_name, device=device)


def _write_audio_slice(audio_path: Path, start: float, end: float) -> str:
    slice_path = audio_path.parent / f"transcription_slice_{start:.3f}_{end:.3f}.wav"
    with sf.SoundFile(str(audio_path), "r") as source:
        source.seek(int(round(start * source.samplerate)))
        frames = int(round((end - start) * source.samplerate))
        data = source.read(frames, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        sf.write(str(slice_path), data, source.samplerate, subtype="PCM_16")
    return str(slice_path)


def _format_srt(segments: list[TranscriptSegment]) -> str:
    lines = []
    for index, segment in enumerate(segments, start=1):
        start_h = _format_timestamp(segment.start)
        end_h = _format_timestamp(segment.end)
        lines.append(f"{index}")
        lines.append(f"{start_h} --> {end_h}")
        lines.append(segment.text.strip() or "[inaudible]")
        lines.append("")
    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transcribe_speaker_timeline(
    audio_path: str | Path,
    timeline_path: str | Path,
    output_dir: str | Path,
    model_name: str = DEFAULT_WHISPER_MODEL,
    language: str = DEFAULT_WHISPER_LANGUAGE,
    device: str | None = None,
    save_json: bool = True,
    save_txt: bool = True,
    save_srt: bool = True,
) -> list[TranscriptSegment]:
    source_path = Path(audio_path)
    output_dir = Path(output_dir)
    timeline_path = Path(timeline_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {source_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline file not found: {timeline_path}")

    model = load_whisper_model(model_name, device=device)
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    segments: list[TranscriptSegment] = []

    for entry in timeline:
        raw_speaker = str(entry.get("speaker_id", entry.get("speaker", "unknown")))
        speaker = _normalize_speaker_label(raw_speaker)
        start = float(entry["start"])
        end = float(entry["end"])
        if end <= start:
            LOGGER.warning("Skipping invalid timeline interval %s-%s for %s", start, end, speaker)
            continue

        slice_path = _write_audio_slice(source_path, start, end)
        try:
            whisper_language = None if str(language).lower() == "auto" else language
            result = model.transcribe(str(slice_path), language=whisper_language, task="transcribe")
        finally:
            Path(slice_path).unlink(missing_ok=True)

        transcribed = _collect_transcription(result, start)
        for segment in transcribed:
            segments.append(TranscriptSegment(speaker=speaker, start=segment[0], end=segment[1], text=segment[2]))

    segments.sort(key=lambda item: (item.start, item.end, item.speaker))
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_json:
        json_path = output_dir / "transcript.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump({"segments": [item.to_dict() for item in segments]}, file, indent=2, ensure_ascii=False)
        LOGGER.info("Saved transcript JSON to %s", json_path)

    if save_txt:
        txt_path = output_dir / "transcript.txt"
        with txt_path.open("w", encoding="utf-8") as file:
            for segment in segments:
                file.write(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.speaker}: {segment.text.strip()}\n")
        LOGGER.info("Saved transcript TXT to %s", txt_path)

    if save_srt:
        srt_path = output_dir / "transcript.srt"
        srt_path.write_text(_format_srt(segments), encoding="utf-8")
        LOGGER.info("Saved transcript SRT to %s", srt_path)

    return segments


def _normalize_speaker_label(label: str) -> str:
    label = label.strip()
    if not label:
        return "Speaker unknown"
    if label.startswith("speaker_"):
        return f"Speaker {label.split('speaker_')[1]}"
    if label.lower().startswith("speaker"):
        return label
    return label


def _collect_transcription(result: Any, offset: float) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    if not isinstance(result, dict):
        return segments
    text = str(result.get("text", "")).strip()
    if "segments" not in result or not isinstance(result["segments"], list):
        if text:
            duration = float(result.get("end", 0.0) or 0.0)
            segments.append((offset, offset + duration, text))
        return segments

    for item in result["segments"]:
        start = float(item.get("start", 0.0)) + offset
        end = float(item.get("end", 0.0)) + offset
        segments.append((start, end, str(item.get("text", "")).strip()))
    if not segments and text:
        duration = float(result.get("end", 0.0) or 0.0)
        segments.append((offset, offset + duration, text))
    return segments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Whisper speaker-attributed transcription.")
    parser.add_argument("--input", required=True, type=Path, help="Processed audio file to transcribe.")
    parser.add_argument("--timeline", required=True, type=Path, help="Speaker timeline JSON produced by diarization.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for transcript outputs.")
    parser.add_argument("--model", default=DEFAULT_WHISPER_MODEL, help="Whisper model size.")
    parser.add_argument("--language", default=DEFAULT_WHISPER_LANGUAGE, help="Language code or auto.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for Whisper inference.")
    parser.add_argument("--no-json", action="store_true", help="Do not save transcript.json.")
    parser.add_argument("--no-txt", action="store_true", help="Do not save transcript.txt.")
    parser.add_argument("--no-srt", action="store_true", help="Do not save transcript.srt.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = None if args.device == "auto" else args.device
    transcribe_speaker_timeline(
        audio_path=args.input,
        timeline_path=args.timeline,
        output_dir=args.output_dir,
        model_name=str(args.model),
        language=str(args.language),
        device=device,
        save_json=not args.no_json,
        save_txt=not args.no_txt,
        save_srt=not args.no_srt,
    )


if __name__ == "__main__":
    main()
