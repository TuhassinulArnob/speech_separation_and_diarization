"""Voice Activity Detection for the diarization pipeline.

The public API is ``detect_speech_regions``. It loads a mono WAV file,
detects speech regions with a configurable backend, post-processes timestamps,
and optionally exports the detected regions as JSON.

Example:
    python -m Project.ssl_pipeline.vad Project/ssl_pipeline/processed.wav \
        --output outputs/vad_regions.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from Project.ssl_pipeline.config import (
    DiarizationConfig,
    get_huggingface_token,
    load_config,
)

LOGGER = logging.getLogger(__name__)

SUPPORTED_BACKENDS = {"auto", "pyannote", "webrtc", "energy"}
WEBRTC_VALID_FRAME_DURATIONS_MS = {10, 20, 30}


@dataclass(frozen=True)
class SpeechRegion:
    """A detected speech interval in seconds."""

    start_time: float
    end_time: float

    def as_tuple(self) -> tuple[float, float]:
        """Return the region as a timestamp tuple."""

        return (self.start_time, self.end_time)


def detect_speech_regions(
    audio_path: str | Path,
    *,
    config_path: str | Path | None = None,
    backend: str | None = None,
    export_json_path: str | Path | None = None,
) -> list[tuple[float, float]]:
    """Detect speech timestamps from a mono WAV audio file.

    Args:
        audio_path: Path to a WAV audio file.
        config_path: Optional project configuration path.
        backend: Optional VAD backend override. Supported values are
            ``auto``, ``pyannote``, ``webrtc``, and ``energy``.
        export_json_path: Optional path where timestamps should be written.

    Returns:
        A list of ``(start_time, end_time)`` tuples in seconds.
    """

    config = load_config(config_path) if config_path else load_config()
    selected_backend = (backend or config.vad_backend).lower()
    if selected_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported VAD backend '{selected_backend}'. "
            f"Choose one of {sorted(SUPPORTED_BACKENDS)}."
        )

    path = Path(audio_path)
    audio, sample_rate = load_wav_audio(path, config)

    regions = _run_backend(path, audio, sample_rate, selected_backend, config)
    processed_regions = postprocess_regions(regions, config)
    output = [region.as_tuple() for region in processed_regions]

    if export_json_path is not None:
        export_speech_regions(output, export_json_path)

    LOGGER.info(
        "Detected %d speech regions in %s using %s backend",
        len(output),
        path,
        selected_backend,
    )
    return output


def load_wav_audio(
    audio_path: str | Path,
    config: DiarizationConfig,
) -> tuple[np.ndarray, int]:
    """Load and validate a WAV file for VAD."""

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sample_rate, channels = _read_wav(path)
    if sample_rate != config.sample_rate:
        raise ValueError(
            f"Expected {config.sample_rate} Hz audio, got {sample_rate} Hz."
        )

    if channels != 1:
        raise ValueError("VAD expects mono audio. Run preprocessing first.")

    LOGGER.debug("Loaded %s for VAD at %d Hz", path, sample_rate)
    return audio, sample_rate


def _read_wav(audio_path: Path) -> tuple[np.ndarray, int, int]:
    """Read a PCM WAV file as mono float32 audio."""

    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(
            f"Unsupported WAV sample width: {sample_width} bytes. "
            "Use 8-bit or 16-bit PCM WAV."
        )

    if channels > 1:
        audio = audio.reshape(-1, channels)

    return audio, sample_rate, channels


def postprocess_regions(
    regions: Iterable[SpeechRegion],
    config: DiarizationConfig,
) -> list[SpeechRegion]:
    """Merge nearby regions and remove speech intervals that are too short."""

    sorted_regions = sorted(regions, key=lambda region: region.start_time)
    if not sorted_regions:
        return []

    merge_gap = max(
        config.vad_min_silence_duration_seconds,
        config.vad_merge_threshold_seconds,
    )

    merged: list[SpeechRegion] = []
    current = sorted_regions[0]
    for region in sorted_regions[1:]:
        gap = region.start_time - current.end_time
        if gap <= merge_gap:
            current = SpeechRegion(
                current.start_time,
                max(current.end_time, region.end_time),
            )
        else:
            merged.append(current)
            current = region
    merged.append(current)

    minimum_duration = config.vad_min_speech_duration_seconds
    filtered = [
        region
        for region in merged
        if region.end_time - region.start_time >= minimum_duration
    ]
    LOGGER.debug(
        "Post-processed %d raw regions into %d speech regions",
        len(sorted_regions),
        len(filtered),
    )
    return filtered


def export_speech_regions(
    regions: list[tuple[float, float]],
    output_path: str | Path,
) -> None:
    """Export speech regions to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"start_time": round(start, 3), "end_time": round(end, 3)}
        for start, end in regions
    ]
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    LOGGER.info("Exported VAD regions to %s", path)


def _run_backend(
    audio_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    backend: str,
    config: DiarizationConfig,
) -> list[SpeechRegion]:
    """Run the selected VAD backend."""

    if backend == "pyannote":
        return _detect_with_pyannote(audio_path, config)
    if backend == "webrtc":
        return _detect_with_webrtc(audio, sample_rate, config)
    if backend == "energy":
        return _detect_with_energy(audio, sample_rate, config)

    for fallback_backend in ("pyannote", "webrtc", "energy"):
        try:
            LOGGER.info("Trying VAD backend: %s", fallback_backend)
            return _run_backend(audio_path, audio, sample_rate, fallback_backend, config)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning("VAD backend %s unavailable: %s", fallback_backend, exc)

    raise RuntimeError("No VAD backend could process the audio.")


def _detect_with_pyannote(
    audio_path: Path,
    config: DiarizationConfig,
) -> list[SpeechRegion]:
    """Detect speech with pyannote.audio VAD."""

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError("pyannote.audio is not installed.") from exc

    token = get_huggingface_token(config.token_env_var, required=False)
    pipeline = _load_pyannote_pipeline(Pipeline, config.vad_pyannote_model, token)
    if pipeline is None:
        raise RuntimeError(f"Could not load pyannote model: {config.vad_pyannote_model}")

    result = pipeline(str(audio_path))
    return _regions_from_pyannote_result(result)


def _load_pyannote_pipeline(
    pipeline_class: object,
    model_name: str,
    token: str | None,
) -> object:
    """Load pyannote Pipeline across old and new authentication APIs."""

    from_pretrained = getattr(pipeline_class, "from_pretrained")
    if not token:
        return from_pretrained(model_name)

    try:
        parameters = inspect.signature(from_pretrained).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "token" in parameters:
        return from_pretrained(model_name, token=token)
    if "use_auth_token" in parameters:
        return from_pretrained(model_name, use_auth_token=token)

    try:
        return from_pretrained(model_name, token=token)
    except TypeError as exc:
        if "token" not in str(exc):
            raise
    return from_pretrained(model_name, use_auth_token=token)


def _regions_from_pyannote_result(result: object) -> list[SpeechRegion]:
    """Convert pyannote output objects into speech regions."""

    if hasattr(result, "get_timeline"):
        timeline = result.get_timeline()
        return [
            SpeechRegion(float(segment.start), float(segment.end))
            for segment in timeline
        ]

    if hasattr(result, "itertracks"):
        return [
            SpeechRegion(float(turn.start), float(turn.end))
            for turn, _, _ in result.itertracks(yield_label=True)
        ]

    if hasattr(result, "itersegments"):
        return [
            SpeechRegion(float(segment.start), float(segment.end))
            for segment in result.itersegments()
        ]

    raise RuntimeError("Unsupported pyannote VAD output format.")


def _detect_with_webrtc(
    audio: np.ndarray,
    sample_rate: int,
    config: DiarizationConfig,
) -> list[SpeechRegion]:
    """Detect speech with WebRTC VAD."""

    try:
        import webrtcvad
    except ImportError as exc:
        raise ImportError("webrtcvad is not installed.") from exc

    if config.vad_frame_duration_ms not in WEBRTC_VALID_FRAME_DURATIONS_MS:
        raise ValueError("WebRTC VAD frame duration must be 10, 20, or 30 ms.")

    vad = webrtcvad.Vad(config.vad_webrtc_aggressiveness)
    pcm_audio = _float_to_pcm16(audio)
    frame_size = int(sample_rate * config.vad_frame_duration_ms / 1000)
    frame_bytes = frame_size * 2
    regions: list[SpeechRegion] = []
    active_start: float | None = None

    for index, frame in enumerate(_iter_pcm_frames(pcm_audio, frame_bytes)):
        start_time = index * config.vad_frame_duration_ms / 1000
        end_time = start_time + config.vad_frame_duration_ms / 1000
        is_speech = vad.is_speech(frame, sample_rate)

        if is_speech and active_start is None:
            active_start = start_time
        elif not is_speech and active_start is not None:
            regions.append(SpeechRegion(active_start, end_time))
            active_start = None

    if active_start is not None:
        duration = len(audio) / sample_rate
        regions.append(SpeechRegion(active_start, duration))

    return regions


def _detect_with_energy(
    audio: np.ndarray,
    sample_rate: int,
    config: DiarizationConfig,
) -> list[SpeechRegion]:
    """Detect speech-like regions with frame energy.

    This lightweight backend keeps tests deterministic and provides a local
    fallback when pyannote and WebRTC are unavailable. It adaptively raises the
    threshold above the background noise floor when dynamic range is present,
    avoiding classifying an entire noisy or peak-normalized recording as speech.
    """

    frame_size = int(sample_rate * config.vad_frame_duration_ms / 1000)
    if frame_size <= 0:
        raise ValueError("VAD frame duration must produce at least one sample.")

    threshold = _compute_adaptive_energy_threshold(
        audio,
        frame_size,
        config.vad_energy_threshold,
    )

    regions: list[SpeechRegion] = []
    active_start: float | None = None

    for index, frame in enumerate(_iter_audio_frames(audio, frame_size)):
        start_time = index * frame_size / sample_rate
        end_time = min((index + 1) * frame_size / sample_rate, len(audio) / sample_rate)
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
        is_speech = rms >= threshold

        if is_speech and active_start is None:
            active_start = start_time
        elif not is_speech and active_start is not None:
            regions.append(SpeechRegion(active_start, start_time))
            active_start = None

    if active_start is not None:
        regions.append(SpeechRegion(active_start, len(audio) / sample_rate))

    return regions


def _compute_adaptive_energy_threshold(
    audio: np.ndarray,
    frame_size: int,
    min_threshold: float,
) -> float:
    """Compute an energy threshold that separates speech from background noise floor."""

    if len(audio) < frame_size:
        return min_threshold

    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return min_threshold

    frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms_values = np.sqrt(np.mean(frames ** 2, axis=1))

    p10 = float(np.percentile(rms_values, 10))
    p95 = float(np.percentile(rms_values, 95))

    if p95 > min_threshold and (p95 - p10) > min_threshold:
        adaptive_threshold = p10 + 0.15 * (p95 - p10)
        return max(min_threshold, adaptive_threshold)

    return min_threshold


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float audio in [-1, 1] to 16-bit PCM bytes."""

    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _iter_pcm_frames(pcm_audio: bytes, frame_bytes: int) -> Iterable[bytes]:
    """Yield complete PCM frames for WebRTC VAD."""

    for start in range(0, len(pcm_audio) - frame_bytes + 1, frame_bytes):
        yield pcm_audio[start : start + frame_bytes]


def _iter_audio_frames(audio: np.ndarray, frame_size: int) -> Iterable[np.ndarray]:
    """Yield fixed-size audio frames, including the final partial frame."""

    for start in range(0, len(audio), frame_size):
        yield audio[start : start + frame_size]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone VAD usage."""

    parser = argparse.ArgumentParser(description="Detect speech regions in a WAV file.")
    parser.add_argument("audio_path", type=Path, help="Path to a 16 kHz mono WAV file.")
    parser.add_argument(
        "--backend",
        choices=sorted(SUPPORTED_BACKENDS),
        default=None,
        help="Override the VAD backend configured in Project/config/default.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file for exported speech timestamps.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON configuration path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run VAD from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    regions = detect_speech_regions(
        args.audio_path,
        config_path=args.config,
        backend=args.backend,
        export_json_path=args.output,
    )
    for start_time, end_time in regions:
        print(f"{start_time:.3f}\t{end_time:.3f}")


if __name__ == "__main__":
    main()
