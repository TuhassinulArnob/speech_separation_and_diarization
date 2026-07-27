"""True speech separation with pretrained separation models.

This module reconstructs separate speaker waveforms from mixed speech. It is
independent from the diarization pipeline and does not cut audio by timeline.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly, spectrogram

from Project.ssl_pipeline.config import (
    DEFAULT_TOKEN_ENV_VAR,
    DiarizationConfig,
    load_config,
)

LOGGER = logging.getLogger(__name__)

SUPPORTED_BACKENDS = {"speechbrain", "asteroid", "auto"}


@dataclass(frozen=True)
class SeparationResult:
    """Separated speaker waveform outputs."""

    input_path: Path
    output_paths: list[Path]
    sample_rate: int
    backend: str
    model_name: str
    duration: float


class SpeechSeparationModelLoader:
    """Lazy pretrained model loader for speech separation."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        backend: str | None = None,
        model_name: str | None = None,
        fallback_model_name: str | None = None,
        device: str | None = None,
        model: object | None = None,
    ) -> None:
        self.config = config or load_config()
        self.backend = (backend or self.config.separation_model_backend).lower()
        self.model_name = model_name or self.config.separation_model_name
        self.fallback_model_name = (
            fallback_model_name or self.config.separation_fallback_model_name
        )
        self.device = resolve_separation_device(device or self.config.separation_device)
        self.model = model
        self.loaded_backend = "injected" if model is not None else self.backend
        self.loaded_model_name = "injected" if model is not None else self.model_name
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported separation backend: {self.backend}")

    def load_model(self) -> object:
        """Load and cache a pretrained separation model."""

        if self.model is not None:
            return self.model

        if self.backend in {"speechbrain", "auto"}:
            try:
                self.model = self._load_speechbrain(self.model_name)
                self.loaded_backend = "speechbrain"
                self.loaded_model_name = self.model_name
                return self.model
            except (ImportError, OSError, RuntimeError) as exc:
                if self.backend == "speechbrain":
                    raise RuntimeError(
                        "SpeechBrain SepFormer could not be loaded. Install "
                        "speechbrain and set HUGGINGFACE_TOKEN if the selected "
                        "model requires access."
                    ) from exc
                LOGGER.warning("SpeechBrain separation unavailable: %s", exc)

        if self.backend in {"asteroid", "auto"}:
            try:
                self.model = self._load_asteroid(self.fallback_model_name)
                self.loaded_backend = "asteroid"
                self.loaded_model_name = self.fallback_model_name
                return self.model
            except (ImportError, OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "No pretrained speech separation backend could be loaded. "
                    "Install speechbrain or asteroid, and set HUGGINGFACE_TOKEN "
                    "only if the model requires authentication."
                ) from exc

        raise RuntimeError("No separation backend was loaded.")

    def _load_speechbrain(self, model_name: str) -> object:
        """Load SpeechBrain SepFormer from pretrained weights."""

        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation

        token = get_huggingface_token(required=False)
        kwargs = {
            "source": model_name,
            "savedir": str(Path("outputs") / "model_cache" / "speechbrain_sepformer"),
            "run_opts": {"device": str(self.device)},
        }
        if token:
            kwargs["use_auth_token"] = token
        return SepformerSeparation.from_hparams(**kwargs)

    def _load_asteroid(self, model_name: str) -> object:
        """Load Asteroid Conv-TasNet from pretrained weights."""

        try:
            from asteroid.models import BaseModel
        except ImportError as exc:
            raise ImportError("asteroid is not installed.") from exc

        model = BaseModel.from_pretrained(model_name)
        if hasattr(model, "to"):
            model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        return model


class SpeechSeparator:
    """Run true speech separation and save reconstructed speaker waveforms."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        model_loader: SpeechSeparationModelLoader | None = None,
        model: object | None = None,
        backend: str | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.model_loader = model_loader or SpeechSeparationModelLoader(
            self.config,
            model=model,
            backend=backend,
            device=device,
        )
        self.sample_rate = self.config.separation_sample_rate
        self.num_speakers = self.config.separation_num_speakers
        self.chunk_duration_seconds = self.config.separation_chunk_duration_seconds
        self.chunk_overlap_seconds = self.config.separation_chunk_overlap_seconds
        if self.num_speakers <= 0:
            raise ValueError("separation_num_speakers must be positive.")
        if self.chunk_duration_seconds < 0.0:
            raise ValueError("separation_chunk_duration_seconds cannot be negative.")
        if self.chunk_overlap_seconds < 0.0:
            raise ValueError("separation_chunk_overlap_seconds cannot be negative.")
        if (
            self.chunk_duration_seconds > 0.0
            and self.chunk_overlap_seconds >= self.chunk_duration_seconds
        ):
            raise ValueError(
                "separation_chunk_overlap_seconds must be smaller than "
                "separation_chunk_duration_seconds."
            )

    def separate_file(
        self,
        audio_path: str | Path,
        *,
        output_dir: str | Path | None = None,
    ) -> SeparationResult:
        """Separate one mixed audio file and save speaker waveforms."""

        path = Path(audio_path)
        audio, sample_rate = load_audio(path, target_sample_rate=self.sample_rate)
        separated = self.separate_waveform(audio, sample_rate)
        target_dir = Path(output_dir or self.config.separation_output_dir)
        output_paths = save_separated_audio(separated, sample_rate, target_dir)
        return SeparationResult(
            input_path=path,
            output_paths=output_paths,
            sample_rate=sample_rate,
            backend=self.model_loader.loaded_backend,
            model_name=self.model_loader.loaded_model_name,
            duration=len(audio) / sample_rate if sample_rate else 0.0,
        )

    def separate_waveform(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> list[np.ndarray]:
        """Separate an in-memory mixed waveform into speaker streams."""

        if len(audio) == 0:
            raise ValueError("Cannot separate empty audio.")
        if np.max(np.abs(audio)) == 0.0:
            return [
                np.zeros_like(audio, dtype=np.float32)
                for _ in range(self.num_speakers)
            ]

        model = self.model_loader.load_model()
        separated = _infer_separation_with_optional_chunking(
            model,
            audio,
            sample_rate,
            expected_num_speakers=self.num_speakers,
            backend=self.model_loader.loaded_backend,
            device=self.model_loader.device,
            chunk_duration_seconds=self.chunk_duration_seconds,
            chunk_overlap_seconds=self.chunk_overlap_seconds,
        )
        return _format_sources(
            separated,
            expected_length=len(audio),
            expected_num_speakers=self.num_speakers,
        )


def separate_speech(
    audio_path: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    backend: str | None = None,
    device: str | None = None,
) -> SeparationResult:
    """Convenience function for true speech separation."""

    config = load_config(config_path) if config_path else load_config()
    separator = SpeechSeparator(config, backend=backend, device=device)
    return separator.separate_file(audio_path, output_dir=output_dir)


def load_audio(
    audio_path: str | Path,
    *,
    target_sample_rate: int,
) -> tuple[np.ndarray, int]:
    """Load mono audio and resample for separation."""

    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sample_rate = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
        sample_rate = target_sample_rate
    return audio, sample_rate


def save_separated_audio(
    sources: Iterable[np.ndarray],
    sample_rate: int,
    output_dir: str | Path,
) -> list[Path]:
    """Save separated sources as speaker1.wav, speaker2.wav, ..."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for index, source in enumerate(sources, start=1):
        output_path = path / f"speaker{index}.wav"
        sf.write(output_path, np.asarray(source, dtype=np.float32), sample_rate)
        output_paths.append(output_path)
    LOGGER.info("Saved %d separated sources to %s", len(output_paths), path)
    return output_paths


def plot_waveform_comparison(
    mixture: np.ndarray,
    separated_sources: Iterable[np.ndarray],
    sample_rate: int,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot original waveform and separated waveforms."""

    sources = list(separated_sources)
    rows = 1 + len(sources)
    figure, axes = plt.subplots(rows, 1, figsize=(10, 2.5 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    time = np.arange(len(mixture)) / sample_rate
    axes[0].plot(time, mixture)
    axes[0].set_title("Original Mixture")
    for index, source in enumerate(sources, start=1):
        source_time = np.arange(len(source)) / sample_rate
        axes[index].plot(source_time, source)
        axes[index].set_title(f"Separated Speaker {index}")
    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_spectrogram_comparison(
    mixture: np.ndarray,
    separated_sources: Iterable[np.ndarray],
    sample_rate: int,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot spectrograms for the original and separated waveforms."""

    signals = [mixture, *list(separated_sources)]
    titles = ["Original Mixture", *[f"Separated Speaker {i}" for i in range(1, len(signals))]]
    figure, axes = plt.subplots(len(signals), 1, figsize=(10, 3 * len(signals)))
    axes = np.atleast_1d(axes)
    for axis, signal, title in zip(axes, signals, titles):
        frequencies, times, values = spectrogram(signal, fs=sample_rate)
        axis.pcolormesh(times, frequencies, 10 * np.log10(values + 1e-10), shading="auto")
        axis.set_title(title)
        axis.set_ylabel("Hz")
    axes[-1].set_xlabel("Time (seconds)")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def resolve_separation_device(device: str) -> torch.device:
    """Resolve configured separation device."""

    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("separation_device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(normalized)


def get_huggingface_token(
    env_var: str = DEFAULT_TOKEN_ENV_VAR,
    *,
    required: bool = False,
) -> str | None:
    """Read a Hugging Face token from environment when needed."""

    token = os.getenv(env_var)
    if token:
        return token
    if required:
        raise OSError(f"Missing Hugging Face token. Set {env_var}.")
    return None


def _infer_separation(
    model: object,
    audio: np.ndarray,
    sample_rate: int,
    *,
    backend: str,
    device: torch.device,
) -> np.ndarray:
    """Run separation model inference and return a source matrix."""

    if hasattr(model, "separate_batch"):
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.separate_batch(waveform)
        if hasattr(output, "detach"):
            return output.detach().cpu().numpy()

    if hasattr(model, "separate_file"):
        raise RuntimeError("Model exposes separate_file only; use injected test model or backend loader.")

    if callable(model):
        result = model(audio, sample_rate)
        if hasattr(result, "detach"):
            return result.detach().cpu().numpy()
        return np.asarray(result)

    if backend == "asteroid":
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(waveform)
        return output.detach().cpu().numpy()

    raise RuntimeError("Unsupported separation model interface.")


def _infer_separation_with_optional_chunking(
    model: object,
    audio: np.ndarray,
    sample_rate: int,
    *,
    expected_num_speakers: int,
    backend: str,
    device: torch.device,
    chunk_duration_seconds: float,
    chunk_overlap_seconds: float,
) -> np.ndarray:
    """Run whole-file or chunked separation depending on configured length."""

    chunk_samples = int(round(chunk_duration_seconds * sample_rate))
    if chunk_samples <= 0 or len(audio) <= chunk_samples:
        return _infer_separation(
            model,
            audio,
            sample_rate,
            backend=backend,
            device=device,
        )

    overlap_samples = int(round(chunk_overlap_seconds * sample_rate))
    return _infer_separation_in_chunks(
        model,
        audio,
        sample_rate,
        expected_num_speakers=expected_num_speakers,
        backend=backend,
        device=device,
        chunk_samples=chunk_samples,
        overlap_samples=overlap_samples,
    )


def _infer_separation_in_chunks(
    model: object,
    audio: np.ndarray,
    sample_rate: int,
    *,
    expected_num_speakers: int,
    backend: str,
    device: torch.device,
    chunk_samples: int,
    overlap_samples: int,
) -> np.ndarray:
    """Separate long audio in overlapped chunks and stitch source streams."""

    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive.")
    if overlap_samples < 0:
        raise ValueError("overlap_samples cannot be negative.")
    if overlap_samples >= chunk_samples:
        raise ValueError("overlap_samples must be smaller than chunk_samples.")

    hop_samples = chunk_samples - overlap_samples
    output = np.zeros((expected_num_speakers, len(audio)), dtype=np.float32)
    weights = np.zeros(len(audio), dtype=np.float32)

    for start in range(0, len(audio), hop_samples):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        raw_sources = _infer_separation(
            model,
            chunk,
            sample_rate,
            backend=backend,
            device=device,
        )
        sources = _format_sources(
            raw_sources,
            expected_length=len(chunk),
            expected_num_speakers=expected_num_speakers,
            normalize_peaks=False,
        )
        window = _chunk_crossfade_window(
            len(chunk),
            overlap_samples=overlap_samples,
            has_previous=start > 0,
            has_next=end < len(audio),
        )
        for source_index, source in enumerate(sources[:expected_num_speakers]):
            output[source_index, start:end] += source * window
        weights[start:end] += window
        if end >= len(audio):
            break

    safe_weights = np.maximum(weights, 1e-8)
    output = output / safe_weights[np.newaxis, :]
    return output


def _chunk_crossfade_window(
    length: int,
    *,
    overlap_samples: int,
    has_previous: bool,
    has_next: bool,
) -> np.ndarray:
    """Create a linear overlap-add window for one chunk."""

    window = np.ones(length, dtype=np.float32)
    fade = min(overlap_samples, length)
    if fade <= 0:
        return window
    if has_previous:
        window[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    if has_next:
        window[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False, dtype=np.float32)
    return window


def _format_sources(
    separated: np.ndarray,
    *,
    expected_length: int,
    expected_num_speakers: int | None = None,
    normalize_peaks: bool = True,
) -> list[np.ndarray]:
    """Normalize model output shape to a list of one-dimensional sources."""

    array = np.asarray(separated, dtype=np.float32)

    if array.ndim == 3:
        if array.shape[0] != 1:
            raise RuntimeError(
                f"Expected one separated batch, got output shape {array.shape}."
            )
        array = array[0]
        if expected_num_speakers is not None and array.shape[-1] == expected_num_speakers:
            array = array.T
        elif (
            expected_num_speakers is not None
            and array.shape[0] == expected_num_speakers
        ):
            pass
        elif array.shape[-1] <= 4:
            array = array.T
        elif array.shape[0] <= 4:
            pass
        else:
            raise RuntimeError(
                f"Cannot determine speaker dimension from output shape {array.shape}."
            )
    elif array.ndim == 2:
        if expected_num_speakers is not None and array.shape[0] == expected_num_speakers:
            pass
        elif (
            expected_num_speakers is not None
            and array.shape[1] == expected_num_speakers
        ):
            array = array.T
        elif array.shape[0] <= 4 and array.shape[1] > array.shape[0]:
            pass
        elif array.shape[1] <= 4 and array.shape[0] > array.shape[1]:
            array = array.T
        else:
            raise RuntimeError(
                f"Cannot determine speaker dimension from output shape {array.shape}."
            )
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    else:
        raise RuntimeError(
            f"Unexpected separation output shape: {array.shape}"
        )

    sources: list[np.ndarray] = []
    for source in array:
        source = np.asarray(source, dtype=np.float32)
        if len(source) < expected_length:
            source = np.pad(source, (0, expected_length - len(source)))
        elif len(source) > expected_length:
            source = source[:expected_length]

        peak = float(np.max(np.abs(source))) if len(source) else 0.0
        if normalize_peaks and peak > 1.0:
            source = source / peak
        sources.append(source.astype(np.float32))

    return sources


def _save_plot_if_requested(
    figure: plt.Figure,
    output_path: str | Path | None,
) -> None:
    """Save a Matplotlib figure if requested."""

    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    LOGGER.info("Saved speech separation plot to %s", path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for speech separation."""

    parser = argparse.ArgumentParser(description="Separate mixed speech with a pretrained model.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=sorted(SUPPORTED_BACKENDS), default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    return parser.parse_args()


def main() -> None:
    """Run speech separation from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    result = separate_speech(
        args.audio_path,
        config_path=args.config,
        output_dir=args.output_dir,
        backend=args.backend,
        device=args.device,
    )
    for path in result.output_paths:
        print(path)


if __name__ == "__main__":
    main()
