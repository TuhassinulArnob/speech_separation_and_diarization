"""WavLM embedding extraction for diarization speech segments.

This module extracts one pretrained WavLM embedding per ``SpeechSegment``. It
does not normalize embeddings, cluster speakers, evaluate DER, or reconstruct
timelines; those are separate pipeline stages.

Example:
    python -m Project.ssl_pipeline.ssl_embedding Project/ssl_pipeline/processed.wav \
        --segments outputs/segments.json --cache outputs/embeddings/sample.npz
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

from Project.ssl_pipeline.config import (
    DEFAULT_TOKEN_ENV_VAR,
    DiarizationConfig,
    load_config,
)
from Project.ssl_pipeline.segmentation import SpeechSegment, load_segments
from Project.ssl_pipeline.vad import load_wav_audio

LOGGER = logging.getLogger(__name__)

SUPPORTED_POOLING_METHODS = {"mean", "max", "cls", "statistics"}


@dataclass(frozen=True)
class SegmentEmbedding:
    """A WavLM speaker embedding aligned to a speech segment."""

    segment_id: int
    start: float
    end: float
    vector: np.ndarray
    dimension: int
    model_name: str

    def to_metadata(self) -> dict[str, float | int | str]:
        """Return serializable metadata without the embedding vector."""

        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "dimension": self.dimension,
            "model_name": self.model_name,
        }


class WavLMEmbeddingExtractor:
    """Lazy WavLM model loader and batch embedding extractor."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        feature_extractor: object | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.config = config or load_config()
        self.device = resolve_device(self.config.embedding_device)
        self.feature_extractor = feature_extractor
        self.model = model
        self.model_name = self.config.embedding_model_name
        _validate_embedding_config(self.config)

    def load_model(self) -> None:
        """Load the configured WavLM model and feature extractor lazily."""

        if self.feature_extractor is not None and self.model is not None:
            return

        token = get_huggingface_token(required=False)
        model_kwargs = {"token": token} if token else {}

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            self.model = WavLMModel.from_pretrained(self.model_name, **model_kwargs)
        except OSError as exc:
            raise OSError(
                "Could not load WavLM model. Set HUGGINGFACE_TOKEN in the "
                "environment or a .env file if the model requires authentication."
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        LOGGER.info("Loaded WavLM model %s on %s", self.model_name, self.device)

    def extract_embeddings(
        self,
        audio_path: str | Path,
        segments: Iterable[SpeechSegment],
        *,
        cache_path: str | Path | None = None,
        use_cache: bool | None = None,
    ) -> list[SegmentEmbedding]:
        """Extract one WavLM embedding for each speech segment."""

        segment_list = list(segments)
        should_use_cache = (
            self.config.embedding_cache_enabled if use_cache is None else use_cache
        )
        if should_use_cache and cache_path is not None:
            path = Path(cache_path)
            if path.exists():
                LOGGER.info("Loading cached embeddings from %s", path)
                return load_embedding_cache(path)

        self.load_model()
        audio, sample_rate = load_wav_audio(audio_path, self.config)
        embeddings: list[SegmentEmbedding] = []

        for batch in _batched(segment_list, self.config.embedding_batch_size):
            audio_batch = [_crop_audio(audio, sample_rate, segment) for segment in batch]
            vectors = self._infer_batch(audio_batch, sample_rate)
            for segment, vector in zip(batch, vectors):
                embeddings.append(
                    SegmentEmbedding(
                        segment_id=segment.segment_id,
                        start=segment.start,
                        end=segment.end,
                        vector=vector,
                        dimension=int(vector.shape[0]),
                        model_name=self.model_name,
                    )
                )

        if should_use_cache and cache_path is not None:
            save_embedding_cache(embeddings, cache_path)

        LOGGER.info("Extracted %d WavLM embeddings", len(embeddings))
        return embeddings

    def _infer_batch(
        self,
        audio_batch: list[np.ndarray],
        sample_rate: int,
    ) -> list[np.ndarray]:
        """Run batch inference and pool the selected hidden layer."""

        if self.feature_extractor is None or self.model is None:
            raise RuntimeError("Model must be loaded before inference.")

        inputs = self.feature_extractor(
            audio_batch,
            sampling_rate=sample_rate,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_state = _select_hidden_state(outputs, self.config.embedding_hidden_layer)
        pooled = _pool_hidden_state(hidden_state, self.config.embedding_pooling_method)
        return [vector.detach().cpu().numpy().astype(np.float32) for vector in pooled]


def extract_segment_embeddings(
    audio_path: str | Path,
    segments: Iterable[SpeechSegment],
    *,
    config_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    use_cache: bool | None = None,
) -> list[SegmentEmbedding]:
    """Convenience function for WavLM embedding extraction."""

    config = load_config(config_path) if config_path else load_config()
    extractor = WavLMEmbeddingExtractor(config)
    return extractor.extract_embeddings(
        audio_path,
        segments,
        cache_path=cache_path,
        use_cache=use_cache,
    )


def extract_embedding(audio_segment: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for older code that embeds one audio segment."""

    config = load_config()
    extractor = WavLMEmbeddingExtractor(config)
    extractor.load_model()
    return extractor._infer_batch([audio_segment], config.sample_rate)[0]


def save_embedding_cache(
    embeddings: Iterable[SegmentEmbedding],
    cache_path: str | Path,
) -> None:
    """Save segment embeddings to a compressed NumPy cache."""

    embedding_list = list(embeddings)
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vectors = (
        np.vstack([embedding.vector for embedding in embedding_list]).astype(np.float32)
        if embedding_list
        else np.empty((0, 0), dtype=np.float32)
    )
    np.savez_compressed(
        path,
        vectors=vectors,
        segment_ids=np.array([item.segment_id for item in embedding_list], dtype=np.int64),
        starts=np.array([item.start for item in embedding_list], dtype=np.float32),
        ends=np.array([item.end for item in embedding_list], dtype=np.float32),
        dimensions=np.array([item.dimension for item in embedding_list], dtype=np.int64),
        model_names=np.array([item.model_name for item in embedding_list]),
    )
    LOGGER.info("Saved %d embeddings to %s", len(embedding_list), path)


def load_embedding_cache(cache_path: str | Path) -> list[SegmentEmbedding]:
    """Load segment embeddings from a compressed NumPy cache."""

    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        vectors = data["vectors"]
        segment_ids = data["segment_ids"]
        starts = data["starts"]
        ends = data["ends"]
        dimensions = data["dimensions"]
        model_names = data["model_names"]

        embeddings = [
            SegmentEmbedding(
                segment_id=int(segment_ids[index]),
                start=float(starts[index]),
                end=float(ends[index]),
                vector=vectors[index].astype(np.float32),
                dimension=int(dimensions[index]),
                model_name=str(model_names[index]),
            )
            for index in range(len(segment_ids))
        ]

    LOGGER.info("Loaded %d embeddings from %s", len(embeddings), path)
    return embeddings


def plot_embedding_statistics(
    embeddings: Iterable[SegmentEmbedding],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot basic embedding statistics without dimensionality reduction."""

    embedding_list = list(embeddings)
    dimensions = [embedding.dimension for embedding in embedding_list]
    norms = [float(np.linalg.norm(embedding.vector)) for embedding in embedding_list]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["segments"], [len(embedding_list)])
    axes[0].set_title("Number of Segments")
    axes[0].set_ylabel("Count")

    axes[1].hist(norms, bins=min(10, max(1, len(norms))))
    dimension_label = dimensions[0] if dimensions else 0
    axes[1].set_title(f"Embedding Norms (dim={dimension_label})")
    axes[1].set_xlabel("L2 norm")
    axes[1].set_ylabel("Count")

    figure.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        LOGGER.info("Saved embedding statistics plot to %s", path)
    return figure


def resolve_device(device: str) -> torch.device:
    """Resolve configured device into a torch device."""

    normalized = device.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("embedding_device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(normalized)


def get_huggingface_token(
    env_var: str = DEFAULT_TOKEN_ENV_VAR,
    *,
    env_path: str | Path = ".env",
    required: bool = False,
) -> str | None:
    """Read a Hugging Face token from environment, .env, or return None."""

    token = os.getenv(env_var)
    if token:
        return token

    dotenv_token = _read_token_from_env_file(env_path, env_var)
    if dotenv_token:
        return dotenv_token

    if required:
        raise OSError(
            f"Missing Hugging Face token. Set {env_var} or add it to .env."
        )
    return None


def _read_token_from_env_file(env_path: str | Path, env_var: str) -> str | None:
    """Read a token from a simple KEY=value .env file."""

    path = Path(env_path)
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == env_var:
            return value.strip().strip('"').strip("'")
    return None


def _crop_audio(
    audio: np.ndarray,
    sample_rate: int,
    segment: SpeechSegment,
) -> np.ndarray:
    """Crop a waveform according to a speech segment timestamp."""

    start_sample = max(0, int(round(segment.start * sample_rate)))
    end_sample = min(len(audio), int(round(segment.end * sample_rate)))
    if end_sample <= start_sample:
        raise ValueError(f"Invalid segment boundaries: {segment}")
    return audio[start_sample:end_sample]


def _select_hidden_state(outputs: object, hidden_layer: int) -> torch.Tensor:
    """Select a WavLM hidden state by layer index."""

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        if hidden_layer not in {-1, 0}:
            raise ValueError("Selected hidden layer is unavailable.")
        return outputs.last_hidden_state

    try:
        return hidden_states[hidden_layer]
    except IndexError as exc:
        raise ValueError(f"Hidden layer {hidden_layer} is unavailable.") from exc


def _pool_hidden_state(hidden_state: torch.Tensor, pooling_method: str) -> torch.Tensor:
    """Pool frame-level WavLM features into one segment embedding."""

    method = pooling_method.lower()
    if method == "mean":
        return hidden_state.mean(dim=1)
    if method == "max":
        return hidden_state.max(dim=1).values
    if method == "cls":
        return hidden_state[:, 0, :]
    if method == "statistics":
        mean = hidden_state.mean(dim=1)
        std = hidden_state.std(dim=1, unbiased=False)
        return torch.cat([mean, std], dim=1)
    raise ValueError(f"Unsupported pooling method: {pooling_method}")


def _validate_embedding_config(config: DiarizationConfig) -> None:
    """Validate embedding extraction parameters."""

    if config.embedding_pooling_method.lower() not in SUPPORTED_POOLING_METHODS:
        raise ValueError(
            f"Unsupported pooling method: {config.embedding_pooling_method}"
        )
    if config.embedding_batch_size <= 0:
        raise ValueError("embedding_batch_size must be greater than zero.")


def _batched(
    values: list[SpeechSegment],
    batch_size: int,
) -> Iterable[list[SpeechSegment]]:
    """Yield fixed-size batches from a list."""

    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for WavLM embedding extraction."""

    parser = argparse.ArgumentParser(description="Extract WavLM segment embeddings.")
    parser.add_argument("audio_path", type=Path, help="Path to preprocessed WAV audio.")
    parser.add_argument(
        "--segments",
        type=Path,
        required=True,
        help="Path to segments.json or segments.csv.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional .npz cache path for saving/loading embeddings.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation even if a cache file exists.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config file.")
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional output path for embedding statistics plot.",
    )
    return parser.parse_args()


def main() -> None:
    """Run embedding extraction from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    segments = load_segments(args.segments)
    embeddings = extract_segment_embeddings(
        args.audio_path,
        segments,
        config_path=args.config,
        cache_path=args.cache,
        use_cache=not args.no_cache,
    )
    if args.plot is not None:
        plot_embedding_statistics(embeddings, args.plot)

    for embedding in embeddings:
        print(
            f"{embedding.segment_id}\t{embedding.start:.3f}\t"
            f"{embedding.end:.3f}\t{embedding.dimension}"
        )


if __name__ == "__main__":
    main()
