"""Configuration helpers for the diarization pipeline."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"
DEFAULT_TOKEN_ENV_VAR = "HUGGINGFACE_TOKEN"


@dataclass(frozen=True)
class DiarizationConfig:
    """Runtime configuration for the current diarization pipeline."""

    sample_rate: int = 16_000
    mono: bool = True
    normalize_audio: bool = True
    segment_duration_seconds: float = 2.0
    segment_window_seconds: float = 2.0
    segment_hop_seconds: float = 1.5
    segment_min_duration_seconds: float = 0.5
    segment_max_duration_seconds: float = 2.0
    segment_include_short_final: bool = True
    pyannote_model: str = "pyannote/speaker-diarization"
    token_env_var: str = DEFAULT_TOKEN_ENV_VAR
    vad_backend: str = "auto"
    vad_pyannote_model: str = "pyannote/voice-activity-detection"
    vad_min_speech_duration_seconds: float = 0.25
    vad_min_silence_duration_seconds: float = 0.15
    vad_merge_threshold_seconds: float = 0.2
    vad_frame_duration_ms: int = 30
    vad_webrtc_aggressiveness: int = 2
    vad_energy_threshold: float = 0.01
    embedding_model_name: str = "microsoft/wavlm-base-plus"
    embedding_pooling_method: str = "mean"
    embedding_hidden_layer: int = -1
    embedding_batch_size: int = 4
    embedding_device: str = "auto"
    embedding_cache_enabled: bool = True
    embedding_cache_dir: str = "outputs/embeddings"
    embedding_normalization_method: str = "l2"
    embedding_pca_enabled: bool = False
    embedding_pca_components: int | None = None
    clustering_algorithm: str = "spectral"
    clustering_affinity_metric: str = "cosine"
    clustering_affinity_threshold: float = 0.0
    clustering_num_speakers: int = 2
    clustering_oracle_num_speakers: bool = True
    clustering_speaker_count_method: str = "eigengap"
    clustering_max_estimated_speakers: int = 8
    clustering_random_seed: int = 42
    timeline_merge_adjacent: bool = True
    timeline_merge_silence_threshold_seconds: float = 0.2
    timeline_min_duration_seconds: float = 0.1
    timeline_label_smoothing_enabled: bool = False
    rttm_recording_id: str = "recording1"
    rttm_channel_id: str = "1"
    rttm_decimal_places: int = 2
    evaluation_collar_seconds: float = 0.0
    evaluation_use_pyannote: bool = True
    baseline_mfcc_coefficients: int = 13
    baseline_frame_length_seconds: float = 0.025
    baseline_frame_hop_seconds: float = 0.010
    baseline_kmeans_random_seed: int = 42
    baseline_num_speakers: int = 2
    baseline_oracle_num_speakers: bool = True
    separation_model_backend: str = "speechbrain"
    separation_model_name: str = "speechbrain/sepformer-whamr"
    separation_fallback_model_name: str = "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k"
    separation_device: str = "auto"
    separation_output_dir: str = "outputs/separated_audio"
    separation_sample_rate: int = 8_000
    separation_num_speakers: int = 2
    separation_chunk_duration_seconds: float = 30.0
    separation_chunk_overlap_seconds: float = 1.0


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> DiarizationConfig:
    """Load diarization configuration from a JSON file.

    Missing files are treated as an error because silent defaults make
    experiments hard to reproduce.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = json.load(file)

    config = DiarizationConfig(
        sample_rate=int(raw_config.get("sample_rate", DiarizationConfig.sample_rate)),
        mono=bool(raw_config.get("mono", DiarizationConfig.mono)),
        normalize_audio=bool(
            raw_config.get("normalize_audio", DiarizationConfig.normalize_audio)
        ),
        segment_duration_seconds=float(
            raw_config.get(
                "segment_duration_seconds",
                DiarizationConfig.segment_duration_seconds,
            )
        ),
        segment_window_seconds=float(
            raw_config.get(
                "segment_window_seconds",
                DiarizationConfig.segment_window_seconds,
            )
        ),
        segment_hop_seconds=float(
            raw_config.get(
                "segment_hop_seconds",
                DiarizationConfig.segment_hop_seconds,
            )
        ),
        segment_min_duration_seconds=float(
            raw_config.get(
                "segment_min_duration_seconds",
                DiarizationConfig.segment_min_duration_seconds,
            )
        ),
        segment_max_duration_seconds=float(
            raw_config.get(
                "segment_max_duration_seconds",
                DiarizationConfig.segment_max_duration_seconds,
            )
        ),
        segment_include_short_final=bool(
            raw_config.get(
                "segment_include_short_final",
                DiarizationConfig.segment_include_short_final,
            )
        ),
        pyannote_model=str(
            raw_config.get("pyannote_model", DiarizationConfig.pyannote_model)
        ),
        vad_backend=str(raw_config.get("vad_backend", DiarizationConfig.vad_backend)),
        vad_pyannote_model=str(
            raw_config.get(
                "vad_pyannote_model",
                DiarizationConfig.vad_pyannote_model,
            )
        ),
        vad_min_speech_duration_seconds=float(
            raw_config.get(
                "vad_min_speech_duration_seconds",
                DiarizationConfig.vad_min_speech_duration_seconds,
            )
        ),
        vad_min_silence_duration_seconds=float(
            raw_config.get(
                "vad_min_silence_duration_seconds",
                DiarizationConfig.vad_min_silence_duration_seconds,
            )
        ),
        vad_merge_threshold_seconds=float(
            raw_config.get(
                "vad_merge_threshold_seconds",
                DiarizationConfig.vad_merge_threshold_seconds,
            )
        ),
        vad_frame_duration_ms=int(
            raw_config.get(
                "vad_frame_duration_ms",
                DiarizationConfig.vad_frame_duration_ms,
            )
        ),
        vad_webrtc_aggressiveness=int(
            raw_config.get(
                "vad_webrtc_aggressiveness",
                DiarizationConfig.vad_webrtc_aggressiveness,
            )
        ),
        vad_energy_threshold=float(
            raw_config.get(
                "vad_energy_threshold",
                DiarizationConfig.vad_energy_threshold,
            )
        ),
        embedding_model_name=str(
            raw_config.get(
                "embedding_model_name",
                DiarizationConfig.embedding_model_name,
            )
        ),
        embedding_pooling_method=str(
            raw_config.get(
                "embedding_pooling_method",
                DiarizationConfig.embedding_pooling_method,
            )
        ),
        embedding_hidden_layer=int(
            raw_config.get(
                "embedding_hidden_layer",
                DiarizationConfig.embedding_hidden_layer,
            )
        ),
        embedding_batch_size=int(
            raw_config.get(
                "embedding_batch_size",
                DiarizationConfig.embedding_batch_size,
            )
        ),
        embedding_device=str(
            raw_config.get(
                "embedding_device",
                DiarizationConfig.embedding_device,
            )
        ),
        embedding_cache_enabled=bool(
            raw_config.get(
                "embedding_cache_enabled",
                DiarizationConfig.embedding_cache_enabled,
            )
        ),
        embedding_cache_dir=str(
            raw_config.get(
                "embedding_cache_dir",
                DiarizationConfig.embedding_cache_dir,
            )
        ),
        embedding_normalization_method=str(
            raw_config.get(
                "embedding_normalization_method",
                DiarizationConfig.embedding_normalization_method,
            )
        ),
        embedding_pca_enabled=bool(
            raw_config.get(
                "embedding_pca_enabled",
                DiarizationConfig.embedding_pca_enabled,
            )
        ),
        embedding_pca_components=(
            None
            if raw_config.get("embedding_pca_components") is None
            else int(raw_config["embedding_pca_components"])
        ),
        clustering_algorithm=str(
            raw_config.get(
                "clustering_algorithm",
                DiarizationConfig.clustering_algorithm,
            )
        ),
        clustering_affinity_metric=str(
            raw_config.get(
                "clustering_affinity_metric",
                DiarizationConfig.clustering_affinity_metric,
            )
        ),
        clustering_affinity_threshold=float(
            raw_config.get(
                "clustering_affinity_threshold",
                DiarizationConfig.clustering_affinity_threshold,
            )
        ),
        clustering_num_speakers=int(
            raw_config.get(
                "clustering_num_speakers",
                DiarizationConfig.clustering_num_speakers,
            )
        ),
        clustering_oracle_num_speakers=bool(
            raw_config.get(
                "clustering_oracle_num_speakers",
                DiarizationConfig.clustering_oracle_num_speakers,
            )
        ),
        clustering_speaker_count_method=str(
            raw_config.get(
                "clustering_speaker_count_method",
                DiarizationConfig.clustering_speaker_count_method,
            )
        ),
        clustering_max_estimated_speakers=int(
            raw_config.get(
                "clustering_max_estimated_speakers",
                DiarizationConfig.clustering_max_estimated_speakers,
            )
        ),
        clustering_random_seed=int(
            raw_config.get(
                "clustering_random_seed",
                DiarizationConfig.clustering_random_seed,
            )
        ),
        timeline_merge_adjacent=bool(
            raw_config.get(
                "timeline_merge_adjacent",
                DiarizationConfig.timeline_merge_adjacent,
            )
        ),
        timeline_merge_silence_threshold_seconds=float(
            raw_config.get(
                "timeline_merge_silence_threshold_seconds",
                DiarizationConfig.timeline_merge_silence_threshold_seconds,
            )
        ),
        timeline_min_duration_seconds=float(
            raw_config.get(
                "timeline_min_duration_seconds",
                DiarizationConfig.timeline_min_duration_seconds,
            )
        ),
        timeline_label_smoothing_enabled=bool(
            raw_config.get(
                "timeline_label_smoothing_enabled",
                DiarizationConfig.timeline_label_smoothing_enabled,
            )
        ),
        rttm_recording_id=str(
            raw_config.get(
                "rttm_recording_id",
                DiarizationConfig.rttm_recording_id,
            )
        ),
        rttm_channel_id=str(
            raw_config.get(
                "rttm_channel_id",
                DiarizationConfig.rttm_channel_id,
            )
        ),
        rttm_decimal_places=int(
            raw_config.get(
                "rttm_decimal_places",
                DiarizationConfig.rttm_decimal_places,
            )
        ),
        evaluation_collar_seconds=float(
            raw_config.get(
                "evaluation_collar_seconds",
                DiarizationConfig.evaluation_collar_seconds,
            )
        ),
        evaluation_use_pyannote=bool(
            raw_config.get(
                "evaluation_use_pyannote",
                DiarizationConfig.evaluation_use_pyannote,
            )
        ),
        baseline_mfcc_coefficients=int(
            raw_config.get(
                "baseline_mfcc_coefficients",
                DiarizationConfig.baseline_mfcc_coefficients,
            )
        ),
        baseline_frame_length_seconds=float(
            raw_config.get(
                "baseline_frame_length_seconds",
                DiarizationConfig.baseline_frame_length_seconds,
            )
        ),
        baseline_frame_hop_seconds=float(
            raw_config.get(
                "baseline_frame_hop_seconds",
                DiarizationConfig.baseline_frame_hop_seconds,
            )
        ),
        baseline_kmeans_random_seed=int(
            raw_config.get(
                "baseline_kmeans_random_seed",
                DiarizationConfig.baseline_kmeans_random_seed,
            )
        ),
        baseline_num_speakers=int(
            raw_config.get(
                "baseline_num_speakers",
                DiarizationConfig.baseline_num_speakers,
            )
        ),
        baseline_oracle_num_speakers=bool(
            raw_config.get(
                "baseline_oracle_num_speakers",
                DiarizationConfig.baseline_oracle_num_speakers,
            )
        ),
        separation_model_backend=str(
            raw_config.get(
                "separation_model_backend",
                DiarizationConfig.separation_model_backend,
            )
        ),
        separation_model_name=str(
            raw_config.get(
                "separation_model_name",
                DiarizationConfig.separation_model_name,
            )
        ),
        separation_fallback_model_name=str(
            raw_config.get(
                "separation_fallback_model_name",
                DiarizationConfig.separation_fallback_model_name,
            )
        ),
        separation_device=str(
            raw_config.get(
                "separation_device",
                DiarizationConfig.separation_device,
            )
        ),
        separation_output_dir=str(
            raw_config.get(
                "separation_output_dir",
                DiarizationConfig.separation_output_dir,
            )
        ),
        separation_sample_rate=int(
            raw_config.get(
                "separation_sample_rate",
                DiarizationConfig.separation_sample_rate,
            )
        ),
        separation_num_speakers=int(
            raw_config.get(
                "separation_num_speakers",
                DiarizationConfig.separation_num_speakers,
            )
        ),
        separation_chunk_duration_seconds=float(
            raw_config.get(
                "separation_chunk_duration_seconds",
                DiarizationConfig.separation_chunk_duration_seconds,
            )
        ),
        separation_chunk_overlap_seconds=float(
            raw_config.get(
                "separation_chunk_overlap_seconds",
                DiarizationConfig.separation_chunk_overlap_seconds,
            )
        ),
    )
    LOGGER.debug("Loaded diarization config from %s", path)
    return config


def get_huggingface_token(
    env_var: str = DEFAULT_TOKEN_ENV_VAR,
    *,
    required: bool = True,
) -> str | None:
    """Return a Hugging Face token from the environment.

    Tokens must not be committed to source files. Set ``HUGGINGFACE_TOKEN``
    before running pyannote-based scripts.
    """

    token = os.getenv(env_var)
    if token:
        return token

    if required:
        raise OSError(
            f"Missing Hugging Face token. Set the {env_var} environment variable."
        )

    LOGGER.debug("No Hugging Face token found in %s", env_var)
    return None
