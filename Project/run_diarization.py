"""End-to-end speaker diarization pipeline runner."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.embedding_normalization import (
    NormalizedEmbedding,
    normalize_embeddings,
)
from Project.ssl_pipeline.evaluation import (
    EvaluationReport,
    evaluate_rttm,
    export_evaluation_json,
)
from Project.ssl_pipeline.preprocessing import preprocess_audio
from Project.ssl_pipeline.rttm import write_rttm
from Project.ssl_pipeline.segmentation import SpeechSegment, save_segments, segment_speech
from Project.ssl_pipeline.spectral_clustering import ClusteredEmbedding, cluster_embeddings
from Project.ssl_pipeline.ssl_embedding import (
    SegmentEmbedding,
    save_embedding_cache,
)
from Project.ssl_pipeline.timeline_reconstruction import (
    SpeakerTimeline,
    export_timeline_csv,
    export_timeline_json,
    export_timeline_txt,
    plot_speaker_activity_chart,
    plot_speaker_duration_histogram,
    plot_timeline,
    reconstruct_speaker_timeline,
)
from Project.ssl_pipeline.vad import detect_speech_regions

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class DiarizationPipelineResult:
    """Paths and in-memory outputs from one diarization run."""

    input_path: Path
    output_dir: Path
    processed_audio_path: Path
    vad_regions: list[tuple[float, float]]
    segments: list[SpeechSegment]
    embeddings: list[SegmentEmbedding]
    normalized_embeddings: list[NormalizedEmbedding]
    clustered_embeddings: list[ClusteredEmbedding]
    timeline: list[SpeakerTimeline]
    rttm_path: Path | None
    evaluation_report: EvaluationReport | None


def run_pipeline(
    input_path: str | Path,
    *,
    output_dir: str | Path = "outputs",
    reference_rttm: str | Path | None = None,
    device: str | None = None,
    backend: str | None = None,
    num_speakers: int | None = None,
    oracle_num_speakers: bool | None = None,
    save_json: bool = True,
    save_csv: bool = True,
    save_rttm: bool = True,
) -> DiarizationPipelineResult:
    """Run the complete diarization pipeline for one audio file."""

    source_path = Path(input_path)
    target_dir = Path(output_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {source_path}")

    print("Loading audio...")
    target_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = target_dir / "embeddings"
    figures_dir = target_dir / "figures"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = _build_runtime_config(
        device=device,
        num_speakers=num_speakers,
        oracle_num_speakers=oracle_num_speakers,
    )

    processed_audio_path = target_dir / "processed.wav"
    vad_json_path = target_dir / "vad_regions.json"
    segments_json_path = target_dir / "segments.json"
    raw_embeddings_path = embeddings_dir / "embeddings.npz"
    normalized_embeddings_path = target_dir / "normalized_embeddings.npz"
    clustering_json_path = target_dir / "clustering.json"
    timeline_json_path = target_dir / "timeline.json"
    timeline_csv_path = target_dir / "timeline.csv"
    timeline_txt_path = target_dir / "timeline.txt"
    hypothesis_rttm_path = target_dir / "hypothesis.rttm"
    evaluation_json_path = target_dir / "evaluation.json"

    _run_stage(
        "preprocessing",
        lambda: preprocess_audio(source_path, processed_audio_path),
    )
    vad_regions = _run_stage(
        "VAD",
        lambda: detect_speech_regions(
            processed_audio_path,
            backend=backend,
            export_json_path=vad_json_path if save_json else None,
        ),
    )
    segments = _run_stage(
        "segmentation",
        lambda: segment_speech(vad_regions),
    )
    if save_json:
        save_segments(segments, segments_json_path)

    embeddings = _run_stage(
        "WavLM embeddings",
        lambda: _extract_embeddings_with_config(
            processed_audio_path,
            segments,
            config,
        ),
    )
    save_embedding_cache(embeddings, raw_embeddings_path)

    normalized_embeddings = _run_stage(
        "embedding normalization",
        lambda: normalize_embeddings_with_config(embeddings, config),
    )
    save_normalized_embeddings(normalized_embeddings, normalized_embeddings_path)

    clustered_embeddings = _run_stage(
        "spectral clustering",
        lambda: cluster_embeddings_with_config(normalized_embeddings, config),
    )
    if save_json:
        save_clustered_embeddings(clustered_embeddings, clustering_json_path)

    timeline = _run_stage(
        "timeline reconstruction",
        lambda: reconstruct_speaker_timeline_with_config(clustered_embeddings, config),
    )
    if save_json:
        export_timeline_json(timeline, timeline_json_path)
    if save_csv:
        export_timeline_csv(timeline, timeline_csv_path)
    export_timeline_txt(timeline, timeline_txt_path)

    _save_figures(timeline, figures_dir)

    rttm_path = None
    if save_rttm:
        _run_stage(
            "RTTM generation",
            lambda: write_rttm(
                timeline,
                hypothesis_rttm_path,
                recording_id=source_path.stem,
            ),
        )
        rttm_path = hypothesis_rttm_path

    evaluation_report = None
    if reference_rttm is not None:
        if rttm_path is None:
            raise ValueError("RTTM output is required when reference_rttm is supplied.")
        evaluation_report = _run_stage(
            "DER/JER evaluation",
            lambda: evaluate_rttm(reference_rttm, rttm_path),
        )
        if save_json:
            export_evaluation_json(evaluation_report, evaluation_json_path)

    print("Done.")
    return DiarizationPipelineResult(
        input_path=source_path,
        output_dir=target_dir,
        processed_audio_path=processed_audio_path,
        vad_regions=vad_regions,
        segments=segments,
        embeddings=embeddings,
        normalized_embeddings=normalized_embeddings,
        clustered_embeddings=clustered_embeddings,
        timeline=timeline,
        rttm_path=rttm_path,
        evaluation_report=evaluation_report,
    )


def _build_runtime_config(
    *,
    device: str | None,
    num_speakers: int | None,
    oracle_num_speakers: bool | None,
) -> DiarizationConfig:
    """Load default configuration and apply CLI overrides."""

    config = load_config()
    updates: dict[str, object] = {}
    if device is not None:
        updates["embedding_device"] = device
    if num_speakers is not None:
        updates["clustering_num_speakers"] = num_speakers
    if oracle_num_speakers is not None:
        updates["clustering_oracle_num_speakers"] = oracle_num_speakers
    return replace(config, **updates) if updates else config


def _extract_embeddings_with_config(
    audio_path: Path,
    segments: list[SpeechSegment],
    config: DiarizationConfig,
) -> list[SegmentEmbedding]:
    """Extract embeddings while honoring runtime config overrides."""

    from Project.ssl_pipeline.ssl_embedding import WavLMEmbeddingExtractor

    extractor = WavLMEmbeddingExtractor(config)
    return extractor.extract_embeddings(audio_path, segments, use_cache=False)


def normalize_embeddings_with_config(
    embeddings: list[SegmentEmbedding],
    config: DiarizationConfig,
) -> list[NormalizedEmbedding]:
    """Normalize embeddings using an already loaded config."""

    from Project.ssl_pipeline.embedding_normalization import EmbeddingNormalizer

    return EmbeddingNormalizer(config).fit_transform(embeddings)


def cluster_embeddings_with_config(
    embeddings: list[NormalizedEmbedding],
    config: DiarizationConfig,
) -> list[ClusteredEmbedding]:
    """Cluster embeddings using an already loaded config."""

    from Project.ssl_pipeline.spectral_clustering import SpectralEmbeddingClusterer

    return SpectralEmbeddingClusterer(config).fit_predict(embeddings)


def reconstruct_speaker_timeline_with_config(
    clustered_embeddings: list[ClusteredEmbedding],
    config: DiarizationConfig,
) -> list[SpeakerTimeline]:
    """Reconstruct speaker timeline using an already loaded config."""

    from Project.ssl_pipeline.timeline_reconstruction import SpeakerTimelineReconstructor

    return SpeakerTimelineReconstructor(config).reconstruct(clustered_embeddings)


def save_normalized_embeddings(
    embeddings: list[NormalizedEmbedding],
    output_path: str | Path,
) -> None:
    """Save normalized embeddings to a compressed NumPy artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = (
        np.vstack([embedding.vector for embedding in embeddings]).astype(np.float32)
        if embeddings
        else np.empty((0, 0), dtype=np.float32)
    )
    np.savez_compressed(
        path,
        vectors=vectors,
        segment_ids=np.array([item.segment_id for item in embeddings], dtype=np.int64),
        starts=np.array([item.start for item in embeddings], dtype=np.float32),
        ends=np.array([item.end for item in embeddings], dtype=np.float32),
        dimensions=np.array([item.dimension for item in embeddings], dtype=np.int64),
        model_names=np.array([item.model_name for item in embeddings]),
        normalization_methods=np.array([item.normalization_method for item in embeddings]),
    )
    LOGGER.info("Saved normalized embeddings to %s", path)


def save_clustered_embeddings(
    clustered_embeddings: list[ClusteredEmbedding],
    output_path: str | Path,
) -> None:
    """Save clustering assignments without embedding vectors."""

    rows = [
        {
            "segment_id": item.segment_id,
            "start": item.start,
            "end": item.end,
            "cluster_id": item.cluster_id,
            "metadata": item.metadata,
        }
        for item in clustered_embeddings
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    LOGGER.info("Saved clustering assignments to %s", path)


def _save_figures(timeline: list[SpeakerTimeline], figures_dir: Path) -> None:
    """Save standard timeline visualizations."""

    figures = [
        plot_timeline(timeline, figures_dir / "timeline.png"),
        plot_speaker_activity_chart(timeline, figures_dir / "speaker_activity.png"),
        plot_speaker_duration_histogram(timeline, figures_dir / "duration_histogram.png"),
    ]
    for figure in figures:
        plt.close(figure)


def _run_stage(stage_name: str, action: Callable[[], T]) -> T:
    """Run one named pipeline stage and re-raise failures with context."""

    print(f"Running {stage_name}...")
    try:
        return action()
    except Exception:
        print(f"{stage_name} failed.")
        raise


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full diarization runner."""

    parser = argparse.ArgumentParser(description="Run the complete diarization pipeline.")
    parser.add_argument("--input", required=True, type=Path, help="Input audio file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--reference-rttm", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--backend", choices=["auto", "pyannote", "webrtc", "energy"], default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument(
        "--oracle-num-speakers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-rttm", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end diarization CLI."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_pipeline(
        args.input,
        output_dir=args.output_dir,
        reference_rttm=args.reference_rttm,
        device=args.device,
        backend=args.backend,
        num_speakers=args.num_speakers,
        oracle_num_speakers=args.oracle_num_speakers,
        save_json=args.save_json,
        save_csv=args.save_csv,
        save_rttm=args.save_rttm,
    )


if __name__ == "__main__":
    main()
