"""Classical MFCC + K-Means speaker diarization baseline.

This module is independent from the proposed WavLM diarization pipeline. It
reuses the timeline, RTTM, and evaluation modules only at the output boundaries.
"""

from __future__ import annotations

import argparse
import math
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.evaluation import EvaluationReport, evaluate_rttm
from Project.ssl_pipeline.rttm import write_rttm
from Project.ssl_pipeline.spectral_clustering import ClusteredEmbedding
from Project.ssl_pipeline.timeline_reconstruction import (
    SpeakerTimeline,
    plot_timeline,
    reconstruct_speaker_timeline,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineFrame:
    """One MFCC frame with a timestamp and optional speaker cluster."""

    frame_id: int
    start: float
    end: float
    vector: np.ndarray
    cluster_id: int | None = None


@dataclass(frozen=True)
class BaselineResult:
    """Outputs produced by the classical baseline."""

    frames: list[BaselineFrame]
    timeline: list[SpeakerTimeline]
    rttm_path: Path | None = None
    evaluation_report: EvaluationReport | None = None


class MFCCKMeansBaseline:
    """Classical speaker diarization baseline using MFCC and K-Means."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        n_mfcc: int | None = None,
        frame_length_seconds: float | None = None,
        frame_hop_seconds: float | None = None,
        num_speakers: int | None = None,
        oracle_num_speakers: bool | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.config = config or load_config()
        self.n_mfcc = (
            self.config.baseline_mfcc_coefficients if n_mfcc is None else int(n_mfcc)
        )
        self.frame_length_seconds = (
            self.config.baseline_frame_length_seconds
            if frame_length_seconds is None
            else float(frame_length_seconds)
        )
        self.frame_hop_seconds = (
            self.config.baseline_frame_hop_seconds
            if frame_hop_seconds is None
            else float(frame_hop_seconds)
        )
        self.num_speakers = (
            self.config.baseline_num_speakers
            if num_speakers is None
            else int(num_speakers)
        )
        self.oracle_num_speakers = (
            self.config.baseline_oracle_num_speakers
            if oracle_num_speakers is None
            else bool(oracle_num_speakers)
        )
        self.random_seed = (
            self.config.baseline_kmeans_random_seed
            if random_seed is None
            else int(random_seed)
        )
        _validate_baseline_config(
            self.n_mfcc,
            self.frame_length_seconds,
            self.frame_hop_seconds,
            self.num_speakers,
        )

    def run(
        self,
        audio_path: str | Path,
        *,
        output_rttm_path: str | Path | None = None,
        reference_rttm_path: str | Path | None = None,
        recording_id: str | None = None,
    ) -> BaselineResult:
        """Run the complete baseline for one audio file."""

        frames = self.extract_mfcc_frames(audio_path)
        clustered_frames = self.cluster_frames(frames)
        timeline = self.reconstruct_timeline(clustered_frames)

        rttm_path = Path(output_rttm_path) if output_rttm_path is not None else None
        if rttm_path is not None:
            write_rttm(
                timeline,
                rttm_path,
                recording_id=recording_id or Path(audio_path).stem,
            )

        evaluation_report = None
        if reference_rttm_path is not None:
            if rttm_path is None:
                raise ValueError("output_rttm_path is required when evaluating baseline.")
            evaluation_report = evaluate_rttm(reference_rttm_path, rttm_path)

        return BaselineResult(
            frames=clustered_frames,
            timeline=timeline,
            rttm_path=rttm_path,
            evaluation_report=evaluation_report,
        )

    def extract_mfcc_frames(self, audio_path: str | Path) -> list[BaselineFrame]:
        """Extract one MFCC vector per audio frame."""

        audio, sample_rate = load_audio_for_baseline(
            audio_path,
            target_sample_rate=self.config.sample_rate,
        )
        return extract_mfcc_frames(
            audio,
            sample_rate,
            n_mfcc=self.n_mfcc,
            frame_length_seconds=self.frame_length_seconds,
            frame_hop_seconds=self.frame_hop_seconds,
        )

    def cluster_frames(self, frames: list[BaselineFrame]) -> list[BaselineFrame]:
        """Cluster MFCC frames and assign speaker labels."""

        if not frames:
            return []

        features = np.vstack([frame.vector for frame in frames]).astype(np.float32)
        n_clusters = self._resolve_speaker_count(features)
        if n_clusters == 1:
            labels = np.zeros(len(frames), dtype=np.int64)
        else:
            scaled = StandardScaler().fit_transform(features)
            model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_seed,
                n_init=10,
            )
            labels = model.fit_predict(scaled)
            labels = _renumber_labels_by_first_seen(labels)

        return [
            BaselineFrame(
                frame_id=frame.frame_id,
                start=frame.start,
                end=frame.end,
                vector=frame.vector,
                cluster_id=int(label),
            )
            for frame, label in zip(frames, labels)
        ]

    def reconstruct_timeline(
        self,
        frames: list[BaselineFrame],
    ) -> list[SpeakerTimeline]:
        """Convert clustered frames into a speaker timeline."""

        clustered_embeddings = baseline_frames_to_clustered_embeddings(frames)
        return reconstruct_speaker_timeline(
            clustered_embeddings,
            merge_silence_threshold_seconds=self.frame_hop_seconds,
            min_duration_seconds=0.0,
            label_smoothing_enabled=True,
        )

    def _resolve_speaker_count(self, features: np.ndarray) -> int:
        """Return oracle or estimated number of speakers for K-Means."""

        sample_count = features.shape[0]
        if sample_count <= 1:
            return 1
        if self.oracle_num_speakers:
            return min(self.num_speakers, sample_count)
        return estimate_kmeans_speaker_count(
            features,
            max_speakers=min(max(2, self.num_speakers), sample_count),
            random_seed=self.random_seed,
        )


def extract_mfcc_frames(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_mfcc: int,
    frame_length_seconds: float,
    frame_hop_seconds: float,
) -> list[BaselineFrame]:
    """Extract MFCC frame features from an audio array."""

    frame_length = max(1, int(round(frame_length_seconds * sample_rate)))
    hop_length = max(1, int(round(frame_hop_seconds * sample_rate)))
    features = compute_mfcc_matrix(
        audio,
        sample_rate,
        n_mfcc=n_mfcc,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return [
        BaselineFrame(
            frame_id=index,
            start=index * hop_length / sample_rate,
            end=min((index * hop_length + frame_length) / sample_rate, len(audio) / sample_rate),
            vector=vector,
        )
        for index, vector in enumerate(features)
    ]


def load_audio_for_baseline(
    audio_path: str | Path,
    *,
    target_sample_rate: int,
) -> tuple[np.ndarray, int]:
    """Load mono audio for the classical baseline."""

    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
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


def compute_mfcc_matrix(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_mfcc: int,
    frame_length: int,
    hop_length: int,
    n_mels: int = 40,
) -> np.ndarray:
    """Compute a simple MFCC matrix without depending on the WavLM pipeline."""

    signal = np.asarray(audio, dtype=np.float32)
    if signal.size == 0:
        return np.empty((0, n_mfcc), dtype=np.float32)
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))

    frames = _frame_signal(signal, frame_length, hop_length)
    window = np.hamming(frame_length).astype(np.float32)
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=frame_length)
    power = (np.abs(spectrum) ** 2) / frame_length
    mel_filters = _mel_filterbank(
        sample_rate,
        n_fft=frame_length,
        n_mels=n_mels,
    )
    mel_energy = np.maximum(power @ mel_filters.T, 1e-10)
    log_mel = np.log(mel_energy)
    coefficients = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return coefficients.astype(np.float32)


def _frame_signal(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """Create overlapping audio frames."""

    frame_count = 1 + max(0, (len(audio) - frame_length) // hop_length)
    frames = np.empty((frame_count, frame_length), dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_length
        frames[index] = audio[start : start + frame_length]
    return frames


def _mel_filterbank(
    sample_rate: int,
    *,
    n_fft: int,
    n_mels: int,
) -> np.ndarray:
    """Create a triangular mel filterbank."""

    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    max_bin = n_fft // 2
    bins = np.clip(bins, 0, max_bin)

    filters = np.zeros((n_mels, max_bin + 1), dtype=np.float32)
    for mel_index in range(1, n_mels + 1):
        left = bins[mel_index - 1]
        center = bins[mel_index]
        right = bins[mel_index + 1]
        if center == left:
            center = min(center + 1, max_bin)
        if right == center:
            right = min(right + 1, max_bin)
        for bin_index in range(left, center):
            filters[mel_index - 1, bin_index] = (bin_index - left) / max(1, center - left)
        for bin_index in range(center, right):
            filters[mel_index - 1, bin_index] = (right - bin_index) / max(1, right - center)
    return filters


def _hz_to_mel(frequency: float | np.ndarray) -> float | np.ndarray:
    """Convert Hertz to mel scale."""

    return 2595.0 * np.log10(1.0 + np.asarray(frequency) / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert mel scale to Hertz."""

    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def cluster_mfcc_features(
    frames: list[BaselineFrame],
    *,
    num_speakers: int,
    random_seed: int = 42,
) -> list[BaselineFrame]:
    """Cluster MFCC frame features with K-Means."""

    baseline = MFCCKMeansBaseline(
        num_speakers=num_speakers,
        oracle_num_speakers=True,
        random_seed=random_seed,
    )
    return baseline.cluster_frames(frames)


def baseline_frames_to_clustered_embeddings(
    frames: list[BaselineFrame],
) -> list[ClusteredEmbedding]:
    """Adapt baseline frames to the existing timeline reconstruction API."""

    embeddings: list[ClusteredEmbedding] = []
    for frame in frames:
        if frame.cluster_id is None:
            raise ValueError("All baseline frames must have cluster_id before timeline reconstruction.")
        embeddings.append(
            ClusteredEmbedding(
                segment_id=frame.frame_id,
                start=frame.start,
                end=frame.end,
                cluster_id=frame.cluster_id,
                vector=frame.vector.astype(np.float32),
                metadata={"baseline": "mfcc_kmeans"},
            )
        )
    return embeddings


def run_baseline(
    audio_path: str | Path,
    *,
    config_path: str | Path | None = None,
    output_rttm_path: str | Path | None = None,
    reference_rttm_path: str | Path | None = None,
    recording_id: str | None = None,
) -> BaselineResult:
    """Run the standalone MFCC + K-Means baseline."""

    config = load_config(config_path) if config_path else load_config()
    baseline = MFCCKMeansBaseline(config)
    return baseline.run(
        audio_path,
        output_rttm_path=output_rttm_path,
        reference_rttm_path=reference_rttm_path,
        recording_id=recording_id,
    )


def estimate_kmeans_speaker_count(
    features: np.ndarray,
    *,
    max_speakers: int,
    random_seed: int,
) -> int:
    """Estimate speaker count with silhouette score over K-Means labels."""

    sample_count = features.shape[0]
    if sample_count <= 2:
        return min(2, sample_count)
    scaled = StandardScaler().fit_transform(features)
    upper = min(max_speakers, sample_count - 1)
    best_k = 2
    best_score = -1.0
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, random_state=random_seed, n_init=10).fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def plot_mfcc_features(
    frames: list[BaselineFrame],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot MFCC features as a coefficient-by-frame heatmap."""

    matrix = np.vstack([frame.vector for frame in frames]).T if frames else np.empty((0, 0))
    figure, axis = plt.subplots(figsize=(9, 4))
    image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
    axis.set_title("MFCC Features")
    axis.set_xlabel("Frame")
    axis.set_ylabel("MFCC coefficient")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_baseline_cluster_distribution(
    frames: list[BaselineFrame],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the number of baseline frames assigned to each cluster."""

    labels = [frame.cluster_id for frame in frames if frame.cluster_id is not None]
    unique, counts = (
        np.unique(labels, return_counts=True)
        if labels
        else (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([str(label) for label in unique], counts)
    axis.set_title("Baseline Cluster Distribution")
    axis.set_xlabel("Cluster ID")
    axis.set_ylabel("Frame count")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_baseline_timeline(
    timeline: list[SpeakerTimeline],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the baseline speaker timeline."""

    return plot_timeline(timeline, output_path)


def _renumber_labels_by_first_seen(labels: np.ndarray) -> np.ndarray:
    """Make cluster IDs deterministic by first appearance."""

    mapping: dict[int, int] = {}
    renumbered: list[int] = []
    for label in labels:
        value = int(label)
        if value not in mapping:
            mapping[value] = len(mapping)
        renumbered.append(mapping[value])
    return np.array(renumbered, dtype=np.int64)


def _validate_baseline_config(
    n_mfcc: int,
    frame_length_seconds: float,
    frame_hop_seconds: float,
    num_speakers: int,
) -> None:
    """Validate baseline configuration."""

    if n_mfcc <= 0:
        raise ValueError("baseline_mfcc_coefficients must be positive.")
    if frame_length_seconds <= 0.0:
        raise ValueError("baseline_frame_length_seconds must be positive.")
    if frame_hop_seconds <= 0.0:
        raise ValueError("baseline_frame_hop_seconds must be positive.")
    if num_speakers <= 0:
        raise ValueError("baseline_num_speakers must be positive.")


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
    LOGGER.info("Saved baseline plot to %s", path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the baseline."""

    parser = argparse.ArgumentParser(description="Run MFCC + K-Means diarization baseline.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--rttm", type=Path, default=None)
    parser.add_argument("--reference-rttm", type=Path, default=None)
    parser.add_argument("--recording-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the classical baseline from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    result = run_baseline(
        args.audio_path,
        config_path=args.config,
        output_rttm_path=args.rttm,
        reference_rttm_path=args.reference_rttm,
        recording_id=args.recording_id,
    )
    print(f"frames={len(result.frames)}\ttimeline_intervals={len(result.timeline)}")
    if result.evaluation_report is not None:
        print(f"DER={result.evaluation_report.der:.4f}\tJER={result.evaluation_report.jer:.4f}")


if __name__ == "__main__":
    main()
