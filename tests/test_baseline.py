"""Tests for the MFCC + K-Means diarization baseline."""

from __future__ import annotations

import unittest
import wave
from pathlib import Path

import numpy as np

from Project.ssl_pipeline.baseline import (
    BaselineFrame,
    MFCCKMeansBaseline,
    cluster_mfcc_features,
    extract_mfcc_frames,
    plot_baseline_cluster_distribution,
    plot_baseline_timeline,
    plot_mfcc_features,
    run_baseline,
)
from Project.ssl_pipeline.evaluation import evaluate_rttm
from Project.ssl_pipeline.rttm import write_rttm


SAMPLE_RATE = 16_000
TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class BaselineTests(unittest.TestCase):
    """Validate the standalone classical baseline."""

    def test_mfcc_extraction_returns_expected_dimensions(self) -> None:
        audio = _tone(1.0, 220.0)

        frames = extract_mfcc_frames(
            audio,
            SAMPLE_RATE,
            n_mfcc=13,
            frame_length_seconds=0.025,
            frame_hop_seconds=0.010,
        )

        self.assertGreater(len(frames), 0)
        self.assertEqual(frames[0].vector.shape, (13,))
        self.assertEqual(frames[0].start, 0.0)
        self.assertGreater(frames[0].end, frames[0].start)

    def test_kmeans_clustering_assigns_two_speaker_labels(self) -> None:
        clustered = cluster_mfcc_features(_synthetic_frames(), num_speakers=2, random_seed=7)
        labels = [frame.cluster_id for frame in clustered]

        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_timeline_generation_reuses_existing_timeline_module(self) -> None:
        baseline = MFCCKMeansBaseline(num_speakers=2, random_seed=7)
        clustered = cluster_mfcc_features(_synthetic_frames(), num_speakers=2, random_seed=7)

        timeline = baseline.reconstruct_timeline(clustered)

        self.assertGreaterEqual(len(timeline), 2)
        self.assertEqual(timeline[0].speaker_id, "speaker_0")
        self.assertEqual(timeline[1].speaker_id, "speaker_1")

    def test_rttm_generation_from_baseline_timeline(self) -> None:
        path = _fixture_path("baseline.rttm")
        baseline = MFCCKMeansBaseline(num_speakers=2, random_seed=7)
        timeline = baseline.reconstruct_timeline(
            cluster_mfcc_features(_synthetic_frames(), num_speakers=2, random_seed=7)
        )

        write_rttm(timeline, path, recording_id="baseline_test")

        text = path.read_text(encoding="utf-8")
        self.assertIn("SPEAKER baseline_test", text)

    def test_evaluation_reuses_existing_evaluation_module(self) -> None:
        path = _fixture_path("baseline_eval.rttm")
        baseline = MFCCKMeansBaseline(num_speakers=2, random_seed=7)
        timeline = baseline.reconstruct_timeline(
            cluster_mfcc_features(_synthetic_frames(), num_speakers=2, random_seed=7)
        )
        write_rttm(timeline, path, recording_id="baseline_eval")

        report = evaluate_rttm(path, path, use_pyannote=False)

        self.assertEqual(report.der, 0.0)
        self.assertEqual(report.jer, 0.0)

    def test_full_baseline_produces_timeline_and_rttm(self) -> None:
        audio_path = _fixture_path("baseline_audio.wav")
        rttm_path = _fixture_path("baseline_full.rttm")
        _write_wav(audio_path, np.concatenate([_tone(0.5, 220.0), _tone(0.5, 880.0)]))

        result = run_baseline(
            audio_path,
            output_rttm_path=rttm_path,
            recording_id="baseline_full",
        )

        self.assertGreater(len(result.frames), 0)
        self.assertGreater(len(result.timeline), 0)
        self.assertEqual(result.rttm_path, rttm_path)
        self.assertTrue(rttm_path.exists())

    def test_plots_return_figures(self) -> None:
        baseline = MFCCKMeansBaseline(num_speakers=2, random_seed=7)
        clustered = cluster_mfcc_features(_synthetic_frames(), num_speakers=2, random_seed=7)
        timeline = baseline.reconstruct_timeline(clustered)

        self.assertEqual(len(plot_mfcc_features(clustered).axes), 2)
        self.assertEqual(len(plot_baseline_cluster_distribution(clustered).axes), 1)
        self.assertEqual(len(plot_baseline_timeline(timeline).axes), 1)


def _synthetic_frames() -> list[BaselineFrame]:
    """Return separable MFCC-like frame fixtures."""

    return [
        BaselineFrame(0, 0.0, 0.5, np.array([0.0, 0.1, 0.0], dtype=np.float32)),
        BaselineFrame(1, 0.5, 1.0, np.array([0.1, 0.0, 0.0], dtype=np.float32)),
        BaselineFrame(2, 1.0, 1.5, np.array([5.0, 5.1, 5.0], dtype=np.float32)),
        BaselineFrame(3, 1.5, 2.0, np.array([5.1, 5.0, 5.0], dtype=np.float32)),
    ]


def _tone(duration_seconds: float, frequency: float) -> np.ndarray:
    """Create a synthetic tone."""

    samples = int(duration_seconds * SAMPLE_RATE)
    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Write a mono PCM WAV fixture."""

    pcm_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_audio.tobytes())


def _fixture_path(file_name: str) -> Path:
    """Return a writable path for generated test fixtures."""

    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return TEST_TMP_ROOT / file_name


if __name__ == "__main__":
    unittest.main()
