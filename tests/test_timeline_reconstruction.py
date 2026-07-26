"""Tests for speaker timeline reconstruction."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from Project.ssl_pipeline.spectral_clustering import ClusteredEmbedding
from Project.ssl_pipeline.timeline_reconstruction import (
    SpeakerTimeline,
    SpeakerTimelineReconstructor,
    export_timeline_csv,
    export_timeline_json,
    export_timeline_txt,
    plot_speaker_activity_chart,
    plot_speaker_duration_histogram,
    plot_timeline,
    reconstruct_speaker_timeline,
)


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class TimelineReconstructionTests(unittest.TestCase):
    """Validate timeline reconstruction without RTTM or DER/JER."""

    def test_adjacent_segments_merge_for_same_speaker(self) -> None:
        timeline = reconstruct_speaker_timeline(
            [
                _embedding(0, 0.0, 1.0, 0),
                _embedding(1, 1.0, 2.5, 0),
            ],
            merge_silence_threshold_seconds=0.0,
            min_duration_seconds=0.0,
        )

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0], SpeakerTimeline("speaker_0", 0.0, 2.5, 2.5, 2))

    def test_small_silence_gap_merges_for_same_speaker(self) -> None:
        timeline = reconstruct_speaker_timeline(
            [
                _embedding(0, 0.0, 1.0, 1),
                _embedding(1, 1.08, 2.0, 1),
            ],
            merge_silence_threshold_seconds=0.1,
            min_duration_seconds=0.0,
        )

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].start, 0.0)
        self.assertEqual(timeline[0].end, 2.0)
        self.assertEqual(timeline[0].segment_count, 2)

    def test_tiny_segment_removal_then_merge(self) -> None:
        timeline = reconstruct_speaker_timeline(
            [
                _embedding(0, 0.0, 1.0, 0),
                _embedding(1, 1.0, 1.05, 1),
                _embedding(2, 1.05, 2.0, 0),
            ],
            merge_silence_threshold_seconds=0.1,
            min_duration_seconds=0.1,
        )

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].speaker_id, "speaker_0")
        self.assertEqual(timeline[0].start, 0.0)
        self.assertEqual(timeline[0].end, 2.0)

    def test_single_speaker_reconstructs_one_interval(self) -> None:
        timeline = reconstruct_speaker_timeline(
            [
                _embedding(0, 0.0, 1.0, 0),
                _embedding(1, 1.0, 2.0, 0),
                _embedding(2, 2.0, 3.0, 0),
            ],
            min_duration_seconds=0.0,
        )

        self.assertEqual(timeline, [SpeakerTimeline("speaker_0", 0.0, 3.0, 3.0, 3)])

    def test_multiple_speakers_keep_chronological_turns(self) -> None:
        timeline = reconstruct_speaker_timeline(
            [
                _embedding(0, 0.0, 1.0, 0),
                _embedding(1, 1.0, 2.0, 1),
                _embedding(2, 2.0, 3.0, 0),
            ],
            min_duration_seconds=0.0,
        )

        self.assertEqual([item.speaker_id for item in timeline], ["speaker_0", "speaker_1", "speaker_0"])
        self.assertEqual([(item.start, item.end) for item in timeline], [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])

    def test_overlapping_different_speakers_are_clamped_chronologically(self) -> None:
        timeline = reconstruct_speaker_timeline(
            [
                _embedding(0, 0.0, 1.5, 0),
                _embedding(1, 1.0, 2.0, 1),
            ],
            min_duration_seconds=0.0,
        )

        self.assertEqual([(item.start, item.end) for item in timeline], [(0.0, 1.5), (1.5, 2.0)])
        self.assertEqual([item.speaker_id for item in timeline], ["speaker_0", "speaker_1"])

    def test_empty_input_returns_empty_timeline(self) -> None:
        self.assertEqual(reconstruct_speaker_timeline([]), [])

    def test_label_smoothing_relabels_isolated_fragment(self) -> None:
        reconstructor = SpeakerTimelineReconstructor(
            label_smoothing_enabled=True,
            min_duration_seconds=0.0,
        )

        timeline = reconstructor.reconstruct(
            [
                _embedding(0, 0.0, 1.0, 0),
                _embedding(1, 1.0, 2.0, 1),
                _embedding(2, 2.0, 3.0, 0),
            ]
        )

        self.assertEqual(timeline, [SpeakerTimeline("speaker_0", 0.0, 3.0, 3.0, 3)])

    def test_exports_write_json_csv_and_txt(self) -> None:
        timeline = [SpeakerTimeline("speaker_0", 0.0, 1.5, 1.5, 2)]
        json_path = _fixture_path("timeline.json")
        csv_path = _fixture_path("timeline.csv")
        txt_path = _fixture_path("timeline.txt")

        export_timeline_json(timeline, json_path)
        export_timeline_csv(timeline, csv_path)
        export_timeline_txt(timeline, txt_path)

        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))[0]["speaker_id"], "speaker_0")
        self.assertIn("speaker_id,start,end,duration,segment_count", csv_path.read_text(encoding="utf-8"))
        self.assertIn("speaker_0", txt_path.read_text(encoding="utf-8"))

    def test_plots_return_figures(self) -> None:
        timeline = [
            SpeakerTimeline("speaker_0", 0.0, 1.0, 1.0, 1),
            SpeakerTimeline("speaker_1", 1.0, 2.5, 1.5, 1),
        ]

        self.assertEqual(len(plot_timeline(timeline).axes), 1)
        self.assertEqual(len(plot_speaker_activity_chart(timeline).axes), 1)
        self.assertEqual(len(plot_speaker_duration_histogram(timeline).axes), 1)


def _embedding(
    segment_id: int,
    start: float,
    end: float,
    cluster_id: int,
) -> ClusteredEmbedding:
    """Create a clustered embedding fixture."""

    return ClusteredEmbedding(
        segment_id=segment_id,
        start=start,
        end=end,
        cluster_id=cluster_id,
        vector=np.array([1.0, 0.0], dtype=np.float32),
        metadata={"source": "unit-test"},
    )


def _fixture_path(file_name: str) -> Path:
    """Return a writable path for generated test fixtures."""

    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return TEST_TMP_ROOT / file_name


if __name__ == "__main__":
    unittest.main()
