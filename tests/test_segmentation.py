"""Tests for VAD-driven speech segmentation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from Project.ssl_pipeline.segmentation import (
    SpeechSegment,
    load_segments,
    save_segments,
    segment_speech,
)


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class SpeechSegmentationTests(unittest.TestCase):
    """Validate segmentation behavior on VAD timestamp inputs."""

    def test_empty_speech_regions_return_empty_segments(self) -> None:
        segments = segment_speech([], config_path=_write_config())

        self.assertEqual(segments, [])

    def test_one_speech_region_creates_expected_windows(self) -> None:
        segments = segment_speech([(0.3, 4.3)], config_path=_write_config())

        self.assertEqual(
            segments,
            [
                SpeechSegment(segment_id=0, start=0.3, end=2.3),
                SpeechSegment(segment_id=1, start=1.8, end=3.8),
                SpeechSegment(segment_id=2, start=3.3, end=4.3),
            ],
        )

    def test_multiple_speech_regions_keep_stable_ids(self) -> None:
        segments = segment_speech(
            [(0.0, 2.0), (5.0, 7.2)],
            config_path=_write_config(),
        )

        self.assertEqual([segment.segment_id for segment in segments], [0, 1, 2])
        self.assertEqual(segments[0], SpeechSegment(0, 0.0, 2.0))
        self.assertEqual(segments[1], SpeechSegment(1, 5.0, 7.0))
        self.assertEqual(segments[2], SpeechSegment(2, 6.5, 7.2))

    def test_overlapping_windows_follow_hop_length(self) -> None:
        config_path = _write_config(
            {
                "segment_window_seconds": 2.0,
                "segment_hop_seconds": 0.5,
                "segment_min_duration_seconds": 0.5,
                "segment_max_duration_seconds": 2.0,
            }
        )

        segments = segment_speech([(0.0, 3.0)], config_path=config_path)

        self.assertEqual(
            [(segment.start, segment.end) for segment in segments],
            [(0.0, 2.0), (0.5, 2.5), (1.0, 3.0)],
        )

    def test_short_regions_are_filtered(self) -> None:
        segments = segment_speech([(1.0, 1.2)], config_path=_write_config())

        self.assertEqual(segments, [])

    def test_long_region_is_split_into_max_duration_segments(self) -> None:
        config_path = _write_config(
            {
                "segment_window_seconds": 1.5,
                "segment_hop_seconds": 1.5,
                "segment_max_duration_seconds": 1.5,
            }
        )

        segments = segment_speech([(0.0, 5.0)], config_path=config_path)

        self.assertEqual(
            [(segment.start, segment.end) for segment in segments],
            [(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 5.0)],
        )

    def test_save_and_load_segments_json(self) -> None:
        path = _fixture_path("segments.json")
        segments = [SpeechSegment(0, 0.0, 2.0), SpeechSegment(1, 2.0, 3.0)]

        save_segments(segments, path)
        loaded = load_segments(path)

        self.assertEqual(loaded, segments)

    def test_save_and_load_segments_csv(self) -> None:
        path = _fixture_path("segments.csv")
        segments = [SpeechSegment(0, 0.0, 2.0), SpeechSegment(1, 2.0, 3.0)]

        save_segments(segments, path)
        loaded = load_segments(path)

        self.assertEqual(loaded, segments)


def _write_config(overrides: dict[str, object] | None = None) -> Path:
    """Write a minimal segmentation config for tests."""

    config = {
        "sample_rate": 16000,
        "mono": True,
        "normalize_audio": True,
        "segment_duration_seconds": 2.0,
        "segment_window_seconds": 2.0,
        "segment_hop_seconds": 1.5,
        "segment_min_duration_seconds": 0.5,
        "segment_max_duration_seconds": 2.0,
        "segment_include_short_final": True,
        "pyannote_model": "pyannote/speaker-diarization",
        "vad_backend": "energy",
        "vad_pyannote_model": "pyannote/voice-activity-detection",
        "vad_min_speech_duration_seconds": 0.25,
        "vad_min_silence_duration_seconds": 0.15,
        "vad_merge_threshold_seconds": 0.2,
        "vad_frame_duration_ms": 30,
        "vad_webrtc_aggressiveness": 2,
        "vad_energy_threshold": 0.01,
    }
    if overrides:
        config.update(overrides)

    path = _fixture_path("segmentation_config.json")
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _fixture_path(file_name: str) -> Path:
    """Return a writable path for generated test fixtures."""

    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return TEST_TMP_ROOT / file_name


if __name__ == "__main__":
    unittest.main()
