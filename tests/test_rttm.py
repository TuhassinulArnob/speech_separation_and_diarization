"""Tests for RTTM reader and writer."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from Project.ssl_pipeline.rttm import (
    RTTMReader,
    RTTMSpeakerTurn,
    RTTMWriter,
    export_rttm_json,
    load_rttm_json,
    parse_rttm,
    rttm_turns_to_timeline,
    timeline_to_rttm_turns,
    validate_rttm,
    validate_rttm_turns,
    write_rttm,
)
from Project.ssl_pipeline.timeline_reconstruction import SpeakerTimeline


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class RTTMTests(unittest.TestCase):
    """Validate RTTM conversion without DER/JER evaluation."""

    def test_write_rttm_uses_standard_speaker_rows(self) -> None:
        path = _fixture_path("written.rttm")

        write_rttm(_timeline(), path, recording_id="recording1", channel_id="1")

        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            lines[0],
            "SPEAKER recording1 1 0.00 2.34 <NA> <NA> speaker_0 <NA> <NA>",
        )
        self.assertEqual(
            lines[1],
            "SPEAKER recording1 1 2.34 1.87 <NA> <NA> speaker_1 <NA> <NA>",
        )

    def test_read_rttm_accepts_written_file(self) -> None:
        path = _fixture_path("read.rttm")
        path.write_text(
            "SPEAKER rec 1 0.00 2.34 <NA> <NA> speaker_0 <NA> <NA>\n",
            encoding="utf-8",
        )

        turns = parse_rttm(path)

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].recording_id, "rec")
        self.assertEqual(turns[0].speaker_id, "speaker_0")
        self.assertEqual(turns[0].duration, 2.34)

    def test_read_rttm_accepts_nine_field_example_rows(self) -> None:
        text = "SPEAKER recording1 1 0.00 2.34 <NA> <NA> speaker_0 <NA>\n"

        turns = RTTMReader().parse_text(text)

        self.assertEqual(turns[0].signal_lookahead_time, "<NA>")

    def test_round_trip_conversion_preserves_timeline_values(self) -> None:
        path = _fixture_path("roundtrip.rttm")
        writer = RTTMWriter(recording_id="rec", channel_id="1", decimal_places=3)

        writer.write(_timeline(), path)
        turns = parse_rttm(path)
        timeline = rttm_turns_to_timeline(turns)

        self.assertEqual([item.speaker_id for item in timeline], ["speaker_0", "speaker_1"])
        self.assertEqual([item.start for item in timeline], [0.0, 2.34])
        self.assertEqual([item.duration for item in timeline], [2.34, 1.87])

    def test_timeline_to_rttm_turns_validates_speaker_names(self) -> None:
        timeline = [SpeakerTimeline("bad speaker", 0.0, 1.0, 1.0, 1)]

        with self.assertRaisesRegex(ValueError, "speaker_id"):
            timeline_to_rttm_turns(timeline, recording_id="rec", channel_id="1")

    def test_malformed_rttm_raises(self) -> None:
        path = _fixture_path("malformed.rttm")
        path.write_text("SPEAKER rec 1 0.00\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Malformed RTTM"):
            parse_rttm(path)

    def test_empty_rttm_raises_by_default(self) -> None:
        path = _fixture_path("empty.rttm")
        path.write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "empty"):
            parse_rttm(path)

    def test_empty_rttm_can_be_allowed(self) -> None:
        path = _fixture_path("empty_allowed.rttm")
        path.write_text("", encoding="utf-8")

        self.assertEqual(parse_rttm(path, allow_empty=True), [])

    def test_negative_start_and_duration_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "start time"):
            validate_rttm_turns([RTTMSpeakerTurn("rec", "1", -0.1, 1.0, "speaker_0")])

        with self.assertRaisesRegex(ValueError, "duration"):
            validate_rttm_turns([RTTMSpeakerTurn("rec", "1", 0.0, 0.0, "speaker_0")])

    def test_overlap_raises(self) -> None:
        turns = [
            RTTMSpeakerTurn("rec", "1", 0.0, 2.0, "speaker_0"),
            RTTMSpeakerTurn("rec", "1", 1.5, 1.0, "speaker_1"),
        ]

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_rttm_turns(turns)

    def test_validate_rttm_returns_true_for_valid_file(self) -> None:
        path = _fixture_path("valid.rttm")
        path.write_text(
            "SPEAKER rec 1 0.00 1.00 <NA> <NA> speaker_0 <NA> <NA>\n",
            encoding="utf-8",
        )

        self.assertTrue(validate_rttm(path))

    def test_json_conversion_round_trip(self) -> None:
        json_path = _fixture_path("rttm.json")
        turns = [RTTMSpeakerTurn("rec", "1", 0.0, 1.0, "speaker_0")]

        export_rttm_json(turns, json_path)
        loaded = load_rttm_json(json_path)

        self.assertEqual(loaded, turns)
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))[0]["speaker_id"], "speaker_0")


def _timeline() -> list[SpeakerTimeline]:
    """Return a small timeline fixture."""

    return [
        SpeakerTimeline("speaker_0", 0.0, 2.34, 2.34, 2),
        SpeakerTimeline("speaker_1", 2.34, 4.21, 1.87, 1),
    ]


def _fixture_path(file_name: str) -> Path:
    """Return a writable path for generated test fixtures."""

    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return TEST_TMP_ROOT / file_name


if __name__ == "__main__":
    unittest.main()
