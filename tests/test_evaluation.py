"""Tests for DER and JER evaluation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from Project.ssl_pipeline.evaluation import (
    DiarizationEvaluator,
    evaluate_rttm,
    evaluate_rttm_directory,
    export_evaluation_csv,
    export_evaluation_json,
    export_evaluation_markdown,
    plot_der_breakdown,
    plot_recording_comparison,
    plot_speaker_confusion_chart,
)


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class EvaluationTests(unittest.TestCase):
    """Validate DER/JER evaluation without baseline or experiment runner code."""

    def test_perfect_match_has_zero_der_and_jer(self) -> None:
        reference = _write_rttm(
            "perfect_ref.rttm",
            [
                ("rec", 0.0, 1.0, "speaker_0"),
                ("rec", 1.0, 1.0, "speaker_1"),
            ],
        )
        hypothesis = _write_rttm(
            "perfect_hyp.rttm",
            [
                ("rec", 0.0, 1.0, "speaker_0"),
                ("rec", 1.0, 1.0, "speaker_1"),
            ],
        )

        report = evaluate_rttm(reference, hypothesis, use_pyannote=False)

        self.assertEqual(report.der, 0.0)
        self.assertEqual(report.jer, 0.0)
        self.assertEqual(report.duration, 2.0)

    def test_small_timing_error_counts_miss_and_false_alarm(self) -> None:
        reference = _write_rttm("timing_ref.rttm", [("rec", 0.0, 1.0, "speaker_0")])
        hypothesis = _write_rttm("timing_hyp.rttm", [("rec", 0.1, 1.0, "speaker_0")])

        report = evaluate_rttm(reference, hypothesis, use_pyannote=False)

        self.assertAlmostEqual(report.missed_speech, 0.1, places=6)
        self.assertAlmostEqual(report.false_alarm, 0.1, places=6)
        self.assertAlmostEqual(report.der, 0.2, places=6)

    def test_wrong_speaker_assignment_counts_confusion(self) -> None:
        reference = _write_rttm(
            "confusion_ref.rttm",
            [
                ("rec", 0.0, 1.0, "speaker_0"),
                ("rec", 1.0, 1.0, "speaker_1"),
            ],
        )
        hypothesis = _write_rttm(
            "confusion_hyp.rttm",
            [
                ("rec", 0.0, 1.0, "speaker_0"),
                ("rec", 1.0, 1.0, "speaker_0"),
            ],
        )

        report = evaluate_rttm(reference, hypothesis, use_pyannote=False)

        self.assertAlmostEqual(report.speaker_confusion, 1.0, places=6)
        self.assertAlmostEqual(report.der, 0.5, places=6)
        self.assertGreater(report.jer, 0.0)

    def test_missing_speech_counts_miss(self) -> None:
        reference = _write_rttm("miss_ref.rttm", [("rec", 0.0, 2.0, "speaker_0")])
        hypothesis = _write_rttm("miss_hyp.rttm", [("rec", 0.0, 1.0, "speaker_0")])

        report = evaluate_rttm(reference, hypothesis, use_pyannote=False)

        self.assertEqual(report.missed_speech, 1.0)
        self.assertEqual(report.false_alarm, 0.0)
        self.assertEqual(report.der, 0.5)

    def test_false_alarm_counts_extra_hypothesis_speech(self) -> None:
        reference = _write_rttm("fa_ref.rttm", [("rec", 0.0, 1.0, "speaker_0")])
        hypothesis = _write_rttm("fa_hyp.rttm", [("rec", 0.0, 2.0, "speaker_0")])

        report = evaluate_rttm(reference, hypothesis, use_pyannote=False)

        self.assertEqual(report.false_alarm, 1.0)
        self.assertEqual(report.missed_speech, 0.0)
        self.assertEqual(report.der, 1.0)

    def test_multiple_recordings_are_reported_and_averaged(self) -> None:
        reference = _write_rttm(
            "multi_ref.rttm",
            [
                ("rec1", 0.0, 1.0, "speaker_0"),
                ("rec2", 0.0, 1.0, "speaker_0"),
            ],
        )
        hypothesis = _write_rttm(
            "multi_hyp.rttm",
            [
                ("rec1", 0.0, 1.0, "speaker_0"),
                ("rec2", 0.0, 0.5, "speaker_0"),
            ],
        )

        report = evaluate_rttm(reference, hypothesis, use_pyannote=False)

        self.assertEqual(len(report.recordings), 2)
        self.assertEqual(report.recording_name, "ALL")
        self.assertEqual(report.der, 0.25)
        self.assertEqual(report.average_der, 0.25)

    def test_evaluate_directory_matches_files_by_name(self) -> None:
        reference_dir = _fixture_dir("reference_dir")
        hypothesis_dir = _fixture_dir("hypothesis_dir")
        _write_rttm_in(reference_dir, "a.rttm", [("a", 0.0, 1.0, "speaker_0")])
        _write_rttm_in(hypothesis_dir, "a.rttm", [("a", 0.0, 1.0, "speaker_0")])

        report = evaluate_rttm_directory(reference_dir, hypothesis_dir, use_pyannote=False)

        self.assertEqual(report.der, 0.0)
        self.assertEqual(report.recordings[0].recording_name, "a")

    def test_empty_reference_rttm_raises(self) -> None:
        reference = _fixture_path("empty_eval_ref.rttm")
        hypothesis = _write_rttm("empty_eval_hyp.rttm", [("rec", 0.0, 1.0, "speaker_0")])
        reference.write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_rttm(reference, hypothesis, use_pyannote=False)

    def test_exports_write_json_csv_and_markdown(self) -> None:
        report = _perfect_report()
        json_path = _fixture_path("evaluation.json")
        csv_path = _fixture_path("evaluation.csv")
        md_path = _fixture_path("evaluation.md")

        export_evaluation_json(report, json_path)
        export_evaluation_csv(report, csv_path)
        export_evaluation_markdown(report, md_path)

        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["der"], 0.0)
        self.assertIn("recording_name,duration,der,jer", csv_path.read_text(encoding="utf-8"))
        self.assertIn("# Diarization Evaluation Report", md_path.read_text(encoding="utf-8"))

    def test_plots_return_figures(self) -> None:
        report = _perfect_report()

        self.assertEqual(len(plot_der_breakdown(report).axes), 1)
        self.assertEqual(len(plot_recording_comparison(report).axes), 1)
        self.assertEqual(len(plot_speaker_confusion_chart(report).axes), 1)

    def test_negative_collar_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "collar"):
            DiarizationEvaluator(collar_seconds=-0.1)

    def test_overlapping_speech_evaluated_with_and_without_skip_overlap(self) -> None:
        reference = _write_rttm(
            "overlap_ref.rttm",
            [
                ("rec", 0.0, 3.0, "speaker_0"),
                ("rec", 2.0, 2.0, "speaker_1"),  # Overlap between 2.0 and 3.0
            ],
        )
        hypothesis = _write_rttm(
            "overlap_hyp.rttm",
            [
                ("rec", 0.0, 3.0, "speaker_0"),
                # speaker_1 is completely missed in hypothesis
            ],
        )

        # When skip_overlap=False, the overlapped region [2.0, 3.0] is scored
        # Total reference duration: 3.0 + 2.0 = 5.0s
        # speaker_1 is missed for 2.0s
        report_with_overlap = evaluate_rttm(
            reference, hypothesis, use_pyannote=False, skip_overlap=False
        )
        self.assertAlmostEqual(report_with_overlap.duration, 5.0, places=6)
        self.assertAlmostEqual(report_with_overlap.missed_speech, 2.0, places=6)
        self.assertAlmostEqual(report_with_overlap.der, 2.0 / 5.0, places=6)

        # When skip_overlap=True, [2.0, 3.0] (where 2 speakers talk) is excluded
        # Scored region: [0.0, 2.0] for speaker_0 (dur 2.0, correct)
        # and [3.0, 4.0] for speaker_1 (dur 1.0, missed)
        # Total scored duration: 2.0 + 1.0 = 3.0s, missed: 1.0s
        report_skip_overlap = evaluate_rttm(
            reference, hypothesis, use_pyannote=False, skip_overlap=True
        )
        self.assertAlmostEqual(report_skip_overlap.duration, 3.0, places=6)
        self.assertAlmostEqual(report_skip_overlap.missed_speech, 1.0, places=6)
        self.assertAlmostEqual(report_skip_overlap.der, 1.0 / 3.0, places=6)

    def test_three_speaker_permutation_invariance(self) -> None:
        reference = _write_rttm(
            "perm_ref.rttm",
            [
                ("rec", 0.0, 1.0, "spk_A"),
                ("rec", 1.0, 1.0, "spk_B"),
                ("rec", 2.0, 1.0, "spk_C"),
            ],
        )
        # Hypothesis permutes speaker labels: A->Y, B->Z, C->X
        hypothesis = _write_rttm(
            "perm_hyp.rttm",
            [
                ("rec", 0.0, 1.0, "hyp_Y"),
                ("rec", 1.0, 1.0, "hyp_Z"),
                ("rec", 2.0, 1.0, "hyp_X"),
            ],
        )

        report = evaluate_rttm(reference, hypothesis, use_pyannote=False)
        self.assertEqual(report.der, 0.0)
        self.assertEqual(report.jer, 0.0)
        self.assertEqual(report.speaker_confusion, 0.0)

    def test_malformed_rttm_fails_safely(self) -> None:
        ref_path = _fixture_path("malformed.rttm")
        ref_path.write_text("INVALID RTTM LINE NOT ENOUGH FIELDS\n", encoding="utf-8")
        hyp_path = _write_rttm("valid.rttm", [("rec", 0.0, 1.0, "speaker_0")])

        with self.assertRaises(ValueError):
            evaluate_rttm(ref_path, hyp_path, use_pyannote=False)


def _perfect_report():
    """Return a simple perfect-match report."""

    reference = _write_rttm("export_ref.rttm", [("rec", 0.0, 1.0, "speaker_0")])
    hypothesis = _write_rttm("export_hyp.rttm", [("rec", 0.0, 1.0, "speaker_0")])
    return evaluate_rttm(reference, hypothesis, use_pyannote=False)


def _write_rttm(
    file_name: str,
    rows: list[tuple[str, float, float, str]],
) -> Path:
    """Write an RTTM fixture under tmp_tests."""

    return _write_rttm_in(TEST_TMP_ROOT, file_name, rows)


def _write_rttm_in(
    directory: Path,
    file_name: str,
    rows: list[tuple[str, float, float, str]],
) -> Path:
    """Write an RTTM fixture in a selected directory."""

    directory.mkdir(exist_ok=True)
    path = directory / file_name
    lines = [
        f"SPEAKER {recording} 1 {start:.2f} {duration:.2f} <NA> <NA> {speaker} <NA> <NA>"
        for recording, start, duration, speaker in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fixture_path(file_name: str) -> Path:
    """Return a writable path for generated test fixtures."""

    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return TEST_TMP_ROOT / file_name


def _fixture_dir(dir_name: str) -> Path:
    """Return a writable directory for generated test fixtures."""

    path = TEST_TMP_ROOT / dir_name
    path.mkdir(exist_ok=True)
    return path


if __name__ == "__main__":
    unittest.main()
