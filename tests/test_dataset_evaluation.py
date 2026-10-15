"""Tests for the dataset-level diarization evaluation harness.

These tests use small synthetic RTTM annotations and a stub diarization
function, so they never download models from Hugging Face and never depend on
the real evaluation dataset.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import soundfile as sf

from Project.ssl_pipeline.dataset_evaluation import (
    DiarizationScorer,
    EvaluationProtocol,
    align_recording_id,
    clear_results,
    discover_evaluation_items,
    find_unpaired_audio,
    load_reference_turns,
    load_results,
    parse_lab_file,
    run_dataset_evaluation,
    write_hypothesis_rttm,
)
from Project.ssl_pipeline.rttm import parse_rttm

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests" / "dataset_evaluation"

# Synthetic two-speaker reference: alternating turns plus one overlapped region,
# so the fixtures exercise the same shapes as a real reference annotation.
REFERENCE_ROWS = [
    ("MEETING_01", 0.0, 2.0, "speakerA"),
    ("MEETING_01", 2.0, 2.0, "speakerB"),
    ("MEETING_01", 4.0, 2.0, "speakerA"),
    ("MEETING_01", 5.0, 1.0, "speakerB"),
]


class DatasetDiscoveryTests(unittest.TestCase):
    """Discovery and pairing of manually placed evaluation files."""

    def test_reference_rttm_loads_with_overlapping_speech(self) -> None:
        root = _fresh_dir("reference_loads")
        reference = _write_rttm(root / "meeting.rttm", REFERENCE_ROWS)

        turns = load_reference_turns(reference)

        self.assertEqual(len(turns), 4)
        self.assertEqual({turn.speaker_id for turn in turns}, {"speakerA", "speakerB"})
        self.assertEqual(turns[0].recording_id, "MEETING_01")

    def test_hypothesis_rttm_loads_and_round_trips(self) -> None:
        root = _fresh_dir("hypothesis_loads")
        source = _write_rttm(
            root / "hyp.rttm",
            [("audio_stem", 0.0, 2.0, "Speaker 1".replace(" ", "_"))],
        )

        turns = parse_rttm(source)
        relabelled = align_recording_id(turns, "MEETING_01")
        output = write_hypothesis_rttm(relabelled, root / "hypothesis.rttm")
        reloaded = parse_rttm(output)

        self.assertEqual(reloaded[0].recording_id, "MEETING_01")
        self.assertEqual(reloaded[0].speaker_id, "Speaker_1")
        self.assertAlmostEqual(reloaded[0].start, 0.0, places=6)

    def test_multiple_evaluation_files_are_discovered(self) -> None:
        root = _fresh_dir("multi_discovery")
        for stem in ("alpha", "beta", "gamma"):
            _write_audio(root / f"{stem}.wav", seconds=6.0)
            _write_rttm(root / f"{stem}.rttm", REFERENCE_ROWS)
        # Audio without a reference must be skipped, not guessed at.
        _write_audio(root / "unlabelled.wav", seconds=3.0)
        # Previously generated output must never be treated as dataset input.
        generated = root / "results" / "old"
        generated.mkdir(parents=True)
        _write_audio(generated / "processed.wav", seconds=1.0)
        _write_rttm(generated / "processed.rttm", REFERENCE_ROWS)

        items = discover_evaluation_items(root)
        orphans = find_unpaired_audio(root)

        self.assertEqual([item.name for item in items], ["alpha", "beta", "gamma"])
        self.assertEqual([path.name for path in orphans], ["unlabelled.wav"])

    def test_optional_lab_file_is_detected_but_not_required(self) -> None:
        root = _fresh_dir("lab_optional")
        _write_audio(root / "with_lab.wav", seconds=6.0)
        _write_rttm(root / "with_lab.rttm", REFERENCE_ROWS)
        (root / "with_lab.lab").write_text("0.0\t2.0\tspeech\n2.5\t4.0\tspeech\n", encoding="utf-8")
        _write_audio(root / "no_lab.wav", seconds=6.0)
        _write_rttm(root / "no_lab.rttm", REFERENCE_ROWS)

        items = {item.name: item for item in discover_evaluation_items(root)}

        self.assertIsNotNone(items["with_lab"].lab_path)
        self.assertIsNone(items["no_lab"].lab_path)
        self.assertEqual(len(parse_lab_file(items["with_lab"].lab_path)), 2)

    def test_missing_reference_is_reported_clearly(self) -> None:
        root = _fresh_dir("missing_reference")
        _write_audio(root / "orphan.wav", seconds=4.0)

        with self.assertRaises(FileNotFoundError):
            load_reference_turns(root / "orphan.rttm")
        with self.assertRaisesRegex(FileNotFoundError, "No audio/reference RTTM pairs"):
            run_dataset_evaluation(root, diarize=_stub_diarize, log=lambda _message: None)

    def test_invalid_rttm_raises_value_error(self) -> None:
        root = _fresh_dir("invalid_rttm")
        broken = root / "broken.rttm"
        broken.write_text("SPEAKER only three fields\n", encoding="utf-8")
        empty = root / "empty.rttm"
        empty.write_text("\n", encoding="utf-8")
        bad_time = root / "bad_time.rttm"
        bad_time.write_text(
            "SPEAKER MEETING_01 1 start 2.00 <NA> <NA> speakerA <NA> <NA>\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "expected 9 or 10 fields"):
            load_reference_turns(broken)
        with self.assertRaises(ValueError):
            load_reference_turns(empty)
        with self.assertRaisesRegex(ValueError, "invalid timestamp"):
            load_reference_turns(bad_time)


class ScoringTests(unittest.TestCase):
    """DER/JER computation and speaker-label handling."""

    def test_der_is_computed_from_reference_and_hypothesis(self) -> None:
        reference = _turns(REFERENCE_ROWS)
        # Hypothesis misses the final 1.0s of speakerA's third turn.
        hypothesis = _turns(
            [
                ("MEETING_01", 0.0, 2.0, "spk0"),
                ("MEETING_01", 2.0, 2.0, "spk1"),
                ("MEETING_01", 4.0, 1.0, "spk0"),
                ("MEETING_01", 5.0, 1.0, "spk1"),
            ]
        )
        scorer = DiarizationScorer(EvaluationProtocol())

        result = scorer.score(reference, hypothesis, uri="MEETING_01", uem_end=6.0)

        self.assertAlmostEqual(result.scored_duration, 7.0, places=6)
        self.assertAlmostEqual(result.missed_speech, 1.0, places=6)
        self.assertAlmostEqual(result.false_alarm, 0.0, places=6)
        self.assertAlmostEqual(result.der, 1.0 / 7.0, places=6)
        self.assertAlmostEqual(scorer.overall_der(), 1.0 / 7.0, places=6)

    def test_speaker_label_permutation_does_not_change_der(self) -> None:
        reference = _turns(REFERENCE_ROWS)
        swapped = _turns(
            [
                ("MEETING_01", 0.0, 2.0, "speakerB"),
                ("MEETING_01", 2.0, 2.0, "speakerA"),
                ("MEETING_01", 4.0, 2.0, "speakerB"),
                ("MEETING_01", 5.0, 1.0, "speakerA"),
            ]
        )
        renamed = _turns(
            [
                ("MEETING_01", 0.0, 2.0, "cluster_7"),
                ("MEETING_01", 2.0, 2.0, "cluster_3"),
                ("MEETING_01", 4.0, 2.0, "cluster_7"),
                ("MEETING_01", 5.0, 1.0, "cluster_3"),
            ]
        )

        swapped_der = DiarizationScorer(EvaluationProtocol()).score(
            reference, swapped, uri="MEETING_01", uem_end=6.0
        ).der
        renamed_der = DiarizationScorer(EvaluationProtocol()).score(
            reference, renamed, uri="MEETING_01", uem_end=6.0
        ).der

        self.assertAlmostEqual(swapped_der, 0.0, places=6)
        self.assertAlmostEqual(renamed_der, 0.0, places=6)

    def test_missing_hypothesis_is_handled_gracefully(self) -> None:
        root = _fresh_dir("missing_hypothesis")
        _write_audio(root / "recording.wav", seconds=6.0)
        _write_rttm(root / "recording.rttm", REFERENCE_ROWS)

        def diarize_without_output(audio_path, work_dir, *, num_speakers):
            work_dir.mkdir(parents=True, exist_ok=True)
            return work_dir / "hypothesis.rttm"

        outcome = run_dataset_evaluation(
            root,
            diarize=diarize_without_output,
            log=lambda _message: None,
        )

        self.assertEqual(outcome.records, [])
        self.assertEqual(len(outcome.failures), 1)
        self.assertIn("Hypothesis RTTM not found", outcome.failures[0]["error"])
        # The failure is recorded in the summary rather than crashing the run.
        self.assertEqual(outcome.summary["failed_count"], 1)
        self.assertIsNone(outcome.summary["overall_der"])

    def test_protocol_is_explicit_and_validated(self) -> None:
        protocol = EvaluationProtocol(collar=0.25, skip_overlap=True)

        self.assertEqual(
            protocol.to_dict(),
            {
                "collar": 0.25,
                "skip_overlap": True,
                "uem_mode": "full_file",
                "speaker_count_mode": "oracle",
            },
        )
        with self.assertRaisesRegex(ValueError, "collar"):
            EvaluationProtocol(collar=-0.1)
        with self.assertRaisesRegex(ValueError, "uem_mode"):
            EvaluationProtocol(uem_mode="whatever")


class ResultPersistenceTests(unittest.TestCase):
    """Generated output layout, aggregation, and cleanup safety."""

    def test_results_are_saved_for_each_file_and_aggregated(self) -> None:
        root = _fresh_dir("results_saved")
        for stem in ("alpha", "beta"):
            _write_audio(root / f"{stem}.wav", seconds=6.0)
            _write_rttm(root / f"{stem}.rttm", REFERENCE_ROWS)

        outcome = run_dataset_evaluation(root, diarize=_stub_diarize, log=lambda _message: None)

        self.assertEqual(len(outcome.records), 2)
        for stem in ("alpha", "beta"):
            result_dir = root / "results" / stem
            self.assertTrue((result_dir / "hypothesis.rttm").exists())
            self.assertTrue((result_dir / "evaluation.json").exists())
            record = json.loads((result_dir / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(record["file"], f"{stem}.wav")
            self.assertEqual(record["recording_id"], "MEETING_01")
            self.assertEqual(record["reference_speakers"], 2)
            self.assertEqual(record["detected_speakers"], 2)
            self.assertAlmostEqual(record["duration"], 6.0, places=3)
            self.assertEqual(record["protocol"]["collar"], 0.0)
            self.assertIs(record["protocol"]["skip_overlap"], False)
            self.assertIs(record["system"]["pretrained_diarization_pipeline_used"], False)
            # The hypothesis is written with the reference recording id so any
            # external scoring tool can match the two files.
            hypothesis = parse_rttm(result_dir / "hypothesis.rttm")
            self.assertTrue(all(turn.recording_id == "MEETING_01" for turn in hypothesis))

        summary = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["file_count"], 2)
        self.assertAlmostEqual(summary["total_duration"], 12.0, places=3)
        self.assertIsNotNone(summary["overall_der"])
        self.assertIn("average_der", summary)
        self.assertIn("min_der", summary)
        self.assertIn("max_der", summary)
        self.assertEqual(len(summary["files"]), 2)

        loaded = load_results(root)
        self.assertEqual(len(loaded["results"]), 2)
        self.assertEqual(loaded["summary"]["file_count"], 2)

    def test_speaker_count_is_read_from_reference_not_assumed(self) -> None:
        root = _fresh_dir("speaker_count")
        _write_audio(root / "three.wav", seconds=9.0)
        _write_rttm(
            root / "three.rttm",
            [
                ("MEETING_02", 0.0, 3.0, "spk1"),
                ("MEETING_02", 3.0, 3.0, "spk2"),
                ("MEETING_02", 6.0, 3.0, "spk3"),
            ],
        )
        requested: list[int | None] = []

        def recording_diarize(audio_path, work_dir, *, num_speakers):
            requested.append(num_speakers)
            return _stub_diarize(audio_path, work_dir, num_speakers=num_speakers)

        outcome = run_dataset_evaluation(root, diarize=recording_diarize, log=lambda _message: None)

        self.assertEqual(requested, [3])
        self.assertEqual(outcome.records[0]["reference_speakers"], 3)

        estimated: list[int | None] = []

        def estimating_diarize(audio_path, work_dir, *, num_speakers):
            estimated.append(num_speakers)
            return _stub_diarize(audio_path, work_dir, num_speakers=num_speakers)

        run_dataset_evaluation(
            root,
            protocol=EvaluationProtocol(speaker_count_mode="estimated"),
            diarize=estimating_diarize,
            log=lambda _message: None,
        )
        self.assertEqual(estimated, [None])

    def test_clear_results_never_deletes_reference_dataset_files(self) -> None:
        root = _fresh_dir("clear_safety")
        audio = _write_audio(root / "keepme.flac", seconds=6.0)
        reference = _write_rttm(root / "keepme.rttm", REFERENCE_ROWS)
        lab = root / "keepme.lab"
        lab.write_text("0.0\t2.0\tspeech\n", encoding="utf-8")
        reference_before = reference.read_text(encoding="utf-8")

        run_dataset_evaluation(root, diarize=_stub_diarize, log=lambda _message: None)
        self.assertTrue((root / "results" / "keepme" / "evaluation.json").exists())

        removed = clear_results(root)

        self.assertFalse((root / "results").exists())
        self.assertTrue(any(name.endswith("evaluation.json") for name in removed))
        # Reference dataset files are untouched, byte for byte.
        self.assertTrue(audio.exists())
        self.assertTrue(lab.exists())
        self.assertTrue(reference.exists())
        self.assertEqual(reference.read_text(encoding="utf-8"), reference_before)
        # Clearing twice is safe.
        self.assertEqual(clear_results(root), [])

    def test_reference_annotation_is_not_modified_by_evaluation(self) -> None:
        root = _fresh_dir("reference_immutable")
        _write_audio(root / "immutable.wav", seconds=6.0)
        reference = _write_rttm(root / "immutable.rttm", REFERENCE_ROWS)
        before = reference.read_bytes()

        run_dataset_evaluation(root, diarize=_stub_diarize, log=lambda _message: None)

        self.assertEqual(reference.read_bytes(), before)

    def test_realtime_log_messages_are_generated_dynamically(self) -> None:
        root = _fresh_dir("dynamic_logs")
        _write_audio(root / "logged.wav", seconds=6.0)
        _write_rttm(root / "logged.rttm", REFERENCE_ROWS)
        messages: list[str] = []

        run_dataset_evaluation(root, diarize=_stub_diarize, log=messages.append)

        joined = "\n".join(messages)
        self.assertIn("logged.wav", joined)
        self.assertIn("logged.rttm", joined)
        self.assertIn("Evaluating logged (1/1)", joined)
        self.assertIn("Calculating DER", joined)
        self.assertTrue(any(line.startswith("Evaluation completed") for line in messages))


def _stub_diarize(audio_path: Path, work_dir: Path, *, num_speakers: int | None) -> Path:
    """Stand in for the diarization pipeline without loading any model.

    Produces a plausible hypothesis with pipeline-style cluster labels and a
    small timing error, so scoring exercises optimal speaker mapping.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "hypothesis.rttm"
    _write_rttm(
        output,
        [
            (audio_path.stem, 0.1, 1.9, "cluster_1"),
            (audio_path.stem, 2.0, 2.0, "cluster_0"),
            (audio_path.stem, 4.0, 2.0, "cluster_1"),
        ],
    )
    return output


def _turns(rows):
    """Build RTTM turns from ``(recording, start, duration, speaker)`` rows."""

    from Project.ssl_pipeline.rttm import RTTMSpeakerTurn

    return [
        RTTMSpeakerTurn(recording_id=recording, channel_id="1", start=start, duration=duration, speaker_id=speaker)
        for recording, start, duration, speaker in rows
    ]


def _write_rttm(path: Path, rows) -> Path:
    """Write a synthetic RTTM file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"SPEAKER {recording} 1 {start:.2f} {duration:.2f} <NA> <NA> {speaker} <NA> <NA>"
        for recording, start, duration, speaker in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_audio(path: Path, *, seconds: float, sample_rate: int = 16_000) -> Path:
    """Write a short silent audio file so duration lookups succeed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    sf.write(path, [0.0] * frames, sample_rate)
    return path


def _fresh_dir(name: str) -> Path:
    """Return an empty temporary directory for one test."""

    import shutil

    path = TEST_TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    unittest.main()
