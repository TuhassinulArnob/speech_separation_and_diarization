"""Unit tests for Whisper speaker-attributed transcription."""

from __future__ import annotations

import json
import shutil
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from Project.transcription.whisper_transcription import (
    TranscriptSegment,
    WhisperUnavailableError,
    load_whisper_model,
    transcribe_speaker_timeline,
)

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests" / "transcription"


class WhisperTranscriptionTests(unittest.TestCase):
    def setUp(self) -> None:
        if TEST_TMP_ROOT.exists():
            shutil.rmtree(TEST_TMP_ROOT)
        TEST_TMP_ROOT.mkdir(parents=True)
        self.audio_path = TEST_TMP_ROOT / "input.wav"
        self.timeline_path = TEST_TMP_ROOT / "timeline.json"
        self.output_dir = TEST_TMP_ROOT / "output"
        self._write_silence(self.audio_path, duration_seconds=1.0)

    def test_transcribe_speaker_timeline_writes_json_and_preserves_speaker_labels(self) -> None:
        self._write_timeline([
            {"speaker_id": "speaker_0", "start": 0.0, "end": 1.0},
        ])

        mock_model = mock.Mock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello world."},
            ],
            "text": "Hello world.",
        }

        with mock.patch("Project.transcription.whisper_transcription.load_whisper_model", return_value=mock_model):
            segments = transcribe_speaker_timeline(
                audio_path=self.audio_path,
                timeline_path=self.timeline_path,
                output_dir=self.output_dir,
                model_name="base",
                language="en",
                device="cpu",
            )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].speaker, "Speaker 0")
        self.assertEqual(segments[0].text, "Hello world.")
        self.assertTrue((self.output_dir / "transcript.json").exists())
        self.assertTrue((self.output_dir / "transcript.txt").exists())
        self.assertTrue((self.output_dir / "transcript.srt").exists())

        transcript = json.loads((self.output_dir / "transcript.json").read_text(encoding="utf-8"))
        self.assertEqual(transcript["segments"][0]["speaker"], "Speaker 0")
        self.assertEqual(transcript["segments"][0]["start"], 0.0)
        self.assertEqual(transcript["segments"][0]["end"], 1.0)

    def test_transcript_timestamps_preserved_from_timeline_entries(self) -> None:
        self._write_timeline([
            {"speaker_id": "speaker_1", "start": 0.0, "end": 0.5},
            {"speaker_id": "speaker_2", "start": 0.5, "end": 1.0},
        ])

        mock_model = mock.Mock()
        mock_model.transcribe.side_effect = [
            {"segments": [{"start": 0.0, "end": 0.5, "text": "First."}], "text": "First."},
            {"segments": [{"start": 0.0, "end": 0.5, "text": "Second."}], "text": "Second."},
        ]

        with mock.patch("Project.transcription.whisper_transcription.load_whisper_model", return_value=mock_model):
            segments = transcribe_speaker_timeline(
                audio_path=self.audio_path,
                timeline_path=self.timeline_path,
                output_dir=self.output_dir,
                model_name="base",
                language="auto",
                device="cpu",
            )

        self.assertEqual([segment.speaker for segment in segments], ["Speaker 1", "Speaker 2"])
        self.assertEqual([segment.start for segment in segments], [0.0, 0.5])
        self.assertEqual([segment.end for segment in segments], [0.5, 1.0])

    def test_transcribe_speaker_timeline_raises_when_whisper_unavailable(self) -> None:
        self._write_timeline([
            {"speaker_id": "speaker_0", "start": 0.0, "end": 1.0},
        ])
        with mock.patch("Project.transcription.whisper_transcription.load_whisper_model", side_effect=WhisperUnavailableError("Unavailable")):
            with self.assertRaises(WhisperUnavailableError):
                transcribe_speaker_timeline(
                    audio_path=self.audio_path,
                    timeline_path=self.timeline_path,
                    output_dir=self.output_dir,
                    model_name="base",
                    language="auto",
                    device="cpu",
                )

    def test_transcribe_speaker_timeline_saves_empty_segments_when_no_output(self) -> None:
        self._write_timeline([
            {"speaker_id": "speaker_0", "start": 0.0, "end": 1.0},
        ])
        mock_model = mock.Mock()
        mock_model.transcribe.return_value = {"text": ""}

        with mock.patch("Project.transcription.whisper_transcription.load_whisper_model", return_value=mock_model):
            segments = transcribe_speaker_timeline(
                audio_path=self.audio_path,
                timeline_path=self.timeline_path,
                output_dir=self.output_dir,
                model_name="base",
                language="auto",
                device="cpu",
            )

        self.assertEqual(segments, [])
        transcript = json.loads((self.output_dir / "transcript.json").read_text(encoding="utf-8"))
        self.assertEqual(transcript["segments"], [])

    def _write_timeline(self, rows: list[dict[str, Any]]) -> None:
        self.timeline_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def _write_silence(self, path: Path, duration_seconds: float) -> None:
        sample_rate = 16000
        num_samples = int(duration_seconds * sample_rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * num_samples)


if __name__ == "__main__":
    unittest.main()
