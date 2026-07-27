"""Integration tests for the end-to-end diarization runner."""

from __future__ import annotations

import json
import shutil
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from Project.run_diarization import run_pipeline
from Project.ssl_pipeline.rttm import parse_rttm
from Project.ssl_pipeline.segmentation import SpeechSegment
from Project.ssl_pipeline.ssl_embedding import SegmentEmbedding


SAMPLE_RATE = 16_000
TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp_tests"


class RunDiarizationPipelineTests(unittest.TestCase):
    """Verify runner integration without downloading pretrained models."""

    def setUp(self) -> None:
        self.output_dir = TEST_TMP_ROOT / "pipeline_outputs"
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        self.input_path = TEST_TMP_ROOT / "pipeline_input.wav"
        _write_wav(self.input_path)

    def test_pipeline_completes_and_writes_expected_outputs(self) -> None:
        with _patched_lightweight_stages():
            result = run_pipeline(
                self.input_path,
                output_dir=self.output_dir,
                backend="energy",
                num_speakers=2,
                oracle_num_speakers=True,
            )

        self.assertEqual(len(result.timeline), 2)
        self.assertTrue((self.output_dir / "processed.wav").exists())
        self.assertTrue((self.output_dir / "vad_regions.json").exists())
        self.assertTrue((self.output_dir / "segments.json").exists())
        self.assertTrue((self.output_dir / "embeddings" / "embeddings.npz").exists())
        self.assertTrue((self.output_dir / "normalized_embeddings.npz").exists())
        self.assertTrue((self.output_dir / "clustering.json").exists())
        self.assertTrue((self.output_dir / "timeline.json").exists())
        self.assertTrue((self.output_dir / "timeline.csv").exists())
        self.assertTrue((self.output_dir / "timeline.txt").exists())
        self.assertTrue((self.output_dir / "hypothesis.rttm").exists())
        self.assertTrue((self.output_dir / "figures" / "timeline.png").exists())
        self.assertTrue((self.output_dir / "figures" / "speaker_activity.png").exists())
        self.assertTrue((self.output_dir / "figures" / "duration_histogram.png").exists())

        turns = parse_rttm(self.output_dir / "hypothesis.rttm")
        self.assertEqual([turn.speaker_id for turn in turns], ["speaker_0", "speaker_1"])

    def test_pipeline_evaluates_when_reference_rttm_exists(self) -> None:
        reference_path = TEST_TMP_ROOT / "pipeline_reference.rttm"
        reference_path.write_text(
            "\n".join(
                [
                    "SPEAKER pipeline_input 1 0.00 1.00 <NA> <NA> speaker_0 <NA> <NA>",
                    "SPEAKER pipeline_input 1 1.00 1.00 <NA> <NA> speaker_1 <NA> <NA>",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with _patched_lightweight_stages():
            result = run_pipeline(
                self.input_path,
                output_dir=self.output_dir,
                reference_rttm=reference_path,
                backend="energy",
                num_speakers=2,
                oracle_num_speakers=True,
            )

        self.assertIsNotNone(result.evaluation_report)
        self.assertTrue((self.output_dir / "evaluation.json").exists())
        evaluation = json.loads((self.output_dir / "evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(evaluation["der"], 0.0)
        self.assertEqual(evaluation["jer"], 0.0)


def _patched_lightweight_stages():
    """Patch heavyweight boundaries while keeping integration modules active."""

    return mock.patch.multiple(
        "Project.run_diarization",
        preprocess_audio=mock.Mock(side_effect=_fake_preprocess_audio),
        detect_speech_regions=mock.Mock(side_effect=_fake_detect_speech_regions),
        segment_speech=mock.Mock(side_effect=_fake_segment_speech),
        _extract_embeddings_with_config=mock.Mock(side_effect=_fake_extract_embeddings),
    )


def _fake_preprocess_audio(input_path: Path, output_path: Path) -> None:
    """Create the preprocessed audio artifact expected by the runner."""

    _write_wav(output_path)


def _fake_detect_speech_regions(
    audio_path: Path,
    *,
    backend: str | None = None,
    export_json_path: Path | None = None,
) -> list[tuple[float, float]]:
    """Return deterministic VAD regions and write the JSON artifact."""

    regions = [(0.0, 2.0)]
    if export_json_path is not None:
        export_json_path.write_text(
            json.dumps(
                [
                    {"start_time": start, "end_time": end}
                    for start, end in regions
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    return regions


def _fake_segment_speech(
    speech_regions: list[tuple[float, float]],
) -> list[SpeechSegment]:
    """Return two deterministic speech segments."""

    return [
        SpeechSegment(segment_id=0, start=0.0, end=1.0),
        SpeechSegment(segment_id=1, start=1.0, end=2.0),
    ]


def _fake_extract_embeddings(
    audio_path: Path,
    segments: list[SpeechSegment],
    config: object,
) -> list[SegmentEmbedding]:
    """Return separable synthetic embeddings."""

    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    return [
        SegmentEmbedding(
            segment_id=segment.segment_id,
            start=segment.start,
            end=segment.end,
            vector=vector,
            dimension=2,
            model_name="fake-wavlm",
        )
        for segment, vector in zip(segments, vectors)
    ]


def _write_wav(path: Path) -> None:
    """Write a small valid mono WAV file."""

    samples = np.zeros(SAMPLE_RATE // 10, dtype="<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())


if __name__ == "__main__":
    unittest.main()
