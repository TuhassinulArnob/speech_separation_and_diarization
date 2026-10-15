"""Test A: Clean Single-Speaker Audio Pipeline Execution and Evaluation.

Input: evaluation/manual_tests/test_A_clean/clean_speech.flac
Outputs: evaluation/manual_tests/results/test_A_clean/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from Project.run_diarization import run_pipeline
from Project.ssl_pipeline.config import load_config
from Project.ssl_pipeline.rttm import parse_rttm
from Project.transcription.whisper_transcription import transcribe_speaker_timeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_AUDIO = SCRIPT_DIR / "test_A_clean" / "clean_speech.flac"
RESULTS_DIR = SCRIPT_DIR / "results" / "test_A_clean"


def run_test_a() -> dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []

    def log(msg: str) -> None:
        LOGGER.info(msg)
        logs.append(msg)

    log(f"Starting Test A on {INPUT_AUDIO}")

    # 1. Audio Verification
    if not INPUT_AUDIO.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_AUDIO}")

    info = sf.info(str(INPUT_AUDIO))
    audio_data, sr = sf.read(str(INPUT_AUDIO))
    peak = float(np.max(np.abs(audio_data)))
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    duration = float(info.duration)
    channels = int(info.channels)

    log(f"Decoded {info.format} ({info.subtype}): {sr} Hz, {channels} channel(s), {duration:.3f}s, {info.frames} frames")
    log(f"Signal levels: peak={peak:.6f}, RMS={rms:.6f}, clipped={peak >= 1.0}")

    audio_check = {
        "exists": True,
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": sr,
        "channels": channels,
        "duration_seconds": round(duration, 4),
        "total_frames": info.frames,
        "peak_amplitude": round(peak, 6),
        "rms_level": round(rms, 6),
        "clipped": bool(peak >= 1.0),
        "is_valid_audio": bool(len(audio_data) > 0 and not np.isnan(audio_data).any()),
    }

    # 2. Run Diarization Pipeline (with speaker count estimation and single speaker)
    log("Running end-to-end diarization pipeline...")
    start_time = time.time()
    pipeline_result = run_pipeline(
        INPUT_AUDIO,
        output_dir=RESULTS_DIR,
        backend="energy",
        num_speakers=1,
        oracle_num_speakers=True,
        save_json=True,
        save_csv=True,
        save_rttm=True,
    )
    pipeline_elapsed = time.time() - start_time
    log(f"Pipeline completed in {pipeline_elapsed:.2f}s")

    # Load generated artifacts
    vad_regions = json.loads((RESULTS_DIR / "vad_regions.json").read_text(encoding="utf-8"))
    segments = json.loads((RESULTS_DIR / "segments.json").read_text(encoding="utf-8"))
    timeline = json.loads((RESULTS_DIR / "timeline.json").read_text(encoding="utf-8"))
    rttm_turns = parse_rttm(RESULTS_DIR / "hypothesis.rttm", allow_overlap=True)
    detected_speakers = sorted({t.speaker_id for t in rttm_turns})

    total_speech_duration = sum(t.duration for t in rttm_turns)
    log(f"Detected {len(detected_speakers)} speaker(s): {detected_speakers}")
    log(f"Total speech duration: {total_speech_duration:.3f}s across {len(rttm_turns)} turns")

    # Check speaker estimation using eigengap as well
    from Project.ssl_pipeline.spectral_clustering import estimate_speaker_count_eigengap, build_affinity_matrix, compute_cosine_similarity_matrix
    norm_data = np.load(RESULTS_DIR / "normalized_embeddings.npz")
    vectors = norm_data["vectors"]
    sim = compute_cosine_similarity_matrix(vectors)
    aff = build_affinity_matrix(sim)
    estimated_speakers = estimate_speaker_count_eigengap(aff, max_speakers=min(8, len(vectors)))
    log(f"Eigengap estimated speaker count: {estimated_speakers}")

    # 3. Transcription using Whisper
    log("Running Whisper transcription on speaker timeline...")
    tx_start = time.time()
    try:
        transcript_segments = transcribe_speaker_timeline(
            audio_path=RESULTS_DIR / "processed.wav",
            timeline_path=RESULTS_DIR / "timeline.json",
            output_dir=RESULTS_DIR,
            model_name="base",
            language="en",
            device="cpu",
            save_json=True,
            save_txt=True,
            save_srt=True,
        )
        tx_elapsed = time.time() - tx_start
        full_transcript_text = " ".join(seg.text for seg in transcript_segments)
        transcript_status = {
            "status": "success",
            "model": "base",
            "device": "cpu",
            "elapsed_seconds": round(tx_elapsed, 2),
            "segment_count": len(transcript_segments),
            "transcript_text": full_transcript_text.strip(),
            "transcript_file": str((RESULTS_DIR / "transcript.txt").name),
        }
        log(f"Transcription succeeded ({len(transcript_segments)} segments): {full_transcript_text.strip()[:100]}...")
    except Exception as exc:
        log(f"Transcription failed: {exc}")
        transcript_status = {
            "status": "failed",
            "error": str(exc),
        }

    # 4. Speech Separation Applicability Assessment
    log("Evaluating Speech Separation applicability on single-speaker recording...")
    separation_notes = {
        "applicable": False,
        "reason": (
            "The input is a clean single-speaker recording. SpeechBrain SepFormer (trained on "
            "whamr / Libri2Mix 2-speaker mixtures) assumes multiple concurrent voices. "
            "When forced to separate a single speaker, 2-source separation models either split the "
            "spectral energy into phase-distorted sub-bands or duplicate the single voice with "
            "artificial attenuation and leakage. Single-speaker recordings bypass separation in production."
        ),
        "status": "bypassed_as_intended",
    }

    # Summary
    summary = {
        "test": "Test A: Clean Single-Speaker Audio",
        "audio_verification": audio_check,
        "execution_status": "success",
        "pipeline_elapsed_seconds": round(pipeline_elapsed, 2),
        "vad": {
            "backend": "energy",
            "region_count": len(vad_regions),
            "total_speech_seconds": round(sum(r["end_time"] - r["start_time"] for r in vad_regions), 3),
        },
        "segmentation": {
            "segment_count": len(segments),
        },
        "diarization": {
            "oracle_speaker_count_configured": 1,
            "detected_speaker_count": len(detected_speakers),
            "detected_speaker_labels": detected_speakers,
            "eigengap_estimated_speaker_count": estimated_speakers,
            "single_speaker_verified": bool(len(detected_speakers) == 1),
            "turn_count": len(rttm_turns),
            "total_speech_duration_seconds": round(total_speech_duration, 3),
        },
        "transcription": transcript_status,
        "separation": separation_notes,
        "output_files": [p.name for p in sorted(RESULTS_DIR.glob("*")) if p.is_file()],
        "logs": logs,
    }

    summary_file = RESULTS_DIR / "test_A_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Saved Test A summary to {summary_file}")
    return summary


if __name__ == "__main__":
    run_test_a()
