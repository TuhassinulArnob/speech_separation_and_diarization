"""Test B: Voice Activity Detection evaluation on Silence and Noise.

Measures:
1. Speech detection duration
2. False speech detection in known silence (leading/trailing 3.0s)
3. Speech detection under 15 dB SNR background noise vs clean baseline
4. Saves full analysis to evaluation/manual_tests/results/test_B_noise/
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import soundfile as sf

from Project.ssl_pipeline.vad import detect_speech_regions
from Project.ssl_pipeline.preprocessing import preprocess_audio


SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_AUDIO = SCRIPT_DIR / "test_A_clean" / "clean_speech.flac"
SILENCE_AUDIO = SCRIPT_DIR / "test_B_noise" / "speech_silence.wav"
NOISE_AUDIO = SCRIPT_DIR / "test_B_noise" / "speech_noise.wav"
RESULTS_DIR = SCRIPT_DIR / "results" / "test_B_noise"


def run_test_b() -> dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = RESULTS_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess clean audio to 16kHz mono WAV for VAD
    clean_wav = temp_dir / "clean_processed.wav"
    preprocess_audio(CLEAN_AUDIO, clean_wav)

    # 1. Clean Speech Baseline VAD
    clean_regions = detect_speech_regions(clean_wav, backend="energy")
    clean_speech_dur = sum(end - start for start, end in clean_regions)
    clean_total_dur = sf.info(str(clean_wav)).duration

    # 2. Speech + Silence VAD
    silence_wav = temp_dir / "silence_processed.wav"
    preprocess_audio(SILENCE_AUDIO, silence_wav)
    silence_regions = detect_speech_regions(silence_wav, backend="energy")
    silence_speech_dur = sum(end - start for start, end in silence_regions)
    silence_total_dur = sf.info(str(silence_wav)).duration

    # Measure false speech in known silence: [0.0, 3.0] and [18.560, 21.560]
    lead_silence_end = 3.0
    trail_silence_start = 3.0 + clean_total_dur

    false_lead_speech = 0.0
    false_trail_speech = 0.0
    for start, end in silence_regions:
        # Check overlap with [0, lead_silence_end]
        overlap_lead = max(0.0, min(end, lead_silence_end) - start)
        false_lead_speech += overlap_lead
        # Check overlap with [trail_silence_start, total]
        overlap_trail = max(0.0, end - max(start, trail_silence_start))
        false_trail_speech += overlap_trail

    total_false_in_silence = false_lead_speech + false_trail_speech

    # 3. Speech + Noise VAD
    noise_wav = temp_dir / "noise_processed.wav"
    preprocess_audio(NOISE_AUDIO, noise_wav)
    noise_regions = detect_speech_regions(noise_wav, backend="energy")
    noise_speech_dur = sum(end - start for start, end in noise_regions)
    noise_total_dur = sf.info(str(noise_wav)).duration

    # Compare speech detected in noise vs clean
    # Map clean regions shifted by +3.0s to compare with silence speech
    shifted_clean_regions = [(s + 3.0, e + 3.0) for s, e in clean_regions]

    results = {
        "test": "Test B: Silence and Background Noise VAD",
        "vad_backend": "energy (adaptive threshold fallback)",
        "clean_baseline": {
            "file": CLEAN_AUDIO.name,
            "duration_seconds": round(clean_total_dur, 4),
            "detected_speech_duration_seconds": round(clean_speech_dur, 4),
            "speech_ratio": round(clean_speech_dur / clean_total_dur, 4),
            "segment_count": len(clean_regions),
            "segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in clean_regions],
        },
        "speech_silence": {
            "file": SILENCE_AUDIO.name,
            "total_duration_seconds": round(silence_total_dur, 4),
            "detected_speech_duration_seconds": round(silence_speech_dur, 4),
            "speech_ratio": round(silence_speech_dur / silence_total_dur, 4),
            "known_silence_duration_seconds": round(lead_silence_end + (silence_total_dur - trail_silence_start), 4),
            "false_speech_in_known_silence_seconds": round(total_false_in_silence, 4),
            "false_speech_in_leading_silence_seconds": round(false_lead_speech, 4),
            "false_speech_in_trailing_silence_seconds": round(false_trail_speech, 4),
            "segment_count": len(silence_regions),
            "segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in silence_regions],
        },
        "speech_noise": {
            "file": NOISE_AUDIO.name,
            "snr_db": 15.0,
            "total_duration_seconds": round(noise_total_dur, 4),
            "detected_speech_duration_seconds": round(noise_speech_dur, 4),
            "speech_ratio": round(noise_speech_dur / noise_total_dur, 4),
            "delta_from_clean_seconds": round(noise_speech_dur - clean_speech_dur, 4),
            "segment_count": len(noise_regions),
            "segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in noise_regions],
        },
        "analysis_notes": {
            "silence_discrimination": "VAD correctly distinguished the known 3.0s leading and 3.0s trailing silence without false triggers in digital silence." if total_false_in_silence == 0.0 else f"VAD triggered {total_false_in_silence:.2f}s of speech in silence.",
            "noise_robustness": f"Under 15 dB Gaussian noise, speech duration changed by {noise_speech_dur - clean_speech_dur:+.2f}s relative to clean baseline.",
            "ground_truth_limitation": "LibriSpeech dev-clean-2 provides utterance-level transcripts rather than frame-level word/phone boundaries. True missed speech and false alarms can only be precisely measured on known silence regions; within the active container, clean VAD detection serves as a reference baseline.",
        },
    }

    out_file = RESULTS_DIR / "vad_evaluation.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_test_b()
