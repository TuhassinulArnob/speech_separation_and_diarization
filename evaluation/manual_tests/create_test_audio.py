"""Reproducible generation script for Test B audio files.

Creates:
1. evaluation/manual_tests/test_B_noise/speech_silence.wav:
   Clean speech from Test A with exactly 3.0s leading silence and 3.0s trailing silence.
2. evaluation/manual_tests/test_B_noise/speech_noise.wav:
   Clean speech from Test A with controlled stationary Gaussian background noise at target SNR=15 dB (seed=42).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import soundfile as sf

SEED = 42
LEADING_SILENCE_SEC = 3.0
TRAILING_SILENCE_SEC = 3.0
TARGET_SNR_DB = 15.0

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_AUDIO_PATH = SCRIPT_DIR / "test_A_clean" / "clean_speech.flac"
OUTPUT_DIR = SCRIPT_DIR / "test_B_noise"


def generate_test_b_audio() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEAN_AUDIO_PATH.exists():
        raise FileNotFoundError(f"Source clean speech audio not found: {CLEAN_AUDIO_PATH}")

    clean_data, sr = sf.read(str(CLEAN_AUDIO_PATH), dtype="float32")
    if clean_data.ndim > 1:
        clean_data = clean_data.mean(axis=1)

    clean_duration = len(clean_data) / sr
    clean_peak = float(np.max(np.abs(clean_data)))
    clean_rms = float(np.sqrt(np.mean(clean_data ** 2)))
    clean_power = float(np.mean(clean_data ** 2))

    # --- 1. Speech + Silence ---
    lead_samples = int(LEADING_SILENCE_SEC * sr)
    trail_samples = int(TRAILING_SILENCE_SEC * sr)
    silence_lead = np.zeros(lead_samples, dtype=np.float32)
    silence_trail = np.zeros(trail_samples, dtype=np.float32)

    speech_silence_data = np.concatenate([silence_lead, clean_data, silence_trail])
    silence_duration = len(speech_silence_data) / sr
    silence_peak = float(np.max(np.abs(speech_silence_data)))
    silence_rms = float(np.sqrt(np.mean(speech_silence_data ** 2)))

    silence_out_path = OUTPUT_DIR / "speech_silence.wav"
    sf.write(str(silence_out_path), speech_silence_data, sr, subtype="PCM_16")

    # --- 2. Speech + Controlled Noise ---
    rng = np.random.RandomState(SEED)
    noise_power_target = clean_power / (10.0 ** (TARGET_SNR_DB / 10.0))
    noise_sigma = np.sqrt(noise_power_target)
    noise = rng.normal(0.0, noise_sigma, size=clean_data.shape).astype(np.float32)

    speech_noise_data = clean_data + noise
    noise_peak = float(np.max(np.abs(speech_noise_data)))
    noise_rms = float(np.sqrt(np.mean(speech_noise_data ** 2)))
    measured_noise_power = float(np.mean(noise ** 2))
    measured_snr_db = float(10.0 * np.log10(clean_power / measured_noise_power))

    # Verify no clipping occurred
    if noise_peak >= 1.0:
        # Scale only if clipping, but clean_peak=0.522 + noise(~0.03) will not clip
        scale = 0.99 / noise_peak
        speech_noise_data = speech_noise_data * scale
        noise_peak = float(np.max(np.abs(speech_noise_data)))

    noise_out_path = OUTPUT_DIR / "speech_noise.wav"
    sf.write(str(noise_out_path), speech_noise_data, sr, subtype="PCM_16")

    metadata = {
        "source_file": CLEAN_AUDIO_PATH.name,
        "sample_rate": sr,
        "clean_duration_seconds": round(clean_duration, 4),
        "clean_peak": round(clean_peak, 6),
        "clean_rms": round(clean_rms, 6),
        "speech_silence": {
            "output_file": silence_out_path.name,
            "leading_silence_seconds": LEADING_SILENCE_SEC,
            "trailing_silence_seconds": TRAILING_SILENCE_SEC,
            "speech_start_seconds": LEADING_SILENCE_SEC,
            "speech_end_seconds": round(LEADING_SILENCE_SEC + clean_duration, 4),
            "total_duration_seconds": round(silence_duration, 4),
            "peak_amplitude": round(silence_peak, 6),
            "rms_level": round(silence_rms, 6),
            "clipped": bool(silence_peak >= 1.0),
        },
        "speech_noise": {
            "output_file": noise_out_path.name,
            "noise_type": "additive_gaussian_white_noise",
            "random_seed": SEED,
            "target_snr_db": TARGET_SNR_DB,
            "measured_snr_db": round(measured_snr_db, 2),
            "total_duration_seconds": round(clean_duration, 4),
            "peak_amplitude": round(noise_peak, 6),
            "rms_level": round(noise_rms, 6),
            "clipped": bool(noise_peak >= 1.0),
        },
    }

    meta_path = OUTPUT_DIR / "generation_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated {silence_out_path} ({silence_duration:.3f}s)")
    print(f"Generated {noise_out_path} ({clean_duration:.3f}s, SNR={measured_snr_db:.2f}dB)")
    print(f"Saved metadata to {meta_path}")
    return metadata


if __name__ == "__main__":
    generate_test_b_audio()
