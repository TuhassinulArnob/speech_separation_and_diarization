"""Analyze RMS energy distribution of LDC2021S10 speech vs silence."""

import wave
import numpy as np
from Project.ssl_pipeline.rttm import parse_rttm

def main():
    wav_path = "evaluation/manual_tests/results/test_C_two_speakers/pipeline/processed.wav"
    rttm_path = "evaluation/LDC2021S10.rttm"

    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n_frames_raw = w.getnframes()
        audio = np.frombuffer(w.readframes(n_frames_raw), dtype=np.int16).astype(np.float32) / 32768.0

    ref_turns = parse_rttm(rttm_path, allow_overlap=True)
    frame_size = int(sr * 0.03)  # 30ms
    n_frames = len(audio) // frame_size
    frames = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    is_ref_speech = np.zeros(n_frames, dtype=bool)
    for t in ref_turns:
        s_idx = max(0, int(t.start / 0.03))
        e_idx = min(n_frames, int(t.end / 0.03))
        is_ref_speech[s_idx:e_idx] = True

    speech_rms = rms[is_ref_speech]
    silence_rms = rms[~is_ref_speech]

    print(f"Audio duration: {len(audio)/sr:.2f}s, Total 30ms frames: {n_frames}")
    print("\nOverall RMS Percentiles:")
    for p in [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95]:
        print(f"  p{p:02d}: {np.percentile(rms, p):.5f}")

    print("\nReference Speech RMS (active speech frames):")
    print(f"  Count: {len(speech_rms)}, Min: {np.min(speech_rms):.5f}, Max: {np.max(speech_rms):.5f}")
    print(f"  Mean: {np.mean(speech_rms):.5f}, Median: {np.median(speech_rms):.5f}")
    print(f"  p05: {np.percentile(speech_rms, 5):.5f}")
    print(f"  p10: {np.percentile(speech_rms, 10):.5f}")
    print(f"  p25: {np.percentile(speech_rms, 25):.5f}")
    print(f"  p50: {np.percentile(speech_rms, 50):.5f}")

    print("\nReference Silence RMS (non-speech frames):")
    print(f"  Count: {len(silence_rms)}, Min: {np.min(silence_rms):.5f}, Max: {np.max(silence_rms):.5f}")
    print(f"  Mean: {np.mean(silence_rms):.5f}, Median: {np.median(silence_rms):.5f}")
    print(f"  p50: {np.percentile(silence_rms, 50):.5f}")
    print(f"  p75: {np.percentile(silence_rms, 75):.5f}")
    print(f"  p90: {np.percentile(silence_rms, 90):.5f}")
    print(f"  p95: {np.percentile(silence_rms, 95):.5f}")
    print(f"  p99: {np.percentile(silence_rms, 99):.5f}")

    p10 = float(np.percentile(rms, 10))
    p95 = float(np.percentile(rms, 95))
    cur_thresh = p10 + 0.15 * (p95 - p10)
    print(f"\nCurrent formula threshold: p10={p10:.5f}, p95={p95:.5f} -> thresh={cur_thresh:.5f}")

    # Test what percentage of speech and silence frames are classified as speech for different thresholds
    print("\nThreshold Sweep Analysis:")
    print(f"{'Threshold':>10} | {'Speech Detected %':>17} | {'Silence False Alarm %':>21}")
    print("-" * 55)
    for factor in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        t_val = p10 + factor * (p95 - p10)
        sp_det = 100.0 * np.mean(speech_rms >= t_val)
        sil_fa = 100.0 * np.mean(silence_rms >= t_val)
        print(f"{t_val:10.5f} (factor {factor:.2f}) | {sp_det:16.1f}% | {sil_fa:20.1f}%")

if __name__ == "__main__":
    main()
