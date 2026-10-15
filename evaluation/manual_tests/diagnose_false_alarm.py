"""Diagnostic script to analyze False Alarm and error breakdown for Test C (LDC2021S10)."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Project.ssl_pipeline.rttm import parse_rttm, RTTMSpeakerTurn

def main():
    proj_root = Path(__file__).resolve().parents[2]
    eval_dir = proj_root / "evaluation"
    res_dir = eval_dir / "manual_tests" / "results" / "test_C_two_speakers"
    pipe_dir = res_dir / "pipeline"

    ref_rttm_path = eval_dir / "LDC2021S10.rttm"
    hyp_rttm_path = res_dir / "hypothesis.rttm"
    vad_json_path = pipe_dir / "vad_regions.json"
    segments_json_path = pipe_dir / "segments.json"
    timeline_json_path = pipe_dir / "timeline.json"

    ref_turns = parse_rttm(ref_rttm_path, allow_overlap=True)
    hyp_turns = parse_rttm(hyp_rttm_path, allow_overlap=True)

    # 1. Pipeline stage durations
    vad_data = json.loads(vad_json_path.read_text(encoding="utf-8")) if vad_json_path.exists() else []
    vad_regions = [{"start": r.get("start_time", r.get("start", 0.0)), "end": r.get("end_time", r.get("end", 0.0))} for r in vad_data]
    vad_duration = sum(r["end"] - r["start"] for r in vad_regions)

    segments_data = json.loads(segments_json_path.read_text(encoding="utf-8")) if segments_json_path.exists() else []
    # Segments may overlap due to window/hop; union duration:
    seg_intervals = sorted([(s["start"], s["end"]) for s in segments_data])
    def union_dur(intervals):
        if not intervals:
            return 0.0
        cur_s, cur_e = intervals[0]
        tot = 0.0
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                tot += cur_e - cur_s
                cur_s, cur_e = s, e
        tot += cur_e - cur_s
        return tot
    seg_union_dur = union_dur(seg_intervals)

    hyp_duration = sum(t.duration for t in hyp_turns)
    ref_duration = sum(t.duration for t in ref_turns)

    print("=== Pipeline Stage Durations ===")
    print(f"Reference Total Speech (sum of turns): {ref_duration:.3f}s")
    ref_union = union_dur(sorted([(t.start, t.end) for t in ref_turns]))
    print(f"Reference Union Speech (time with >=1 speaker): {ref_union:.3f}s")
    print(f"VAD Output Duration: {vad_duration:.3f}s (across {len(vad_data)} regions)")
    print(f"Segments Union Duration: {seg_union_dur:.3f}s (across {len(segments_data)} segments)")
    print(f"Hypothesis Total Duration: {hyp_duration:.3f}s (across {len(hyp_turns)} turns)")

    # 2. Optimal mapping and atomic interval breakdown
    # Mapping from evaluation: speaker_0 -> speaker117, speaker_1 -> speaker118
    # Let's verify mapping via overlap
    ref_speakers = sorted({t.speaker_id for t in ref_turns})
    hyp_speakers = sorted({t.speaker_id for t in hyp_turns})
    
    # Clip reference to 120.0s audio duration if any turn exceeds 120.0s
    audio_dur = 120.0
    clipped_ref = []
    for t in ref_turns:
        if t.start >= audio_dur:
            continue
        clipped_ref.append(RTTMSpeakerTurn(
            recording_id=t.recording_id,
            channel_id=t.channel_id,
            start=t.start,
            duration=min(t.duration, audio_dur - t.start),
            speaker_id=t.speaker_id,
        ))

    boundaries = sorted({t for turn in [*clipped_ref, *hyp_turns] for t in (turn.start, turn.end)} | {0.0, audio_dur})
    
    fa_intervals = []
    miss_intervals = []
    conf_intervals = []
    correct_intervals = []
    true_silence_intervals = []

    # Map hyp to ref based on optimal overlap
    mapping = {"speaker_0": "speaker117", "speaker_1": "speaker118"}

    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        mid = (start + end) / 2.0
        dur = end - start
        r_spks = {t.speaker_id for t in clipped_ref if t.start <= mid < t.end}
        h_spks = {t.speaker_id for t in hyp_turns if t.start <= mid < t.end}

        n_ref = len(r_spks)
        n_hyp = len(h_spks)

        mapped_h = {mapping.get(s, s) for s in h_spks}
        n_corr = len(r_spks & mapped_h)

        fa = max(0, n_hyp - n_ref) * dur
        miss = max(0, n_ref - n_hyp) * dur
        conf = (min(n_ref, n_hyp) - n_corr) * dur
        corr = n_corr * dur

        if fa > 0:
            fa_intervals.append((start, end, dur, r_spks, h_spks))
        if miss > 0:
            miss_intervals.append((start, end, dur, r_spks, h_spks))
        if conf > 0:
            conf_intervals.append((start, end, dur, r_spks, h_spks))
        if corr > 0:
            correct_intervals.append((start, end, dur, r_spks, h_spks))
        if n_ref == 0 and n_hyp == 0:
            true_silence_intervals.append((start, end, dur))

    tot_fa = sum(i[2] for i in fa_intervals)
    tot_miss = sum(i[2] for i in miss_intervals)
    tot_conf = sum(i[2] for i in conf_intervals)
    tot_corr = sum(i[2] for i in correct_intervals)
    tot_true_sil = sum(i[2] for i in true_silence_intervals)

    print(f"\n=== Metric Verification ===")
    print(f"Total Correct: {tot_corr:.3f}s")
    print(f"Total Miss: {tot_miss:.3f}s")
    print(f"Total False Alarm: {tot_fa:.3f}s")
    print(f"Total Speaker Confusion: {tot_conf:.3f}s")
    print(f"Total True Silence: {tot_true_sil:.3f}s")
    print(f"Total Scored Ref Duration: {tot_corr + tot_miss + tot_conf:.3f}s")

    print(f"\n=== False Alarm Interval Breakdown (Total = {tot_fa:.3f}s) ===")
    # Classify False Alarm into:
    # A) In Reference Silence (n_ref == 0, n_hyp >= 1)
    # B) Over-detecting during active speech (n_ref >= 1, n_hyp > n_ref)
    fa_in_silence = [i for i in fa_intervals if len(i[3]) == 0]
    fa_in_speech = [i for i in fa_intervals if len(i[3]) > 0]
    print(f"FA in Reference Silence (ref = 0, hyp >= 1): {sum(i[2] for i in fa_in_silence):.3f}s across {len(fa_in_silence)} intervals")
    print(f"FA during Reference Speech (ref >= 1, hyp > ref): {sum(i[2] for i in fa_in_speech):.3f}s across {len(fa_in_speech)} intervals")

    print("\nTop 15 Longest False Alarm Intervals:")
    fa_sorted = sorted(fa_intervals, key=lambda x: x[2], reverse=True)
    for s, e, d, r, h in fa_sorted[:15]:
        print(f"  [{s:6.3f}s - {e:6.3f}s] ({d:5.3f}s) | Ref: {list(r)} | Hyp: {list(h)}")

    # 3. Check VAD correlation with False Alarm
    # For each FA interval, check if VAD was active
    fa_during_vad = 0.0
    for s, e, d, r, h in fa_intervals:
        mid = (s + e) / 2.0
        vad_active = any(vr["start"] <= mid < vr["end"] for vr in vad_regions)
        if vad_active:
            fa_during_vad += d
    print(f"\nFalse alarm occurring directly inside VAD speech regions: {fa_during_vad:.3f}s / {tot_fa:.3f}s ({fa_during_vad/tot_fa*100:.1f}%)")

    # 4. Generate comparison plot
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    
    # Plot Ref Speaker 117
    for t in clipped_ref:
        if t.speaker_id == "speaker117":
            axes[0].barh(0, t.duration, left=t.start, height=0.6, color="tab:blue", align="center")
    axes[0].set_ylabel("Ref 117")
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_yticks([])

    # Plot Ref Speaker 118
    for t in clipped_ref:
        if t.speaker_id == "speaker118":
            axes[1].barh(0, t.duration, left=t.start, height=0.6, color="tab:cyan", align="center")
    axes[1].set_ylabel("Ref 118")
    axes[1].set_ylim(-0.5, 0.5)
    axes[1].set_yticks([])

    # Plot VAD regions
    for r in vad_regions:
        axes[2].barh(0, r["end"] - r["start"], left=r["start"], height=0.6, color="tab:green", align="center")
    axes[2].set_ylabel("VAD Regions")
    axes[2].set_ylim(-0.5, 0.5)
    axes[2].set_yticks([])

    # Plot Hyp (Mapped: spk0 -> 117, spk1 -> 118)
    for t in hyp_turns:
        c = "tab:blue" if t.speaker_id == "speaker_0" else "tab:cyan"
        y = 0.2 if t.speaker_id == "speaker_0" else -0.2
        axes[3].barh(y, t.duration, left=t.start, height=0.35, color=c, align="center", label=t.speaker_id)
    axes[3].set_ylabel("Hypothesis")
    axes[3].set_ylim(-0.5, 0.5)
    axes[3].set_yticks([0.2, -0.2])
    axes[3].set_yticklabels(["spk_0 (117)", "spk_1 (118)"])

    # Plot False Alarm and Missed error regions
    for s, e, d, r, h in fa_intervals:
        axes[4].barh(0.2, d, left=s, height=0.35, color="tab:red", align="center")
    for s, e, d, r, h in miss_intervals:
        axes[4].barh(-0.2, d, left=s, height=0.35, color="tab:orange", align="center")
    axes[4].set_ylabel("Errors")
    axes[4].set_ylim(-0.5, 0.5)
    axes[4].set_yticks([0.2, -0.2])
    axes[4].set_yticklabels(["False Alarm", "Missed"])
    axes[4].set_xlabel("Time (seconds)")

    fig.suptitle("Test C Diagnostic: Ground Truth vs VAD vs Diarization Hypothesis (LDC2021S10)", fontsize=13)
    fig.tight_layout()
    plot_path = res_dir / "diagnostic_timeline.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved diagnostic timeline plot to {plot_path}")

if __name__ == "__main__":
    main()
