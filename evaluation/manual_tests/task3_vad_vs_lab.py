"""Task 3: Separate VAD evaluation against LAB from Diarization evaluation against RTTM."""

import json
from pathlib import Path
from Project.ssl_pipeline.rttm import parse_rttm

def main():
    root = Path(__file__).resolve().parents[2]
    eval_dir = root / "evaluation"
    lab_file = eval_dir / "LDC2021S10.lab"
    rttm_file = eval_dir / "LDC2021S10.rttm"
    vad_json_file = eval_dir / "manual_tests" / "results" / "test_C_two_speakers" / "pipeline" / "vad_regions.json"
    audio_dur = 120.0

    # 1. Parse LAB speech intervals
    lab_intervals = []
    with open(lab_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3 and p[2] == "speech":
                lab_intervals.append((float(p[0]), min(audio_dur, float(p[1]))))

    # 2. Parse RTTM turns (clipped to 120.0s)
    rttm_turns = parse_rttm(rttm_file, allow_overlap=True)
    rttm_clipped = [(t.start, min(audio_dur, t.end), t.speaker_id) for t in rttm_turns if t.start < audio_dur]

    # 3. Parse VAD output
    vad_data = json.loads(vad_json_file.read_text(encoding="utf-8"))
    vad_intervals = [(r.get("start_time", r.get("start")), min(audio_dur, r.get("end_time", r.get("end")))) for r in vad_data]

    # Timeline slicing
    boundaries = sorted({0.0, audio_dur} | 
                        {t for seg in lab_intervals for t in seg} | 
                        {t for seg in rttm_clipped for t in seg[:2]} | 
                        {t for seg in vad_intervals for t in seg})

    # A) VAD vs LAB (Speech Activity Ground Truth)
    vad_tp_lab = 0.0  # VAD speech, LAB speech
    vad_fp_lab = 0.0  # VAD speech, LAB silence (True False Positive)
    vad_fn_lab = 0.0  # VAD silence, LAB speech (True False Negative / Miss)
    vad_tn_lab = 0.0  # VAD silence, LAB silence (True Negative)

    # B) VAD vs RTTM Union
    vad_tp_rttm = 0.0
    vad_fp_rttm = 0.0
    vad_fn_rttm = 0.0
    vad_tn_rttm = 0.0

    for s, e in zip(boundaries, boundaries[1:]):
        if e <= s or s >= audio_dur:
            continue
        mid = (s + e) / 2.0
        dur = e - s

        in_vad = any(vs <= mid < ve for vs, ve in vad_intervals)
        in_lab = any(ls <= mid < le for ls, le in lab_intervals)
        in_rttm = any(rs <= mid < re for rs, re, _ in rttm_clipped)

        # vs LAB
        if in_vad and in_lab:
            vad_tp_lab += dur
        elif in_vad and not in_lab:
            vad_fp_lab += dur
        elif not in_vad and in_lab:
            vad_fn_lab += dur
        else:
            vad_tn_lab += dur

        # vs RTTM
        if in_vad and in_rttm:
            vad_tp_rttm += dur
        elif in_vad and not in_rttm:
            vad_fp_rttm += dur
        elif not in_vad and in_rttm:
            vad_fn_rttm += dur
        else:
            vad_tn_rttm += dur

    print("=== 1. VAD EVALUATION RELATIVE TO LAB (SPEECH ACTIVITY GROUND TRUTH) ===")
    print(f"LAB Reference Speech Duration: {vad_tp_lab + vad_fn_lab:.3f}s (73.9% of audio)")
    print(f"LAB Reference Silence Duration: {vad_fp_lab + vad_tn_lab:.3f}s (26.1% of audio)")
    print(f"VAD Output Duration: {vad_tp_lab + vad_fp_lab:.3f}s")
    print(f"  True Positives (Speech correctly detected): {vad_tp_lab:.3f}s")
    print(f"  False Positives (Speech detected in genuine LAB silence): {vad_fp_lab:.3f}s")
    print(f"  False Negatives (Missed speech present in LAB): {vad_fn_lab:.3f}s")
    print(f"  True Negatives (Silence correctly rejected): {vad_tn_lab:.3f}s")

    precision_lab = vad_tp_lab / (vad_tp_lab + vad_fp_lab) if (vad_tp_lab + vad_fp_lab) else 0.0
    recall_lab = vad_tp_lab / (vad_tp_lab + vad_fn_lab) if (vad_tp_lab + vad_fn_lab) else 0.0
    f1_lab = 2 * precision_lab * recall_lab / (precision_lab + recall_lab) if (precision_lab + recall_lab) else 0.0
    acc_lab = (vad_tp_lab + vad_tn_lab) / audio_dur

    print(f"  VAD Precision: {precision_lab * 100:.2f}%")
    print(f"  VAD Recall: {recall_lab * 100:.2f}%")
    print(f"  VAD F1-Score: {f1_lab * 100:.2f}%")
    print(f"  VAD Accuracy: {acc_lab * 100:.2f}%")

    print("\n=== 2. VAD EVALUATION RELATIVE TO RTTM (SPEAKER TURN ANNOTATION) ===")
    print(f"RTTM Union Reference Speech Duration: {vad_tp_rttm + vad_fn_rttm:.3f}s (45.5% of audio)")
    print(f"RTTM Union Reference Silence Duration: {vad_fp_rttm + vad_tn_rttm:.3f}s (54.5% of audio)")
    print(f"  Speech in RTTM correctly detected by VAD: {vad_tp_rttm:.3f}s")
    print(f"  Speech detected by VAD outside RTTM: {vad_fp_rttm:.3f}s")
    print(f"  Speech in RTTM missed by VAD: {vad_fn_rttm:.3f}s")
    print(f"  Silence in RTTM rejected by VAD: {vad_tn_rttm:.3f}s")

    print("\n=== 3. ANNOTATION GAP VS VAD ERROR SUMMARY ===")
    print(f"Out of the {vad_fp_rttm:.3f}s of VAD speech outside RTTM:")
    unannotated_real_speech = vad_tp_lab - vad_tp_rttm
    print(f"  - Genuine audible speech confirmed by LAB: {unannotated_real_speech:.3f}s ({unannotated_real_speech / vad_fp_rttm * 100:.1f}%)")
    print(f"  - Genuine acoustic non-speech false alarm: {vad_fp_lab:.3f}s ({vad_fp_lab / vad_fp_rttm * 100:.1f}%)")

if __name__ == "__main__":
    main()
