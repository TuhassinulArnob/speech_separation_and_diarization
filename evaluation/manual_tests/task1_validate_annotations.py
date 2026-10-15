"""Task 1: Detailed timeline comparison between LDC2021S10.lab and LDC2021S10.rttm."""

from pathlib import Path
from Project.ssl_pipeline.rttm import parse_rttm

def main():
    root = Path(__file__).resolve().parents[2]
    eval_dir = root / "evaluation"
    lab_file = eval_dir / "LDC2021S10.lab"
    rttm_file = eval_dir / "LDC2021S10.rttm"
    audio_dur = 120.0

    # 1. Parse LAB
    lab_intervals = []
    with open(lab_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3 and p[2] == "speech":
                s = float(p[0])
                e = float(p[1])
                lab_intervals.append((s, min(audio_dur, e)))

    # 2. Parse RTTM
    rttm_turns = parse_rttm(rttm_file, allow_overlap=True)
    rttm_clipped = []
    for t in rttm_turns:
        if t.start < audio_dur:
            rttm_clipped.append((t.start, min(audio_dur, t.end), t.speaker_id))

    # 3. Compute metrics:
    # (a) Total speaker time: sum of all turn durations (each speaker counted separately during overlap)
    total_ref_speaker_time_raw = sum(t.duration for t in rttm_turns)
    total_ref_speaker_time_clipped = sum(e - s for s, e, spk in rttm_clipped)

    # (b) Union speech duration: duration of time where >=1 speaker is speaking
    def compute_union(intervals):
        if not intervals:
            return 0.0
        sorted_int = sorted(intervals, key=lambda x: x[0])
        tot = 0.0
        cur_s, cur_e = sorted_int[0][:2]
        for item in sorted_int[1:]:
            s, e = item[:2]
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                tot += (cur_e - cur_s)
                cur_s, cur_e = s, e
        tot += (cur_e - cur_s)
        return tot

    rttm_union_dur = compute_union([(s, e) for s, e, _ in rttm_clipped])
    lab_total_speech_dur = compute_union(lab_intervals)

    print("=== DURATION DEFINITIONS & DENOMINATOR ===")
    print(f"1. Total Reference Speaker Time (unclipped): {total_ref_speaker_time_raw:.3f}s")
    print(f"2. Total Reference Speaker Time (clipped to 120.0s): {total_ref_speaker_time_clipped:.3f}s")
    print(f"3. Reference Union Speech Duration (time >= 1 speaker): {rttm_union_dur:.3f}s")
    print(f"4. Scorer Denominator in NIST md-eval (collar=0.0s, skip_overlap=False): {total_ref_speaker_time_clipped:.3f}s")
    print(f"5. Total LAB Speech Activity Duration (clipped to 120.0s): {lab_total_speech_dur:.3f}s")
    print(f"   Delta (LAB speech - RTTM speaker time): {lab_total_speech_dur - total_ref_speaker_time_clipped:+.3f}s")
    print(f"   Delta (LAB speech - RTTM union time): {lab_total_speech_dur - rttm_union_dur:+.3f}s")

    # 4. Atomic interval comparison
    boundaries = sorted({0.0, audio_dur} | {t for seg in lab_intervals for t in seg} | {t for seg in rttm_clipped for t in seg[:2]})
    
    lab_only_intervals = []
    rttm_only_intervals = []
    both_intervals = []
    neither_intervals = []

    for s, e in zip(boundaries, boundaries[1:]):
        if e <= s or s >= audio_dur:
            continue
        mid = (s + e) / 2.0
        dur = e - s
        in_lab = any(ls <= mid < le for ls, le in lab_intervals)
        r_spks = [spk for rs, re, spk in rttm_clipped if rs <= mid < re]

        if in_lab and not r_spks:
            lab_only_intervals.append((s, e, dur))
        elif not in_lab and r_spks:
            rttm_only_intervals.append((s, e, dur, r_spks))
        elif in_lab and r_spks:
            both_intervals.append((s, e, dur, r_spks))
        else:
            neither_intervals.append((s, e, dur))

    tot_lab_only = sum(i[2] for i in lab_only_intervals)
    tot_rttm_only = sum(i[2] for i in rttm_only_intervals)
    tot_both = sum(i[2] for i in both_intervals)
    tot_neither = sum(i[2] for i in neither_intervals)

    print("\n=== TIMELINE INTERSECTION BREAKDOWN ===")
    print(f"Both LAB & RTTM mark Speech: {tot_both:.3f}s")
    print(f"LAB marks Speech, RTTM has NO annotation (Missing in RTTM): {tot_lab_only:.3f}s ({len(lab_only_intervals)} intervals)")
    print(f"RTTM marks Speech, LAB has Silence: {tot_rttm_only:.3f}s ({len(rttm_only_intervals)} intervals)")
    print(f"Both mark Non-Speech / Silence: {tot_neither:.3f}s")
    print(f"Total time verified: {tot_both + tot_lab_only + tot_rttm_only + tot_neither:.3f}s == {audio_dur:.3f}s")

    print("\n=== TOP INTERVALS MARKED AS SPEECH IN LAB BUT MISSING FROM RTTM ===")
    for s, e, d in sorted(lab_only_intervals, key=lambda x: x[2], reverse=True)[:15]:
        print(f"  [{s:7.3f}s - {e:7.3f}s] ({d:6.3f}s missing from RTTM)")

if __name__ == "__main__":
    main()
