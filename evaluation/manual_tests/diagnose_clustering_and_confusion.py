"""Diagnose clustering accuracy, speaker confusion, and missed speech on LDC2021S10."""

import json
from pathlib import Path
import numpy as np
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
    clustering_json_path = pipe_dir / "clustering.json"

    ref_turns = parse_rttm(ref_rttm_path, allow_overlap=True)
    hyp_turns = parse_rttm(hyp_rttm_path, allow_overlap=True)
    vad_data = json.loads(vad_json_path.read_text(encoding="utf-8"))
    vad_regions = [{"start": r.get("start_time", r.get("start")), "end": r.get("end_time", r.get("end"))} for r in vad_data]
    clusters = json.loads(clustering_json_path.read_text(encoding="utf-8"))

    # Clip reference to 120.0s
    clipped_ref = [t for t in ref_turns if t.start < 120.0]

    # 1. Missed speech analysis:
    # A reference segment is missed if hyp has no speaker at that time.
    # Check whether the missed speech was:
    # (a) Missed by VAD
    # (b) Present in VAD but lost in segmentation / windowing
    # (c) Lost in timeline overlap resolution (when two speakers talk, pipeline picks only one)
    boundaries = sorted({t for turn in [*clipped_ref, *hyp_turns] for t in (turn.start, turn.end)} | {0.0, 120.0})
    mapping = {"speaker_0": "speaker117", "speaker_1": "speaker118"}

    missed_vad = 0.0
    missed_overlap = 0.0
    missed_other = 0.0

    for s, e in zip(boundaries, boundaries[1:]):
        if e <= s or s >= 120.0:
            continue
        mid = (s + e) / 2.0
        dur = e - s
        r_spks = {t.speaker_id for t in clipped_ref if t.start <= mid < t.end}
        h_spks = {t.speaker_id for t in hyp_turns if t.start <= mid < t.end}

        n_ref = len(r_spks)
        n_hyp = len(h_spks)

        if n_ref > n_hyp:
            miss_dur = (n_ref - n_hyp) * dur
            # Is VAD active at this midpoint?
            in_vad = any(vr["start"] <= mid < vr["end"] for vr in vad_regions)
            if not in_vad:
                missed_vad += miss_dur
            elif n_ref > 1 and n_hyp == 1:
                # Reference had overlapping speech, but hypothesis is single-speaker partition!
                missed_overlap += miss_dur
            else:
                missed_other += miss_dur

    print("=== Missed Speech Breakdown (Total: 15.068s) ===")
    print(f"  Missed due to VAD classifying speech as silence: {missed_vad:.3f}s ({missed_vad/(missed_vad+missed_overlap+missed_other)*100:.1f}%)")
    print(f"  Missed due to overlapping speech (pipeline lacks multi-speaker output): {missed_overlap:.3f}s ({missed_overlap/(missed_vad+missed_overlap+missed_other)*100:.1f}%)")
    print(f"  Missed due to segmentation/timeline trimming: {missed_other:.3f}s ({missed_other/(missed_vad+missed_overlap+missed_other)*100:.1f}%)")

    # 2. Speaker Confusion analysis:
    # Confusion occurs when both ref and hyp are active, but hyp speaker != ref speaker
    confusion_intervals = []
    for s, e in zip(boundaries, boundaries[1:]):
        if e <= s or s >= 120.0:
            continue
        mid = (s + e) / 2.0
        dur = e - s
        r_spks = {t.speaker_id for t in clipped_ref if t.start <= mid < t.end}
        h_spks = {t.speaker_id for t in hyp_turns if t.start <= mid < t.end}

        mapped_h = {mapping.get(spk, spk) for spk in h_spks}
        n_corr = len(r_spks & mapped_h)
        n_ref = len(r_spks)
        n_hyp = len(h_spks)
        conf = (min(n_ref, n_hyp) - n_corr) * dur
        if conf > 0:
            confusion_intervals.append((s, e, dur, r_spks, mapped_h, h_spks))

    tot_conf = sum(i[2] for i in confusion_intervals)
    print(f"\n=== Speaker Confusion Breakdown (Total: {tot_conf:.3f}s) ===")
    # Check confusion by reference speaker
    conf_117_as_118 = sum(i[2] for i in confusion_intervals if "speaker117" in i[3] and "speaker118" in i[4])
    conf_118_as_117 = sum(i[2] for i in confusion_intervals if "speaker118" in i[3] and "speaker117" in i[4])
    print(f"  Ref speaker117 misclassified as speaker_1 (mapped to speaker118): {conf_117_as_118:.3f}s")
    print(f"  Ref speaker118 misclassified as speaker_0 (mapped to speaker117): {conf_118_as_117:.3f}s")

    print("\nTop 10 Longest Speaker Confusion Intervals:")
    for s, e, d, r, mh, h in sorted(confusion_intervals, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  [{s:6.3f}s - {e:6.3f}s] ({d:5.3f}s) | True: {list(r)} | Pred: {list(h)} -> Mapped: {list(mh)}")

    # 3. Cluster purity on pure reference segments
    # For segments that fall entirely within a single reference speaker turn:
    pure_117_clusters = []
    pure_118_clusters = []
    for c in clusters:
        c_mid = (c["start"] + c["end"]) / 2.0
        # which ref speaker is active at c_mid?
        active_ref = [t.speaker_id for t in clipped_ref if t.start <= c_mid < t.end]
        if len(active_ref) == 1:
            if active_ref[0] == "speaker117":
                pure_117_clusters.append(c["cluster_id"])
            elif active_ref[0] == "speaker118":
                pure_118_clusters.append(c["cluster_id"])

    print(f"\n=== Spectral Clustering Segment Purity ===")
    print(f"Segments in pure speaker117 turns ({len(pure_117_clusters)}):")
    print(f"  Assigned cluster 0 (spk_0): {pure_117_clusters.count(0)} ({pure_117_clusters.count(0)/max(1, len(pure_117_clusters))*100:.1f}%)")
    print(f"  Assigned cluster 1 (spk_1): {pure_117_clusters.count(1)} ({pure_117_clusters.count(1)/max(1, len(pure_117_clusters))*100:.1f}%)")

    print(f"Segments in pure speaker118 turns ({len(pure_118_clusters)}):")
    print(f"  Assigned cluster 0 (spk_0): {pure_118_clusters.count(0)} ({pure_118_clusters.count(0)/max(1, len(pure_118_clusters))*100:.1f}%)")
    print(f"  Assigned cluster 1 (spk_1): {pure_118_clusters.count(1)} ({pure_118_clusters.count(1)/max(1, len(pure_118_clusters))*100:.1f}%)")

if __name__ == "__main__":
    main()
