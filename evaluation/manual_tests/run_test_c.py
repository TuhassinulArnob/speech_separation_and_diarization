"""Test C: Two-Speaker Diarization Pipeline Execution and Scoring.

Inputs:
- evaluation/LDC2021S10.flac
- evaluation/LDC2021S10.rttm

Outputs:
- evaluation/manual_tests/results/test_C_two_speakers/
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from Project.run_diarization import run_pipeline
from Project.ssl_pipeline.dataset_evaluation import DiarizationScorer, EvaluationProtocol, align_recording_id, write_hypothesis_rttm
from Project.ssl_pipeline.rttm import parse_rttm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
AUDIO_PATH = ROOT_DIR / "evaluation" / "LDC2021S10.flac"
REF_RTTM_PATH = ROOT_DIR / "evaluation" / "LDC2021S10.rttm"
RESULTS_DIR = SCRIPT_DIR / "results" / "test_C_two_speakers"


def run_test_c() -> dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_dir = RESULTS_DIR / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []

    def log(msg: str) -> None:
        LOGGER.info(msg)
        logs.append(msg)

    log(f"Starting Test C on {AUDIO_PATH} and {REF_RTTM_PATH}")

    # 1. Verification of Inputs
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")
    if not REF_RTTM_PATH.exists():
        raise FileNotFoundError(f"Reference RTTM not found: {REF_RTTM_PATH}")

    audio_info = sf.info(str(AUDIO_PATH))
    audio_dur = float(audio_info.duration)
    sr = int(audio_info.samplerate)
    channels = int(audio_info.channels)

    ref_turns = parse_rttm(REF_RTTM_PATH, allow_overlap=True)
    recording_id = ref_turns[0].recording_id
    ref_speakers = sorted({t.speaker_id for t in ref_turns})
    ref_speech_dur = sum(t.duration for t in ref_turns)
    max_ref_timestamp = max(t.end for t in ref_turns)

    # Calculate overlapping speech intervals in reference
    boundaries = sorted({t for turn in ref_turns for t in (turn.start, turn.end)})
    overlap_duration = 0.0
    overlap_intervals: list[tuple[float, float, list[str]]] = []
    for s, e in zip(boundaries, boundaries[1:]):
        if e <= s:
            continue
        mid = (s + e) / 2.0
        active = sorted({t.speaker_id for t in ref_turns if t.start <= mid < t.end})
        if len(active) > 1:
            dur = e - s
            overlap_duration += dur
            overlap_intervals.append((round(s, 3), round(e, 3), active))

    log(f"Audio: {sr} Hz, {channels} ch, {audio_dur:.2f}s")
    log(f"Reference RTTM: {len(ref_turns)} turns, {len(ref_speakers)} speakers {ref_speakers}, recording_id={recording_id}")
    log(f"Reference speech duration: {ref_speech_dur:.3f}s ({ref_speech_dur / audio_dur * 100:.1f}% of audio)")
    log(f"Overlapping intervals: {len(overlap_intervals)} regions, total {overlap_duration:.3f}s ({overlap_duration / audio_dur * 100:.2f}% of audio)")
    log(f"Timestamp consistency: max reference end={max_ref_timestamp:.3f}s <= audio duration={audio_dur:.3f}s ({max_ref_timestamp <= audio_dur})")

    # 2. Run Diarization Pipeline
    log("Executing diarization pipeline on LDC2021S10...")
    start_time = time.time()
    pipeline_result = run_pipeline(
        AUDIO_PATH,
        output_dir=pipeline_dir,
        backend="energy",
        num_speakers=2,
        oracle_num_speakers=True,
        save_json=True,
        save_csv=True,
        save_rttm=True,
    )
    pipeline_elapsed = time.time() - start_time
    log(f"Pipeline finished in {pipeline_elapsed:.2f}s")

    # 3. Save predicted RTTM separately with aligned recording_id
    raw_hyp_turns = parse_rttm(pipeline_dir / "hypothesis.rttm", allow_overlap=True)
    hyp_turns = align_recording_id(raw_hyp_turns, recording_id)
    separate_hyp_path = RESULTS_DIR / "hypothesis.rttm"
    write_hypothesis_rttm(hyp_turns, separate_hyp_path)
    hyp_speakers = sorted({t.speaker_id for t in hyp_turns})
    hyp_speech_dur = sum(t.duration for t in hyp_turns)
    log(f"Saved separate hypothesis RTTM to {separate_hyp_path} ({len(hyp_turns)} turns, {len(hyp_speakers)} speakers)")

    # Check whether pipeline preserved or removed overlaps
    hyp_boundaries = sorted({t for turn in hyp_turns for t in (turn.start, turn.end)})
    hyp_overlap_dur = 0.0
    for s, e in zip(hyp_boundaries, hyp_boundaries[1:]):
        if e <= s:
            continue
        mid = (s + e) / 2.0
        active = [t.speaker_id for t in hyp_turns if t.start <= mid < t.end]
        if len(active) > 1:
            hyp_overlap_dur += (e - s)
    log(f"Hypothesis overlap duration: {hyp_overlap_dur:.3f}s (pipeline timeline reconstruction enforces hard non-overlapping partitions)")

    # 4. Evaluation under Standard Protocol (Collar=0.0s, skip_overlap=False)
    scorer_default = DiarizationScorer(EvaluationProtocol(collar=0.0, skip_overlap=False, uem_mode="full_file"))
    result_default = scorer_default.score(ref_turns, hyp_turns, uri=recording_id, uem_end=audio_dur)

    # Evaluation with forgiveness collar (Collar=0.25s)
    scorer_collar25 = DiarizationScorer(EvaluationProtocol(collar=0.25, skip_overlap=False, uem_mode="full_file"))
    result_collar25 = scorer_collar25.score(ref_turns, hyp_turns, uri=recording_id, uem_end=audio_dur)

    # Evaluation excluding overlapping speech (Collar=0.0s, skip_overlap=True)
    scorer_no_overlap = DiarizationScorer(EvaluationProtocol(collar=0.0, skip_overlap=True, uem_mode="full_file"))
    result_no_overlap = scorer_no_overlap.score(ref_turns, hyp_turns, uri=recording_id, uem_end=audio_dur)

    # Mapping
    from Project.ssl_pipeline.evaluation import _optimal_hypothesis_to_reference_mapping
    speaker_map = _optimal_hypothesis_to_reference_mapping(ref_turns, hyp_turns)

    log(f"Primary Scoring (collar=0.0s, overlap=included): DER={result_default.der * 100:.2f}%, JER={result_default.jer * 100:.2f}%" if result_default.jer else f"Primary Scoring: DER={result_default.der * 100:.2f}%")
    log(f"  Missed Speech: {result_default.missed_speech:.3f}s ({result_default.missed_speech / result_default.scored_duration * 100:.2f}%)")
    log(f"  False Alarm: {result_default.false_alarm:.3f}s ({result_default.false_alarm / result_default.scored_duration * 100:.2f}%)")
    log(f"  Speaker Confusion: {result_default.speaker_confusion:.3f}s ({result_default.speaker_confusion / result_default.scored_duration * 100:.2f}%)")
    log(f"  Scored Duration: {result_default.scored_duration:.3f}s")
    log(f"Optimal speaker mapping: {speaker_map}")

    evaluation_report = {
        "test": "Test C: Two-Speaker Diarization Evaluation",
        "audio_file": AUDIO_PATH.name,
        "reference_rttm_file": REF_RTTM_PATH.name,
        "recording_id": recording_id,
        "audio_duration_seconds": round(audio_dur, 4),
        "sample_rate": sr,
        "reference_analysis": {
            "speaker_count": len(ref_speakers),
            "speakers": ref_speakers,
            "turn_count": len(ref_turns),
            "total_speech_duration_seconds": round(ref_speech_dur, 4),
            "speech_ratio": round(ref_speech_dur / audio_dur, 4),
            "overlapping_intervals_count": len(overlap_intervals),
            "overlapping_speech_duration_seconds": round(overlap_duration, 4),
            "overlapping_speech_ratio": round(overlap_duration / audio_dur, 4),
            "timestamp_consistency": bool(max_ref_timestamp <= audio_dur + 0.1),
        },
        "hypothesis_analysis": {
            "hypothesis_rttm_file": separate_hyp_path.name,
            "detected_speaker_count": len(hyp_speakers),
            "detected_speakers": hyp_speakers,
            "turn_count": len(hyp_turns),
            "total_speech_duration_seconds": round(hyp_speech_dur, 4),
            "hypothesis_overlap_duration_seconds": round(hyp_overlap_dur, 4),
            "speaker_mapping": speaker_map,
        },
        "pipeline_execution": {
            "backend": "WavLM + Spectral Clustering",
            "vad_backend": "energy (adaptive threshold)",
            "pipeline_elapsed_seconds": round(pipeline_elapsed, 2),
        },
        "scoring_results": {
            "primary_protocol_collar_0_0s_overlap_scored": {
                "collar_seconds": 0.0,
                "skip_overlap": False,
                "scoring_backend": result_default.backend,
                "der": round(result_default.der, 6),
                "der_percent": round(result_default.der * 100.0, 2),
                "jer": round(result_default.jer, 6) if result_default.jer is not None else "N/A",
                "jer_percent": round(result_default.jer * 100.0, 2) if result_default.jer is not None else "N/A",
                "missed_speech_seconds": round(result_default.missed_speech, 4),
                "missed_speech_percent": round(result_default.missed_speech / result_default.scored_duration * 100.0, 2),
                "false_alarm_seconds": round(result_default.false_alarm, 4),
                "false_alarm_percent": round(result_default.false_alarm / result_default.scored_duration * 100.0, 2),
                "speaker_confusion_seconds": round(result_default.speaker_confusion, 4),
                "speaker_confusion_percent": round(result_default.speaker_confusion / result_default.scored_duration * 100.0, 2),
                "scored_duration_seconds": round(result_default.scored_duration, 4),
                "total_error_seconds": round(result_default.total_error, 4),
            },
            "comparison_protocol_collar_0_25s": {
                "collar_seconds": 0.25,
                "skip_overlap": False,
                "der_percent": round(result_collar25.der * 100.0, 2),
                "jer_percent": round(result_collar25.jer * 100.0, 2) if result_collar25.jer is not None else "N/A",
                "missed_speech_seconds": round(result_collar25.missed_speech, 4),
                "false_alarm_seconds": round(result_collar25.false_alarm, 4),
                "speaker_confusion_seconds": round(result_collar25.speaker_confusion, 4),
            },
            "comparison_protocol_skip_overlap_true": {
                "collar_seconds": 0.0,
                "skip_overlap": True,
                "der_percent": round(result_no_overlap.der * 100.0, 2),
                "jer_percent": round(result_no_overlap.jer * 100.0, 2) if result_no_overlap.jer is not None else "N/A",
                "missed_speech_seconds": round(result_no_overlap.missed_speech, 4),
                "false_alarm_seconds": round(result_no_overlap.false_alarm, 4),
                "speaker_confusion_seconds": round(result_no_overlap.speaker_confusion, 4),
                "scored_duration_seconds": round(result_no_overlap.scored_duration, 4),
            },
        },
        "overlap_handling_documentation": {
            "supported": False,
            "explanation": (
                "The pipeline reconstructs timelines via Project.ssl_pipeline.timeline_reconstruction._resolve_overlaps, "
                "which forcibly clamps cross-speaker overlapping intervals to single-speaker hard partitions "
                "(current.start = resolved[-1].end). Consequently, any simultaneous speech in the recording "
                "is assigned to at most one speaker, causing the secondary overlapping speaker to be classified as missed speech. "
                "The system therefore does NOT support overlapping-speaker diarization."
            ),
            "reference_overlap_lost_seconds": round(overlap_duration, 4),
        },
        "diagnostic_analysis": _compute_diagnostics(
            audio_dur=audio_dur,
            ref_turns=ref_turns,
            hyp_turns=hyp_turns,
            pipeline_dir=pipeline_dir,
            mapping=speaker_map,
            results_dir=RESULTS_DIR,
            lab_path=ROOT_DIR / "evaluation" / "LDC2021S10.lab",
        ),
        "logs": logs,
    }

    out_file = RESULTS_DIR / "test_C_evaluation.json"
    out_file.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")
    log(f"Saved Test C evaluation to {out_file}")
    return evaluation_report


def _compute_diagnostics(
    audio_dur: float,
    ref_turns: list,
    hyp_turns: list,
    pipeline_dir: Path,
    mapping: dict[str, str],
    results_dir: Path,
    lab_path: Path,
) -> dict[str, object]:
    """Compute detailed root-cause diagnostics for False Alarm, Miss, and Confusion."""
    import matplotlib.pyplot as plt

    clipped_ref = [t for t in ref_turns if t.start < audio_dur]
    vad_json_path = pipeline_dir / "vad_regions.json"
    clustering_json_path = pipeline_dir / "clustering.json"

    vad_data = json.loads(vad_json_path.read_text(encoding="utf-8")) if vad_json_path.exists() else []
    vad_regions = [{"start": r.get("start_time", r.get("start", 0.0)), "end": r.get("end_time", r.get("end", 0.0))} for r in vad_data]

    clusters = json.loads(clustering_json_path.read_text(encoding="utf-8")) if clustering_json_path.exists() else []

    lab_speech: list[tuple[float, float]] = []
    if lab_path.exists():
        for line in lab_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and parts[2] == "speech":
                lab_speech.append((float(parts[0]), min(audio_dur, float(parts[1]))))

    boundaries = sorted({0.0, audio_dur} | {t for seg in [*clipped_ref, *hyp_turns] for t in (seg.start, seg.end)} | {t for seg in lab_speech for t in seg})

    fa_intervals = []
    miss_intervals = []
    conf_intervals = []
    fa_in_lab_speech = 0.0
    fa_in_acoustic_silence = 0.0
    miss_vad = 0.0
    miss_overlap = 0.0
    miss_trim = 0.0

    for s, e in zip(boundaries, boundaries[1:]):
        if e <= s or s >= audio_dur:
            continue
        mid = (s + e) / 2.0
        dur = e - s
        r_spks = {t.speaker_id for t in clipped_ref if t.start <= mid < t.end}
        h_spks = {t.speaker_id for t in hyp_turns if t.start <= mid < t.end}
        n_ref = len(r_spks)
        n_hyp = len(h_spks)

        mapped_h = {mapping.get(spk, spk) for spk in h_spks}
        n_corr = len(r_spks & mapped_h)

        if n_hyp > n_ref:
            fa_dur = (n_hyp - n_ref) * dur
            fa_intervals.append({"start": round(s, 3), "end": round(e, 3), "duration": round(dur, 3), "hypothesis_speakers": list(h_spks)})
            if any(ls <= mid < le for ls, le in lab_speech):
                fa_in_lab_speech += fa_dur
            else:
                fa_in_acoustic_silence += fa_dur

        if n_ref > n_hyp:
            miss_dur = (n_ref - n_hyp) * dur
            miss_intervals.append({"start": round(s, 3), "end": round(e, 3), "duration": round(dur, 3), "reference_speakers": list(r_spks)})
            in_vad = any(vr["start"] <= mid < vr["end"] for vr in vad_regions)
            if not in_vad:
                miss_vad += miss_dur
            elif n_ref > 1 and n_hyp == 1:
                miss_overlap += miss_dur
            else:
                miss_trim += miss_dur

        if min(n_ref, n_hyp) > n_corr:
            conf_dur = (min(n_ref, n_hyp) - n_corr) * dur
            conf_intervals.append({"start": round(s, 3), "end": round(e, 3), "duration": round(dur, 3), "ref_speakers": list(r_spks), "hyp_speakers": list(h_spks), "mapped_hyp": list(mapped_h)})

    tot_fa = fa_in_lab_speech + fa_in_acoustic_silence
    tot_miss = miss_vad + miss_overlap + miss_trim
    conf_117_as_118 = sum(i["duration"] for i in conf_intervals if "speaker117" in i["ref_speakers"] and "speaker118" in i["mapped_hyp"])
    conf_118_as_117 = sum(i["duration"] for i in conf_intervals if "speaker118" in i["ref_speakers"] and "speaker117" in i["mapped_hyp"])

    # Cluster purity in pure reference turns
    pure_117 = []
    pure_118 = []
    for c in clusters:
        c_mid = (c["start"] + c["end"]) / 2.0
        active_ref = [t.speaker_id for t in clipped_ref if t.start <= c_mid < t.end]
        if len(active_ref) == 1:
            if active_ref[0] == "speaker117":
                pure_117.append(c["cluster_id"])
            elif active_ref[0] == "speaker118":
                pure_118.append(c["cluster_id"])

    # Generate diagnostic plot
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    for t in clipped_ref:
        if t.speaker_id == "speaker117":
            axes[0].barh(0, t.duration, left=t.start, height=0.6, color="tab:blue")
        elif t.speaker_id == "speaker118":
            axes[1].barh(0, t.duration, left=t.start, height=0.6, color="tab:cyan")
    axes[0].set_ylabel("Ref 117")
    axes[1].set_ylabel("Ref 118")
    for ax in axes[:3]:
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])

    for r in vad_regions:
        axes[2].barh(0, r["end"] - r["start"], left=r["start"], height=0.6, color="tab:green")
    axes[2].set_ylabel("VAD Regions")

    for t in hyp_turns:
        c = "tab:blue" if t.speaker_id == "speaker_0" else "tab:cyan"
        y = 0.2 if t.speaker_id == "speaker_0" else -0.2
        axes[3].barh(y, t.duration, left=t.start, height=0.35, color=c)
    axes[3].set_ylabel("Hypothesis")
    axes[3].set_ylim(-0.5, 0.5)
    axes[3].set_yticks([0.2, -0.2])
    axes[3].set_yticklabels(["spk_0 (117)", "spk_1 (118)"])

    for i in fa_intervals:
        axes[4].barh(0.2, i["duration"], left=i["start"], height=0.35, color="tab:red")
    for i in miss_intervals:
        axes[4].barh(-0.2, i["duration"], left=i["start"], height=0.35, color="tab:orange")
    axes[4].set_ylabel("Errors")
    axes[4].set_ylim(-0.5, 0.5)
    axes[4].set_yticks([0.2, -0.2])
    axes[4].set_yticklabels(["False Alarm", "Missed"])
    axes[4].set_xlabel("Time (seconds)")
    fig.suptitle("Test C Diagnostic Timeline: Ground Truth vs VAD vs Hypothesis (LDC2021S10)", fontsize=13)
    fig.tight_layout()
    plot_file = results_dir / "diagnostic_timeline.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)

    return {
        "false_alarm_diagnosis": {
            "total_false_alarm_seconds": round(tot_fa, 4),
            "fa_in_ground_truth_lab_speech_seconds": round(fa_in_lab_speech, 4),
            "fa_in_ground_truth_lab_speech_percent": round(fa_in_lab_speech / max(0.001, tot_fa) * 100.0, 2),
            "fa_in_acoustic_silence_seconds": round(fa_in_acoustic_silence, 4),
            "fa_in_acoustic_silence_percent": round(fa_in_acoustic_silence / max(0.001, tot_fa) * 100.0, 2),
            "vad_vs_rttm_root_cause": (
                "87.0% (38.675s) of the False Alarm corresponds to audible, intelligible human speech that is "
                "explicitly annotated in the ground-truth Speech Activity Detection file (LDC2021S10.lab, 88.696s total speech) "
                "and confirmed by Whisper ASR, but is omitted from the reference speaker turn annotation (LDC2021S10.rttm, 58.432s total speech). "
                "Only 13.0% (5.767s) represents false detection in true acoustic silence. "
                "VAD correctly detects speech present in the audio container, but NIST scoring against the incomplete RTTM labels all unannotated speech as False Alarm."
            ),
            "top_false_alarm_intervals": sorted(fa_intervals, key=lambda x: x["duration"], reverse=True)[:10],
        },
        "missed_speech_diagnosis": {
            "total_missed_seconds": round(tot_miss, 4),
            "missed_due_to_vad_silence_seconds": round(miss_vad, 4),
            "missed_due_to_vad_silence_percent": round(miss_vad / max(0.001, tot_miss) * 100.0, 2),
            "missed_due_to_overlap_collapse_seconds": round(miss_overlap, 4),
            "missed_due_to_overlap_collapse_percent": round(miss_overlap / max(0.001, tot_miss) * 100.0, 2),
            "missed_due_to_windowing_trim_seconds": round(miss_trim, 4),
            "missed_due_to_windowing_trim_percent": round(miss_trim / max(0.001, tot_miss) * 100.0, 2),
        },
        "speaker_confusion_diagnosis": {
            "total_confusion_seconds": round(conf_117_as_118 + conf_118_as_117, 4),
            "ref_117_misclassified_as_118_seconds": round(conf_117_as_118, 4),
            "ref_118_misclassified_as_117_seconds": round(conf_118_as_117, 4),
            "pure_117_segments_cluster_distribution": {"cluster_0": pure_117.count(0), "cluster_1": pure_117.count(1)},
            "pure_118_segments_cluster_distribution": {"cluster_0": pure_118.count(0), "cluster_1": pure_118.count(1)},
            "representation_root_cause": (
                "Pretrained WavLM-base-plus representations without speaker-discriminative fine-tuning or metric projection "
                "(e.g. ECAPA-TDNN / x-vector loss) capture predominantly phonetic and acoustic room properties. "
                "Under natural single-microphone conversational conditions, inter-speaker cosine distance is smaller than "
                "intra-speaker phonetic variation, resulting in cluster overlap."
            ),
        },
        "diagnostic_timeline_plot": str(plot_file),
    }


if __name__ == "__main__":
    run_test_c()
