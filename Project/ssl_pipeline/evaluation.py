"""DER and JER evaluation for diarization RTTM files.

This module compares reference RTTM files against hypothesis RTTM files. It does
not implement a baseline system or an experiment runner.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.rttm import RTTMSpeakerTurn, parse_rttm

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingEvaluation:
    """Evaluation metrics for one recording."""

    recording_name: str
    duration: float
    der: float
    jer: float
    missed_speech: float
    false_alarm: float
    speaker_confusion: float
    total_error: float

    def to_dict(self) -> dict[str, str | float]:
        """Return a serializable recording report row."""

        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    """Overall DER/JER report with recording-wise details."""

    recording_name: str
    duration: float
    der: float
    jer: float
    missed_speech: float
    false_alarm: float
    speaker_confusion: float
    total_error: float
    average_der: float
    average_jer: float
    recordings: list[RecordingEvaluation]
    backend: str

    def to_dict(self) -> dict[str, object]:
        """Return a serializable evaluation report."""

        data = asdict(self)
        data["recordings"] = [item.to_dict() for item in self.recordings]
        return data


class DiarizationEvaluator:
    """Evaluate diarization RTTM files with DER and JER."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        collar_seconds: float | None = None,
        use_pyannote: bool | None = None,
        skip_overlap: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.collar_seconds = (
            self.config.evaluation_collar_seconds
            if collar_seconds is None
            else float(collar_seconds)
        )
        self.use_pyannote = (
            self.config.evaluation_use_pyannote
            if use_pyannote is None
            else bool(use_pyannote)
        )
        self.skip_overlap = bool(skip_overlap)
        if self.collar_seconds < 0.0:
            raise ValueError("evaluation_collar_seconds must be non-negative.")

    def evaluate_files(
        self,
        reference_rttm: str | Path,
        hypothesis_rttm: str | Path,
    ) -> EvaluationReport:
        """Evaluate one pair of RTTM files."""

        reference_turns = parse_rttm(reference_rttm, allow_overlap=True)
        hypothesis_turns = parse_rttm(hypothesis_rttm, allow_overlap=True)
        return self.evaluate_turns(reference_turns, hypothesis_turns)

    def evaluate_turns(
        self,
        reference_turns: Iterable[RTTMSpeakerTurn],
        hypothesis_turns: Iterable[RTTMSpeakerTurn],
    ) -> EvaluationReport:
        """Evaluate parsed RTTM turns."""

        reference = list(reference_turns)
        hypothesis = list(hypothesis_turns)
        if not reference:
            raise ValueError("Reference RTTM is empty.")
        if not hypothesis:
            raise ValueError("Hypothesis RTTM is empty.")

        recordings = _evaluate_recordings(
            reference,
            hypothesis,
            collar_seconds=self.collar_seconds,
            use_pyannote=self.use_pyannote,
            skip_overlap=self.skip_overlap,
        )
        return _build_report(recordings, backend=_backend_name(self.use_pyannote))


def evaluate_rttm(
    reference_rttm: str | Path,
    hypothesis_rttm: str | Path,
    *,
    config_path: str | Path | None = None,
    collar_seconds: float | None = None,
    use_pyannote: bool | None = None,
    skip_overlap: bool = False,
) -> EvaluationReport:
    """Evaluate one reference/hypothesis RTTM pair."""

    config = load_config(config_path) if config_path else load_config()
    evaluator = DiarizationEvaluator(
        config,
        collar_seconds=collar_seconds,
        use_pyannote=use_pyannote,
        skip_overlap=skip_overlap,
    )
    return evaluator.evaluate_files(reference_rttm, hypothesis_rttm)


def evaluate_rttm_directory(
    reference_dir: str | Path,
    hypothesis_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    collar_seconds: float | None = None,
    use_pyannote: bool | None = None,
) -> EvaluationReport:
    """Evaluate matching RTTM files in two directories by file name."""

    reference_root = Path(reference_dir)
    hypothesis_root = Path(hypothesis_dir)
    reference_files = sorted(reference_root.glob("*.rttm"))
    if not reference_files:
        raise ValueError(f"No reference RTTM files found in {reference_root}")

    config = load_config(config_path) if config_path else load_config()
    evaluator = DiarizationEvaluator(
        config,
        collar_seconds=collar_seconds,
        use_pyannote=use_pyannote,
    )
    recordings: list[RecordingEvaluation] = []
    for reference_file in reference_files:
        hypothesis_file = hypothesis_root / reference_file.name
        if not hypothesis_file.exists():
            raise FileNotFoundError(f"Missing hypothesis RTTM: {hypothesis_file}")
        report = evaluator.evaluate_files(reference_file, hypothesis_file)
        recordings.extend(report.recordings)
    return _build_report(recordings, backend=_backend_name(evaluator.use_pyannote))


def export_evaluation_json(
    report: EvaluationReport,
    output_path: str | Path,
) -> None:
    """Export an evaluation report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Saved evaluation JSON to %s", path)


def export_evaluation_csv(
    report: EvaluationReport,
    output_path: str | Path,
) -> None:
    """Export recording-wise evaluation rows as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "recording_name",
                "duration",
                "der",
                "jer",
                "missed_speech",
                "false_alarm",
                "speaker_confusion",
                "total_error",
            ],
        )
        writer.writeheader()
        writer.writerows([item.to_dict() for item in report.recordings])
    LOGGER.info("Saved evaluation CSV to %s", path)


def export_evaluation_markdown(
    report: EvaluationReport,
    output_path: str | Path,
) -> None:
    """Export a compact Markdown evaluation summary."""

    lines = [
        "# Diarization Evaluation Report",
        "",
        f"- Backend: {report.backend}",
        f"- Duration: {report.duration:.3f}s",
        f"- DER: {report.der:.4f}",
        f"- JER: {report.jer:.4f}",
        f"- Missed Speech: {report.missed_speech:.3f}s",
        f"- False Alarm: {report.false_alarm:.3f}s",
        f"- Speaker Confusion: {report.speaker_confusion:.3f}s",
        "",
        "| Recording | Duration | DER | JER | Miss | False Alarm | Confusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.recordings:
        lines.append(
            "| "
            f"{item.recording_name} | {item.duration:.3f} | {item.der:.4f} | "
            f"{item.jer:.4f} | {item.missed_speech:.3f} | "
            f"{item.false_alarm:.3f} | {item.speaker_confusion:.3f} |"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved evaluation Markdown to %s", path)


def plot_der_breakdown(
    report: EvaluationReport,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot DER component durations."""

    figure, axis = plt.subplots(figsize=(7, 4))
    labels = ["Miss", "False Alarm", "Confusion"]
    values = [report.missed_speech, report.false_alarm, report.speaker_confusion]
    axis.bar(labels, values)
    axis.set_title("DER Breakdown")
    axis.set_ylabel("Seconds")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_recording_comparison(
    report: EvaluationReport,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot recording-wise DER and JER."""

    names = [item.recording_name for item in report.recordings]
    der_values = [item.der for item in report.recordings]
    jer_values = [item.jer for item in report.recordings]
    x = np.arange(len(names))

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - 0.2, der_values, width=0.4, label="DER")
    axis.bar(x + 0.2, jer_values, width=0.4, label="JER")
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=30, ha="right")
    axis.set_ylabel("Error rate")
    axis.set_title("Recording Comparison")
    axis.legend()
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_speaker_confusion_chart(
    report: EvaluationReport,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot speaker confusion duration per recording."""

    names = [item.recording_name for item in report.recordings]
    values = [item.speaker_confusion for item in report.recordings]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(names, values)
    axis.set_title("Speaker Confusion")
    axis.set_xlabel("Recording")
    axis.set_ylabel("Seconds")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def _evaluate_recordings(
    reference_turns: list[RTTMSpeakerTurn],
    hypothesis_turns: list[RTTMSpeakerTurn],
    *,
    collar_seconds: float,
    use_pyannote: bool,
    skip_overlap: bool = False,
) -> list[RecordingEvaluation]:
    """Evaluate each recording present in the reference RTTM."""

    reference_by_recording = _group_by_recording(reference_turns)
    hypothesis_by_recording = _group_by_recording(hypothesis_turns)
    recordings: list[RecordingEvaluation] = []

    for recording_name, reference in sorted(reference_by_recording.items()):
        hypothesis = hypothesis_by_recording.get(recording_name, [])
        if not hypothesis:
            raise ValueError(f"Missing hypothesis turns for recording {recording_name}")
        local_report = _evaluate_one_recording(
            recording_name,
            _apply_collar(reference, collar_seconds),
            _apply_collar(hypothesis, collar_seconds),
            skip_overlap=skip_overlap,
        )
        recordings.append(
            _with_pyannote_metrics(
                local_report,
                reference,
                hypothesis,
                collar_seconds=collar_seconds,
                use_pyannote=use_pyannote,
            )
        )
    return recordings


def _evaluate_one_recording(
    recording_name: str,
    reference: list[RTTMSpeakerTurn],
    hypothesis: list[RTTMSpeakerTurn],
    *,
    skip_overlap: bool = False,
) -> RecordingEvaluation:
    """Evaluate one recording using atomic time intervals."""

    if not reference:
        raise ValueError(f"Reference recording {recording_name} is empty after collar.")
    mapping = _optimal_hypothesis_to_reference_mapping(reference, hypothesis)
    boundaries = sorted(
        {
            time
            for turn in [*reference, *hypothesis]
            for time in (turn.start, turn.end)
        }
    )
    missed = 0.0
    false_alarm = 0.0
    confusion = 0.0
    scored_ref_duration = 0.0

    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        duration = end - start
        ref_speakers = {turn.speaker_id for turn in reference if turn.start <= midpoint < turn.end}
        hyp_speakers = {turn.speaker_id for turn in hypothesis if turn.start <= midpoint < turn.end}

        n_ref = len(ref_speakers)
        n_hyp = len(hyp_speakers)

        if skip_overlap and n_ref > 1:
            continue

        scored_ref_duration += n_ref * duration

        mapped_hyp = {mapping.get(s, s) for s in hyp_speakers}
        n_correct = len(ref_speakers & mapped_hyp)

        missed += max(0, n_ref - n_hyp) * duration
        false_alarm += max(0, n_hyp - n_ref) * duration
        confusion += (min(n_ref, n_hyp) - n_correct) * duration

    reference_duration = scored_ref_duration if skip_overlap else sum(turn.duration for turn in reference)
    if reference_duration <= 0.0:
        raise ValueError(f"Reference recording {recording_name} has zero duration.")

    total_error = missed + false_alarm + confusion
    der = total_error / reference_duration
    jer = _compute_jer(reference, hypothesis, mapping)
    return RecordingEvaluation(
        recording_name=recording_name,
        duration=round(reference_duration, 6),
        der=round(der, 6),
        jer=round(jer, 6),
        missed_speech=round(missed, 6),
        false_alarm=round(false_alarm, 6),
        speaker_confusion=round(confusion, 6),
        total_error=round(total_error, 6),
    )


def _compute_jer(
    reference: list[RTTMSpeakerTurn],
    hypothesis: list[RTTMSpeakerTurn],
    mapping: dict[str, str],
) -> float:
    """Compute Jaccard Error Rate averaged over reference speakers."""

    ref_speakers = sorted({turn.speaker_id for turn in reference})
    if not ref_speakers:
        return 0.0
    errors: list[float] = []
    for ref_speaker in ref_speakers:
        mapped_hyp_speakers = [
            hyp_speaker
            for hyp_speaker, mapped_ref in mapping.items()
            if mapped_ref == ref_speaker
        ]
        ref_intervals = [(turn.start, turn.end) for turn in reference if turn.speaker_id == ref_speaker]
        hyp_intervals = [
            (turn.start, turn.end)
            for turn in hypothesis
            if turn.speaker_id in mapped_hyp_speakers
        ]
        intersection = _interval_intersection_duration(ref_intervals, hyp_intervals)
        union = _interval_union_duration([*ref_intervals, *hyp_intervals])
        errors.append(0.0 if union == 0.0 else 1.0 - intersection / union)
    return float(np.mean(errors))


def _optimal_hypothesis_to_reference_mapping(
    reference: list[RTTMSpeakerTurn],
    hypothesis: list[RTTMSpeakerTurn],
) -> dict[str, str]:
    """Map hypothesis speaker names to reference speaker names by max overlap."""

    ref_speakers = sorted({turn.speaker_id for turn in reference})
    hyp_speakers = sorted({turn.speaker_id for turn in hypothesis})
    if not ref_speakers or not hyp_speakers:
        return {}

    overlaps = {
        (hyp, ref): _speaker_overlap(reference, hypothesis, ref, hyp)
        for hyp in hyp_speakers
        for ref in ref_speakers
    }

    try:
        from scipy.optimize import linear_sum_assignment

        cost_matrix = np.zeros((len(hyp_speakers), len(ref_speakers)), dtype=np.float64)
        for i, hyp in enumerate(hyp_speakers):
            for j, ref in enumerate(ref_speakers):
                cost_matrix[i, j] = -overlaps[(hyp, ref)]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {hyp_speakers[r]: ref_speakers[c] for r, c in zip(row_ind, col_ind)}
        used_refs = set(mapping.values())
        fallback_ref = ref_speakers[0]
        for hyp in hyp_speakers:
            if hyp not in mapping:
                remaining = [ref for ref in ref_speakers if ref not in used_refs]
                mapping[hyp] = remaining[0] if remaining else fallback_ref
        return mapping
    except Exception:
        if len(hyp_speakers) <= 8 and len(ref_speakers) <= 8:
            return _brute_force_mapping(hyp_speakers, ref_speakers, overlaps)
        return _greedy_mapping(hyp_speakers, ref_speakers, overlaps)


def _brute_force_mapping(
    hyp_speakers: list[str],
    ref_speakers: list[str],
    overlaps: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Find the highest-overlap one-to-one mapping for small speaker sets."""

    best_score = -1.0
    best_mapping: dict[str, str] = {}
    for refs in itertools.permutations(ref_speakers, min(len(hyp_speakers), len(ref_speakers))):
        mapping = dict(zip(hyp_speakers, refs))
        score = sum(overlaps[(hyp, ref)] for hyp, ref in mapping.items())
        if score > best_score:
            best_score = score
            best_mapping = mapping
    if len(hyp_speakers) > len(ref_speakers):
        used_refs = set(best_mapping.values())
        fallback_ref = ref_speakers[0]
        for hyp in hyp_speakers:
            if hyp not in best_mapping:
                remaining = [ref for ref in ref_speakers if ref not in used_refs]
                best_mapping[hyp] = remaining[0] if remaining else fallback_ref
    return best_mapping


def _greedy_mapping(
    hyp_speakers: list[str],
    ref_speakers: list[str],
    overlaps: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Greedy fallback mapping for large speaker sets."""

    pairs = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
    mapping: dict[str, str] = {}
    used_refs: set[str] = set()
    for (hyp, ref), _score in pairs:
        if hyp not in mapping and ref not in used_refs:
            mapping[hyp] = ref
            used_refs.add(ref)
    for hyp in hyp_speakers:
        mapping.setdefault(hyp, ref_speakers[0])
    return mapping


def _speaker_overlap(
    reference: list[RTTMSpeakerTurn],
    hypothesis: list[RTTMSpeakerTurn],
    ref_speaker: str,
    hyp_speaker: str,
) -> float:
    """Return total overlap between one reference and hypothesis speaker."""

    ref_intervals = [(turn.start, turn.end) for turn in reference if turn.speaker_id == ref_speaker]
    hyp_intervals = [(turn.start, turn.end) for turn in hypothesis if turn.speaker_id == hyp_speaker]
    return _interval_intersection_duration(ref_intervals, hyp_intervals)


def _interval_intersection_duration(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    """Return total intersection duration for two interval sets."""

    total = 0.0
    for left_start, left_end in left:
        for right_start, right_end in right:
            total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
    return total


def _interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    """Return total union duration of intervals."""

    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total


def _active_speaker(turns: list[RTTMSpeakerTurn], time: float) -> str | None:
    """Return the active speaker at a point in time."""

    for turn in turns:
        if turn.start <= time < turn.end:
            return turn.speaker_id
    return None


def _apply_collar(
    turns: list[RTTMSpeakerTurn],
    collar_seconds: float,
) -> list[RTTMSpeakerTurn]:
    """Trim a collar from both sides of every turn."""

    if collar_seconds == 0.0:
        return turns
    trimmed: list[RTTMSpeakerTurn] = []
    for turn in turns:
        start = turn.start + collar_seconds
        end = turn.end - collar_seconds
        if end > start:
            trimmed.append(
                RTTMSpeakerTurn(
                    recording_id=turn.recording_id,
                    channel_id=turn.channel_id,
                    start=start,
                    duration=end - start,
                    speaker_id=turn.speaker_id,
                    orthography=turn.orthography,
                    speaker_type=turn.speaker_type,
                    confidence=turn.confidence,
                    signal_lookahead_time=turn.signal_lookahead_time,
                )
            )
    return trimmed


def _group_by_recording(
    turns: Iterable[RTTMSpeakerTurn],
) -> dict[str, list[RTTMSpeakerTurn]]:
    """Group RTTM turns by recording name."""

    grouped: dict[str, list[RTTMSpeakerTurn]] = {}
    for turn in turns:
        grouped.setdefault(turn.recording_id, []).append(turn)
    return grouped


def _with_pyannote_metrics(
    local_report: RecordingEvaluation,
    reference: list[RTTMSpeakerTurn],
    hypothesis: list[RTTMSpeakerTurn],
    *,
    collar_seconds: float,
    use_pyannote: bool,
) -> RecordingEvaluation:
    """Replace local metric values with pyannote.metrics values when available."""

    if not use_pyannote or not _pyannote_metrics_available():
        return local_report

    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

        reference_annotation = Annotation(uri=local_report.recording_name)
        hypothesis_annotation = Annotation(uri=local_report.recording_name)
        for turn in reference:
            reference_annotation[Segment(turn.start, turn.end), turn.speaker_id] = turn.speaker_id
        for turn in hypothesis:
            hypothesis_annotation[Segment(turn.start, turn.end), turn.speaker_id] = turn.speaker_id

        der_components = DiarizationErrorRate(collar=collar_seconds)(
            reference_annotation,
            hypothesis_annotation,
            detailed=True,
            uri=local_report.recording_name,
        )
        jer_components = JaccardErrorRate(collar=collar_seconds)(
            reference_annotation,
            hypothesis_annotation,
            detailed=True,
            uri=local_report.recording_name,
        )
    except Exception as exc:
        LOGGER.warning("Falling back to local evaluator because pyannote failed: %s", exc)
        return local_report

    missed = float(der_components.get("missed detection", local_report.missed_speech))
    false_alarm = float(der_components.get("false alarm", local_report.false_alarm))
    confusion = float(der_components.get("confusion", local_report.speaker_confusion))
    total_error = missed + false_alarm + confusion
    der = float(der_components.get("diarization error rate", local_report.der))
    jer = float(jer_components.get("jaccard error rate", local_report.jer))
    return RecordingEvaluation(
        recording_name=local_report.recording_name,
        duration=local_report.duration,
        der=round(der, 6),
        jer=round(jer, 6),
        missed_speech=round(missed, 6),
        false_alarm=round(false_alarm, 6),
        speaker_confusion=round(confusion, 6),
        total_error=round(total_error, 6),
    )


def _build_report(
    recordings: list[RecordingEvaluation],
    *,
    backend: str,
) -> EvaluationReport:
    """Build overall and average metrics from recording reports."""

    if not recordings:
        raise ValueError("No recordings were evaluated.")
    duration = sum(item.duration for item in recordings)
    missed = sum(item.missed_speech for item in recordings)
    false_alarm = sum(item.false_alarm for item in recordings)
    confusion = sum(item.speaker_confusion for item in recordings)
    total_error = missed + false_alarm + confusion
    der = total_error / duration if duration else 0.0
    jer = float(np.mean([item.jer for item in recordings]))
    average_der = float(np.mean([item.der for item in recordings]))
    average_jer = jer
    recording_name = recordings[0].recording_name if len(recordings) == 1 else "ALL"
    return EvaluationReport(
        recording_name=recording_name,
        duration=round(duration, 6),
        der=round(der, 6),
        jer=round(jer, 6),
        missed_speech=round(missed, 6),
        false_alarm=round(false_alarm, 6),
        speaker_confusion=round(confusion, 6),
        total_error=round(total_error, 6),
        average_der=round(average_der, 6),
        average_jer=round(average_jer, 6),
        recordings=recordings,
        backend=backend,
    )


def _backend_name(use_pyannote: bool) -> str:
    """Return the active metric backend name."""

    if use_pyannote and _pyannote_metrics_available():
        return "pyannote.metrics-compatible"
    if use_pyannote:
        return "fallback-pyannote-unavailable"
    return "fallback"


def _pyannote_metrics_available() -> bool:
    """Return whether pyannote.metrics can be imported."""

    try:
        import pyannote.metrics  # noqa: F401
    except Exception:
        return False
    return True


def _save_plot_if_requested(
    figure: plt.Figure,
    output_path: str | Path | None,
) -> None:
    """Save a Matplotlib figure if requested."""

    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    LOGGER.info("Saved evaluation plot to %s", path)


def _parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""

    parser = argparse.ArgumentParser(description="Evaluate diarization RTTM files.")
    parser.add_argument("reference_rttm", type=Path)
    parser.add_argument("hypothesis_rttm", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--collar", type=float, default=None)
    parser.add_argument("--fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run one RTTM evaluation from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    report = evaluate_rttm(
        args.reference_rttm,
        args.hypothesis_rttm,
        collar_seconds=args.collar,
        use_pyannote=not args.fallback,
    )
    print(f"DER={report.der:.4f}\tJER={report.jer:.4f}")
    if args.json is not None:
        export_evaluation_json(report, args.json)
    if args.csv is not None:
        export_evaluation_csv(report, args.csv)
    if args.markdown is not None:
        export_evaluation_markdown(report, args.markdown)


if __name__ == "__main__":
    main()
