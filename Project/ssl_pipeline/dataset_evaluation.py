"""Dataset-level speaker diarization evaluation (DER/JER).

This module scores reference RTTM annotations that ship with an evaluation
dataset against hypothesis RTTM files produced by **this project's own**
diarization pipeline:

    audio -> preprocessing -> VAD -> segmentation -> WavLM embeddings ->
    embedding normalization -> spectral clustering -> speaker timeline -> RTTM

``pyannote.metrics`` is used only as the scoring library (optimal speaker
mapping, DER/JER accumulation). No pretrained diarization pipeline is ever used
to produce the hypothesis. When ``pyannote.metrics`` cannot be imported the
module falls back to the local evaluator in
:mod:`Project.ssl_pipeline.evaluation` and labels the backend accordingly.

Scoring protocol
----------------
* ``collar`` -- forgiveness collar in seconds applied around reference speaker
  boundaries. Default ``0.0`` (no forgiveness), matching
  ``evaluation_collar_seconds`` in the project configuration and the DIHARD
  style of scoring.
* ``skip_overlap`` -- whether overlapping reference speech is excluded from
  scoring. Default ``False``, so overlapped speech is scored.
* ``uem_mode`` -- ``"full_file"`` (default) scores the span from 0.0 to the end
  of the recording, extended when the reference annotation runs past the
  decoded audio duration so that reference speech is never silently cropped.
  ``"annotated"`` leaves the scored region to ``pyannote.metrics``, which
  approximates it with the union of the reference and hypothesis extents.
* ``speaker_count_mode`` -- ``"oracle"`` (default) reads the speaker count from
  the reference RTTM and gives it to the clustering stage. ``"estimated"`` lets
  the pipeline estimate the speaker count on its own.

The protocol actually applied to a run is always written into the generated
``evaluation.json`` so results stay reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.rttm import RTTMSpeakerTurn, RTTMWriter, parse_rttm

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR_NAME = "results"
SUMMARY_FILE_NAME = "summary.json"
EVALUATION_FILE_NAME = "evaluation.json"
HYPOTHESIS_FILE_NAME = "hypothesis.rttm"
PIPELINE_DIR_NAME = "pipeline"

#: Audio containers that may be paired with a reference RTTM.
AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".flac",
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".sph",
)

#: Stages of the diarization system under evaluation, recorded for reporting.
PIPELINE_STAGES: tuple[str, ...] = (
    "preprocessing",
    "voice activity detection",
    "segmentation",
    "WavLM embeddings",
    "embedding normalization",
    "spectral clustering",
    "speaker timeline reconstruction",
    "RTTM export",
)

LogFunction = Callable[[str], None]


class DiarizeFunction(Protocol):
    """Callable that runs a diarization system and returns a hypothesis RTTM."""

    def __call__(
        self,
        audio_path: Path,
        work_dir: Path,
        *,
        num_speakers: int | None,
    ) -> Path:
        """Diarize ``audio_path`` into ``work_dir`` and return the RTTM path."""


@dataclass(frozen=True)
class EvaluationProtocol:
    """Explicit, serializable scoring protocol."""

    collar: float = 0.0
    skip_overlap: bool = False
    uem_mode: str = "full_file"
    speaker_count_mode: str = "oracle"

    def __post_init__(self) -> None:
        if self.collar < 0.0:
            raise ValueError("collar must be non-negative.")
        if self.uem_mode not in {"full_file", "annotated"}:
            raise ValueError("uem_mode must be 'full_file' or 'annotated'.")
        if self.speaker_count_mode not in {"oracle", "estimated"}:
            raise ValueError("speaker_count_mode must be 'oracle' or 'estimated'.")

    @classmethod
    def from_config(cls, config: DiarizationConfig | None = None) -> "EvaluationProtocol":
        """Build a protocol from the project configuration defaults."""

        active = config or load_config()
        return cls(collar=float(active.evaluation_collar_seconds))

    def to_dict(self) -> dict[str, object]:
        """Return a serializable protocol description."""

        return {
            "collar": self.collar,
            "skip_overlap": self.skip_overlap,
            "uem_mode": self.uem_mode,
            "speaker_count_mode": self.speaker_count_mode,
        }


@dataclass(frozen=True)
class EvaluationItem:
    """One evaluation audio file paired with its reference annotation."""

    name: str
    audio_path: Path
    reference_rttm_path: Path
    lab_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a serializable description of the dataset item."""

        return {
            "name": self.name,
            "file": self.audio_path.name,
            "audio_path": self.audio_path.as_posix(),
            "reference_rttm": self.reference_rttm_path.as_posix(),
            "lab_file": self.lab_path.name if self.lab_path else None,
        }


@dataclass(frozen=True)
class MetricResult:
    """Scored DER/JER components for one recording."""

    der: float
    missed_speech: float
    false_alarm: float
    speaker_confusion: float
    scored_duration: float
    correct: float | None = None
    jer: float | None = None
    backend: str = "unknown"
    backend_version: str | None = None

    @property
    def total_error(self) -> float:
        """Return the summed error duration in seconds."""

        return self.missed_speech + self.false_alarm + self.speaker_confusion

    @property
    def jer_available(self) -> bool:
        """Return whether a Jaccard Error Rate was computed."""

        return self.jer is not None


@dataclass
class EvaluationOutcome:
    """Result of evaluating a whole dataset."""

    records: list[dict[str, object]] = field(default_factory=list)
    failures: list[dict[str, object]] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def discover_evaluation_items(
    evaluation_dir: str | Path = DEFAULT_EVALUATION_DIR,
) -> list[EvaluationItem]:
    """Discover audio files paired with a reference RTTM by common file stem.

    Generated output under ``<evaluation_dir>/results`` is ignored so previous
    runs can never be mistaken for dataset input.
    """

    root = Path(evaluation_dir)
    if not root.exists():
        return []

    results_root = (root / RESULTS_DIR_NAME).resolve()
    items: list[EvaluationItem] = []
    for audio_path in sorted(root.rglob("*")):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if _is_within(audio_path, results_root) or any(part.startswith("manual_test") for part in audio_path.parts):
            continue
        reference_rttm = audio_path.with_suffix(".rttm")
        if not reference_rttm.exists():
            LOGGER.debug("Skipping %s: no matching .rttm reference.", audio_path.name)
            continue
        lab_path = audio_path.with_suffix(".lab")
        items.append(
            EvaluationItem(
                name=audio_path.stem,
                audio_path=audio_path,
                reference_rttm_path=reference_rttm,
                lab_path=lab_path if lab_path.exists() else None,
            )
        )
    return items


def find_unpaired_audio(
    evaluation_dir: str | Path = DEFAULT_EVALUATION_DIR,
) -> list[Path]:
    """Return evaluation audio files that have no matching reference RTTM."""

    root = Path(evaluation_dir)
    if not root.exists():
        return []
    results_root = (root / RESULTS_DIR_NAME).resolve()
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and not _is_within(path, results_root)
        and not any(part.startswith("manual_test") for part in path.parts)
        and not path.with_suffix(".rttm").exists()
    ]


def describe_evaluation_item(item: EvaluationItem) -> dict[str, object]:
    """Describe one dataset item without running diarization."""

    description = item.to_dict()
    try:
        turns = load_reference_turns(item.reference_rttm_path)
        description["reference_speakers"] = len({turn.speaker_id for turn in turns})
        description["reference_speaker_labels"] = sorted({turn.speaker_id for turn in turns})
        description["reference_turns"] = len(turns)
        description["reference_speech_duration"] = round(
            sum(turn.duration for turn in turns), 3
        )
        description["recording_id"] = turns[0].recording_id if turns else None
        description["reference_error"] = None
    except (OSError, ValueError) as exc:
        description["reference_speakers"] = None
        description["reference_error"] = str(exc)

    duration = read_audio_duration(item.audio_path)
    description["duration"] = None if duration is None else round(duration, 3)
    return description


def read_audio_duration(audio_path: str | Path) -> float | None:
    """Return audio duration in seconds, or ``None`` when it cannot be read."""

    try:
        import soundfile as sf

        return float(sf.info(str(audio_path)).duration)
    except Exception as exc:  # pragma: no cover - depends on local codecs
        LOGGER.warning("Could not read audio duration for %s: %s", audio_path, exc)
        return None


def parse_lab_file(lab_path: str | Path) -> list[tuple[float, float, str]]:
    """Parse a ``.lab`` annotation into ``(start, end, label)`` rows.

    The ``.lab`` file is optional metadata. It is never required to compute DER.
    """

    rows: list[tuple[float, float, str]] = []
    for line_number, raw_line in enumerate(
        Path(lab_path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed .lab line {line_number}: expected at least 2 fields.")
        try:
            start = float(fields[0])
            end = float(fields[1])
        except ValueError as exc:
            raise ValueError(f"Malformed .lab line {line_number}: invalid timestamp.") from exc
        rows.append((start, end, fields[2] if len(fields) > 2 else ""))
    return rows


# ---------------------------------------------------------------------------
# RTTM loading and alignment
# ---------------------------------------------------------------------------


def load_reference_turns(reference_rttm: str | Path) -> list[RTTMSpeakerTurn]:
    """Load a reference RTTM, allowing overlapping speech.

    Reference annotations legitimately contain simultaneous speakers, so the
    strict non-overlap validation used for generated RTTM output is relaxed.
    """

    path = Path(reference_rttm)
    if not path.exists():
        raise FileNotFoundError(f"Reference RTTM not found: {path}")
    turns = parse_rttm(path, allow_overlap=True)
    if not turns:
        raise ValueError(f"Reference RTTM is empty: {path}")
    return turns


def load_hypothesis_turns(hypothesis_rttm: str | Path) -> list[RTTMSpeakerTurn]:
    """Load a hypothesis RTTM produced by the project pipeline."""

    path = Path(hypothesis_rttm)
    if not path.exists():
        raise FileNotFoundError(f"Hypothesis RTTM not found: {path}")
    turns = parse_rttm(path, allow_overlap=True)
    if not turns:
        raise ValueError(f"Hypothesis RTTM is empty: {path}")
    return turns


def align_recording_id(
    turns: Iterable[RTTMSpeakerTurn],
    recording_id: str,
) -> list[RTTMSpeakerTurn]:
    """Relabel turns so they share one recording identifier.

    Reference datasets name the recording independently of the audio file name
    (``LDC2021S10.flac`` carries recording id ``DH_0052``). Scoring tools match
    reference and hypothesis by recording id, so the hypothesis is relabelled to
    the reference identifier. The reference file itself is never modified.
    """

    return [replace(turn, recording_id=recording_id) for turn in turns]


def write_hypothesis_rttm(
    turns: Sequence[RTTMSpeakerTurn],
    output_path: str | Path,
    *,
    config: DiarizationConfig | None = None,
) -> Path:
    """Write hypothesis turns to an RTTM file, preserving their recording id."""

    if not turns:
        raise ValueError("Cannot write an empty hypothesis RTTM.")
    active = config or load_config()
    writer = RTTMWriter(
        active,
        recording_id=turns[0].recording_id,
        channel_id=turns[0].channel_id,
        decimal_places=active.rttm_decimal_places,
    )
    text = "\n".join(writer.format_turn(turn) for turn in turns) + "\n"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    LOGGER.info("Saved hypothesis RTTM to %s", path)
    return path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class DiarizationScorer:
    """Score reference/hypothesis pairs and accumulate global metrics.

    A single scorer instance is reused across a dataset so the overall DER is a
    properly accumulated (duration-weighted) figure rather than a mean of
    per-file percentages.
    """

    def __init__(self, protocol: EvaluationProtocol | None = None) -> None:
        self.protocol = protocol or EvaluationProtocol()
        self._pyannote = _build_pyannote_metrics(self.protocol)
        self._totals = {"total": 0.0, "error": 0.0}
        self._jer_values: list[float] = []

    @property
    def backend(self) -> str:
        """Return the active scoring backend name."""

        return "pyannote.metrics" if self._pyannote is not None else "local-fallback"

    @property
    def backend_version(self) -> str | None:
        """Return the scoring backend version when known."""

        return self._pyannote["version"] if self._pyannote is not None else None

    def score(
        self,
        reference_turns: Sequence[RTTMSpeakerTurn],
        hypothesis_turns: Sequence[RTTMSpeakerTurn],
        *,
        uri: str,
        uem_end: float | None = None,
    ) -> MetricResult:
        """Score one recording and accumulate it into the global metrics."""

        if not reference_turns:
            raise ValueError("Reference RTTM is empty.")
        if not hypothesis_turns:
            raise ValueError("Hypothesis RTTM is empty.")

        result = (
            self._score_with_pyannote(reference_turns, hypothesis_turns, uri, uem_end)
            if self._pyannote is not None
            else self._score_with_fallback(reference_turns, hypothesis_turns, uri, uem_end)
        )
        self._totals["total"] += result.scored_duration
        self._totals["error"] += result.total_error
        if result.jer is not None:
            self._jer_values.append(result.jer)
        return result

    def overall_der(self) -> float | None:
        """Return the accumulated (duration-weighted) DER across all scored files."""

        if self._pyannote is not None:
            metric = self._pyannote["der"]
            try:
                if len(metric) == 0:
                    return None
                return float(abs(metric))
            except (TypeError, ZeroDivisionError):  # pragma: no cover - defensive
                pass
        if self._totals["total"] <= 0.0:
            return None
        return self._totals["error"] / self._totals["total"]

    def overall_jer(self) -> float | None:
        """Return the accumulated JER across all scored files."""

        if self._pyannote is not None and self._pyannote["jer"] is not None:
            metric = self._pyannote["jer"]
            try:
                if len(metric) == 0:
                    return None
                return float(abs(metric))
            except (TypeError, ZeroDivisionError):  # pragma: no cover - defensive
                pass
        if not self._jer_values:
            return None
        return sum(self._jer_values) / len(self._jer_values)

    def _score_with_pyannote(
        self,
        reference_turns: Sequence[RTTMSpeakerTurn],
        hypothesis_turns: Sequence[RTTMSpeakerTurn],
        uri: str,
        uem_end: float | None,
    ) -> MetricResult:
        """Score with ``pyannote.metrics`` using optimal speaker mapping."""

        assert self._pyannote is not None
        annotation_factory = self._pyannote["annotation"]
        reference = annotation_factory(reference_turns, uri)
        hypothesis = annotation_factory(hypothesis_turns, uri)
        kwargs: dict[str, object] = {"detailed": True, "uri": uri}
        uem = self._pyannote["uem"](uri, uem_end) if uem_end is not None else None
        if uem is not None:
            kwargs["uem"] = uem

        components = self._pyannote["der"](reference, hypothesis, **kwargs)
        jer_value: float | None = None
        jer_metric = self._pyannote["jer"]
        if jer_metric is not None:
            try:
                jer_components = jer_metric(reference, hypothesis, **kwargs)
                jer_value = float(jer_components["jaccard error rate"])
            except Exception as exc:  # pragma: no cover - version dependent
                LOGGER.warning("JER could not be computed: %s", exc)
                self._pyannote["jer"] = None

        return MetricResult(
            der=float(components["diarization error rate"]),
            missed_speech=float(components["missed detection"]),
            false_alarm=float(components["false alarm"]),
            speaker_confusion=float(components["confusion"]),
            scored_duration=float(components["total"]),
            correct=float(components["correct"]) if "correct" in components else None,
            jer=jer_value,
            backend="pyannote.metrics",
            backend_version=self._pyannote["version"],
        )

    def _score_with_fallback(
        self,
        reference_turns: Sequence[RTTMSpeakerTurn],
        hypothesis_turns: Sequence[RTTMSpeakerTurn],
        uri: str,
        uem_end: float | None = None,
    ) -> MetricResult:
        """Score with the project's local evaluator when pyannote is missing."""

        from Project.ssl_pipeline.evaluation import DiarizationEvaluator

        if uem_end is not None:
            def clip(turn: RTTMSpeakerTurn) -> RTTMSpeakerTurn | None:
                if turn.start >= uem_end or turn.end <= 0.0:
                    return None
                start = max(0.0, turn.start)
                end = min(uem_end, turn.end)
                if end <= start:
                    return None
                return replace(turn, start=start, duration=end - start)

            clipped_ref = [c for t in reference_turns if (c := clip(t)) is not None]
            clipped_hyp = [c for t in hypothesis_turns if (c := clip(t)) is not None]
        else:
            clipped_ref = list(reference_turns)
            clipped_hyp = list(hypothesis_turns)

        evaluator = DiarizationEvaluator(
            collar_seconds=self.protocol.collar,
            use_pyannote=False,
            skip_overlap=self.protocol.skip_overlap,
        )
        report = evaluator.evaluate_turns(
            align_recording_id(clipped_ref, uri),
            align_recording_id(clipped_hyp, uri),
        )
        return MetricResult(
            der=report.der,
            missed_speech=report.missed_speech,
            false_alarm=report.false_alarm,
            speaker_confusion=report.speaker_confusion,
            scored_duration=report.duration,
            jer=report.jer,
            backend="local-fallback",
        )


def _build_pyannote_metrics(protocol: EvaluationProtocol) -> dict[str, object] | None:
    """Build reusable pyannote.metrics objects, or ``None`` when unavailable."""

    try:
        import pyannote.metrics as pyannote_metrics
        from pyannote.core import Annotation, Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate
    except Exception as exc:
        LOGGER.warning("pyannote.metrics unavailable, using local evaluator: %s", exc)
        return None

    def build_annotation(turns: Sequence[RTTMSpeakerTurn], uri: str):
        """Build a pyannote Annotation with one track per RTTM turn."""

        annotation = Annotation(uri=uri)
        for index, turn in enumerate(turns):
            if turn.duration <= 0.0:
                continue
            annotation[Segment(turn.start, turn.end), index] = turn.speaker_id
        return annotation

    def build_uem(uri: str, end: float):
        """Build an explicit scored region from 0.0 to ``end``."""

        return Timeline([Segment(0.0, end)], uri=uri)

    jer_metric = None
    try:
        from pyannote.metrics.diarization import JaccardErrorRate

        jer_metric = JaccardErrorRate(
            collar=protocol.collar,
            skip_overlap=protocol.skip_overlap,
        )
    except Exception as exc:  # pragma: no cover - version dependent
        LOGGER.warning("JaccardErrorRate unavailable: %s", exc)

    return {
        "der": DiarizationErrorRate(
            collar=protocol.collar,
            skip_overlap=protocol.skip_overlap,
        ),
        "jer": jer_metric,
        "annotation": build_annotation,
        "uem": build_uem,
        "version": getattr(pyannote_metrics, "__version__", None),
    }


# ---------------------------------------------------------------------------
# Diarization driver
# ---------------------------------------------------------------------------


def run_project_diarization(
    audio_path: Path,
    work_dir: Path,
    *,
    num_speakers: int | None,
) -> Path:
    """Run this project's diarization pipeline and return its hypothesis RTTM.

    This is the only place the hypothesis is produced. It calls
    :func:`Project.run_diarization.run_pipeline`, i.e. the existing
    WavLM + spectral clustering system, and never a pretrained diarization
    pipeline.
    """

    from Project.run_diarization import run_pipeline

    result = run_pipeline(
        audio_path,
        output_dir=work_dir,
        num_speakers=num_speakers,
        oracle_num_speakers=None if num_speakers is None else True,
        save_json=True,
        save_csv=True,
        save_rttm=True,
    )
    if result.rttm_path is None:
        raise RuntimeError("Diarization pipeline did not produce an RTTM file.")
    return Path(result.rttm_path)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_item(
    item: EvaluationItem,
    *,
    results_dir: Path,
    scorer: DiarizationScorer,
    protocol: EvaluationProtocol,
    diarize: DiarizeFunction = run_project_diarization,
    log: LogFunction = LOGGER.info,
    config: DiarizationConfig | None = None,
) -> dict[str, object]:
    """Evaluate one dataset item and write its generated artifacts."""

    active_config = config or load_config()
    output_dir = results_dir / item.name
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Found evaluation file: {item.audio_path.name}")
    reference_turns = load_reference_turns(item.reference_rttm_path)
    recording_id = reference_turns[0].recording_id
    reference_speakers = sorted({turn.speaker_id for turn in reference_turns})
    log(
        f"Found reference RTTM: {item.reference_rttm_path.name} "
        f"({len(reference_turns)} turns, {len(reference_speakers)} speakers)"
    )
    if item.lab_path is not None:
        log(f"Found optional .lab annotation: {item.lab_path.name}")

    num_speakers = len(reference_speakers) if protocol.speaker_count_mode == "oracle" else None
    if num_speakers is not None:
        log(f"Speaker count mode: oracle ({num_speakers} speakers from reference)")
    else:
        log("Speaker count mode: estimated by the clustering stage")

    log(f"Running diarization pipeline on {item.audio_path.name}")
    pipeline_dir = output_dir / PIPELINE_DIR_NAME
    raw_hypothesis = diarize(item.audio_path, pipeline_dir, num_speakers=num_speakers)

    hypothesis_turns = align_recording_id(
        load_hypothesis_turns(raw_hypothesis),
        recording_id,
    )
    hypothesis_path = write_hypothesis_rttm(
        hypothesis_turns,
        output_dir / HYPOTHESIS_FILE_NAME,
        config=active_config,
    )
    detected_speakers = sorted({turn.speaker_id for turn in hypothesis_turns})
    log(
        f"Hypothesis RTTM generated: {hypothesis_path.name} "
        f"({len(hypothesis_turns)} turns, {len(detected_speakers)} speakers)"
    )

    audio_duration = read_audio_duration(item.audio_path)
    uem_end = _resolve_uem_end(
        protocol,
        audio_duration=audio_duration,
        reference_turns=reference_turns,
        hypothesis_turns=hypothesis_turns,
    )

    log(f"Calculating DER (collar={protocol.collar}s, skip_overlap={protocol.skip_overlap})")
    metrics = scorer.score(
        reference_turns,
        hypothesis_turns,
        uri=recording_id,
        uem_end=uem_end,
    )
    log(
        f"DER calculation completed for {item.name}: "
        f"DER={metrics.der * 100:.2f}%"
        + (f", JER={metrics.jer * 100:.2f}%" if metrics.jer is not None else ", JER unavailable")
    )

    record = _build_record(
        item,
        metrics=metrics,
        protocol=protocol,
        recording_id=recording_id,
        reference_turns=reference_turns,
        hypothesis_turns=hypothesis_turns,
        reference_speakers=reference_speakers,
        detected_speakers=detected_speakers,
        audio_duration=audio_duration,
        uem_end=uem_end,
        hypothesis_path=hypothesis_path,
        results_dir=results_dir,
        config=active_config,
    )

    evaluation_path = output_dir / EVALUATION_FILE_NAME
    evaluation_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"Saved evaluation result: {evaluation_path.name}")
    return record


def run_dataset_evaluation(
    evaluation_dir: str | Path = DEFAULT_EVALUATION_DIR,
    *,
    protocol: EvaluationProtocol | None = None,
    diarize: DiarizeFunction = run_project_diarization,
    log: LogFunction = LOGGER.info,
    only: Sequence[str] | None = None,
    config: DiarizationConfig | None = None,
) -> EvaluationOutcome:
    """Evaluate every paired file in ``evaluation_dir`` and write results."""

    root = Path(evaluation_dir)
    active_config = config or load_config()
    active_protocol = protocol or EvaluationProtocol.from_config(active_config)
    results_dir = root / RESULTS_DIR_NAME

    log("Evaluation started")
    if not root.exists():
        raise FileNotFoundError(
            f"Evaluation directory not found: {root}. "
            "Create it and place <name>.flac and <name>.rttm files inside."
        )

    items = discover_evaluation_items(root)
    if only:
        wanted = set(only)
        items = [item for item in items if item.name in wanted]
    for orphan in find_unpaired_audio(root):
        log(f"Skipping {orphan.name}: no matching .rttm reference annotation")
    if not items:
        raise FileNotFoundError(
            f"No audio/reference RTTM pairs found in {root}. "
            "Each audio file needs a reference RTTM with the same file stem."
        )
    log(f"Discovered {len(items)} evaluation file(s) with reference annotations")

    scorer = DiarizationScorer(active_protocol)
    log(f"Scoring backend: {scorer.backend}" + (f" {scorer.backend_version}" if scorer.backend_version else ""))

    outcome = EvaluationOutcome()
    results_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        log(f"Evaluating {item.name} ({index}/{len(items)})")
        try:
            outcome.records.append(
                evaluate_item(
                    item,
                    results_dir=results_dir,
                    scorer=scorer,
                    protocol=active_protocol,
                    diarize=diarize,
                    log=log,
                    config=active_config,
                )
            )
        except Exception as exc:
            LOGGER.exception("Evaluation failed for %s", item.name)
            outcome.failures.append(
                {
                    "name": item.name,
                    "file": item.audio_path.name,
                    "error": str(exc) or exc.__class__.__name__,
                }
            )
            log(f"Evaluation failed for {item.name}: {exc}")

    outcome.summary = build_summary(
        outcome.records,
        failures=outcome.failures,
        scorer=scorer,
        protocol=active_protocol,
        evaluation_dir=root,
        config=active_config,
    )
    summary_path = results_dir / SUMMARY_FILE_NAME
    summary_path.write_text(json.dumps(outcome.summary, indent=2), encoding="utf-8")
    log(f"Saved evaluation summary: {summary_path.name}")

    if not outcome.records:
        log("Evaluation completed with no successful files")
    else:
        overall = outcome.summary.get("overall_der_percent")
        log(f"Evaluation completed ({len(outcome.records)} file(s), overall DER {overall}%)")
    return outcome


def build_summary(
    records: Sequence[dict[str, object]],
    *,
    failures: Sequence[dict[str, object]],
    scorer: DiarizationScorer,
    protocol: EvaluationProtocol,
    evaluation_dir: Path,
    config: DiarizationConfig | None = None,
) -> dict[str, object]:
    """Aggregate per-file records into a labelled dataset summary."""

    der_values = [float(row["der"]) for row in records if row.get("der") is not None]
    jer_values = [float(row["jer"]) for row in records if row.get("jer") is not None]
    durations = [float(row["duration"]) for row in records if row.get("duration") is not None]
    scored = [float(row["scored_duration"]) for row in records if row.get("scored_duration") is not None]
    reference_counts = [
        int(row["reference_speakers"]) for row in records if row.get("reference_speakers") is not None
    ]
    detected_counts = [
        int(row["detected_speakers"]) for row in records if row.get("detected_speakers") is not None
    ]

    overall_der = scorer.overall_der()
    overall_jer = scorer.overall_jer()
    lowest = min(records, key=lambda row: row["der"], default=None) if der_values else None
    highest = max(records, key=lambda row: row["der"], default=None) if der_values else None

    summary: dict[str, object] = {
        "generated_at": _now_iso(),
        "evaluation_dir": evaluation_dir.as_posix(),
        "metrics_backend": scorer.backend,
        "metrics_backend_version": scorer.backend_version,
        "protocol": protocol.to_dict(),
        "system": _system_description(config),
        "file_count": len(records),
        "failed_count": len(failures),
        "failures": list(failures),
        "total_duration": _rounded(sum(durations)) if durations else None,
        "total_scored_duration": _rounded(sum(scored)) if scored else None,
        # Duration-weighted metric accumulated over every scored file. This is
        # the figure to report as the dataset result.
        "overall_der": _rounded(overall_der, 6),
        "overall_der_percent": _rounded(_percent(overall_der)),
        "overall_jer": _rounded(overall_jer, 6),
        "overall_jer_percent": _rounded(_percent(overall_jer)),
        # Unweighted mean of per-file rates. Kept separate because averaging
        # percentages over files of different length is not the dataset DER.
        "average_der": _rounded(_mean(der_values), 6),
        "average_der_percent": _rounded(_percent(_mean(der_values))),
        "average_jer": _rounded(_mean(jer_values), 6),
        "average_jer_percent": _rounded(_percent(_mean(jer_values))),
        "min_der": _rounded(min(der_values), 6) if der_values else None,
        "min_der_percent": _rounded(_percent(min(der_values))) if der_values else None,
        "min_der_file": lowest["file"] if lowest else None,
        "max_der": _rounded(max(der_values), 6) if der_values else None,
        "max_der_percent": _rounded(_percent(max(der_values))) if der_values else None,
        "max_der_file": highest["file"] if highest else None,
        "average_reference_speakers": _rounded(_mean(reference_counts), 2),
        "average_detected_speakers": _rounded(_mean(detected_counts), 2),
        "total_false_alarm": _rounded(sum(float(row.get("false_alarm") or 0.0) for row in records)),
        "total_missed_speech": _rounded(sum(float(row.get("missed_speech") or 0.0) for row in records)),
        "total_speaker_confusion": _rounded(
            sum(float(row.get("speaker_confusion") or 0.0) for row in records)
        ),
        "jer_available": bool(jer_values),
        "files": [_summary_row(row) for row in records],
    }
    return summary


def load_results(
    evaluation_dir: str | Path = DEFAULT_EVALUATION_DIR,
) -> dict[str, object]:
    """Load previously generated evaluation results from disk."""

    results_dir = Path(evaluation_dir) / RESULTS_DIR_NAME
    summary_path = results_dir / SUMMARY_FILE_NAME
    records: list[dict[str, object]] = []
    if results_dir.exists():
        for child in sorted(results_dir.iterdir()):
            evaluation_path = child / EVALUATION_FILE_NAME
            if child.is_dir() and evaluation_path.exists():
                try:
                    records.append(json.loads(evaluation_path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    LOGGER.warning("Ignoring unreadable evaluation JSON: %s", evaluation_path)
    summary: dict[str, object] | None = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring unreadable summary JSON: %s", summary_path)
    return {"summary": summary, "results": records}


def clear_results(
    evaluation_dir: str | Path = DEFAULT_EVALUATION_DIR,
) -> list[str]:
    """Delete generated evaluation output only.

    Reference dataset files placed directly in ``evaluation/`` (audio, ``.rttm``,
    ``.lab``) are never touched: only ``evaluation/results`` is removed.
    """

    root = Path(evaluation_dir)
    results_dir = root / RESULTS_DIR_NAME
    if not results_dir.exists():
        return []

    resolved_root = root.resolve()
    resolved_results = results_dir.resolve()
    if resolved_results == resolved_root or resolved_root not in resolved_results.parents:
        raise ValueError(f"Refusing to clear unexpected path: {resolved_results}")

    removed = [
        path.relative_to(results_dir).as_posix()
        for path in sorted(results_dir.rglob("*"))
        if path.is_file()
    ]
    shutil.rmtree(results_dir)
    LOGGER.info("Cleared %d generated evaluation file(s) from %s", len(removed), results_dir)
    return removed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_record(
    item: EvaluationItem,
    *,
    metrics: MetricResult,
    protocol: EvaluationProtocol,
    recording_id: str,
    reference_turns: Sequence[RTTMSpeakerTurn],
    hypothesis_turns: Sequence[RTTMSpeakerTurn],
    reference_speakers: Sequence[str],
    detected_speakers: Sequence[str],
    audio_duration: float | None,
    uem_end: float | None,
    hypothesis_path: Path,
    results_dir: Path,
    config: DiarizationConfig | None = None,
) -> dict[str, object]:
    """Build the per-file ``evaluation.json`` payload."""

    hypothesis_speech = sum(turn.duration for turn in hypothesis_turns)
    record: dict[str, object] = {
        "file": item.audio_path.name,
        "name": item.name,
        "recording_id": recording_id,
        "evaluated_at": _now_iso(),
        "duration": _rounded(audio_duration),
        "scored_duration": _rounded(metrics.scored_duration),
        "reference_speech_duration": _rounded(sum(turn.duration for turn in reference_turns)),
        "hypothesis_speech_duration": _rounded(hypothesis_speech),
        # Share of the recording the pipeline labelled as speech. A ratio near
        # 1.0 means voice activity detection found almost no silence, which
        # shows up as false alarm in the DER breakdown.
        "hypothesis_speech_ratio": (
            _rounded(hypothesis_speech / audio_duration, 4)
            if audio_duration
            else None
        ),
        "reference_speakers": len(reference_speakers),
        "detected_speakers": len(detected_speakers),
        "reference_speaker_labels": list(reference_speakers),
        "detected_speaker_labels": list(detected_speakers),
        "reference_turns": len(reference_turns),
        "hypothesis_turns": len(hypothesis_turns),
        "der": _rounded(metrics.der, 6),
        "der_percent": _rounded(_percent(metrics.der)),
        "jer": _rounded(metrics.jer, 6),
        "jer_percent": _rounded(_percent(metrics.jer)),
        "jer_available": metrics.jer_available,
        "false_alarm": _rounded(metrics.false_alarm),
        "missed_speech": _rounded(metrics.missed_speech),
        "speaker_confusion": _rounded(metrics.speaker_confusion),
        "total_error": _rounded(metrics.total_error),
        "metrics_backend": metrics.backend,
        "metrics_backend_version": metrics.backend_version,
        "protocol": {**protocol.to_dict(), "uem_start": 0.0, "uem_end": _rounded(uem_end)},
        "system": _system_description(config),
        "reference_rttm": item.reference_rttm_path.as_posix(),
        "hypothesis_rttm": _relative_or_absolute(hypothesis_path, results_dir),
        "lab_file": item.lab_path.name if item.lab_path else None,
    }
    if metrics.correct is not None:
        record["correct"] = _rounded(metrics.correct)
    return record


def _summary_row(record: dict[str, object]) -> dict[str, object]:
    """Return the compact per-file row used for report tables."""

    keys = (
        "name",
        "file",
        "duration",
        "reference_speakers",
        "detected_speakers",
        "der",
        "der_percent",
        "jer",
        "jer_percent",
        "false_alarm",
        "missed_speech",
        "speaker_confusion",
    )
    return {key: record.get(key) for key in keys}


def _system_description(config: DiarizationConfig | None = None) -> dict[str, object]:
    """Describe the diarization system that produced the hypothesis.

    The pipeline settings that most affect DER are recorded alongside the
    metrics so a result can be interpreted and reproduced later.
    """

    description: dict[str, object] = {
        "hypothesis_source": "project diarization pipeline (WavLM + spectral clustering)",
        "stages": list(PIPELINE_STAGES),
        "pretrained_diarization_pipeline_used": False,
        "training_performed": False,
    }
    if config is not None:
        description["settings"] = {
            "vad_backend": config.vad_backend,
            "embedding_model": config.embedding_model_name,
            "embedding_normalization": config.embedding_normalization_method,
            "clustering_algorithm": config.clustering_algorithm,
            "clustering_affinity_metric": config.clustering_affinity_metric,
            "segment_window_seconds": config.segment_window_seconds,
            "segment_hop_seconds": config.segment_hop_seconds,
        }
    return description


def _resolve_uem_end(
    protocol: EvaluationProtocol,
    *,
    audio_duration: float | None,
    reference_turns: Sequence[RTTMSpeakerTurn],
    hypothesis_turns: Sequence[RTTMSpeakerTurn],
) -> float | None:
    """Return the end of the scored region for the configured UEM mode."""

    if protocol.uem_mode != "full_file":
        return None
    candidates = [
        value
        for value in (
            audio_duration,
            max((turn.end for turn in reference_turns), default=None),
            max((turn.end for turn in hypothesis_turns), default=None),
        )
        if value is not None
    ]
    if not candidates:
        return None
    return float(max(candidates))


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` lives inside ``root``."""

    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - defensive
        return False
    return resolved == root or root in resolved.parents


def _relative_or_absolute(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - defensive
        return path.as_posix()


def _percent(value: float | None) -> float | None:
    """Convert a rate to a percentage."""

    return None if value is None else value * 100.0


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean, or ``None`` for an empty sequence."""

    return sum(values) / len(values) if values else None


def _rounded(value: float | None, digits: int = 3) -> float | None:
    """Round a value while preserving ``None``."""

    return None if value is None else round(float(value), digits)


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_args() -> argparse.Namespace:
    """Parse dataset evaluation CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate the project diarization pipeline against reference RTTM files.",
    )
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--collar", type=float, default=None, help="Forgiveness collar in seconds.")
    parser.add_argument("--skip-overlap", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--uem-mode", choices=["full_file", "annotated"], default=None)
    parser.add_argument("--speaker-count-mode", choices=["oracle", "estimated"], default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Evaluate selected file stems only.")
    return parser.parse_args()


def main() -> None:
    """Run the dataset evaluation from the command line."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    config = load_config()
    protocol = EvaluationProtocol.from_config(config)
    overrides: dict[str, object] = {}
    if args.collar is not None:
        overrides["collar"] = args.collar
    if args.skip_overlap is not None:
        overrides["skip_overlap"] = args.skip_overlap
    if args.uem_mode is not None:
        overrides["uem_mode"] = args.uem_mode
    if args.speaker_count_mode is not None:
        overrides["speaker_count_mode"] = args.speaker_count_mode
    if overrides:
        protocol = replace(protocol, **overrides)

    outcome = run_dataset_evaluation(
        args.evaluation_dir,
        protocol=protocol,
        log=print,
        only=args.only,
        config=config,
    )
    summary = outcome.summary
    print(
        f"Files={summary.get('file_count')} "
        f"OverallDER={summary.get('overall_der_percent')}% "
        f"OverallJER={summary.get('overall_jer_percent')}%"
    )
    if outcome.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
