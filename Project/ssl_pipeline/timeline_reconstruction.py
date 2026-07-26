"""Speaker timeline reconstruction from clustered speech segments.

This module converts ``ClusteredEmbedding`` objects into chronological speaker
intervals. It does not write RTTM files, compute DER/JER, or implement a
baseline system.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.spectral_clustering import ClusteredEmbedding

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerTimeline:
    """A clean chronological interval assigned to one speaker."""

    speaker_id: str
    start: float
    end: float
    duration: float
    segment_count: int

    def to_dict(self) -> dict[str, str | float | int]:
        """Return a serializable timeline row."""

        return asdict(self)


@dataclass
class _TimelineFragment:
    """Mutable internal timeline fragment."""

    speaker_id: str
    start: float
    end: float
    segment_count: int

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerTimelineReconstructor:
    """Post-process clustered embeddings into speaker timeline intervals."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        merge_adjacent: bool | None = None,
        merge_silence_threshold_seconds: float | None = None,
        min_duration_seconds: float | None = None,
        label_smoothing_enabled: bool | None = None,
    ) -> None:
        self.config = config or load_config()
        self.merge_adjacent = (
            self.config.timeline_merge_adjacent
            if merge_adjacent is None
            else bool(merge_adjacent)
        )
        self.merge_silence_threshold_seconds = (
            self.config.timeline_merge_silence_threshold_seconds
            if merge_silence_threshold_seconds is None
            else float(merge_silence_threshold_seconds)
        )
        self.min_duration_seconds = (
            self.config.timeline_min_duration_seconds
            if min_duration_seconds is None
            else float(min_duration_seconds)
        )
        self.label_smoothing_enabled = (
            self.config.timeline_label_smoothing_enabled
            if label_smoothing_enabled is None
            else bool(label_smoothing_enabled)
        )
        _validate_timeline_config(
            self.merge_silence_threshold_seconds,
            self.min_duration_seconds,
        )

    def reconstruct(
        self,
        clustered_embeddings: Iterable[ClusteredEmbedding],
    ) -> list[SpeakerTimeline]:
        """Create a chronological speaker timeline from clustered embeddings."""

        fragments = _clustered_to_fragments(clustered_embeddings)
        if not fragments:
            return []

        if self.label_smoothing_enabled:
            fragments = _smooth_isolated_labels(fragments)

        fragments = _resolve_overlaps(fragments)
        fragments = _merge_fragments(
            fragments,
            merge_adjacent=self.merge_adjacent,
            silence_threshold=self.merge_silence_threshold_seconds,
        )
        fragments = _remove_tiny_fragments(fragments, self.min_duration_seconds)
        fragments = _merge_fragments(
            fragments,
            merge_adjacent=self.merge_adjacent,
            silence_threshold=self.merge_silence_threshold_seconds,
        )

        timeline = [_fragment_to_timeline(fragment) for fragment in fragments]
        LOGGER.info("Reconstructed %d speaker timeline intervals", len(timeline))
        return timeline


def reconstruct_speaker_timeline(
    clustered_embeddings: Iterable[ClusteredEmbedding],
    *,
    config_path: str | Path | None = None,
    merge_adjacent: bool | None = None,
    merge_silence_threshold_seconds: float | None = None,
    min_duration_seconds: float | None = None,
    label_smoothing_enabled: bool | None = None,
) -> list[SpeakerTimeline]:
    """Convenience function for speaker timeline reconstruction."""

    config = load_config(config_path) if config_path else load_config()
    reconstructor = SpeakerTimelineReconstructor(
        config,
        merge_adjacent=merge_adjacent,
        merge_silence_threshold_seconds=merge_silence_threshold_seconds,
        min_duration_seconds=min_duration_seconds,
        label_smoothing_enabled=label_smoothing_enabled,
    )
    return reconstructor.reconstruct(clustered_embeddings)


def export_timeline_json(
    timeline: Iterable[SpeakerTimeline],
    output_path: str | Path,
) -> None:
    """Export speaker timeline intervals as JSON."""

    rows = [item.to_dict() for item in timeline]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    LOGGER.info("Saved speaker timeline JSON to %s", path)


def export_timeline_csv(
    timeline: Iterable[SpeakerTimeline],
    output_path: str | Path,
) -> None:
    """Export speaker timeline intervals as CSV."""

    rows = [item.to_dict() for item in timeline]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["speaker_id", "start", "end", "duration", "segment_count"],
        )
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved speaker timeline CSV to %s", path)


def export_timeline_txt(
    timeline: Iterable[SpeakerTimeline],
    output_path: str | Path,
) -> None:
    """Export speaker timeline intervals as human-readable text."""

    lines = [
        (
            f"{item.speaker_id}\t{item.start:.2f}-{item.end:.2f}\t"
            f"duration={item.duration:.2f}s\tsegments={item.segment_count}"
        )
        for item in timeline
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    LOGGER.info("Saved speaker timeline text to %s", path)


def plot_timeline(
    timeline: Iterable[SpeakerTimeline],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot chronological speaker intervals on a horizontal timeline."""

    items = list(timeline)
    speaker_ids = _speaker_order(items)
    speaker_to_y = {speaker_id: index for index, speaker_id in enumerate(speaker_ids)}

    figure, axis = plt.subplots(figsize=(10, 3 + 0.4 * max(1, len(speaker_ids))))
    for item in items:
        axis.barh(
            speaker_to_y[item.speaker_id],
            item.duration,
            left=item.start,
            height=0.35,
            align="center",
        )
    axis.set_yticks(range(len(speaker_ids)))
    axis.set_yticklabels(speaker_ids)
    axis.set_xlabel("Time (seconds)")
    axis.set_title("Speaker Timeline")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_speaker_activity_chart(
    timeline: Iterable[SpeakerTimeline],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot total speaking duration per speaker."""

    totals = _speaker_duration_totals(timeline)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(list(totals.keys()), list(totals.values()))
    axis.set_title("Speaker Activity")
    axis.set_xlabel("Speaker")
    axis.set_ylabel("Total duration (seconds)")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def plot_speaker_duration_histogram(
    timeline: Iterable[SpeakerTimeline],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the distribution of timeline interval durations."""

    durations = [item.duration for item in timeline]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(durations, bins=min(20, max(1, len(durations))))
    axis.set_title("Speaker Interval Durations")
    axis.set_xlabel("Duration (seconds)")
    axis.set_ylabel("Count")
    figure.tight_layout()
    _save_plot_if_requested(figure, output_path)
    return figure


def _clustered_to_fragments(
    clustered_embeddings: Iterable[ClusteredEmbedding],
) -> list[_TimelineFragment]:
    """Convert clustered embeddings into sorted timeline fragments."""

    fragments = [
        _TimelineFragment(
            speaker_id=f"speaker_{embedding.cluster_id}",
            start=float(embedding.start),
            end=float(embedding.end),
            segment_count=1,
        )
        for embedding in clustered_embeddings
        if float(embedding.end) > float(embedding.start)
    ]
    return sorted(fragments, key=lambda item: (item.start, item.end, item.speaker_id))


def _smooth_isolated_labels(
    fragments: list[_TimelineFragment],
) -> list[_TimelineFragment]:
    """Replace one-fragment label islands between matching speakers."""

    if len(fragments) < 3:
        return fragments

    smoothed = [
        _TimelineFragment(item.speaker_id, item.start, item.end, item.segment_count)
        for item in fragments
    ]
    for index in range(1, len(smoothed) - 1):
        previous_item = smoothed[index - 1]
        current_item = smoothed[index]
        next_item = smoothed[index + 1]
        if (
            previous_item.speaker_id == next_item.speaker_id
            and current_item.speaker_id != previous_item.speaker_id
        ):
            current_item.speaker_id = previous_item.speaker_id
    return smoothed


def _resolve_overlaps(fragments: list[_TimelineFragment]) -> list[_TimelineFragment]:
    """Clamp cross-speaker overlaps to keep intervals chronological."""

    if not fragments:
        return []

    resolved: list[_TimelineFragment] = []
    for fragment in fragments:
        current = _TimelineFragment(
            fragment.speaker_id,
            fragment.start,
            fragment.end,
            fragment.segment_count,
        )
        if resolved and current.start < resolved[-1].end:
            if current.speaker_id == resolved[-1].speaker_id:
                resolved[-1].end = max(resolved[-1].end, current.end)
                resolved[-1].segment_count += current.segment_count
                continue
            current.start = resolved[-1].end
        if current.end > current.start:
            resolved.append(current)
    return resolved


def _merge_fragments(
    fragments: list[_TimelineFragment],
    *,
    merge_adjacent: bool,
    silence_threshold: float,
) -> list[_TimelineFragment]:
    """Merge adjacent or near-adjacent fragments from the same speaker."""

    if not fragments:
        return []

    merged = [
        _TimelineFragment(
            fragments[0].speaker_id,
            fragments[0].start,
            fragments[0].end,
            fragments[0].segment_count,
        )
    ]
    for fragment in fragments[1:]:
        previous = merged[-1]
        gap = fragment.start - previous.end
        same_speaker = fragment.speaker_id == previous.speaker_id
        should_merge = same_speaker and (
            (merge_adjacent and gap <= 0.0) or 0.0 < gap <= silence_threshold
        )
        if should_merge:
            previous.end = max(previous.end, fragment.end)
            previous.segment_count += fragment.segment_count
        else:
            merged.append(
                _TimelineFragment(
                    fragment.speaker_id,
                    fragment.start,
                    fragment.end,
                    fragment.segment_count,
                )
            )
    return merged


def _remove_tiny_fragments(
    fragments: list[_TimelineFragment],
    min_duration_seconds: float,
) -> list[_TimelineFragment]:
    """Remove fragments shorter than the configured duration threshold."""

    return [
        fragment
        for fragment in fragments
        if fragment.duration >= min_duration_seconds
    ]


def _fragment_to_timeline(fragment: _TimelineFragment) -> SpeakerTimeline:
    """Convert an internal fragment to an immutable timeline row."""

    return SpeakerTimeline(
        speaker_id=fragment.speaker_id,
        start=round(fragment.start, 6),
        end=round(fragment.end, 6),
        duration=round(fragment.duration, 6),
        segment_count=fragment.segment_count,
    )


def _speaker_order(timeline: Iterable[SpeakerTimeline]) -> list[str]:
    """Return speakers in first-appearance order."""

    ordered: list[str] = []
    for item in timeline:
        if item.speaker_id not in ordered:
            ordered.append(item.speaker_id)
    return ordered


def _speaker_duration_totals(
    timeline: Iterable[SpeakerTimeline],
) -> dict[str, float]:
    """Return total duration per speaker in first-appearance order."""

    totals: dict[str, float] = {}
    for item in timeline:
        totals.setdefault(item.speaker_id, 0.0)
        totals[item.speaker_id] += item.duration
    return totals


def _validate_timeline_config(
    merge_silence_threshold_seconds: float,
    min_duration_seconds: float,
) -> None:
    """Validate timeline reconstruction configuration."""

    if merge_silence_threshold_seconds < 0.0:
        raise ValueError("timeline_merge_silence_threshold_seconds must be non-negative.")
    if min_duration_seconds < 0.0:
        raise ValueError("timeline_min_duration_seconds must be non-negative.")


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
    LOGGER.info("Saved timeline plot to %s", path)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for exporting a timeline from JSON rows."""

    parser = argparse.ArgumentParser(description="Export speaker timeline rows.")
    parser.add_argument("timeline_json", type=Path, help="Input timeline JSON rows.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output.")
    parser.add_argument("--txt", type=Path, default=None, help="Optional TXT output.")
    return parser.parse_args()


def main() -> None:
    """Export timeline JSON rows to CSV/TXT from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    raw_rows = json.loads(args.timeline_json.read_text(encoding="utf-8"))
    timeline = [SpeakerTimeline(**row) for row in raw_rows]
    if args.csv is not None:
        export_timeline_csv(timeline, args.csv)
    if args.txt is not None:
        export_timeline_txt(timeline, args.txt)


if __name__ == "__main__":
    main()
