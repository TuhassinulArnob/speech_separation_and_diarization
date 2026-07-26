"""RTTM reader and writer for reconstructed speaker timelines.

This module converts ``SpeakerTimeline`` rows to standard RTTM ``SPEAKER``
records and reads existing RTTM files. It does not compute DER/JER or implement
a baseline system.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from Project.ssl_pipeline.config import DiarizationConfig, load_config
from Project.ssl_pipeline.timeline_reconstruction import SpeakerTimeline

LOGGER = logging.getLogger(__name__)

RTTM_SPEAKER_TYPE = "SPEAKER"
RTTM_NA = "<NA>"
SPEAKER_NAME_PATTERN = re.compile(r"^[^\s]+$")


@dataclass(frozen=True)
class RTTMSpeakerTurn:
    """One RTTM speaker turn."""

    recording_id: str
    channel_id: str
    start: float
    duration: float
    speaker_id: str
    orthography: str = RTTM_NA
    speaker_type: str = RTTM_NA
    confidence: str = RTTM_NA
    signal_lookahead_time: str = RTTM_NA

    @property
    def end(self) -> float:
        """Return end time in seconds."""

        return self.start + self.duration

    def to_timeline(self) -> SpeakerTimeline:
        """Convert this RTTM turn into a SpeakerTimeline row."""

        return SpeakerTimeline(
            speaker_id=self.speaker_id,
            start=round(self.start, 6),
            end=round(self.end, 6),
            duration=round(self.duration, 6),
            segment_count=1,
        )

    def to_dict(self) -> dict[str, str | float]:
        """Return a serializable RTTM turn dictionary."""

        return asdict(self)


class RTTMWriter:
    """Write SpeakerTimeline rows as RTTM speaker turns."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        *,
        recording_id: str | None = None,
        channel_id: str | None = None,
        decimal_places: int | None = None,
    ) -> None:
        self.config = config or load_config()
        self.recording_id = recording_id or self.config.rttm_recording_id
        self.channel_id = channel_id or self.config.rttm_channel_id
        self.decimal_places = (
            self.config.rttm_decimal_places
            if decimal_places is None
            else int(decimal_places)
        )
        _validate_writer_config(self.recording_id, self.channel_id, self.decimal_places)

    def format_turn(self, turn: RTTMSpeakerTurn) -> str:
        """Format one RTTM turn as a standard 10-field SPEAKER line."""

        start = f"{turn.start:.{self.decimal_places}f}"
        duration = f"{turn.duration:.{self.decimal_places}f}"
        return " ".join(
            [
                RTTM_SPEAKER_TYPE,
                turn.recording_id,
                turn.channel_id,
                start,
                duration,
                turn.orthography,
                turn.speaker_type,
                turn.speaker_id,
                turn.confidence,
                turn.signal_lookahead_time,
            ]
        )

    def format_timeline(self, timeline: Iterable[SpeakerTimeline]) -> str:
        """Format timeline rows as RTTM text."""

        turns = timeline_to_rttm_turns(
            timeline,
            recording_id=self.recording_id,
            channel_id=self.channel_id,
        )
        validate_rttm_turns(turns, allow_empty=False)
        return "\n".join(self.format_turn(turn) for turn in turns) + "\n"

    def write(
        self,
        timeline: Iterable[SpeakerTimeline],
        output_path: str | Path,
    ) -> None:
        """Write timeline rows to an RTTM file."""

        text = self.format_timeline(timeline)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        LOGGER.info("Saved RTTM to %s", path)


class RTTMReader:
    """Read RTTM speaker turns from text or files."""

    def parse_text(
        self,
        text: str,
        *,
        allow_empty: bool = False,
    ) -> list[RTTMSpeakerTurn]:
        """Parse RTTM text into speaker turns."""

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not lines:
            if allow_empty:
                return []
            raise ValueError("RTTM content is empty.")

        turns = [_parse_rttm_line(line, line_number=index + 1) for index, line in enumerate(lines)]
        validate_rttm_turns(turns, allow_empty=allow_empty)
        return turns

    def read(
        self,
        rttm_path: str | Path,
        *,
        allow_empty: bool = False,
    ) -> list[RTTMSpeakerTurn]:
        """Read RTTM speaker turns from a file."""

        path = Path(rttm_path)
        text = path.read_text(encoding="utf-8")
        return self.parse_text(text, allow_empty=allow_empty)


def write_rttm(
    timeline: Iterable[SpeakerTimeline],
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
    recording_id: str | None = None,
    channel_id: str | None = None,
    decimal_places: int | None = None,
) -> None:
    """Write a speaker timeline to an RTTM file."""

    config = load_config(config_path) if config_path else load_config()
    writer = RTTMWriter(
        config,
        recording_id=recording_id,
        channel_id=channel_id,
        decimal_places=decimal_places,
    )
    writer.write(timeline, output_path)


def parse_rttm(
    rttm_path: str | Path,
    *,
    allow_empty: bool = False,
) -> list[RTTMSpeakerTurn]:
    """Read and parse an RTTM file."""

    return RTTMReader().read(rttm_path, allow_empty=allow_empty)


def validate_rttm(
    rttm_path: str | Path,
    *,
    allow_empty: bool = False,
) -> bool:
    """Validate an RTTM file, raising ValueError on invalid input."""

    parse_rttm(rttm_path, allow_empty=allow_empty)
    return True


def validate_rttm_turns(
    turns: Iterable[RTTMSpeakerTurn],
    *,
    allow_empty: bool = False,
    allow_overlap: bool = False,
) -> None:
    """Validate RTTM turn fields and chronological overlaps."""

    turn_list = list(turns)
    if not turn_list:
        if allow_empty:
            return
        raise ValueError("RTTM content is empty.")

    for turn in turn_list:
        _validate_token("recording_id", turn.recording_id)
        _validate_token("channel_id", turn.channel_id)
        _validate_token("speaker_id", turn.speaker_id)
        if turn.start < 0.0:
            raise ValueError("RTTM start time must be non-negative.")
        if turn.duration <= 0.0:
            raise ValueError("RTTM duration must be positive.")

    if not allow_overlap:
        _validate_no_overlap(turn_list)


def timeline_to_rttm_turns(
    timeline: Iterable[SpeakerTimeline],
    *,
    recording_id: str,
    channel_id: str,
) -> list[RTTMSpeakerTurn]:
    """Convert SpeakerTimeline rows to RTTM speaker turns."""

    turns = [
        RTTMSpeakerTurn(
            recording_id=recording_id,
            channel_id=channel_id,
            start=float(item.start),
            duration=float(item.duration),
            speaker_id=item.speaker_id,
        )
        for item in timeline
    ]
    validate_rttm_turns(turns, allow_empty=False)
    return turns


def rttm_turns_to_timeline(
    turns: Iterable[RTTMSpeakerTurn],
) -> list[SpeakerTimeline]:
    """Convert RTTM speaker turns to timeline rows."""

    return [turn.to_timeline() for turn in sorted(turns, key=lambda item: item.start)]


def export_rttm_json(
    turns: Iterable[RTTMSpeakerTurn],
    output_path: str | Path,
) -> None:
    """Export RTTM turns as JSON."""

    rows = [turn.to_dict() for turn in turns]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    LOGGER.info("Saved RTTM JSON to %s", path)


def load_rttm_json(
    input_path: str | Path,
) -> list[RTTMSpeakerTurn]:
    """Load RTTM turns from JSON exported by this module."""

    path = Path(input_path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    turns = [RTTMSpeakerTurn(**row) for row in rows]
    validate_rttm_turns(turns)
    return turns


def _parse_rttm_line(line: str, *, line_number: int) -> RTTMSpeakerTurn:
    """Parse a 9-field or 10-field RTTM SPEAKER line."""

    fields = line.split()
    if len(fields) not in {9, 10}:
        raise ValueError(f"Malformed RTTM line {line_number}: expected 9 or 10 fields.")
    if fields[0] != RTTM_SPEAKER_TYPE:
        raise ValueError(f"Malformed RTTM line {line_number}: expected SPEAKER record.")

    try:
        start = float(fields[3])
        duration = float(fields[4])
    except ValueError as exc:
        raise ValueError(f"Malformed RTTM line {line_number}: invalid timestamp.") from exc

    return RTTMSpeakerTurn(
        recording_id=fields[1],
        channel_id=fields[2],
        start=start,
        duration=duration,
        orthography=fields[5],
        speaker_type=fields[6],
        speaker_id=fields[7],
        confidence=fields[8],
        signal_lookahead_time=fields[9] if len(fields) == 10 else RTTM_NA,
    )


def _validate_no_overlap(turns: list[RTTMSpeakerTurn]) -> None:
    """Validate that turns do not overlap within recording/channel."""

    grouped: dict[tuple[str, str], list[RTTMSpeakerTurn]] = {}
    for turn in turns:
        grouped.setdefault((turn.recording_id, turn.channel_id), []).append(turn)

    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.start, item.end, item.speaker_id))
        previous: RTTMSpeakerTurn | None = None
        for turn in ordered:
            if previous is not None and turn.start < previous.end:
                raise ValueError("RTTM turns must not overlap within a recording/channel.")
            previous = turn


def _validate_token(field_name: str, value: str) -> None:
    """Validate a whitespace-free RTTM token."""

    if not value or not SPEAKER_NAME_PATTERN.match(value):
        raise ValueError(f"Invalid RTTM {field_name}: {value!r}")


def _validate_writer_config(
    recording_id: str,
    channel_id: str,
    decimal_places: int,
) -> None:
    """Validate writer configuration."""

    _validate_token("recording_id", recording_id)
    _validate_token("channel_id", channel_id)
    if decimal_places < 0:
        raise ValueError("rttm_decimal_places must be non-negative.")


def _parse_args() -> argparse.Namespace:
    """Parse RTTM conversion CLI arguments."""

    parser = argparse.ArgumentParser(description="Read, write, and convert RTTM files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_json = subparsers.add_parser("to-json", help="Convert RTTM to JSON.")
    to_json.add_argument("rttm_path", type=Path)
    to_json.add_argument("json_path", type=Path)

    validate = subparsers.add_parser("validate", help="Validate an RTTM file.")
    validate.add_argument("rttm_path", type=Path)

    return parser.parse_args()


def main() -> None:
    """Run RTTM conversion helpers from the command line."""

    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if args.command == "to-json":
        turns = parse_rttm(args.rttm_path)
        export_rttm_json(turns, args.json_path)
    elif args.command == "validate":
        validate_rttm(args.rttm_path)
        print("RTTM is valid.")


if __name__ == "__main__":
    main()
