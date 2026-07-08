"""Deterministically characterise event boxes and event sequences.

This module contains no model calls. It derives geometry, ordering, interval,
overlap, density, and boundary fields from event boxes and explicit clip/source
metadata. Behavioural interpretations are not ground truth in the Australia
dataset and are permitted only as clearly marked exploratory hypotheses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from event_characterisation_models import (
    EventCharacterisation,
    GroundedEventInterpretation,
    RetrievedAnnotationCase,
    RetrievedLiteratureEvidence,
    SequenceCharacterisation,
)


def load_jsonl_records(path: str | Path, model: type[BaseModel]) -> list[BaseModel]:
    """Load non-empty JSONL rows and validate each row with ``model``."""

    records: list[BaseModel] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(model.model_validate(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}: {exc}") from exc
    return records


def _required_number(event: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in event and event[key] is not None:
            return float(event[key])
    raise ValueError(f"Event is missing required field; expected one of {keys}")


def _normalise_event(event: dict[str, Any], index: int) -> dict[str, Any]:
    start = _required_number(event, "start_time_seconds", "start_time")
    end = _required_number(event, "end_time_seconds", "end_time")
    low = _required_number(event, "low_frequency_hz", "low_frequency")
    high = _required_number(event, "high_frequency_hz", "high_frequency")
    if start < 0 or end <= start:
        raise ValueError(f"Invalid time geometry for event {event.get('event_id', index)}")
    if low < 0 or high <= low:
        raise ValueError(
            f"Invalid frequency geometry for event {event.get('event_id', index)}"
        )
    return {
        "event_id": str(event.get("event_id") or f"event_{index:03d}"),
        "label": event.get("label"),
        "start": start,
        "end": end,
        "low": low,
        "high": high,
        "truncation_side": event.get("truncation_side"),
        "is_truncated": event.get("is_truncated_by_clip_boundary"),
        "source_start": event.get("source_start_time"),
        "source_end": event.get("source_end_time"),
    }


def _boundary_truncation(
    event: dict[str, Any],
    *,
    clip_source_start: float | None,
    clip_source_end: float | None,
) -> tuple[bool, bool, bool, str]:
    side = event.get("truncation_side")
    if side in {"none", "left", "right", "both"}:
        return side in {"left", "both"}, side in {"right", "both"}, True, "explicit_metadata"

    source_start = event.get("source_start")
    source_end = event.get("source_end")
    if (
        source_start is not None
        and source_end is not None
        and clip_source_start is not None
        and clip_source_end is not None
    ):
        return (
            float(source_start) < clip_source_start,
            float(source_end) > clip_source_end,
            True,
            "source_time_comparison",
        )

    return False, False, False, "unknown"


def _event_density_category(event_count: int) -> str:
    if event_count >= 5:
        return "high"
    if event_count >= 3:
        return "medium"
    if event_count >= 1:
        return "low"
    return "zero"


def characterise_events(
    *,
    clip_id: str,
    clip_duration_seconds: float,
    events: list[dict[str, Any]],
    clip_source_start_seconds: float | None = None,
    clip_source_end_seconds: float | None = None,
) -> SequenceCharacterisation:
    """Calculate deterministic features for event boxes in one clip.

    Events are sorted by start time, then end time and ID. Inter-event intervals
    are signed temporal gaps, so overlapping neighbours produce negative values.
    ``event_overlap`` means positive temporal overlap with another event.
    Boundary truncation is marked known only when explicit truncation metadata or
    source-time comparisons are available; touching a boundary alone is not
    treated as proof of truncation.
    """

    if clip_duration_seconds <= 0:
        raise ValueError("clip_duration_seconds must be greater than zero")

    normalised = [_normalise_event(event, index) for index, event in enumerate(events, 1)]
    normalised.sort(key=lambda item: (item["start"], item["end"], item["event_id"]))

    for event in normalised:
        if event["end"] > clip_duration_seconds:
            raise ValueError(
                f"Event {event['event_id']} ends after clip duration "
                f"({event['end']} > {clip_duration_seconds})"
            )

    characterised: list[EventCharacterisation] = []
    tolerance = 1e-9
    for index, event in enumerate(normalised):
        previous_interval = None
        if index > 0:
            previous_interval = (event["start"] - normalised[index - 1]["end"]) * 1000
        next_interval = None
        if index + 1 < len(normalised):
            next_interval = (normalised[index + 1]["start"] - event["end"]) * 1000

        overlapping_ids = [
            other["event_id"]
            for other in normalised
            if other["event_id"] != event["event_id"]
            and max(event["start"], other["start"])
            < min(event["end"], other["end"])
        ]
        left_truncated, right_truncated, truncation_known, truncation_basis = (
            _boundary_truncation(
                event,
                clip_source_start=clip_source_start_seconds,
                clip_source_end=clip_source_end_seconds,
            )
        )
        temporal_center = (event["start"] + event["end"]) / 2
        characterised.append(
            EventCharacterisation(
                event_id=event["event_id"],
                label=str(event["label"]) if event["label"] is not None else None,
                start_time_seconds=event["start"],
                end_time_seconds=event["end"],
                low_frequency_hz=event["low"],
                high_frequency_hz=event["high"],
                duration_ms=(event["end"] - event["start"]) * 1000,
                bandwidth_hz=event["high"] - event["low"],
                temporal_center_seconds=temporal_center,
                frequency_center_hz=(event["low"] + event["high"]) / 2,
                event_order=index + 1,
                previous_inter_event_interval_ms=previous_interval,
                next_inter_event_interval_ms=next_interval,
                clip_relative_position=temporal_center / clip_duration_seconds,
                touches_left_clip_boundary=event["start"] <= tolerance,
                touches_right_clip_boundary=(
                    clip_duration_seconds - event["end"] <= tolerance
                ),
                left_boundary_truncated=left_truncated,
                right_boundary_truncated=right_truncated,
                boundary_truncation_known=truncation_known,
                boundary_truncation_basis=truncation_basis,
                event_overlap=bool(overlapping_ids),
                overlapping_event_ids=overlapping_ids,
            )
        )

    event_count = len(characterised)
    return SequenceCharacterisation(
        clip_id=clip_id,
        clip_duration_seconds=clip_duration_seconds,
        event_count=event_count,
        event_density_events_per_second=event_count / clip_duration_seconds,
        event_density_category=_event_density_category(event_count),
        events=characterised,
    )


def characterise_payload(
    payload: dict[str, Any],
    *,
    clip_duration_seconds: float | None = None,
) -> SequenceCharacterisation:
    """Characterise an evaluation-set or prediction-style JSON payload."""

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("Input JSON must contain an events list")

    duration = clip_duration_seconds
    if duration is None:
        for key in ("clip_duration_seconds", "clip_duration"):
            if payload.get(key) is not None:
                duration = float(payload[key])
                break
    source_start = payload.get("source_start_time")
    source_end = payload.get("source_end_time")
    if duration is None and source_start is not None and source_end is not None:
        duration = float(source_end) - float(source_start)
    if duration is None:
        raise ValueError(
            "Clip duration is required via payload metadata or --clip-duration-seconds"
        )

    clip_id = str(payload.get("clip_id") or payload.get("audio_path") or "unknown_clip")
    return characterise_events(
        clip_id=clip_id,
        clip_duration_seconds=duration,
        events=events,
        clip_source_start_seconds=float(source_start) if source_start is not None else None,
        clip_source_end_seconds=float(source_end) if source_end is not None else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically characterise event boxes in a clip JSON."
    )
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--clip-duration-seconds", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_json}. Use --overwrite to replace it."
        )
    with args.input_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    result = characterise_payload(
        payload, clip_duration_seconds=args.clip_duration_seconds
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {result.event_count} event characterisations to {args.output_json}")


if __name__ == "__main__":
    main()
