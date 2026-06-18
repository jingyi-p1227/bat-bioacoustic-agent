"""Build a reproducible single-species clip evaluation set from AOEF data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf


DEFAULT_SPECIES = "Ozimops petersi"
DEFAULT_SOURCE_PREFIX = "pseudo_petersi_"
DEFAULT_CLIP_SECONDS = 1.0
TASK = "single_species_event_detection"
ANNOTATION_STANDARD = (
    "AOEF BoundingBox [start_time, low_frequency, end_time, high_frequency]"
)
MANIFEST_FIELDS = [
    "clip_id",
    "clip_path",
    "ground_truth_path",
    "source_recording",
    "source_start_time",
    "source_end_time",
    "clip_duration",
    "species",
    "has_target_event",
    "num_gt_events",
    "event_density",
    "auto_scenario",
    "manual_scenario",
    "notes",
]


@dataclass(frozen=True)
class SourceEvent:
    event_id: str
    source_start_time: float
    source_end_time: float
    low_frequency: float
    high_frequency: float
    label: str
    tags: list[dict[str, Any]]


def load_project(dataset_dir: str | Path) -> dict[str, Any]:
    """Load the AOEF project stored in ``annotations.json``."""
    annotation_path = Path(dataset_dir) / "annotations.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    project = payload.get("data")
    if not isinstance(project, dict):
        raise ValueError("annotations.json must contain a top-level 'data' object")
    return project


def build_annotation_indexes(
    project: dict[str, Any],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    """Build tag, annotation, and recording-event indexes."""
    tags_by_id = {tag["id"]: tag for tag in project.get("tags", [])}
    annotations_by_event = {
        annotation["sound_event"]: annotation
        for annotation in project.get("sound_event_annotations", [])
    }
    events_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in project.get("sound_events", []):
        events_by_recording[event["recording"]].append(event)
    return tags_by_id, annotations_by_event, events_by_recording


def resolved_tags(
    event: dict[str, Any],
    *,
    tags_by_id: dict[int, dict[str, Any]],
    annotations_by_event: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve shared AOEF tag IDs for one sound event."""
    annotation = annotations_by_event.get(event["uuid"], {})
    return [
        dict(tags_by_id[tag_id])
        for tag_id in annotation.get("tags", [])
        if tag_id in tags_by_id
    ]


def scientific_name(tags: list[dict[str, Any]]) -> str | None:
    """Return the scientific-name tag value when present."""
    preferred_keys = ("dwc:scientificName", "species", "soundevent:class")
    for key in preferred_keys:
        for tag in tags:
            if tag.get("key") == key and tag.get("value"):
                return str(tag["value"])
    return None


def parse_source_event(
    event: dict[str, Any],
    *,
    tags: list[dict[str, Any]],
    species: str,
) -> SourceEvent | None:
    """Parse one target-species AOEF bounding box."""
    if scientific_name(tags) != species:
        return None

    geometry = event.get("geometry", {})
    if geometry.get("type") != "BoundingBox":
        raise ValueError(
            f"Unsupported geometry type for event {event.get('uuid')}: "
            f"{geometry.get('type')}"
        )

    coordinates = geometry.get("coordinates", [])
    if len(coordinates) != 4:
        raise ValueError(
            f"Expected four BoundingBox coordinates for event {event.get('uuid')}"
        )

    start_time, low_frequency, end_time, high_frequency = map(float, coordinates)
    if end_time < start_time:
        raise ValueError(f"Invalid event time range for {event.get('uuid')}")
    if high_frequency < low_frequency:
        raise ValueError(f"Invalid event frequency range for {event.get('uuid')}")

    return SourceEvent(
        event_id=str(event["uuid"]),
        source_start_time=start_time,
        source_end_time=end_time,
        low_frequency=low_frequency,
        high_frequency=high_frequency,
        label=species,
        tags=tags,
    )


def clip_event(
    event: SourceEvent,
    *,
    clip_start_time: float,
    clip_end_time: float,
) -> dict[str, Any] | None:
    """Convert an overlapping source event to clip-relative coordinates."""
    overlap_start = max(event.source_start_time, clip_start_time)
    overlap_end = min(event.source_end_time, clip_end_time)
    if overlap_end <= overlap_start:
        return None

    return {
        "event_id": event.event_id,
        "start_time": overlap_start - clip_start_time,
        "end_time": overlap_end - clip_start_time,
        "low_frequency": event.low_frequency,
        "high_frequency": event.high_frequency,
        "label": event.label,
        "tags": event.tags,
        "source_start_time": event.source_start_time,
        "source_end_time": event.source_end_time,
    }


def event_density(num_gt_events: int) -> str:
    if num_gt_events >= 5:
        return "high"
    if num_gt_events >= 3:
        return "medium"
    if num_gt_events >= 1:
        return "low"
    return "zero"


def auto_scenario(num_gt_events: int) -> str:
    if num_gt_events >= 5:
        return "multi_event"
    if num_gt_events >= 3:
        return "positive"
    if num_gt_events >= 1:
        return "low_activity"
    return "negative"


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Create an empty output directory, refusing replacement by default."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    (output_dir / "audio").mkdir(parents=True)
    (output_dir / "ground_truth").mkdir(parents=True)


def build_evaluation_set(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    species: str = DEFAULT_SPECIES,
    source_prefix: str = DEFAULT_SOURCE_PREFIX,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build clips, clip-level ground truth, and a portable manifest."""
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be greater than 0")

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    audio_dir = dataset_dir / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    project = load_project(dataset_dir)
    tags_by_id, annotations_by_event, events_by_recording = (
        build_annotation_indexes(project)
    )
    recordings = sorted(
        [
            recording
            for recording in project.get("recordings", [])
            if Path(recording.get("path", "")).name.startswith(source_prefix)
        ],
        key=lambda recording: Path(recording["path"]).name,
    )
    if not recordings:
        raise ValueError(
            f"No recordings found with source prefix {source_prefix!r}"
        )

    prepare_output_dir(output_dir, overwrite=overwrite)

    manifest_rows: list[dict[str, Any]] = []
    unique_event_ids: set[str] = set()
    clip_event_instances = 0
    clip_number = 0

    for recording in recordings:
        source_recording = Path(recording["path"]).name
        source_audio_path = audio_dir / recording["path"]
        if not source_audio_path.exists():
            raise FileNotFoundError(f"Source WAV not found: {source_audio_path}")

        source_events: list[SourceEvent] = []
        for raw_event in events_by_recording.get(recording["uuid"], []):
            tags = resolved_tags(
                raw_event,
                tags_by_id=tags_by_id,
                annotations_by_event=annotations_by_event,
            )
            parsed_event = parse_source_event(
                raw_event,
                tags=tags,
                species=species,
            )
            if parsed_event is not None:
                source_events.append(parsed_event)
                unique_event_ids.add(parsed_event.event_id)
        source_events.sort(
            key=lambda event: (
                event.source_start_time,
                event.source_end_time,
                event.event_id,
            )
        )

        audio, sample_rate = sf.read(str(source_audio_path), always_2d=False)
        audio_info = sf.info(str(source_audio_path))
        total_samples = len(audio)
        clip_length_samples = int(round(clip_seconds * sample_rate))
        if clip_length_samples <= 0:
            raise ValueError("clip_seconds is too short for the source sample rate")

        num_clips = math.ceil(total_samples / clip_length_samples)
        for recording_clip_index in range(num_clips):
            start_sample = recording_clip_index * clip_length_samples
            end_sample = min(start_sample + clip_length_samples, total_samples)
            source_start_time = start_sample / sample_rate
            source_end_time = end_sample / sample_rate
            clip_audio = audio[start_sample:end_sample]

            clip_number += 1
            clip_id = f"OP_{clip_number:03d}"
            relative_clip_path = Path("audio") / f"{clip_id}.wav"
            relative_ground_truth_path = (
                Path("ground_truth") / f"{clip_id}_ground_truth.json"
            )
            output_clip_path = output_dir / relative_clip_path
            output_ground_truth_path = output_dir / relative_ground_truth_path

            sf.write(
                str(output_clip_path),
                clip_audio,
                sample_rate,
                subtype=audio_info.subtype,
            )

            clip_events = [
                converted
                for event in source_events
                if (
                    converted := clip_event(
                        event,
                        clip_start_time=source_start_time,
                        clip_end_time=source_end_time,
                    )
                )
                is not None
            ]
            clip_event_instances += len(clip_events)

            ground_truth_payload = {
                "clip_id": clip_id,
                "clip_path": relative_clip_path.as_posix(),
                "source_recording": source_recording,
                "source_start_time": source_start_time,
                "source_end_time": source_end_time,
                "species": species,
                "task": TASK,
                "annotation_standard": ANNOTATION_STANDARD,
                "events": clip_events,
            }
            with output_ground_truth_path.open("w", encoding="utf-8") as f:
                json.dump(ground_truth_payload, f, indent=2, ensure_ascii=False)
                f.write("\n")

            num_gt_events = len(clip_events)
            clip_duration = (end_sample - start_sample) / sample_rate
            manifest_rows.append(
                {
                    "clip_id": clip_id,
                    "clip_path": relative_clip_path.as_posix(),
                    "ground_truth_path": relative_ground_truth_path.as_posix(),
                    "source_recording": source_recording,
                    "source_start_time": source_start_time,
                    "source_end_time": source_end_time,
                    "clip_duration": clip_duration,
                    "species": species,
                    "has_target_event": str(num_gt_events > 0).lower(),
                    "num_gt_events": num_gt_events,
                    "event_density": event_density(num_gt_events),
                    "auto_scenario": auto_scenario(num_gt_events),
                    "manual_scenario": "",
                    "notes": "",
                }
            )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "output_dir": str(output_dir),
        "recording_count": len(recordings),
        "clip_count": len(manifest_rows),
        "unique_source_event_count": len(unique_event_ids),
        "clip_event_instance_count": clip_event_instances,
        "manifest_path": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-species clipped evaluation set from AOEF annotations."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory containing annotations.json and audio/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for clips, ground truth, and manifest.csv.",
    )
    parser.add_argument("--species", default=DEFAULT_SPECIES)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_evaluation_set(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        species=args.species,
        source_prefix=args.source_prefix,
        clip_seconds=args.clip_seconds,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
