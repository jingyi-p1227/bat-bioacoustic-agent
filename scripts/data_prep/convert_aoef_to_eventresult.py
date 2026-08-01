"""Convert one AOEF/Whombat recording to the demo EventResult JSON schema.

This is a small ground-truth conversion helper. It does not call the agent and
does not run any external classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GROUND_TRUTH_DIR = Path("ground_truth")


def load_aoef(annotation_json_path: str | Path) -> dict:
    """Load the annotation project stored under the top-level AOEF data field."""
    with open(annotation_json_path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"]


def tag_lookup(project: dict) -> dict[int, dict]:
    """Map integer tag IDs to tag objects."""
    if "_tag_lookup" not in project:
        project["_tag_lookup"] = {tag["id"]: tag for tag in project.get("tags", [])}
    return project["_tag_lookup"]


def annotations_by_sound_event(project: dict) -> dict[str, dict]:
    """Map sound_event UUIDs to their sound_event_annotation objects."""
    if "_annotations_by_sound_event" not in project:
        project["_annotations_by_sound_event"] = {
            annotation["sound_event"]: annotation
            for annotation in project.get("sound_event_annotations", [])
        }
    return project["_annotations_by_sound_event"]


def tags_for_sound_event(sound_event: dict, project: dict) -> list[dict]:
    """Resolve the shared tag IDs attached to one sound event."""
    annotations = annotations_by_sound_event(project)
    annotation = annotations.get(sound_event["uuid"], {})
    tags = tag_lookup(project)
    return [tags[tag_id] for tag_id in annotation.get("tags", []) if tag_id in tags]


def tag_values(tags: list[dict], keys: set[str]) -> list[str]:
    """Return sorted unique tag values matching any key in keys."""
    return sorted({tag["value"] for tag in tags if tag.get("key") in keys})


def label_from_tags(tags: list[dict]) -> str:
    """Choose the most specific available label from AOEF tags."""
    preferred_keys = [
        "dwc:scientificName",
        "species",
        "soundevent:class",
        "dwc:genus",
        "genus",
        "dwc:family",
        "family",
        "dwc:order",
        "order",
        "soundevent:call_type",
        "call_type",
    ]
    for key in preferred_keys:
        values = tag_values(tags, {key})
        if values:
            return values[0]
    return "unknown"


def recording_audio_path(
    annotation_json_path: Path,
    recording: dict,
    audio_dir: str | Path | None,
) -> Path:
    """Resolve and verify the real WAV path for a recording."""
    audio_root = Path(audio_dir) if audio_dir else annotation_json_path.parent / "audio"
    audio_path = audio_root / recording["path"]
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path.resolve()


def find_recording(project: dict, recording_id_or_path: str) -> dict:
    """Find a recording by UUID or by its AOEF path field."""
    for recording in project.get("recordings", []):
        if recording.get("uuid") == recording_id_or_path:
            return recording
        if recording.get("path") == recording_id_or_path:
            return recording
        if Path(recording.get("path", "")).name == recording_id_or_path:
            return recording
    raise ValueError(f"Recording not found: {recording_id_or_path}")


def sound_events_for_recording(project: dict, recording_uuid: str) -> list[dict]:
    """Return sound events attached to one recording UUID."""
    return [
        event
        for event in project.get("sound_events", [])
        if event.get("recording") == recording_uuid
    ]


def event_from_sound_event(sound_event: dict, project: dict) -> dict:
    """Convert one AOEF BoundingBox sound event to a SpectrogramEvent dict."""
    geometry = sound_event.get("geometry", {})
    if geometry.get("type") != "BoundingBox":
        raise ValueError(f"Unsupported geometry type for {sound_event['uuid']}")

    # AOEF BoundingBox coordinates are ordered as:
    # [start time seconds, low frequency Hz, end time seconds, high frequency Hz].
    # Times are relative to the start of the recording, not the clip.
    coordinates = geometry.get("coordinates", [])
    if len(coordinates) != 4:
        raise ValueError(f"Expected four BoundingBox coordinates for {sound_event['uuid']}")

    start_time, low_frequency, end_time, high_frequency = map(float, coordinates)
    if end_time < start_time:
        raise ValueError(f"Invalid time bounds for {sound_event['uuid']}: {coordinates}")
    if high_frequency < low_frequency:
        raise ValueError(f"Invalid frequency bounds for {sound_event['uuid']}: {coordinates}")

    tags = tags_for_sound_event(sound_event, project)
    label = label_from_tags(tags)
    tag_text = ", ".join(f"{tag['key']}={tag['value']}" for tag in tags)

    return {
        "event_id": sound_event["uuid"],
        "start_time_seconds": start_time,
        "end_time_seconds": end_time,
        "low_frequency_hz": low_frequency,
        "high_frequency_hz": high_frequency,
        "label": label,
        "confidence": 1.0,
        "evidence": f"Converted from AOEF annotation tags: {tag_text}",
        "tools_used": ["convert_aoef_to_eventresult"],
        "human_review_needed": False,
        "review_reason": "Ground-truth annotation converted from AOEF.",
    }


def recording_tags(project: dict, recording_uuid: str) -> tuple[list[str], list[str]]:
    """Collect species/common-name tags seen on sound events for one recording."""
    species_keys = {"dwc:scientificName", "species", "soundevent:class"}
    common_name_keys = {
        "dwc:vernacularName",
        "vernacularName",
        "common_name",
        "commonName",
    }

    species = set()
    common_names = set()
    for sound_event in sound_events_for_recording(project, recording_uuid):
        tags = tags_for_sound_event(sound_event, project)
        species.update(tag_values(tags, species_keys))
        common_names.update(tag_values(tags, common_name_keys))

    return sorted(species), sorted(common_names)


def list_recordings(project: dict) -> None:
    """Print available recordings and useful metadata."""
    for recording in project.get("recordings", []):
        species, common_names = recording_tags(project, recording["uuid"])
        print(
            "\t".join(
                [
                    recording["uuid"],
                    recording["path"],
                    f"duration={recording.get('duration')}",
                    f"samplerate={recording.get('samplerate')}",
                    f"species={';'.join(species) if species else ''}",
                    f"common_name={';'.join(common_names) if common_names else ''}",
                ]
            )
        )


def convert_recording(
    annotation_json_path: str | Path,
    recording_id_or_path: str,
    audio_dir: str | Path | None = None,
    output_dir: str | Path = GROUND_TRUTH_DIR,
) -> Path:
    """Convert one AOEF recording to EventResult and save it as JSON."""
    annotation_json_path = Path(annotation_json_path)
    project = load_aoef(annotation_json_path)
    recording = find_recording(project, recording_id_or_path)
    audio_path = recording_audio_path(annotation_json_path, recording, audio_dir)

    events = [
        event_from_sound_event(sound_event, project)
        for sound_event in sound_events_for_recording(project, recording["uuid"])
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}_ground_truth.json"

    payload = {
        "audio_path": str(audio_path),
        "events": events,
        "notes": (
            f"Converted from AOEF recording {recording['uuid']} "
            f"({recording['path']})."
        ),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AOEF/Whombat annotations to EventResult JSON."
    )
    parser.add_argument("annotation_json", help="Path to AOEF annotations.json.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available recordings and exit.",
    )
    parser.add_argument(
        "--recording",
        help="Recording UUID or AOEF recording path to convert.",
    )
    parser.add_argument(
        "--audio-dir",
        help="Audio directory. Defaults to <annotation_json_parent>/audio.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(GROUND_TRUTH_DIR),
        help="Directory for <audio_stem>_ground_truth.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = load_aoef(args.annotation_json)

    if args.list:
        list_recordings(project)
        return

    if not args.recording:
        raise SystemExit("Use --list or provide --recording <uuid-or-path>.")

    output_path = convert_recording(
        args.annotation_json,
        args.recording,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
    )
    print(f"Saved EventResult ground truth to {output_path}")


if __name__ == "__main__":
    main()
