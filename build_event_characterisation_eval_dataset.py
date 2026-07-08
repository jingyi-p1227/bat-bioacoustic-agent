"""Build the Pydantic Evals-compatible event-characterisation dataset.

Expected outputs are generated exclusively from frozen ground-truth geometry and
deterministic feature extraction. Behavioural hypotheses never appear in the
expected output. The same cases are reused across four retrieval conditions by
changing only the two condition flags in ``EventCharacterisationInput``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic_evals import Case, Dataset

from event_characterisation_models import (
    EventBox,
    EventCharacterisationCaseMetadata,
    EventCharacterisationExpected,
    EventCharacterisationInput,
    ExpectedEventFeatures,
    ExpectedSequenceFeatures,
)
from extract_event_characterisation_features import characterise_events


SCHEMA_VERSION = "1.0.0"
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_INPUT_DIR = Path("outputs/agent_inputs/prompt_v2_full_grid_v2")
DEFAULT_OUTPUT_DIR = Path("outputs/evaluation_sets/event_characterisation_v1")
REPRESENTATIVE_CLIP_IDS = (
    "OP_001",
    "OP_003",
    "OP_004",
    "OP_010",
    "OP_016",
    "OP_045",
)
HELD_OUT_CLIP_IDS = ("OP_032", "OP_042")
ConditionName = Literal[
    "baseline", "annotation_memory_only", "literature_only", "combined"
]
CONDITION_SETTINGS: dict[str, dict[str, bool]] = {
    "baseline": {
        "annotation_memory_enabled": False,
        "literature_evidence_enabled": False,
    },
    "annotation_memory_only": {
        "annotation_memory_enabled": True,
        "literature_evidence_enabled": False,
    },
    "literature_only": {
        "annotation_memory_enabled": False,
        "literature_evidence_enabled": True,
    },
    "combined": {
        "annotation_memory_enabled": True,
        "literature_evidence_enabled": True,
    },
}
EventCharacterisationDataset = Dataset[
    EventCharacterisationInput,
    EventCharacterisationExpected,
    EventCharacterisationCaseMetadata,
]


def load_manifest_rows(eval_dir: Path) -> dict[str, dict[str, str]]:
    """Load evaluation-set manifest rows keyed by clip ID."""

    with (eval_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def direct_scientific_name(event: dict) -> str | None:
    """Read a scientific name only from a direct event tag."""

    for tag in event.get("tags", []):
        if tag.get("key") == "dwc:scientificName" and tag.get("value"):
            return str(tag["value"])
    return None


def event_box_from_ground_truth(event: dict) -> EventBox:
    """Convert one evaluation-set event into a frozen input box."""

    scientific_name = direct_scientific_name(event)
    return EventBox(
        event_id=str(event["event_id"]),
        start_time_seconds=float(event["start_time"]),
        end_time_seconds=float(event["end_time"]),
        low_frequency_hz=float(event["low_frequency"]),
        high_frequency_hz=float(event["high_frequency"]),
        scientific_name=scientific_name,
        scientific_name_source=(
            "direct_annotation" if scientific_name is not None else "unavailable"
        ),
        source_start_time_seconds=(
            float(event["source_start_time"])
            if event.get("source_start_time") is not None
            else None
        ),
        source_end_time_seconds=(
            float(event["source_end_time"])
            if event.get("source_end_time") is not None
            else None
        ),
        truncation_side=event.get("truncation_side"),
    )


def build_expected_output(
    *,
    ground_truth: dict,
    clip_duration_seconds: float,
) -> EventCharacterisationExpected:
    """Create expected features using deterministic geometry calculations."""

    sequence = characterise_events(
        clip_id=str(ground_truth["clip_id"]),
        clip_duration_seconds=clip_duration_seconds,
        events=ground_truth["events"],
        clip_source_start_seconds=float(ground_truth["source_start_time"]),
        clip_source_end_seconds=float(ground_truth["source_end_time"]),
    )
    direct_names = {
        str(event["event_id"]): direct_scientific_name(event)
        for event in ground_truth["events"]
    }
    expected_events = [
        ExpectedEventFeatures(
            event_id=event.event_id,
            duration_ms=event.duration_ms,
            bandwidth_hz=event.bandwidth_hz,
            temporal_center_seconds=event.temporal_center_seconds,
            frequency_center_hz=event.frequency_center_hz,
            event_order=event.event_order,
            previous_inter_event_interval_ms=event.previous_inter_event_interval_ms,
            next_inter_event_interval_ms=event.next_inter_event_interval_ms,
            clip_relative_position=event.clip_relative_position,
            left_boundary_truncated=event.left_boundary_truncated,
            right_boundary_truncated=event.right_boundary_truncated,
            boundary_truncation_known=event.boundary_truncation_known,
            event_overlap=event.event_overlap,
            overlapping_event_ids=event.overlapping_event_ids,
            scientific_name=direct_names[event.event_id],
            scientific_name_directly_annotated=(
                direct_names[event.event_id] is not None
            ),
        )
        for event in sequence.events
    ]
    return EventCharacterisationExpected(
        clip_id=sequence.clip_id,
        events=expected_events,
        sequence=ExpectedSequenceFeatures(
            event_count=sequence.event_count,
            event_density_events_per_second=(
                sequence.event_density_events_per_second
            ),
            event_density_category=sequence.event_density_category,
        ),
    )


def build_case(
    *,
    clip_id: str,
    split: Literal["representative", "heldout"],
    eval_dir: Path,
    input_dir: Path,
    manifest_row: dict[str, str],
) -> Case[
    EventCharacterisationInput,
    EventCharacterisationExpected,
    EventCharacterisationCaseMetadata,
]:
    """Build one typed Pydantic Evals case from frozen evaluation artifacts."""

    ground_truth_path = eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"
    with ground_truth_path.open(encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    duration = float(manifest_row["clip_duration"])
    boxes = [event_box_from_ground_truth(event) for event in ground_truth["events"]]
    expected = build_expected_output(
        ground_truth=ground_truth,
        clip_duration_seconds=duration,
    )
    boundary_case = any(
        event.truncation_side in {"left", "right", "both"} for event in boxes
    )
    fields = [
        "event_geometry",
        "duration",
        "bandwidth",
        "temporal_center",
        "frequency_center",
        "event_order",
        "inter_event_interval",
        "event_overlap",
        "boundary_truncation",
    ]
    if any(event.scientific_name_source == "direct_annotation" for event in boxes):
        fields.append("scientific_name")

    return Case(
        name=clip_id,
        inputs=EventCharacterisationInput(
            clip_id=clip_id,
            clip_duration_seconds=duration,
            frozen_event_boxes=boxes,
            spectrogram_path=(input_dir / f"{clip_id}_spectrogram.png").as_posix(),
            annotation_memory_enabled=False,
            literature_evidence_enabled=False,
        ),
        expected_output=expected,
        metadata=EventCharacterisationCaseMetadata(
            split=(
                "diagnostic_development"
                if split == "representative"
                else "held_out_validation"
            ),
            scenario=manifest_row["auto_scenario"],
            species=manifest_row.get("species") or None,
            event_count=len(boxes),
            boundary_case=boundary_case,
            representative_or_heldout=split,
            ground_truth_fields_available=fields,
        ),
    )


def build_dataset(
    *,
    eval_dir: Path = DEFAULT_EVAL_DIR,
    input_dir: Path = DEFAULT_INPUT_DIR,
) -> EventCharacterisationDataset:
    """Build the eight-case dataset once, independent of retrieval condition."""

    manifest = load_manifest_rows(eval_dir)
    cases = [
        build_case(
            clip_id=clip_id,
            split="representative",
            eval_dir=eval_dir,
            input_dir=input_dir,
            manifest_row=manifest[clip_id],
        )
        for clip_id in REPRESENTATIVE_CLIP_IDS
    ]
    cases.extend(
        build_case(
            clip_id=clip_id,
            split="heldout",
            eval_dir=eval_dir,
            input_dir=input_dir,
            manifest_row=manifest[clip_id],
        )
        for clip_id in HELD_OUT_CLIP_IDS
    )
    return EventCharacterisationDataset(
        name="event_characterisation_v1", cases=cases
    )


def inputs_for_condition(
    inputs: EventCharacterisationInput,
    condition: ConditionName,
) -> EventCharacterisationInput:
    """Return condition-specific task settings without duplicating a Case."""

    settings = CONDITION_SETTINGS[condition]
    return inputs.model_copy(update=settings)


def write_case_summary(dataset: Dataset, path: Path) -> None:
    """Write a compact human-readable case summary."""

    rows = [
        {
            "case_id": case.name,
            "split": case.metadata.split,
            "representative_or_heldout": case.metadata.representative_or_heldout,
            "scenario": case.metadata.scenario,
            "species": case.metadata.species or "",
            "event_count": case.metadata.event_count,
            "boundary_case": str(case.metadata.boundary_case).lower(),
            "clip_duration_seconds": case.inputs.clip_duration_seconds,
            "spectrogram_path": case.inputs.spectrogram_path,
        }
        for case in dataset.cases
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_dataset_snapshot(dataset: Dataset, output_dir: Path) -> None:
    """Save deterministic dataset, summary, and schema metadata artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_file(
        output_dir / "dataset.json",
        schema_path="dataset_schema.json",
    )
    write_case_summary(dataset, output_dir / "case_summary.csv")
    schema_metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset.name,
        "case_count": len(dataset.cases),
        "representative_case_ids": list(REPRESENTATIVE_CLIP_IDS),
        "held_out_case_ids": list(HELD_OUT_CLIP_IDS),
        "condition_settings": CONDITION_SETTINGS,
        "expected_output_ground_truth_boundary": {
            "included": [
                "direct_event_geometry",
                "deterministic_event_and_sequence_features",
                "direct_scientific_name_when_available",
            ],
            "excluded": [
                "behaviour",
                "call_phase",
                "social_call",
                "individual_identity",
                "environment",
                "signal_quality",
                "echo_or_artifact_type",
                "rationale_quality",
            ],
        },
        "dataset_audit": [
            "docs/dataset_audits/batdetect2_australia_annotation_schema_audit.md",
            "docs/dataset_audits/batdetect2_australia_annotation_schema_summary.json",
        ],
    }
    (output_dir / "schema_version.json").write_text(
        json.dumps(schema_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic event-characterisation eval dataset."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {args.output_dir}. Use --overwrite."
            )
        shutil.rmtree(args.output_dir)
    dataset = build_dataset(eval_dir=args.eval_dir, input_dir=args.input_dir)
    save_dataset_snapshot(dataset, args.output_dir)
    print(f"Saved {len(dataset.cases)} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
