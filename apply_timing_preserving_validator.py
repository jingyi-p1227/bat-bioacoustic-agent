"""Apply independent timing and frequency proposal-preservation rules.

This P6E.3 shadow-mode variant writes new predictions and never modifies its
P6D.2 inputs, proposals, ground truth, prompts, or the official evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from apply_proposal_preserving_validator import (
    DEFAULT_AUDIT_CSV,
    DEFAULT_CLIP_IDS,
    DEFAULT_PREDICTION_DIR,
    DEFAULT_PROPOSAL_DIR,
    HIGH_CONFIDENCE_RESTORE_THRESHOLD,
    clipped_proposal_geometry,
    geometry_snapshot,
    load_audit_rows,
    parse_bool,
    parse_clip_ids,
    proposal_index,
    restore_high_confidence_proposal,
)


DEFAULT_OUTPUT_DIR = Path(
    "outputs/agent_runs/p6_timing_preserving_validator_representative6"
)

TIMING_DECISIONS = (
    "kept_vlm_timing",
    "preserved_proposal_timing",
    "restored_proposal_timing",
    "not_applicable_new_event",
)
FREQUENCY_DECISIONS = (
    "kept_vlm_frequency",
    "preserved_proposal_frequency",
    "restored_proposal_frequency",
    "not_applicable_new_event",
)
VALIDATION_DECISIONS = (
    "kept_vlm_geometry",
    "preserved_proposal_timing",
    "preserved_proposal_frequency",
    "preserved_proposal_timing_and_frequency",
    "restored_high_confidence_proposal",
    "kept_new_vlm_event",
)


def timing_preservation_rule(audit_row: dict[str, str]) -> tuple[bool, str]:
    """Return whether proposal start/end should replace VLM timing."""
    reasons: list[str] = []
    if float(audit_row["time_iou_between_proposal_and_prediction"]) < 0.5:
        reasons.append("proposal-prediction time IoU is below 0.5")
    if abs(float(audit_row["delta_start_ms"])) > 10:
        reasons.append("absolute start shift exceeds 10 ms")
    if abs(float(audit_row["delta_end_ms"])) > 10:
        reasons.append("absolute end shift exceeds 10 ms")
    ratio = float(audit_row["duration_ratio"])
    if ratio > 2.0 or ratio < 0.5:
        reasons.append("duration ratio is outside [0.5, 2.0]")
    return bool(reasons), "; ".join(dict.fromkeys(reasons))


def frequency_preservation_rule(audit_row: dict[str, str]) -> tuple[bool, str]:
    """Return whether proposal low/high frequency bounds should be restored."""
    preserve = parse_bool(audit_row["frequency_shift_large"])
    reason = "frequency-boundary shift exceeds 10 kHz" if preserve else ""
    return preserve, reason


def apply_linked_event(
    event: dict[str, Any],
    proposal: dict[str, Any],
    audit_row: dict[str, str],
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Apply timing and frequency decisions independently to one linked event."""
    output = deepcopy(event)
    proposal_id = str(proposal["proposal_id"])
    proposal_geometry = clipped_proposal_geometry(proposal, clip_duration_seconds)
    preserve_timing, timing_reason = timing_preservation_rule(audit_row)
    preserve_frequency, frequency_reason = frequency_preservation_rule(audit_row)

    output["original_vlm_geometry"] = geometry_snapshot(event)
    output["source_proposal_id"] = proposal_id
    output["det_prob"] = float(proposal["det_prob"])
    output["class_prob"] = float(proposal["class_prob"])

    if preserve_timing:
        output["start_time_seconds"] = proposal_geometry["start_time_seconds"]
        output["end_time_seconds"] = proposal_geometry["end_time_seconds"]
        output["timing_decision"] = "preserved_proposal_timing"
    else:
        output["timing_decision"] = "kept_vlm_timing"

    if preserve_frequency:
        output["low_frequency_hz"] = proposal_geometry["low_frequency_hz"]
        output["high_frequency_hz"] = proposal_geometry["high_frequency_hz"]
        output["frequency_decision"] = "preserved_proposal_frequency"
    else:
        output["frequency_decision"] = "kept_vlm_frequency"

    if preserve_timing and preserve_frequency:
        output["validation_decision"] = "preserved_proposal_timing_and_frequency"
    elif preserve_timing:
        output["validation_decision"] = "preserved_proposal_timing"
    elif preserve_frequency:
        output["validation_decision"] = "preserved_proposal_frequency"
    else:
        output["validation_decision"] = "kept_vlm_geometry"

    reasons = [reason for reason in (timing_reason, frequency_reason) if reason]
    output["validation_reason"] = (
        "; ".join(reasons)
        if reasons
        else "VLM timing and frequency stayed within independent deviation limits."
    )
    if preserve_timing or preserve_frequency:
        output["human_review_needed"] = True
        output["review_reason"] = _append_reason(
            str(output.get("review_reason") or ""),
            "Timing-only shadow validator restored one or more proposal bounds.",
        )
    return output


def _append_reason(existing: str, addition: str) -> str:
    return f"{existing} {addition}".strip() if existing else addition


def keep_new_vlm_event(event: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(event)
    output["original_vlm_geometry"] = geometry_snapshot(event)
    output["validation_decision"] = "kept_new_vlm_event"
    output["timing_decision"] = "not_applicable_new_event"
    output["frequency_decision"] = "not_applicable_new_event"
    output["validation_reason"] = "VLM event has no explicit source proposal."
    output["source_proposal_id"] = ""
    output["human_review_needed"] = True
    output["review_reason"] = _append_reason(
        str(output.get("review_reason") or ""),
        "New VLM event requires human review because it lacks detector provenance.",
    )
    return output


def restore_rejected_proposal(
    proposal: dict[str, Any], clip_duration_seconds: float
) -> dict[str, Any]:
    output = restore_high_confidence_proposal(proposal, clip_duration_seconds)
    output["timing_decision"] = "restored_proposal_timing"
    output["frequency_decision"] = "restored_proposal_frequency"
    output["original_vlm_geometry"] = None
    return output


def validate_output_event(event: dict[str, Any], clip_duration: float) -> None:
    required = {
        "event_id",
        "start_time_seconds",
        "end_time_seconds",
        "low_frequency_hz",
        "high_frequency_hz",
        "label",
        "confidence",
        "evidence",
        "human_review_needed",
        "review_reason",
        "validation_decision",
        "timing_decision",
        "frequency_decision",
        "validation_reason",
        "source_proposal_id",
        "original_vlm_geometry",
    }
    missing = sorted(required - event.keys())
    if missing:
        raise ValueError(f"Constrained event missing fields: {', '.join(missing)}")
    start = float(event["start_time_seconds"])
    end = float(event["end_time_seconds"])
    low = float(event["low_frequency_hz"])
    high = float(event["high_frequency_hz"])
    if not 0 <= start < end <= clip_duration:
        raise ValueError(f"Invalid constrained time geometry: {start}, {end}")
    if not 0 <= low < high:
        raise ValueError(f"Invalid constrained frequency geometry: {low}, {high}")
    if event["validation_decision"] not in VALIDATION_DECISIONS:
        raise ValueError(f"Invalid validation_decision: {event['validation_decision']}")
    if event["timing_decision"] not in TIMING_DECISIONS:
        raise ValueError(f"Invalid timing_decision: {event['timing_decision']}")
    if event["frequency_decision"] not in FREQUENCY_DECISIONS:
        raise ValueError(f"Invalid frequency_decision: {event['frequency_decision']}")


def constrain_clip_payload(
    *,
    clip_id: str,
    prediction_payload: dict[str, Any],
    proposal_payload: dict[str, Any],
    audit_rows: dict[tuple[str, str], dict[str, str]],
    restore_threshold: float = HIGH_CONFIDENCE_RESTORE_THRESHOLD,
) -> dict[str, Any]:
    if prediction_payload.get("clip_id") != clip_id:
        raise ValueError(f"Prediction payload clip_id does not match {clip_id}")
    clip_duration = float(prediction_payload["clip_duration_seconds"])
    proposals = proposal_index(proposal_payload, clip_id)
    source_events = prediction_payload.get("events")
    if not isinstance(source_events, list):
        raise ValueError("Prediction payload events must be a list")

    output_events: list[dict[str, Any]] = []
    explicitly_linked_ids: set[str] = set()
    for event in source_events:
        event_id = str(event.get("event_id") or "")
        used_id = str(event.get("used_proposal_id") or "")
        if used_id and used_id in proposals:
            explicitly_linked_ids.add(used_id)
            audit_row = audit_rows.get((clip_id, event_id))
            if audit_row is None:
                raise ValueError(f"Missing P6E.1 audit row for {clip_id}/{event_id}")
            output_event = apply_linked_event(
                event, proposals[used_id], audit_row, clip_duration
            )
        else:
            output_event = keep_new_vlm_event(event)
        validate_output_event(output_event, clip_duration)
        output_events.append(output_event)

    for proposal_id, proposal in proposals.items():
        if proposal_id in explicitly_linked_ids:
            continue
        if float(proposal["det_prob"]) >= restore_threshold:
            restored = restore_rejected_proposal(proposal, clip_duration)
            validate_output_event(restored, clip_duration)
            output_events.append(restored)

    output_events.sort(
        key=lambda event: (
            float(event["start_time_seconds"]),
            float(event["end_time_seconds"]),
            str(event["event_id"]),
        )
    )
    return {
        "clip_id": clip_id,
        "prompt_version": prediction_payload.get("prompt_version"),
        "model_name": prediction_payload.get("model_name"),
        "backend": "timing_preserving_validator_shadow_mode",
        "input_image_path": prediction_payload.get("input_image_path", ""),
        "proposal_source": "batdetect2",
        "proposal_threshold": proposal_payload.get("proposal_threshold"),
        "high_confidence_restore_threshold": restore_threshold,
        "clip_duration_seconds": clip_duration,
        "events": output_events,
    }


def count_decisions(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timing_total: Counter[str] = Counter()
    frequency_total: Counter[str] = Counter()
    validation_total: Counter[str] = Counter()
    for payload in payloads:
        timing = Counter(event["timing_decision"] for event in payload["events"])
        frequency = Counter(event["frequency_decision"] for event in payload["events"])
        validation = Counter(event["validation_decision"] for event in payload["events"])
        timing_total.update(timing)
        frequency_total.update(frequency)
        validation_total.update(validation)
        rows.append(_decision_row(payload["clip_id"], timing, frequency, validation))
    rows.append(_decision_row("ALL", timing_total, frequency_total, validation_total))
    return rows


def _decision_row(
    clip_id: str,
    timing: Counter[str],
    frequency: Counter[str],
    validation: Counter[str],
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        **{f"timing__{decision}": timing[decision] for decision in TIMING_DECISIONS},
        **{
            f"frequency__{decision}": frequency[decision]
            for decision in FREQUENCY_DECISIONS
        },
        **{
            f"validation__{decision}": validation[decision]
            for decision in VALIDATION_DECISIONS
        },
        "total_events": sum(validation.values()),
    }


def write_decision_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        *(f"timing__{decision}" for decision in TIMING_DECISIONS),
        *(f"frequency__{decision}" for decision in FREQUENCY_DECISIONS),
        *(f"validation__{decision}" for decision in VALIDATION_DECISIONS),
        "total_events",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_validator(
    *,
    proposal_dir: Path,
    prediction_dir: Path,
    audit_csv: Path,
    output_dir: Path,
    clip_ids: list[str],
    overwrite: bool,
) -> tuple[list[Path], Path]:
    prediction_output_dir = output_dir / "predictions"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = load_audit_rows(audit_csv)
    paths: list[Path] = []
    payloads: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        source_prediction_path = prediction_dir / f"{clip_id}_predictions.json"
        proposal_path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
        output_path = prediction_output_dir / f"{clip_id}_predictions.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
        prediction_payload = json.loads(source_prediction_path.read_text(encoding="utf-8"))
        proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        constrained = constrain_clip_payload(
            clip_id=clip_id,
            prediction_payload=prediction_payload,
            proposal_payload=proposal_payload,
            audit_rows=audit_rows,
        )
        constrained["source_prediction_path"] = source_prediction_path.as_posix()
        constrained["proposal_metadata_path"] = proposal_path.as_posix()
        output_path.write_text(
            json.dumps(constrained, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths.append(output_path)
        payloads.append(constrained)
    summary_path = output_dir / "validation_decision_summary.csv"
    write_decision_summary(summary_path, count_decisions(payloads))
    return paths, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-list", default=",".join(DEFAULT_CLIP_IDS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, summary_path = apply_validator(
        proposal_dir=args.proposal_dir,
        prediction_dir=args.prediction_dir,
        audit_csv=args.audit_csv,
        output_dir=args.output_dir,
        clip_ids=parse_clip_ids(args.clip_list),
        overwrite=args.overwrite,
    )
    print(f"Created {len(paths)} timing-preserving prediction files:")
    for path in paths:
        print(path)
    print(f"Decision summary: {summary_path}")


if __name__ == "__main__":
    main()
