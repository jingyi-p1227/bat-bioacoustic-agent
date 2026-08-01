"""Apply deterministic proposal-preserving rules in shadow mode.

This script writes new constrained prediction artifacts. It never modifies the
unconstrained P6D.2 predictions, source proposals, ground truth, or evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")
DEFAULT_PROPOSAL_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6"
)
DEFAULT_PREDICTION_DIR = Path(
    "outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/predictions"
)
DEFAULT_AUDIT_CSV = Path(
    "outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/"
    "proposal_deviation_analysis/proposal_deviation_events.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/agent_runs/p6_proposal_preserving_validator_representative6"
)
HIGH_CONFIDENCE_RESTORE_THRESHOLD = 0.7

GEOMETRY_FIELDS = (
    "start_time_seconds",
    "end_time_seconds",
    "low_frequency_hz",
    "high_frequency_hz",
)
DECISIONS = (
    "kept_vlm_geometry",
    "preserved_proposal_geometry",
    "restored_high_confidence_proposal",
    "kept_new_vlm_event",
)


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_audit_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Index P6E.1 rows by clip and prediction event id."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["clip_id"], row["prediction_id"])
        if key in result:
            raise ValueError(f"Duplicate deviation audit row: {key}")
        result[key] = row
    return result


def proposal_index(payload: dict[str, Any], clip_id: str) -> dict[str, dict[str, Any]]:
    if payload.get("clip_id") != clip_id:
        raise ValueError(f"Proposal payload clip_id does not match {clip_id}")
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("Proposal payload events must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for event in events:
        proposal_id = str(event.get("proposal_id") or "")
        if not proposal_id:
            raise ValueError("Proposal is missing proposal_id")
        if proposal_id in indexed:
            raise ValueError(f"Duplicate proposal_id: {proposal_id}")
        indexed[proposal_id] = event
    return indexed


def should_preserve_proposal_geometry(audit_row: dict[str, str]) -> tuple[bool, str]:
    """Apply P6E.2 geometry-preservation thresholds to one linked prediction."""
    reasons: list[str] = []
    if float(audit_row["time_iou_between_proposal_and_prediction"]) < 0.5:
        reasons.append("proposal-prediction time IoU is below 0.5")
    if parse_bool(audit_row["unsupported_geometry_change"]):
        reasons.append("P6E.1 marked the geometry change unsupported")
    if abs(float(audit_row["delta_start_ms"])) > 10:
        reasons.append("absolute start shift exceeds 10 ms")
    if abs(float(audit_row["delta_end_ms"])) > 10:
        reasons.append("absolute end shift exceeds 10 ms")
    ratio = float(audit_row["duration_ratio"])
    if ratio > 2.0 or ratio < 0.5:
        reasons.append("duration ratio is outside [0.5, 2.0]")
    if parse_bool(audit_row["frequency_shift_large"]):
        reasons.append("frequency-boundary shift exceeds 10 kHz")
    return bool(reasons), "; ".join(dict.fromkeys(reasons))


def clipped_proposal_geometry(
    proposal: dict[str, Any], clip_duration_seconds: float
) -> dict[str, float]:
    """Return source geometry clipped only to physical clip time bounds."""
    start = max(0.0, float(proposal["start_time_seconds"]))
    end = min(clip_duration_seconds, float(proposal["end_time_seconds"]))
    low = float(proposal["low_frequency_hz"])
    high = float(proposal["high_frequency_hz"])
    if start >= end or low >= high:
        raise ValueError(f"Invalid proposal geometry for {proposal.get('proposal_id')}")
    return {
        "start_time_seconds": start,
        "end_time_seconds": end,
        "low_frequency_hz": low,
        "high_frequency_hz": high,
    }


def geometry_snapshot(event: dict[str, Any]) -> dict[str, float]:
    return {field: float(event[field]) for field in GEOMETRY_FIELDS}


def preserve_linked_event(
    event: dict[str, Any],
    proposal: dict[str, Any],
    audit_row: dict[str, str],
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Keep or replace linked VLM geometry and attach shadow-mode provenance."""
    output = deepcopy(event)
    proposal_id = str(proposal["proposal_id"])
    preserve, reason = should_preserve_proposal_geometry(audit_row)
    output["source_proposal_id"] = proposal_id
    output["det_prob"] = float(proposal["det_prob"])
    output["class_prob"] = float(proposal["class_prob"])
    if preserve:
        output["original_vlm_geometry"] = geometry_snapshot(event)
        output.update(clipped_proposal_geometry(proposal, clip_duration_seconds))
        output["validation_decision"] = "preserved_proposal_geometry"
        output["validation_reason"] = reason
        output["human_review_needed"] = True
        output["review_reason"] = _append_reason(
            str(output.get("review_reason") or ""),
            "Deterministic validator restored BatDetect2 geometry in shadow mode.",
        )
    else:
        output["validation_decision"] = "kept_vlm_geometry"
        output["validation_reason"] = "VLM geometry stayed within proposal-deviation limits."
    return output


def _append_reason(existing: str, addition: str) -> str:
    return f"{existing} {addition}".strip() if existing else addition


def keep_new_vlm_event(event: dict[str, Any]) -> dict[str, Any]:
    """Retain an explicitly unlinked VLM event but require review."""
    output = deepcopy(event)
    output["validation_decision"] = "kept_new_vlm_event"
    output["validation_reason"] = "VLM event has no explicit source proposal."
    output["source_proposal_id"] = ""
    output["human_review_needed"] = True
    output["review_reason"] = _append_reason(
        str(output.get("review_reason") or ""),
        "New VLM event requires human review because it lacks detector provenance.",
    )
    return output


def restore_high_confidence_proposal(
    proposal: dict[str, Any],
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Create an evaluator-compatible reviewed event for a rejected strong proposal."""
    proposal_id = str(proposal["proposal_id"])
    det_prob = float(proposal["det_prob"])
    output: dict[str, Any] = {
        "event_id": f"restored_{proposal_id}",
        **clipped_proposal_geometry(proposal, clip_duration_seconds),
        "label": "bat_call",
        "confidence": det_prob,
        "evidence": "High-confidence BatDetect2 proposal restored by shadow validator.",
        "human_review_needed": True,
        "review_reason": "VLM rejected a BatDetect2 proposal with det_prob >= 0.7.",
        "used_proposal_id": proposal_id,
        "proposal_source": "batdetect2",
        "refinement_note": "Original proposal geometry restored without VLM refinement.",
        "validation_decision": "restored_high_confidence_proposal",
        "validation_reason": (
            f"Rejected proposal det_prob {det_prob:.3f} met restoration threshold "
            f"{HIGH_CONFIDENCE_RESTORE_THRESHOLD:.3f}."
        ),
        "source_proposal_id": proposal_id,
        "det_prob": det_prob,
        "class_prob": float(proposal["class_prob"]),
        "original_label": str(proposal.get("label") or ""),
    }
    return output


def validate_evaluator_compatible_event(event: dict[str, Any], clip_duration: float) -> None:
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
        "validation_reason",
        "source_proposal_id",
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
    if event["validation_decision"] not in DECISIONS:
        raise ValueError(f"Invalid validation_decision: {event['validation_decision']}")


def constrain_clip_payload(
    *,
    clip_id: str,
    prediction_payload: dict[str, Any],
    proposal_payload: dict[str, Any],
    audit_rows: dict[tuple[str, str], dict[str, str]],
    restore_threshold: float = HIGH_CONFIDENCE_RESTORE_THRESHOLD,
) -> dict[str, Any]:
    """Create one constrained payload while preserving source artifacts unchanged."""
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
            output_event = preserve_linked_event(
                event,
                proposals[used_id],
                audit_row,
                clip_duration,
            )
        else:
            output_event = keep_new_vlm_event(event)
        validate_evaluator_compatible_event(output_event, clip_duration)
        output_events.append(output_event)

    for proposal_id, proposal in proposals.items():
        if proposal_id in explicitly_linked_ids:
            continue
        if float(proposal["det_prob"]) >= restore_threshold:
            restored = restore_high_confidence_proposal(proposal, clip_duration)
            validate_evaluator_compatible_event(restored, clip_duration)
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
        "backend": "proposal_preserving_validator_shadow_mode",
        "source_prediction_path": prediction_payload.get("input_image_path", ""),
        "input_image_path": prediction_payload.get("input_image_path", ""),
        "proposal_source": "batdetect2",
        "proposal_threshold": proposal_payload.get("proposal_threshold"),
        "high_confidence_restore_threshold": restore_threshold,
        "clip_duration_seconds": clip_duration,
        "events": output_events,
    }


def count_decisions(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-clip and aggregate decision counts in stable column order."""
    rows: list[dict[str, Any]] = []
    aggregate: Counter[str] = Counter()
    for payload in payloads:
        counts = Counter(event["validation_decision"] for event in payload["events"])
        aggregate.update(counts)
        rows.append(
            {
                "clip_id": payload["clip_id"],
                **{decision: counts[decision] for decision in DECISIONS},
                "total_events": len(payload["events"]),
            }
        )
    rows.append(
        {
            "clip_id": "ALL",
            **{decision: aggregate[decision] for decision in DECISIONS},
            "total_events": sum(aggregate.values()),
        }
    )
    return rows


def write_decision_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip_id", *DECISIONS, "total_events"]
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
    print(f"Created {len(paths)} constrained prediction files:")
    for path in paths:
        print(path)
    print(f"Decision summary: {summary_path}")


if __name__ == "__main__":
    main()
