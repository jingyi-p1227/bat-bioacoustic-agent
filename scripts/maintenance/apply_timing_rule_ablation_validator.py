"""Generate and summarize P6E.4 deterministic timing-rule ablations.

The script operates only on existing proposal, VLM prediction, and P6E.1 audit
artifacts. It writes new shadow-mode outputs and never runs a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.maintenance.apply_timing_preserving_validator as timing


DEFAULT_OUTPUT_DIR = Path(
    "outputs/agent_runs/p6_timing_rule_ablation_representative6"
)
TARGET_CLIPS = ("OP_016", "OP_004", "OP_045")
ANCHORED_BOUNDARY_TOLERANCE_MS = 6.0
SHORT_PROPOSAL_THRESHOLD_MS = 8.0
SHORT_PROPOSAL_MIN_CONFIDENCE = 0.7

POLICIES = {
    "policy_a_p6e3_baseline": {"kind": "policy_a", "target_duration_ms": None},
    "policy_b_anchored_expansion": {"kind": "policy_b", "target_duration_ms": None},
    "policy_c_min_duration_8ms": {"kind": "policy_c", "target_duration_ms": 8.0},
    "policy_c_min_duration_10ms": {"kind": "policy_c", "target_duration_ms": 10.0},
    "policy_c_min_duration_12ms": {"kind": "policy_c", "target_duration_ms": 12.0},
}


def policy_output_dir(base_dir: Path, policy_name: str) -> Path:
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown timing policy: {policy_name}")
    return base_dir / policy_name


def classify_policy_b_timing(audit_row: dict[str, str]) -> tuple[str, str]:
    """Classify anchored expansion before near-rigid translation."""
    delta_start = float(audit_row["delta_start_ms"])
    delta_end = float(audit_row["delta_end_ms"])
    ratio = float(audit_row["duration_ratio"])
    same_direction = delta_start * delta_end > 0
    moderate_ratio = 0.5 <= ratio <= 2.0
    abs_start = abs(delta_start)
    abs_end = abs(delta_end)
    one_boundary_anchored = min(abs_start, abs_end) <= ANCHORED_BOUNDARY_TOLERANCE_MS
    other_boundary_expands = max(abs_start, abs_end) > ANCHORED_BOUNDARY_TOLERANCE_MS

    if ratio > 1.0 and moderate_ratio and one_boundary_anchored and other_boundary_expands:
        return (
            "keep_anchored_moderate_expansion",
            "One boundary is within 6 ms and duration expansion remains at or below 2x.",
        )
    if same_direction and moderate_ratio:
        return (
            "preserve_near_rigid_translation",
            "Start and end shift in the same direction with duration ratio in [0.5, 2.0].",
        )
    preserve, reason = timing.timing_preservation_rule(audit_row)
    if preserve:
        return "preserve_p6e3_fallback", reason
    return "keep_p6e3_fallback", "P6E.3 timing thresholds did not fire."


def expand_interval_around_center(
    start: float,
    end: float,
    target_duration_ms: float,
    clip_duration_seconds: float,
) -> tuple[float, float]:
    """Expand to a target duration around center, clipped to physical bounds."""
    if not 0 <= start < end <= clip_duration_seconds:
        raise ValueError("Source interval must lie within clip bounds")
    target_seconds = target_duration_ms / 1000.0
    if target_seconds <= 0:
        raise ValueError("Target duration must be positive")
    if end - start >= target_seconds:
        return start, end
    center = (start + end) / 2.0
    expanded_start = max(0.0, center - target_seconds / 2.0)
    expanded_end = min(clip_duration_seconds, center + target_seconds / 2.0)
    return expanded_start, expanded_end


def apply_policy_b_linked_event(
    event: dict[str, Any],
    proposal: dict[str, Any],
    audit_row: dict[str, str],
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Apply anchored-expansion and rigid-translation rules independently of frequency."""
    output = timing.apply_linked_event(
        event, proposal, audit_row, clip_duration_seconds
    )
    classification, reason = classify_policy_b_timing(audit_row)
    proposal_geometry = timing.clipped_proposal_geometry(proposal, clip_duration_seconds)
    if classification.startswith("keep_"):
        output["start_time_seconds"] = float(event["start_time_seconds"])
        output["end_time_seconds"] = float(event["end_time_seconds"])
        output["timing_decision"] = classification
    else:
        output["start_time_seconds"] = proposal_geometry["start_time_seconds"]
        output["end_time_seconds"] = proposal_geometry["end_time_seconds"]
        output["timing_decision"] = classification

    preserve_frequency, frequency_reason = timing.frequency_preservation_rule(audit_row)
    if preserve_frequency:
        output["low_frequency_hz"] = proposal_geometry["low_frequency_hz"]
        output["high_frequency_hz"] = proposal_geometry["high_frequency_hz"]
        output["frequency_decision"] = "preserved_proposal_frequency"
    else:
        output["low_frequency_hz"] = float(event["low_frequency_hz"])
        output["high_frequency_hz"] = float(event["high_frequency_hz"])
        output["frequency_decision"] = "kept_vlm_frequency"

    timing_preserved = classification.startswith("preserve_")
    if timing_preserved and preserve_frequency:
        output["validation_decision"] = "preserved_proposal_timing_and_frequency"
    elif timing_preserved:
        output["validation_decision"] = "preserved_proposal_timing"
    elif preserve_frequency:
        output["validation_decision"] = "preserved_proposal_frequency"
    else:
        output["validation_decision"] = "kept_vlm_geometry"
    output["validation_reason"] = "; ".join(
        item for item in (reason, frequency_reason) if item
    )
    if timing_preserved or preserve_frequency:
        output["human_review_needed"] = True
        output["review_reason"] = _append_reason(
            str(output.get("review_reason") or ""),
            "P6E.4 deterministic timing policy restored one or more proposal bounds.",
        )
    return output


def _append_reason(existing: str, addition: str) -> str:
    return f"{existing} {addition}".strip() if existing else addition


def apply_short_proposal_prior(
    event: dict[str, Any],
    proposal: dict[str, Any],
    clip_duration_seconds: float,
    target_duration_ms: float,
) -> dict[str, Any]:
    """Apply an exploratory minimum duration prior when eligibility is satisfied."""
    output = deepcopy(event)
    output.setdefault("source_proposal_id", str(proposal["proposal_id"]))
    output.setdefault("det_prob", float(proposal["det_prob"]))
    output.setdefault("class_prob", float(proposal["class_prob"]))
    output.setdefault("original_vlm_geometry", timing.geometry_snapshot(event))
    proposal_geometry = timing.clipped_proposal_geometry(proposal, clip_duration_seconds)
    proposal_duration_ms = (
        proposal_geometry["end_time_seconds"] - proposal_geometry["start_time_seconds"]
    ) * 1000.0
    det_prob = float(proposal["det_prob"])
    eligible = (
        det_prob >= SHORT_PROPOSAL_MIN_CONFIDENCE
        and proposal_duration_ms < SHORT_PROPOSAL_THRESHOLD_MS
    )
    if not eligible or proposal_duration_ms >= target_duration_ms:
        output["duration_prior_applied"] = False
        output["duration_prior_target_ms"] = target_duration_ms
        return output

    new_start, new_end = expand_interval_around_center(
        proposal_geometry["start_time_seconds"],
        proposal_geometry["end_time_seconds"],
        target_duration_ms,
        clip_duration_seconds,
    )
    output["start_time_seconds"] = new_start
    output["end_time_seconds"] = new_end
    output["timing_decision"] = f"expanded_short_proposal_to_{target_duration_ms:g}ms"
    output["validation_decision"] = "exploratory_short_proposal_duration_prior"
    output["validation_reason"] = (
        f"Exploratory shadow prior expanded det_prob={det_prob:.3f}, "
        f"duration={proposal_duration_ms:.3f} ms proposal toward {target_duration_ms:g} ms."
    )
    output["duration_prior_applied"] = True
    output["duration_prior_target_ms"] = target_duration_ms
    output["human_review_needed"] = True
    output["review_reason"] = _append_reason(
        str(output.get("review_reason") or ""),
        "Exploratory duration prior requires human review.",
    )
    return output


def validate_event(event: dict[str, Any], clip_duration: float) -> None:
    required = {
        "event_id",
        "start_time_seconds",
        "end_time_seconds",
        "low_frequency_hz",
        "high_frequency_hz",
        "label",
        "confidence",
        "validation_decision",
        "timing_decision",
        "frequency_decision",
        "validation_reason",
        "source_proposal_id",
        "original_vlm_geometry",
        "human_review_needed",
        "review_reason",
    }
    missing = sorted(required - event.keys())
    if missing:
        raise ValueError(f"Policy event missing fields: {', '.join(missing)}")
    start = float(event["start_time_seconds"])
    end = float(event["end_time_seconds"])
    low = float(event["low_frequency_hz"])
    high = float(event["high_frequency_hz"])
    if not 0 <= start < end <= clip_duration:
        raise ValueError(f"Invalid policy time geometry: {start}, {end}")
    if not 0 <= low < high:
        raise ValueError(f"Invalid policy frequency geometry: {low}, {high}")


def constrain_clip_payload(
    *,
    clip_id: str,
    prediction_payload: dict[str, Any],
    proposal_payload: dict[str, Any],
    audit_rows: dict[tuple[str, str], dict[str, str]],
    policy_name: str,
) -> dict[str, Any]:
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}")
    policy = POLICIES[policy_name]
    clip_duration = float(prediction_payload["clip_duration_seconds"])
    proposals = timing.proposal_index(proposal_payload, clip_id)
    events = prediction_payload.get("events")
    if not isinstance(events, list):
        raise ValueError("Prediction events must be a list")
    output_events: list[dict[str, Any]] = []
    explicitly_linked: set[str] = set()

    for event in events:
        event_id = str(event.get("event_id") or "")
        proposal_id = str(event.get("used_proposal_id") or "")
        if proposal_id and proposal_id in proposals:
            explicitly_linked.add(proposal_id)
            audit_row = audit_rows.get((clip_id, event_id))
            if audit_row is None:
                raise ValueError(f"Missing P6E.1 row for {clip_id}/{event_id}")
            if policy["kind"] == "policy_a":
                output = timing.apply_linked_event(
                    event, proposals[proposal_id], audit_row, clip_duration
                )
            else:
                output = apply_policy_b_linked_event(
                    event, proposals[proposal_id], audit_row, clip_duration
                )
                if policy["kind"] == "policy_c":
                    output = apply_short_proposal_prior(
                        output,
                        proposals[proposal_id],
                        clip_duration,
                        float(policy["target_duration_ms"]),
                    )
        else:
            output = timing.keep_new_vlm_event(event)
            if policy["kind"] == "policy_c":
                output["duration_prior_applied"] = False
                output["duration_prior_target_ms"] = policy["target_duration_ms"]
        validate_event(output, clip_duration)
        output_events.append(output)

    for proposal_id, proposal in proposals.items():
        if proposal_id in explicitly_linked:
            continue
        if float(proposal["det_prob"]) >= timing.HIGH_CONFIDENCE_RESTORE_THRESHOLD:
            restored = timing.restore_rejected_proposal(proposal, clip_duration)
            if policy["kind"] == "policy_c":
                restored = apply_short_proposal_prior(
                    restored,
                    proposal,
                    clip_duration,
                    float(policy["target_duration_ms"]),
                )
            validate_event(restored, clip_duration)
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
        "backend": "timing_rule_ablation_shadow_mode",
        "policy_name": policy_name,
        "input_image_path": prediction_payload.get("input_image_path", ""),
        "clip_duration_seconds": clip_duration,
        "events": output_events,
    }


def write_policy_decision_summary(path: Path, payloads: list[dict[str, Any]]) -> None:
    counts: Counter[tuple[str, str]] = Counter()
    for payload in payloads:
        for event in payload["events"]:
            counts[("timing", event["timing_decision"])] += 1
            counts[("frequency", event["frequency_decision"])] += 1
            counts[("validation", event["validation_decision"])] += 1
    if any("duration_prior_applied" in event for payload in payloads for event in payload["events"]):
        applied = sum(
            bool(event.get("duration_prior_applied"))
            for payload in payloads
            for event in payload["events"]
        )
        counts[("duration_prior", "applied")] = applied
        counts[("duration_prior", "not_applied")] = sum(
            len(payload["events"]) for payload in payloads
        ) - applied
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["decision_axis", "decision", "count"])
        writer.writeheader()
        for (axis, decision), count in sorted(counts.items()):
            writer.writerow({"decision_axis": axis, "decision": decision, "count": count})


def generate_policy_outputs(
    *,
    proposal_dir: Path,
    prediction_dir: Path,
    audit_csv: Path,
    output_dir: Path,
    clip_ids: list[str],
    overwrite: bool,
) -> dict[str, list[Path]]:
    audit_rows = timing.load_audit_rows(audit_csv)
    generated: dict[str, list[Path]] = {}
    for policy_name in POLICIES:
        run_dir = policy_output_dir(output_dir, policy_name)
        prediction_output_dir = run_dir / "predictions"
        prediction_output_dir.mkdir(parents=True, exist_ok=True)
        payloads: list[dict[str, Any]] = []
        paths: list[Path] = []
        for clip_id in clip_ids:
            prediction_path = prediction_dir / f"{clip_id}_predictions.json"
            proposal_path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
            output_path = prediction_output_dir / f"{clip_id}_predictions.json"
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
            prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
            output = constrain_clip_payload(
                clip_id=clip_id,
                prediction_payload=prediction_payload,
                proposal_payload=proposal_payload,
                audit_rows=audit_rows,
                policy_name=policy_name,
            )
            output["source_prediction_path"] = prediction_path.as_posix()
            output["proposal_metadata_path"] = proposal_path.as_posix()
            output_path.write_text(
                json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            payloads.append(output)
            paths.append(output_path)
        write_policy_decision_summary(run_dir / "validation_decision_summary.csv", payloads)
        generated[policy_name] = paths
    return generated


def build_summary_rows(base_dir: Path, policy_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name in policy_names:
        evaluation_dir = policy_output_dir(base_dir, policy_name) / "evaluation"
        aggregate = json.loads(
            (evaluation_dir / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        with (evaluation_dir / "per_clip_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            per_clip = {row["clip_id"]: row for row in csv.DictReader(handle)}
        row: dict[str, Any] = {
            "policy_name": policy_name,
            "clip_scope": "representative6",
            "prediction_count": aggregate["total_predictions"],
            "TP": aggregate["total_tp"],
            "FP": aggregate["total_fp"],
            "FN": aggregate["total_fn"],
            "precision": aggregate["precision"],
            "recall": aggregate["recall"],
            "F1": aggregate["f1"],
            "mean_time_iou": aggregate["mean_time_iou"],
            "mean_frequency_iou": aggregate["mean_frequency_iou"],
            "mean_box_iou": aggregate["mean_box_iou"],
            "box_iou_gte_0_3": aggregate["strict_box_iou_0_3_count"],
            "box_iou_gte_0_5": aggregate["strict_box_iou_0_5_count"],
        }
        for clip_id in TARGET_CLIPS:
            metrics = per_clip[clip_id]
            for field in ("tp", "fp", "fn", "f1"):
                row[f"{clip_id}_{field.upper()}"] = metrics[field]
        rows.append(row)
    return rows


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("At least one policy summary row is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-dir", type=Path, default=timing.DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=timing.DEFAULT_PREDICTION_DIR)
    parser.add_argument("--audit-csv", type=Path, default=timing.DEFAULT_AUDIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-list", default=",".join(timing.DEFAULT_CLIP_IDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only:
        rows = build_summary_rows(args.output_dir, list(POLICIES))
        path = args.output_dir / "timing_rule_ablation_summary.csv"
        write_summary_csv(path, rows)
        print(f"Summary: {path}")
        return
    generated = generate_policy_outputs(
        proposal_dir=args.proposal_dir,
        prediction_dir=args.prediction_dir,
        audit_csv=args.audit_csv,
        output_dir=args.output_dir,
        clip_ids=timing.parse_clip_ids(args.clip_list),
        overwrite=args.overwrite,
    )
    for policy_name, paths in generated.items():
        print(f"{policy_name}: {len(paths)} prediction files")


if __name__ == "__main__":
    main()
