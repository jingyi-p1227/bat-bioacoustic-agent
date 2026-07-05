"""Audit BatDetect2-to-VLM geometry changes without altering predictions.

The geometry flags depend only on proposal and prediction artifacts. Existing
evaluation CSVs are read separately for retrospective outcome attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")
DEFAULT_PROPOSAL_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6"
)
DEFAULT_PREDICTION_DIR = Path(
    "outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/predictions"
)
DEFAULT_ASSISTED_EVALUATION_DIR = Path(
    "outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/evaluation"
)
DEFAULT_PROPOSAL_EVALUATION_DIR = Path(
    "outputs/agent_runs/p6_batdetect2_proposal_only_representative6/evaluation"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/"
    "proposal_deviation_analysis"
)

LARGE_TIME_SHIFT_MS = 10.0
DURATION_EXPANSION_RATIO = 2.0
DURATION_SHRINKAGE_RATIO = 0.5
LOW_SOURCE_TIME_IOU = 0.5
LARGE_FREQUENCY_SHIFT_HZ = 10000.0


class ProposalDeviationFlags(BaseModel):
    """Deterministic proposal-deviation rule results."""

    large_start_shift: bool
    large_end_shift: bool
    duration_expansion: bool
    duration_shrinkage: bool
    low_time_iou_with_source_proposal: bool
    frequency_shift_large: bool
    unsupported_geometry_change: bool

    @model_validator(mode="after")
    def validate_aggregate_flag(self) -> "ProposalDeviationFlags":
        component_flags = (
            self.large_start_shift,
            self.large_end_shift,
            self.duration_expansion,
            self.duration_shrinkage,
            self.low_time_iou_with_source_proposal,
            self.frequency_shift_large,
        )
        if self.unsupported_geometry_change != any(component_flags):
            raise ValueError("unsupported_geometry_change must equal any component flag")
        return self


class ValidatedPredictionDecision(BaseModel):
    """Non-mutating validator recommendation for one prediction geometry."""

    action: Literal[
        "accept_prediction_geometry",
        "preserve_original_proposal_geometry",
        "require_human_review",
        "unsupported_change",
    ]
    reason: str


class ProposalDeviationRecord(BaseModel):
    """Validated, flat proposal-to-prediction audit record."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    prediction_id: str
    linked_proposal_id: str
    link_method: str
    proposal_start_time_seconds: float
    predicted_start_time_seconds: float
    delta_start_ms: float
    proposal_end_time_seconds: float
    predicted_end_time_seconds: float
    delta_end_ms: float
    proposal_duration_ms: float = Field(gt=0)
    predicted_duration_ms: float = Field(gt=0)
    duration_ratio: float = Field(gt=0)
    proposal_low_frequency_hz: float
    predicted_low_frequency_hz: float
    delta_low_frequency_hz: float
    proposal_high_frequency_hz: float
    predicted_high_frequency_hz: float
    delta_high_frequency_hz: float
    time_iou_between_proposal_and_prediction: float = Field(ge=0, le=1)
    frequency_iou_between_proposal_and_prediction: float = Field(ge=0, le=1)
    box_iou_between_proposal_and_prediction: float = Field(ge=0, le=1)
    large_start_shift: bool
    large_end_shift: bool
    duration_expansion: bool
    duration_shrinkage: bool
    low_time_iou_with_source_proposal: bool
    frequency_shift_large: bool
    unsupported_geometry_change: bool
    original_proposal_matched_gt: bool
    refined_prediction_matched_gt: bool
    outcome: str
    validator_decision: str
    validator_reason: str


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _interval_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - intersection
    return intersection / union if union > 0 else 0.0


def temporal_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    return _interval_iou(
        float(a["start_time_seconds"]),
        float(a["end_time_seconds"]),
        float(b["start_time_seconds"]),
        float(b["end_time_seconds"]),
    )


def frequency_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    return _interval_iou(
        float(a["low_frequency_hz"]),
        float(a["high_frequency_hz"]),
        float(b["low_frequency_hz"]),
        float(b["high_frequency_hz"]),
    )


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    time_intersection = max(
        0.0,
        min(float(a["end_time_seconds"]), float(b["end_time_seconds"]))
        - max(float(a["start_time_seconds"]), float(b["start_time_seconds"])),
    )
    frequency_intersection = max(
        0.0,
        min(float(a["high_frequency_hz"]), float(b["high_frequency_hz"]))
        - max(float(a["low_frequency_hz"]), float(b["low_frequency_hz"])),
    )
    intersection = time_intersection * frequency_intersection
    area_a = (
        float(a["end_time_seconds"]) - float(a["start_time_seconds"])
    ) * (float(a["high_frequency_hz"]) - float(a["low_frequency_hz"]))
    area_b = (
        float(b["end_time_seconds"]) - float(b["start_time_seconds"])
    ) * (float(b["high_frequency_hz"]) - float(b["low_frequency_hz"]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def validate_geometry(event: dict[str, Any], prefix: str) -> None:
    start = _finite_float(event.get("start_time_seconds"), f"{prefix}.start")
    end = _finite_float(event.get("end_time_seconds"), f"{prefix}.end")
    low = _finite_float(event.get("low_frequency_hz"), f"{prefix}.low_frequency")
    high = _finite_float(event.get("high_frequency_hz"), f"{prefix}.high_frequency")
    if start >= end or low >= high:
        raise ValueError(f"{prefix} has invalid geometry")


def link_prediction_to_proposal(
    prediction: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Link by explicit id, then temporal overlap, then nearest time center."""
    if not proposals:
        return None, "unlinked_no_proposals"
    used_id = str(prediction.get("used_proposal_id") or "")
    if used_id:
        for proposal in proposals:
            if str(proposal.get("proposal_id") or "") == used_id:
                return proposal, "used_proposal_id"

    overlap_ranked = sorted(
        proposals,
        key=lambda proposal: (
            temporal_iou(prediction, proposal),
            -abs(
                (float(prediction["start_time_seconds"]) + float(prediction["end_time_seconds"])) / 2
                - (float(proposal["start_time_seconds"]) + float(proposal["end_time_seconds"])) / 2
            ),
        ),
        reverse=True,
    )
    if temporal_iou(prediction, overlap_ranked[0]) > 0:
        return overlap_ranked[0], "maximum_temporal_overlap"

    prediction_center = (
        float(prediction["start_time_seconds"]) + float(prediction["end_time_seconds"])
    ) / 2
    nearest = min(
        proposals,
        key=lambda proposal: abs(
            prediction_center
            - (float(proposal["start_time_seconds"]) + float(proposal["end_time_seconds"])) / 2
        ),
    )
    return nearest, "nearest_temporal_center"


def calculate_flags(
    *,
    delta_start_ms: float,
    delta_end_ms: float,
    duration_ratio: float,
    source_time_iou: float,
    delta_low_frequency_hz: float,
    delta_high_frequency_hz: float,
) -> ProposalDeviationFlags:
    components = {
        "large_start_shift": abs(delta_start_ms) > LARGE_TIME_SHIFT_MS,
        "large_end_shift": abs(delta_end_ms) > LARGE_TIME_SHIFT_MS,
        "duration_expansion": duration_ratio > DURATION_EXPANSION_RATIO,
        "duration_shrinkage": duration_ratio < DURATION_SHRINKAGE_RATIO,
        "low_time_iou_with_source_proposal": source_time_iou < LOW_SOURCE_TIME_IOU,
        "frequency_shift_large": (
            abs(delta_low_frequency_hz) > LARGE_FREQUENCY_SHIFT_HZ
            or abs(delta_high_frequency_hz) > LARGE_FREQUENCY_SHIFT_HZ
        ),
    }
    return ProposalDeviationFlags(
        **components,
        unsupported_geometry_change=any(components.values()),
    )


def classify_outcome(proposal_matched: bool, prediction_matched: bool) -> str:
    if proposal_matched and prediction_matched:
        return "both_matched"
    if proposal_matched and not prediction_matched:
        return "proposal_was_good_but_prediction_broke_match"
    if not proposal_matched and prediction_matched:
        return "prediction_improved_bad_proposal"
    return "both_failed"


def choose_validator_decision(
    *,
    linked: bool,
    flags: ProposalDeviationFlags | None,
    outcome: str,
) -> ValidatedPredictionDecision:
    """Return an audit decision without changing the prediction artifact."""
    if not linked or flags is None:
        return ValidatedPredictionDecision(
            action="require_human_review",
            reason="Prediction could not be linked to a source proposal.",
        )
    if outcome == "proposal_was_good_but_prediction_broke_match":
        return ValidatedPredictionDecision(
            action="preserve_original_proposal_geometry",
            reason="Retrospective evaluation shows the proposal matched but the refinement did not.",
        )
    severe_change = (
        flags.large_start_shift
        or flags.large_end_shift
        or flags.duration_expansion
        or flags.duration_shrinkage
        or flags.frequency_shift_large
    )
    if severe_change:
        return ValidatedPredictionDecision(
            action="unsupported_change",
            reason="Prediction exceeds a deterministic geometry-deviation limit.",
        )
    if flags.low_time_iou_with_source_proposal:
        return ValidatedPredictionDecision(
            action="preserve_original_proposal_geometry",
            reason="Prediction has time IoU below 0.5 with its source proposal.",
        )
    return ValidatedPredictionDecision(
        action="accept_prediction_geometry",
        reason="Prediction remains within configured proposal-deviation limits.",
    )


def load_matched_ids(evaluation_dir: Path) -> dict[str, set[str]]:
    """Load official event-level prediction ids from matched_events.csv."""
    path = evaluation_dir / "matched_events.csv"
    matched: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            matched[row["clip_id"]].add(row["prediction_id"])
    return dict(matched)


def build_deviation_record(
    *,
    clip_id: str,
    prediction: dict[str, Any],
    proposal: dict[str, Any],
    link_method: str,
    proposal_matched: bool,
    prediction_matched: bool,
) -> ProposalDeviationRecord:
    validate_geometry(prediction, "prediction")
    validate_geometry(proposal, "proposal")
    proposal_start = float(proposal["start_time_seconds"])
    prediction_start = float(prediction["start_time_seconds"])
    proposal_end = float(proposal["end_time_seconds"])
    prediction_end = float(prediction["end_time_seconds"])
    proposal_duration_ms = (proposal_end - proposal_start) * 1000
    prediction_duration_ms = (prediction_end - prediction_start) * 1000
    duration_ratio = prediction_duration_ms / proposal_duration_ms
    proposal_low = float(proposal["low_frequency_hz"])
    prediction_low = float(prediction["low_frequency_hz"])
    proposal_high = float(proposal["high_frequency_hz"])
    prediction_high = float(prediction["high_frequency_hz"])
    delta_start_ms = (prediction_start - proposal_start) * 1000
    delta_end_ms = (prediction_end - proposal_end) * 1000
    delta_low = prediction_low - proposal_low
    delta_high = prediction_high - proposal_high
    time_iou = temporal_iou(proposal, prediction)
    flags = calculate_flags(
        delta_start_ms=delta_start_ms,
        delta_end_ms=delta_end_ms,
        duration_ratio=duration_ratio,
        source_time_iou=time_iou,
        delta_low_frequency_hz=delta_low,
        delta_high_frequency_hz=delta_high,
    )
    outcome = classify_outcome(proposal_matched, prediction_matched)
    decision = choose_validator_decision(linked=True, flags=flags, outcome=outcome)
    return ProposalDeviationRecord(
        clip_id=clip_id,
        prediction_id=str(prediction.get("event_id") or ""),
        linked_proposal_id=str(proposal.get("proposal_id") or ""),
        link_method=link_method,
        proposal_start_time_seconds=proposal_start,
        predicted_start_time_seconds=prediction_start,
        delta_start_ms=delta_start_ms,
        proposal_end_time_seconds=proposal_end,
        predicted_end_time_seconds=prediction_end,
        delta_end_ms=delta_end_ms,
        proposal_duration_ms=proposal_duration_ms,
        predicted_duration_ms=prediction_duration_ms,
        duration_ratio=duration_ratio,
        proposal_low_frequency_hz=proposal_low,
        predicted_low_frequency_hz=prediction_low,
        delta_low_frequency_hz=delta_low,
        proposal_high_frequency_hz=proposal_high,
        predicted_high_frequency_hz=prediction_high,
        delta_high_frequency_hz=delta_high,
        time_iou_between_proposal_and_prediction=time_iou,
        frequency_iou_between_proposal_and_prediction=frequency_iou(proposal, prediction),
        box_iou_between_proposal_and_prediction=box_iou(proposal, prediction),
        **flags.model_dump(),
        original_proposal_matched_gt=proposal_matched,
        refined_prediction_matched_gt=prediction_matched,
        outcome=outcome,
        validator_decision=decision.action,
        validator_reason=decision.reason,
    )


def analyze_clip(
    *,
    clip_id: str,
    proposal_dir: Path,
    prediction_dir: Path,
    proposal_match_ids: dict[str, set[str]],
    assisted_match_ids: dict[str, set[str]],
) -> tuple[list[ProposalDeviationRecord], int, int]:
    proposal_payload = json.loads(
        (proposal_dir / f"{clip_id}_batdetect2_proposals.json").read_text(encoding="utf-8")
    )
    prediction_payload = json.loads(
        (prediction_dir / f"{clip_id}_predictions.json").read_text(encoding="utf-8")
    )
    proposals = proposal_payload.get("events", [])
    predictions = prediction_payload.get("events", [])
    records: list[ProposalDeviationRecord] = []
    for prediction in predictions:
        proposal, method = link_prediction_to_proposal(prediction, proposals)
        if proposal is None:
            continue
        proposal_id = str(proposal.get("proposal_id") or "")
        prediction_id = str(prediction.get("event_id") or "")
        records.append(
            build_deviation_record(
                clip_id=clip_id,
                prediction=prediction,
                proposal=proposal,
                link_method=method,
                proposal_matched=proposal_id in proposal_match_ids.get(clip_id, set()),
                prediction_matched=prediction_id in assisted_match_ids.get(clip_id, set()),
            )
        )
    return records, len(proposals), len(predictions)


def _mean(records: list[ProposalDeviationRecord], field: str) -> float:
    values = [float(getattr(record, field)) for record in records]
    return fmean(values) if values else 0.0


def build_clip_summary(
    clip_id: str,
    records: list[ProposalDeviationRecord],
    proposal_count: int,
    prediction_count: int,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "proposal_count": proposal_count,
        "prediction_count": prediction_count,
        "linked_prediction_count": len(records),
        "unsupported_geometry_change_count": sum(
            record.unsupported_geometry_change for record in records
        ),
        "proposal_good_prediction_broke_count": sum(
            record.outcome == "proposal_was_good_but_prediction_broke_match"
            for record in records
        ),
        "prediction_improved_bad_proposal_count": sum(
            record.outcome == "prediction_improved_bad_proposal" for record in records
        ),
        "mean_abs_delta_start_ms": _mean_absolute(records, "delta_start_ms"),
        "mean_abs_delta_end_ms": _mean_absolute(records, "delta_end_ms"),
        "mean_duration_ratio": _mean(records, "duration_ratio"),
        "mean_proposal_prediction_time_iou": _mean(
            records, "time_iou_between_proposal_and_prediction"
        ),
        "mean_proposal_prediction_box_iou": _mean(
            records, "box_iou_between_proposal_and_prediction"
        ),
    }


def _mean_absolute(records: list[ProposalDeviationRecord], field: str) -> float:
    values = [abs(float(getattr(record, field))) for record in records]
    return fmean(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write deterministic CSV output, including an empty header-only table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames are required when writing no rows")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count(records: list[ProposalDeviationRecord], field: str, value: Any = True) -> int:
    return sum(getattr(record, field) == value for record in records)


def write_report(
    path: Path,
    records: list[ProposalDeviationRecord],
    summaries: list[dict[str, Any]],
) -> None:
    total = len(records)
    unsupported = _count(records, "unsupported_geometry_change")
    broke = _count(records, "outcome", "proposal_was_good_but_prediction_broke_match")
    improved = _count(records, "outcome", "prediction_improved_bad_proposal")
    both_matched = _count(records, "outcome", "both_matched")
    both_failed = _count(records, "outcome", "both_failed")
    op016 = [record for record in records if record.clip_id == "OP_016"]
    op045 = [record for record in records if record.clip_id == "OP_045"]
    harmful = sorted(
        (record for record in records if record.outcome == "proposal_was_good_but_prediction_broke_match"),
        key=lambda record: record.time_iou_between_proposal_and_prediction,
    )[:5]
    lines = [
        "# P6E.1 Proposal-Deviation Validation Report",
        "",
        "## Scope",
        "",
        "This is a deterministic, non-mutating audit of P6D.2. Geometry flags use only BatDetect2 proposals and VLM predictions. Outcome labels reuse the frozen event-level evaluation CSVs; no inference or GT modification was performed.",
        "",
        "## Aggregate Deviation Summary",
        "",
        f"- Linked predictions: {total}",
        f"- Unsupported geometry changes: {unsupported}",
        f"- Proposal matched but prediction broke match: {broke}",
        f"- Prediction improved an unmatched proposal: {improved}",
        f"- Both matched: {both_matched}",
        f"- Both failed: {both_failed}",
        f"- Mean absolute start shift: {_mean_absolute(records, 'delta_start_ms'):.3f} ms",
        f"- Mean absolute end shift: {_mean_absolute(records, 'delta_end_ms'):.3f} ms",
        f"- Mean duration ratio: {_mean(records, 'duration_ratio'):.3f}",
        f"- Mean proposal-prediction time IoU: {_mean(records, 'time_iou_between_proposal_and_prediction'):.3f}",
        f"- Mean proposal-prediction box IoU: {_mean(records, 'box_iou_between_proposal_and_prediction'):.3f}",
        "",
        "## Clip Summary",
        "",
        "| Clip | Proposals | Predictions | Unsupported | Good proposal broken | Bad proposal improved | Mean abs. start shift (ms) | Mean time IoU | Mean box IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['clip_id']} | {row['proposal_count']} | {row['prediction_count']} | "
            f"{row['unsupported_geometry_change_count']} | "
            f"{row['proposal_good_prediction_broke_count']} | "
            f"{row['prediction_improved_bad_proposal_count']} | "
            f"{row['mean_abs_delta_start_ms']:.3f} | "
            f"{row['mean_proposal_prediction_time_iou']:.3f} | "
            f"{row['mean_proposal_prediction_box_iou']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## OP_016",
            "",
            f"All {len(op016)} predictions linked to explicit BatDetect2 proposal IDs. "
            f"{_count(op016, 'unsupported_geometry_change')} were flagged, and "
            f"{_count(op016, 'outcome', 'proposal_was_good_but_prediction_broke_match')} previously matched proposals were broken by refinement. "
            "The repeated rightward shifts reduced proposal-prediction temporal overlap below the 0.5 guard for most events. The VLM still did not add the left-boundary event.",
            "",
            "## OP_045",
            "",
            f"All {len(op045)} predictions remained closely anchored to their proposals; "
            f"{_count(op045, 'unsupported_geometry_change')} triggered deviation flags. "
            "Both proposal-only and assisted predictions failed because the detector intervals were too short relative to the evaluation boxes. This is a source-proposal limitation, not an unsupported VLM shift, so a deviation guard alone cannot repair OP_045.",
            "",
            "## Harmful Shift Examples",
            "",
            "| Clip | Prediction | Proposal | Delta start ms | Delta end ms | Source time IoU | Decision |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for record in harmful:
        lines.append(
            f"| {record.clip_id} | {record.prediction_id} | {record.linked_proposal_id} | "
            f"{record.delta_start_ms:.3f} | {record.delta_end_ms:.3f} | "
            f"{record.time_iou_between_proposal_and_prediction:.3f} | "
            f"{record.validator_decision} |"
        )
    lines.extend(
        [
            "",
            "## Proposed Deterministic Validator",
            "",
            "Flag a change when start or end moves by more than 10 ms, duration expands above 2x or shrinks below 0.5x, proposal-prediction time IoU falls below 0.5, or either frequency boundary moves by more than 10 kHz.",
            "",
            "- `accept_prediction_geometry`: all deviation checks pass.",
            "- `preserve_original_proposal_geometry`: proposal-prediction time IoU is below 0.5, or retrospective analysis shows a good proposal was broken.",
            "- `unsupported_change`: a severe time, duration, or frequency limit is exceeded.",
            "- `require_human_review`: prediction cannot be linked reliably to a proposal.",
            "",
            "The deployment version must not use GT outcomes; those labels are included here only to validate whether the geometry-only rules catch harmful changes.",
            "",
            "## Recommendation for P6E.2",
            "",
            "Implement a proposal-preserving post-processor in shadow mode. Keep original proposal geometry whenever the VLM's source time IoU is below 0.5 or a severe deviation flag fires, while retaining VLM decisions to reject proposals or add genuinely new events. Evaluate original, unconstrained, and constrained outputs side by side on the same six clips before any 45-clip expansion.",
            "",
            "Do not introduce a critic agent yet. First establish whether deterministic geometry preservation recovers OP_016 without sacrificing the genuine OP_004 improvements. Handle OP_045 separately as a proposal-duration expansion problem requiring visible-evidence rules rather than simple preservation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument(
        "--assisted-evaluation-dir", type=Path, default=DEFAULT_ASSISTED_EVALUATION_DIR
    )
    parser.add_argument(
        "--proposal-evaluation-dir", type=Path, default=DEFAULT_PROPOSAL_EVALUATION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-list", default=",".join(DEFAULT_CLIP_IDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_ids = parse_clip_ids(args.clip_list)
    proposal_matches = load_matched_ids(args.proposal_evaluation_dir)
    assisted_matches = load_matched_ids(args.assisted_evaluation_dir)
    records: list[ProposalDeviationRecord] = []
    summaries: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        clip_records, proposal_count, prediction_count = analyze_clip(
            clip_id=clip_id,
            proposal_dir=args.proposal_dir,
            prediction_dir=args.prediction_dir,
            proposal_match_ids=proposal_matches,
            assisted_match_ids=assisted_matches,
        )
        records.extend(clip_records)
        summaries.append(
            build_clip_summary(clip_id, clip_records, proposal_count, prediction_count)
        )

    event_path = args.output_dir / "proposal_deviation_events.csv"
    summary_path = args.output_dir / "proposal_deviation_clip_summary.csv"
    report_path = args.output_dir / "p6e1_proposal_deviation_validation_report.md"
    event_rows = [record.model_dump() for record in records]
    write_csv(
        event_path,
        event_rows,
        fieldnames=list(ProposalDeviationRecord.model_fields),
    )
    write_csv(summary_path, summaries)
    write_report(report_path, records, summaries)
    print(f"Analyzed {len(records)} linked predictions across {len(clip_ids)} clips.")
    print(f"Unsupported geometry changes: {_count(records, 'unsupported_geometry_change')}")
    print(
        "Good proposals broken by prediction: "
        f"{_count(records, 'outcome', 'proposal_was_good_but_prediction_broke_match')}"
    )
    print(f"Event CSV: {event_path}")
    print(f"Clip summary CSV: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
