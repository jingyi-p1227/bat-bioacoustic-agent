"""Summarise completed and missing single-agent full45 localisation runs.

This script is intentionally no-inference. It audits expected full45
conditions, reuses the explicit P8 multi-protocol evaluator, and writes a
consolidated report without modifying frozen predictions or ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_multi_protocol_detection import (  # noqa: E402
    FrozenRun,
    all_clip_ids,
    evaluate_runs,
    write_csv,
)


DEFAULT_EVAL_DIR = REPO_ROOT / "outputs/evaluation_sets/ozimops_petersi_v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/analysis_reports/single_agent_full45_summary"
)
PROTOCOL_ORDER = (
    "temporal_iou_0.3",
    "temporal_iou_0.1",
    "start_time_proximity_10ms",
)


@dataclass(frozen=True)
class ExpectedCondition:
    """One expected single-agent full45 condition."""

    condition_id: str
    label: str
    method: str
    model: str
    input_condition: str
    prediction_dir: Path | None
    uses_batdetect2_proposals: bool
    uses_vlm: bool
    notes: str
    comparable_to_bbox_protocol: bool = True


def expected_conditions() -> list[ExpectedCondition]:
    """Return the full45 conditions requested for audit."""

    return [
        ExpectedCondition(
            "grid_v2_baseline",
            "grid_v2 baseline",
            "fixed_grid_v2",
            "qwen3.6:latest",
            "clean dB spectrogram + grid_v2",
            REPO_ROOT / "outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2",
            False,
            True,
            "Frozen P5 full45 baseline.",
        ),
        ExpectedCondition(
            "pcen_grid_v2",
            "PCEN + grid_v2",
            "pcen_grid_v2",
            "qwen3.6:latest",
            "PCEN spectrogram + grid_v2",
            REPO_ROOT / "outputs/agent_runs/p8c_pcen_grid_v2_qwen3_6_full45/predictions",
            False,
            True,
            "Completed P8C confirmatory PCEN+grid_v2 run.",
        ),
        ExpectedCondition(
            "tiled_spectrogram",
            "tiled spectrogram",
            "0.5s_tiled",
            "qwen3.6:latest",
            "clean dB grid_v2 0.5s tiles merged to clip level",
            REPO_ROOT
            / "outputs/agent_runs/p6_tiled_qwen3_6_tile_0p5_overlap_0p1_full45/merged_predictions",
            False,
            True,
            "Full45 0.5s tiled run merged from tile-level qwen3.6 predictions when present.",
        ),
        ExpectedCondition(
            "batdetect2_proposal_only",
            "BatDetect2 proposal-only",
            "batdetect2_proposal_only",
            "batdetect2",
            "BatDetect2 proposal metadata converted to prediction JSON",
            REPO_ROOT
            / "outputs/agent_runs/p6_batdetect2_proposal_only_full45/predictions",
            True,
            False,
            "Full45 proposal-only output directory; generated from BatDetect2 1.3.1 proposals at det_prob >= 0.30 when present.",
        ),
        ExpectedCondition(
            "batdetect2_proposal_constrained_vlm",
            "BatDetect2-proposal-constrained VLM",
            "batdetect2_proposal_constrained_vlm",
            "qwen3.6:latest",
            "clean dB grid_v2 with compact BatDetect2 proposal metadata",
            REPO_ROOT
            / "outputs/agent_runs/p7c_full45_proposal_constrained_qwen3_6/predictions",
            True,
            True,
            "Full45 VLM condition using BatDetect2 proposals as candidate metadata, not visual overlays.",
        ),
        ExpectedCondition(
            "default_no_library",
            "default no-library workflow",
            "fixed_grid_v2",
            "qwen3.6:latest",
            "clean dB spectrogram + grid_v2, no retrieval context",
            REPO_ROOT / "outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2",
            False,
            True,
            "Equivalent to the frozen grid_v2 baseline for this localisation task.",
        ),
        ExpectedCondition(
            "acoustic_library",
            "acoustic library",
            "acoustic_library",
            "qwen3.6:latest",
            "clean dB grid_v2 with compact Walters acoustic-reference context",
            REPO_ROOT
            / "outputs/agent_runs/p7c_full45_walters_acoustic_qwen3_6/predictions",
            False,
            True,
            "Full45 run using compact Walters generic acoustic guidance when present.",
        ),
        ExpectedCondition(
            "annotation_example_library",
            "annotation example library",
            "annotation_example_library",
            "qwen3.6:latest",
            "clean dB grid_v2 with leakage-safe annotation memory",
            REPO_ROOT
            / "outputs/agent_runs/p7c_full45_annotation_memory_qwen3_6/predictions",
            False,
            True,
            "Missing full45 run; requires leakage-safe retrieval policy before inference.",
        ),
        ExpectedCondition(
            "acoustic_annotation_combined",
            "acoustic + annotation combined",
            "acoustic_annotation_combined",
            "qwen3.6:latest",
            "clean dB grid_v2 with both knowledge stores",
            REPO_ROOT
            / "outputs/agent_runs/p7c_full45_combined_library_qwen3_6/predictions",
            False,
            True,
            "Missing full45 run; requires the same leakage-safe retrieval policy.",
        ),
        ExpectedCondition(
            "best_stack_qwen3_6",
            "best-stack Qwen3.6",
            "best_stack_proposal_conservative",
            "qwen3.6:latest",
            "clean dB grid_v2 with conservative BatDetect2 proposal metadata",
            REPO_ROOT
            / "outputs/agent_runs/p14_best_stack_qwen3_6_proposal_constrained_conservative/predictions",
            True,
            True,
            "Final selected Qwen3.6 single-agent stack: clean grid_v2 plus conservative BatDetect2 proposal verification.",
        ),
    ]


def prediction_candidates(prediction_dir: Path, clip_id: str) -> tuple[Path, ...]:
    """Return known prediction filename variants."""

    return (
        prediction_dir / f"{clip_id}_predictions.json",
        prediction_dir / f"{clip_id}_prediction.json",
        prediction_dir / f"{clip_id}.json",
    )


def prediction_file_for(prediction_dir: Path, clip_id: str) -> Path | None:
    for path in prediction_candidates(prediction_dir, clip_id):
        if path.is_file():
            return path
    return None


def is_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed)


def is_valid_bbox(event: dict[str, Any]) -> bool:
    """Return whether an event has valid evaluator-compatible bbox geometry."""

    start = event.get("start_time_seconds", event.get("start_time"))
    end = event.get("end_time_seconds", event.get("end_time"))
    low = event.get("low_frequency_hz", event.get("low_frequency"))
    high = event.get("high_frequency_hz", event.get("high_frequency"))
    if not all(is_number(value) for value in (start, end, low, high)):
        return False
    return float(start) < float(end) and float(low) < float(high)


def prediction_quality_counts(
    prediction_dir: Path | None, clip_ids: tuple[str, ...]
) -> dict[str, int]:
    """Count parse success/failure and invalid bbox events without mutating files."""

    counts = {
        "prediction_files_found": 0,
        "parse_success_count": 0,
        "parse_failure_count": 0,
        "raw_event_count": 0,
        "invalid_bbox_count": 0,
    }
    if prediction_dir is None or not prediction_dir.exists():
        counts["parse_failure_count"] = len(clip_ids)
        return counts
    for clip_id in clip_ids:
        path = prediction_file_for(prediction_dir, clip_id)
        if path is None:
            counts["parse_failure_count"] += 1
            continue
        counts["prediction_files_found"] += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            counts["parse_failure_count"] += 1
            continue
        if payload.get("parse_status") and payload.get("parse_status") != "success":
            counts["parse_failure_count"] += 1
            continue
        counts["parse_success_count"] += 1
        events = payload.get("events", [])
        if not isinstance(events, list):
            counts["invalid_bbox_count"] += 1
            continue
        counts["raw_event_count"] += len(events)
        counts["invalid_bbox_count"] += sum(
            1 for event in events if not isinstance(event, dict) or not is_valid_bbox(event)
        )
    return counts


def audit_conditions(clip_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    """Audit expected full45 conditions for existence and compatibility."""

    rows: list[dict[str, Any]] = []
    for condition in expected_conditions():
        exists = condition.prediction_dir is not None and condition.prediction_dir.exists()
        counts = prediction_quality_counts(condition.prediction_dir, clip_ids)
        complete = counts["prediction_files_found"] == len(clip_ids)
        rows.append(
            {
                "condition_id": condition.condition_id,
                "label": condition.label,
                "status": "complete" if exists and complete else "missing_or_incomplete",
                "prediction_dir": ""
                if condition.prediction_dir is None
                else condition.prediction_dir.relative_to(REPO_ROOT).as_posix(),
                "prediction_files_found": counts["prediction_files_found"],
                "expected_clip_count": len(clip_ids),
                "parse_success_count": counts["parse_success_count"],
                "parse_failure_count": counts["parse_failure_count"],
                "invalid_bbox_count": counts["invalid_bbox_count"],
                "uses_batdetect2_proposals": condition.uses_batdetect2_proposals,
                "uses_vlm": condition.uses_vlm,
                "comparable_to_bbox_protocol": condition.comparable_to_bbox_protocol,
                "notes": condition.notes,
            }
        )
    return rows


def runs_for_complete_conditions(clip_ids: tuple[str, ...]) -> list[FrozenRun]:
    """Build FrozenRun records for completed, comparable full45 conditions."""

    runs: list[FrozenRun] = []
    seen_dirs: set[Path] = set()
    for condition in expected_conditions():
        if not condition.comparable_to_bbox_protocol or condition.prediction_dir is None:
            continue
        counts = prediction_quality_counts(condition.prediction_dir, clip_ids)
        if counts["prediction_files_found"] != len(clip_ids):
            continue
        # Avoid duplicate metric rows when two audited conditions intentionally
        # point at the same frozen prediction directory.
        if condition.prediction_dir in seen_dirs:
            continue
        seen_dirs.add(condition.prediction_dir)
        runs.append(
            FrozenRun(
                experiment_id=f"single_agent_full45_{condition.condition_id}",
                experiment_group="single_agent_full45",
                clip_scope="full_45",
                method=condition.method,
                model=condition.model,
                prediction_dir=condition.prediction_dir,
                clip_ids=clip_ids,
                notes=condition.notes,
            )
        )
    return runs


def mean_abs(values: list[str]) -> float:
    parsed = [abs(float(value)) for value in values if value not in {"", None}]
    return sum(parsed) / len(parsed) if parsed else 0.0


def enrich_metric_rows(
    experiment_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add parse/invalid counts and error summaries to aggregate metric rows."""

    audit_by_id = {f"single_agent_full45_{row['condition_id']}": row for row in audit_rows}
    pairs_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pair_rows:
        pairs_by_key.setdefault((row["experiment_id"], row["protocol"]), []).append(row)

    enriched: list[dict[str, Any]] = []
    for row in experiment_rows:
        audit = audit_by_id.get(row["experiment_id"], {})
        pairs = pairs_by_key.get((row["experiment_id"], row["protocol"]), [])
        output = dict(row)
        output.update(
            {
                "parse_success_rate": (
                    float(audit.get("parse_success_count", row.get("parse_success_count", 0)))
                    / float(row["clip_count"])
                    if row["clip_count"]
                    else 0.0
                ),
                "invalid_bbox_count": audit.get("invalid_bbox_count", 0),
                "mean_abs_start_time_error_ms": mean_abs(
                    [str(item["start_time_error_ms"]) for item in pairs]
                ),
                "mean_abs_end_time_error_ms": mean_abs(
                    [str(item["end_time_error_ms"]) for item in pairs]
                ),
                "mean_abs_duration_error_ms": mean_abs(
                    [str(item["duration_error_ms"]) for item in pairs]
                ),
            }
        )
        enriched.append(output)
    return enriched


def write_interpretation_notes(path: Path, metrics: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    """Write concise interpretation notes from currently available full45 runs."""

    rows_03 = [row for row in metrics if row["protocol"] == "temporal_iou_0.3"]
    ranked = sorted(rows_03, key=lambda row: float(row["F1"]), reverse=True)
    missing = [row for row in audit_rows if row["status"] != "complete"]
    completed_ids = {row["experiment_id"] for row in rows_03}
    lines = [
        "# Single-Agent Full45 Interpretation Notes",
        "",
        "This note is generated from existing prediction JSON files only. It does not run inference and does not modify ground truth.",
        "",
        "## Completed comparable full45 rows",
        "",
    ]
    for row in ranked:
        lines.append(
            f"- `{row['experiment_id']}`: F1={float(row['F1']):.3f}, precision={float(row['precision']):.3f}, recall={float(row['recall']):.3f}, mean box IoU={float(row['mean_box_iou']):.3f} under temporal IoU >= 0.3."
        )
    lines.extend(["", "## Missing or incomplete requested conditions", ""])
    for row in missing:
        lines.append(
            f"- `{row['label']}`: {row['prediction_files_found']}/{row['expected_clip_count']} prediction files found. {row['notes']}"
        )
    lines.extend(
        [
            "",
            "## Answers from currently available evidence",
            "",
            "- Visual representation: among completed full45 conditions, the strongest visual baseline is identified by the highest F1 row above.",
            "- Tiled spectrogram full45: compare the `0.5s_tiled` row against fixed grid_v2 before deciding whether tiling is worth keeping.",
            "- PCEN: compare the PCEN+grid_v2 row against fixed grid_v2; previous P8C notes should remain the source for whether PCEN is worth keeping.",
            (
                "- BatDetect2 proposal-only: full45 proposal-only is now included in this table."
                if "single_agent_full45_batdetect2_proposal_only" in completed_ids
                else "- BatDetect2 proposal-only: full45 proposal-only outputs are not present; existing representative/held-out results should not be treated as full45."
            ),
            "- Libraries: compare completed Walters acoustic-library and annotation-library rows against fixed grid_v2; do not interpret missing combined-library outputs as negative evidence.",
            "- Best-stack: should not be frozen until tiled, proposal-only, and any approved library conditions are either completed or explicitly ruled out.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(path: Path, metrics: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    """Write a compact markdown report for dissertation planning."""

    rows_03 = [row for row in metrics if row["protocol"] == "temporal_iou_0.3"]
    rows_03.sort(key=lambda row: float(row["F1"]), reverse=True)
    lines = [
        "# Single-Agent Full45 Localisation Summary",
        "",
        "This report audits the requested Qwen3.6 single-agent full45 localisation conditions and consolidates metrics for conditions that already have complete, evaluator-compatible prediction JSON files.",
        "",
        "## Output Audit",
        "",
        "| Condition | Status | Files | Parse success | Invalid boxes | Notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['label']} | {row['status']} | {row['prediction_files_found']}/{row['expected_clip_count']} | {row['parse_success_count']} | {row['invalid_bbox_count']} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Main Full45 Results: Temporal IoU >= 0.3",
            "",
            "| Experiment | Method | Model | TP | FP | FN | Precision | Recall | F1 | Time IoU | Freq IoU | Box IoU | Parse success | Invalid boxes |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows_03:
        lines.append(
            f"| {row['experiment_id']} | {row['method']} | {row['model']} | {row['TP']} | {row['FP']} | {row['FN']} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['F1']):.3f} | {float(row['mean_time_iou']):.3f} | {float(row['mean_frequency_iou']):.3f} | {float(row['mean_box_iou']):.3f} | {float(row['parse_success_rate']):.3f} | {row['invalid_bbox_count']} |"
        )
    lines.extend(
        [
            "",
            "## Protocol Coverage",
            "",
            "The CSV companion includes `temporal_iou_0.3`, `temporal_iou_0.1`, and `start_time_proximity_10ms` rows for every completed condition.",
            "",
            "## Current Limitation",
            "",
            "Any missing VLM full45 condition should be written into a new experiment-specific output directory and then incorporated by re-running this summary script. Frozen runs are not overwritten.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    clip_ids = all_clip_ids(args.eval_dir)
    audit_rows = audit_conditions(clip_ids)
    runs = runs_for_complete_conditions(clip_ids)
    experiment_rows, case_rows, pair_rows = evaluate_runs(runs, args.eval_dir)
    metric_rows = enrich_metric_rows(experiment_rows, pair_rows, audit_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "condition_audit.csv", audit_rows)
    write_csv(args.output_dir / "single_agent_full45_metrics.csv", metric_rows)
    write_csv(args.output_dir / "case_level_results.csv", case_rows)
    write_csv(args.output_dir / "matched_pair_errors.csv", pair_rows)
    write_summary_report(
        args.output_dir / "single_agent_full45_summary.md",
        metric_rows,
        audit_rows,
    )
    write_interpretation_notes(
        args.output_dir / "interpretation_notes.md",
        metric_rows,
        audit_rows,
    )

    print(f"Audited {len(audit_rows)} expected condition(s).")
    print(f"Evaluated {len(runs)} complete full45 condition(s).")
    print(f"Wrote reports to {args.output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
