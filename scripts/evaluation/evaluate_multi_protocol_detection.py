"""Evaluate frozen prediction runs under multiple temporal matching protocols.

P8A uses this script to re-score existing prediction JSON files without
changing frozen model outputs. The 10 ms protocol follows BatDetect2's cached
evaluation code: predictions are assigned by start-time proximity with the
default detection-overlap threshold of 0.01 seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_audio_agent.evaluation.event_matching import (  # noqa: E402
    ClipEvaluation,
    EventBox,
    MatchingProtocol,
    aggregate_clip_evaluations,
    evaluate_clip,
)


DEFAULT_EVAL_DIR = REPO_ROOT / "outputs/evaluation_sets/ozimops_petersi_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/analysis_reports/p8a_multi_protocol_detection"
REPRESENTATIVE_SIX = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")
HELDOUT_TEN = (
    "OP_009",
    "OP_015",
    "OP_018",
    "OP_020",
    "OP_025",
    "OP_027",
    "OP_032",
    "OP_036",
    "OP_041",
    "OP_042",
)
PROTOCOLS = (
    MatchingProtocol.TEMPORAL_IOU_0_1,
    MatchingProtocol.TEMPORAL_IOU_0_3,
    MatchingProtocol.START_TIME_PROXIMITY_10MS,
)


@dataclass(frozen=True)
class FrozenRun:
    experiment_id: str
    experiment_group: str
    clip_scope: str
    method: str
    model: str
    prediction_dir: Path
    clip_ids: tuple[str, ...]
    notes: str = ""


def all_clip_ids(eval_dir: Path = DEFAULT_EVAL_DIR) -> tuple[str, ...]:
    manifest = eval_dir / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(row["clip_id"] for row in rows)


def default_frozen_runs(eval_dir: Path = DEFAULT_EVAL_DIR) -> list[FrozenRun]:
    full_45 = all_clip_ids(eval_dir)
    return [
        FrozenRun("p5_qwen_grid_v1_full", "full_45", "full_45", "fixed_grid_v1", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v1", full_45),
        FrozenRun("p5_qwen_grid_v2_full", "full_45", "full_45", "fixed_grid_v2", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2", full_45),
        FrozenRun("p5_gemma4_grid_v2_full", "full_45", "full_45", "fixed_grid_v2", "gemma4:31b", REPO_ROOT / "outputs/agent_runs/prompt_v2_full_gemma4_31b_grid_v2", full_45),
        FrozenRun("p5_gated_adaptive_full", "full_45", "full_45", "gated_adaptive_zoom", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/prompt_v2_gated_adaptive_zoom_qwen3_6_full", full_45),
        FrozenRun("p5_gated_overview_only_full", "full_45", "full_45", "gated_overview_only", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/prompt_v2_gated_overview_only_qwen3_6_full", full_45),
        FrozenRun("p8b_pcen_full45", "full_45", "full_45", "PCEN", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/p8b_pcen_qwen3_6_full45/predictions", full_45),
        FrozenRun("p5_qwen_grid_v2_representative6", "representative_6", "representative_6", "fixed_grid_v2", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2", REPRESENTATIVE_SIX),
        FrozenRun("p6_tiled_0p5_representative6", "representative_6", "representative_6", "0.5s_tiled", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/p6_tiled_qwen3_6_tile_0p5_overlap_0p1/merged_predictions", REPRESENTATIVE_SIX),
        FrozenRun("p6_tiled_0p25_OP016", "targeted_OP016", "OP_016_only", "0.25s_tiled", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/p6_tiled_qwen3_6_tile_0p25_overlap_0p05_OP016/merged_predictions", ("OP_016",)),
        FrozenRun("p6_pcen_representative6", "representative_6", "representative_6", "PCEN", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/p6_pcen_qwen3_6_representative6/predictions", REPRESENTATIVE_SIX),
        FrozenRun("p6_batdetect2_proposal_only_representative6", "representative_6", "representative_6", "BatDetect2_proposal_only", "batdetect2", REPO_ROOT / "outputs/agent_runs/p6_batdetect2_proposal_only_representative6/predictions", REPRESENTATIVE_SIX),
        FrozenRun("p6_metadata_assisted_representative6", "representative_6", "representative_6", "BatDetect2_metadata_assisted", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/predictions", REPRESENTATIVE_SIX),
        FrozenRun("p6_proposal_preserving_validator_representative6", "representative_6", "representative_6", "proposal_preserving_validator", "deterministic_validator", REPO_ROOT / "outputs/agent_runs/p6_proposal_preserving_validator_representative6/predictions", REPRESENTATIVE_SIX),
        FrozenRun("p6_timing_preserving_validator_representative6", "representative_6", "representative_6", "timing_preserving_validator", "deterministic_validator", REPO_ROOT / "outputs/agent_runs/p6_timing_preserving_validator_representative6/predictions", REPRESENTATIVE_SIX),
        FrozenRun("p6_policy_b_representative6", "representative_6", "representative_6", "policy_b_anchored_validator", "deterministic_validator", REPO_ROOT / "outputs/agent_runs/p6_timing_rule_ablation_representative6/policy_b_anchored_expansion/predictions", REPRESENTATIVE_SIX),
        FrozenRun("p6e5_proposal_only_heldout", "heldout_10", "heldout_10", "BatDetect2_proposal_only", "batdetect2", REPO_ROOT / "outputs/agent_runs/p6e5_batdetect2_proposal_only_heldout/predictions", HELDOUT_TEN),
        FrozenRun("p6e5_metadata_assisted_heldout", "heldout_10", "heldout_10", "BatDetect2_metadata_assisted", "qwen3.6:latest", REPO_ROOT / "outputs/agent_runs/p6e5_batdetect2_metadata_assisted_heldout/predictions", HELDOUT_TEN),
        FrozenRun("p6e5_policy_b_heldout", "heldout_10", "heldout_10", "policy_b_anchored_validator", "deterministic_validator", REPO_ROOT / "outputs/agent_runs/p6e5_policy_b_anchored_validator_heldout/predictions", HELDOUT_TEN),
    ]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_ground_truth(clip_id: str, eval_dir: Path) -> list[EventBox]:
    path = eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = []
    for index, event in enumerate(payload.get("events", [])):
        events.append(
            EventBox(
                event_id=str(event.get("event_id") or f"gt_{index + 1:03d}"),
                start_time=float(event["start_time"]),
                end_time=float(event["end_time"]),
                low_frequency=as_float(event.get("low_frequency")),
                high_frequency=as_float(event.get("high_frequency")),
                label=event.get("label"),
                source_index=index,
                metadata=event,
            )
        )
    return events


def prediction_file_for(prediction_dir: Path, clip_id: str) -> Path | None:
    candidates = (
        prediction_dir / f"{clip_id}_predictions.json",
        prediction_dir / f"{clip_id}_prediction.json",
        prediction_dir / f"{clip_id}.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_predictions(clip_id: str, prediction_dir: Path) -> tuple[list[EventBox], str, str, Path | None]:
    path = prediction_file_for(prediction_dir, clip_id)
    if path is None:
        return [], "missing_prediction_file", "No prediction JSON file found", None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], "json_parse_failure", str(exc), path
    if payload.get("parse_status") and payload.get("parse_status") != "success":
        return [], str(payload.get("parse_status")), str(payload.get("parse_error", "")), path
    events = []
    for index, event in enumerate(payload.get("events", [])):
        start = as_float(event.get("start_time_seconds", event.get("start_time")))
        end = as_float(event.get("end_time_seconds", event.get("end_time")))
        if start is None or end is None:
            continue
        events.append(
            EventBox(
                event_id=str(event.get("event_id") or event.get("proposal_id") or f"pred_{index + 1:03d}"),
                start_time=start,
                end_time=end,
                low_frequency=as_float(event.get("low_frequency_hz", event.get("low_frequency"))),
                high_frequency=as_float(event.get("high_frequency_hz", event.get("high_frequency"))),
                label=event.get("label"),
                confidence=as_float(event.get("confidence", event.get("det_prob"))),
                source_index=index,
                metadata=event,
            )
        )
    return events, "success", "", path


def clip_row(run: FrozenRun, result: ClipEvaluation, prediction_path: Path | None) -> dict[str, Any]:
    return {
        "experiment_id": run.experiment_id,
        "experiment_group": run.experiment_group,
        "clip_scope": run.clip_scope,
        "method": run.method,
        "model": run.model,
        "protocol": result.protocol.value,
        "clip_id": result.clip_id,
        "prediction_path": str(prediction_path.relative_to(REPO_ROOT)) if prediction_path else "",
        "parse_status": result.parse_status,
        "predicted_count": result.predicted_count,
        "ground_truth_count": result.ground_truth_count,
        "TP": result.tp,
        "FP": result.fp,
        "FN": result.fn,
        "precision": result.precision,
        "recall": result.recall,
        "F1": result.f1,
    }


def matched_pair_rows(run: FrozenRun, result: ClipEvaluation) -> list[dict[str, Any]]:
    rows = []
    for pair in result.matched:
        rows.append(
            {
                "experiment_id": run.experiment_id,
                "experiment_group": run.experiment_group,
                "clip_scope": run.clip_scope,
                "method": run.method,
                "model": run.model,
                "protocol": result.protocol.value,
                "clip_id": result.clip_id,
                "prediction_event_id": pair.prediction.event_id,
                "ground_truth_event_id": pair.ground_truth.event_id,
                "match_score": pair.match_score,
                "start_time_error_ms": pair.start_time_error_ms,
                "end_time_error_ms": pair.end_time_error_ms,
                "center_time_error_ms": pair.center_time_error_ms,
                "duration_error_ms": pair.duration_error_ms,
                "temporal_iou": pair.temporal_iou,
                "frequency_iou": pair.frequency_iou,
                "box_iou": pair.box_iou,
                "prediction_start_time": pair.prediction.start_time,
                "prediction_end_time": pair.prediction.end_time,
                "ground_truth_start_time": pair.ground_truth.start_time,
                "ground_truth_end_time": pair.ground_truth.end_time,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_runs(runs: list[FrozenRun], eval_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    experiment_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for run in runs:
        if not run.prediction_dir.exists():
            continue
        for protocol in PROTOCOLS:
            clip_results: list[ClipEvaluation] = []
            for clip_id in run.clip_ids:
                predictions, parse_status, parse_error, prediction_path = load_predictions(clip_id, run.prediction_dir)
                ground_truth = load_ground_truth(clip_id, eval_dir)
                result = evaluate_clip(
                    clip_id,
                    predictions,
                    ground_truth,
                    protocol,
                    parse_status=parse_status,
                    parse_error=parse_error,
                )
                clip_results.append(result)
                case_rows.append(clip_row(run, result, prediction_path))
                pair_rows.extend(matched_pair_rows(run, result))
            aggregate = aggregate_clip_evaluations(clip_results)
            experiment_rows.append(
                {
                    "experiment_id": run.experiment_id,
                    "experiment_group": run.experiment_group,
                    "clip_scope": run.clip_scope,
                    "method": run.method,
                    "model": run.model,
                    "protocol": protocol.value,
                    **aggregate,
                    "prediction_dir": str(run.prediction_dir.relative_to(REPO_ROOT)),
                    "notes": run.notes,
                }
            )
    return experiment_rows, case_rows, pair_rows


def write_audit(output_dir: Path, runs: list[FrozenRun]) -> None:
    lines = [
        "# P8A Matching Protocol Audit",
        "",
        "This audit re-evaluates frozen prediction JSON files only; no model inference is run.",
        "",
        "## Protocols",
        "",
        "- `temporal_iou_0.1`: confidence-ordered one-to-one matching; temporal IoU must be at least 0.1.",
        "- `temporal_iou_0.3`: same matching protocol as the existing evaluator, with the original threshold of 0.3.",
        "- `start_time_proximity_10ms`: BatDetect2-style start-time affinity; absolute start-time difference must be <= 0.010 seconds.",
        "",
        "## BatDetect2 10 ms source",
        "",
        "The cached BatDetect2 1.3.1 evaluation code defines `compute_affinity_1d` using `abs(pred_box[0] - gt_boxes[:, 0])` and considers a detection valid when the minimum score is <= `threshold`. The CLI default for `--iou_thresh` in `evaluate_models.py` is `0.01`, which is passed as `detection_overlap`. Despite the CLI name, this is start-time proximity rather than box IoU.",
        "",
        "## Frozen runs considered",
        "",
    ]
    for run in runs:
        status = "present" if run.prediction_dir.exists() else "missing"
        lines.append(f"- `{run.experiment_id}`: {status}, `{run.prediction_dir.relative_to(REPO_ROOT)}`")
    (output_dir / "matching_protocol_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(output_dir: Path, experiment_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> None:
    comparison_rows = [
        {
            "experiment_id": row["experiment_id"],
            "experiment_group": row["experiment_group"],
            "method": row["method"],
            "model": row["model"],
            "clip_scope": row["clip_scope"],
            "protocol": row["protocol"],
            "ground_truth_count": row["ground_truth_count"],
            "predicted_count": row["predicted_count"],
            "TP": row["TP"],
            "FP": row["FP"],
            "FN": row["FN"],
            "precision": row["precision"],
            "recall": row["recall"],
            "F1": row["F1"],
        }
        for row in experiment_rows
    ]
    write_csv(output_dir / "protocol_comparison.csv", comparison_rows)
    rep_heldout_rows = [row for row in experiment_rows if row["experiment_group"] in {"representative_6", "heldout_10"}]
    write_csv(output_dir / "representative_heldout_summary.csv", rep_heldout_rows)

    lines = [
        "# P8A Multi-Protocol Detection Report",
        "",
        "P8A re-scores frozen predictions under three event-matching views. Metrics are aggregate event-level values from pooled TP, FP, and FN counts.",
        "",
        "## Protocol comparison",
        "",
        "| Experiment | Scope | Protocol | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in experiment_rows:
        lines.append(
            f"| {row['experiment_id']} | {row['clip_scope']} | {row['protocol']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['F1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `temporal_iou_0.3` is the frozen protocol used by the existing evaluator.",
            "- `temporal_iou_0.1` tests sensitivity to loose temporal overlap.",
            "- `start_time_proximity_10ms` follows the BatDetect2 start-time affinity interpretation documented in `matching_protocol_audit.md`.",
            "- Parse failures or missing prediction files are tracked and not counted as valid zero-event predictions.",
        ]
    )
    (output_dir / "p8a_multi_protocol_detection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = default_frozen_runs(args.eval_dir)
    experiment_rows, case_rows, pair_rows = evaluate_runs(runs, args.eval_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_audit(args.output_dir, runs)
    write_csv(args.output_dir / "experiment_level_protocol_summary.csv", experiment_rows)
    write_csv(args.output_dir / "case_level_protocol_results.csv", case_rows)
    write_csv(args.output_dir / "matched_pair_box_quality.csv", pair_rows)
    write_reports(args.output_dir, experiment_rows, case_rows)

    print(f"Wrote {len(experiment_rows)} experiment-protocol rows to {args.output_dir}")


if __name__ == "__main__":
    main()
