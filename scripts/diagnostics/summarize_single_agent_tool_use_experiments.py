"""Consolidate completed P6 single-agent and tool-use evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(
    "outputs/analysis_reports/p6_single_agent_tool_use_summary"
)

REQUIRED_AGGREGATE_FIELDS = {
    "clip_count",
    "total_predictions",
    "total_tp",
    "total_fp",
    "total_fn",
    "precision",
    "recall",
    "f1",
    "mean_time_iou",
    "mean_frequency_iou",
    "mean_box_iou",
    "strict_box_iou_0_3_count",
    "strict_box_iou_0_5_count",
}

CONSOLIDATED_FIELDS = (
    "experiment_id",
    "experiment_group",
    "clip_scope",
    "clip_count",
    "method",
    "model",
    "input_condition",
    "uses_batdetect2_proposals",
    "uses_vlm",
    "uses_validator",
    "TP",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "mean_time_iou",
    "mean_frequency_iou",
    "mean_box_iou",
    "box_iou_ge_0_3",
    "box_iou_ge_0_5",
    "notes",
)

CASE_FIELDS = (
    "clip_id",
    "key_issue",
    "method",
    "TP",
    "FP",
    "FN",
    "F1",
    "interpretation",
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    experiment_group: str
    clip_scope: str
    method: str
    model: str
    input_condition: str
    uses_batdetect2_proposals: bool
    uses_vlm: bool
    uses_validator: bool
    evaluation_dir: Path
    notes: str


EXPERIMENTS = (
    ExperimentSpec(
        "rep_fixed_qwen3_6_grid_v2",
        "representative_six",
        "representative6",
        "Fixed qwen3.6 + grid_v2",
        "qwen3.6:latest",
        "Clean dB full-view grid_v2 spectrogram",
        False,
        True,
        False,
        Path("outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation_representative6"),
        "Frozen fixed-view baseline on the representative six.",
    ),
    ExperimentSpec(
        "rep_tiled_0p5_qwen3_6",
        "representative_six",
        "representative6",
        "0.5s tiled qwen3.6",
        "qwen3.6:latest",
        "0.5 s tiles with 0.1 s overlap, merged by 2D NMS",
        False,
        True,
        False,
        Path("outputs/agent_runs/p6_tiled_qwen3_6_tile_0p5_overlap_0p1/evaluation"),
        "Higher recall, with duplicate and localisation risk.",
    ),
    ExperimentSpec(
        "target_op016_tiled_0p25_qwen3_6",
        "targeted_single_clip",
        "OP_016_only",
        "0.25s tiled qwen3.6 targeted OP_016",
        "qwen3.6:latest",
        "0.25 s tiles with 0.05 s overlap, merged by 2D NMS",
        False,
        True,
        False,
        Path("outputs/agent_runs/p6_tiled_qwen3_6_tile_0p25_overlap_0p05_OP016/evaluation"),
        "Single-clip diagnostic result; not directly rankable against six-clip aggregates.",
    ),
    ExperimentSpec(
        "rep_pcen_qwen3_6",
        "representative_six",
        "representative6",
        "PCEN qwen3.6",
        "qwen3.6:latest",
        "Clean PCEN-enhanced full-view grid_v2 spectrogram",
        False,
        True,
        False,
        Path("outputs/agent_runs/p6_pcen_qwen3_6_representative6/evaluation"),
        "PCEN visual preprocessing pilot.",
    ),
    ExperimentSpec(
        "rep_batdetect2_proposal_only",
        "representative_six",
        "representative6",
        "BatDetect2 proposal-only",
        "batdetect2 1.3.1",
        "Structured detector proposals at det_prob >= 0.30",
        True,
        False,
        False,
        Path("outputs/agent_runs/p6_batdetect2_proposal_only_representative6/evaluation"),
        "UK taxonomy ignored; proposals evaluated as generic bat_call boxes.",
    ),
    ExperimentSpec(
        "rep_batdetect2_metadata_assisted_qwen3_6",
        "representative_six",
        "representative6",
        "BatDetect2 metadata-assisted qwen3.6",
        "qwen3.6:latest + batdetect2 1.3.1",
        "Clean grid_v2 overview plus structured detector metadata",
        True,
        True,
        False,
        Path("outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/evaluation"),
        "Unconstrained VLM verification and refinement.",
    ),
    ExperimentSpec(
        "rep_p6e2_full_proposal_preserving",
        "representative_six",
        "representative6",
        "P6E.2 proposal-preserving validator",
        "qwen3.6:latest + batdetect2 1.3.1",
        "Metadata-assisted predictions with full proposal geometry preservation",
        True,
        True,
        True,
        Path("outputs/agent_runs/p6_proposal_preserving_validator_representative6/evaluation"),
        "Deterministic full-geometry shadow validator.",
    ),
    ExperimentSpec(
        "rep_p6e3_timing_preserving",
        "representative_six",
        "representative6",
        "P6E.3 timing-preserving validator",
        "qwen3.6:latest + batdetect2 1.3.1",
        "Metadata-assisted predictions with independent timing preservation",
        True,
        True,
        True,
        Path("outputs/agent_runs/p6_timing_preserving_validator_representative6/evaluation"),
        "Timing-only deterministic shadow validator.",
    ),
    ExperimentSpec(
        "rep_p6e4_policy_b_anchored",
        "representative_six",
        "representative6",
        "P6E.4 Policy B anchored validator",
        "qwen3.6:latest + batdetect2 1.3.1",
        "Anchored moderate expansion plus rigid-translation preservation",
        True,
        True,
        True,
        Path("outputs/agent_runs/p6_timing_rule_ablation_representative6/policy_b_anchored_expansion/evaluation"),
        "Rule developed on the representative six; held-out qualification required.",
    ),
    ExperimentSpec(
        "heldout_p6e5_proposal_only",
        "held_out",
        "p6e5_heldout10",
        "P6E.5 proposal-only",
        "batdetect2 1.3.1",
        "Held-out detector proposals at det_prob >= 0.30",
        True,
        False,
        False,
        Path("outputs/agent_runs/p6e5_batdetect2_proposal_only_heldout/evaluation"),
        "Ten clips excluded from Policy B rule development.",
    ),
    ExperimentSpec(
        "heldout_p6e5_metadata_assisted",
        "held_out",
        "p6e5_heldout10",
        "P6E.5 unconstrained metadata-assisted qwen3.6",
        "qwen3.6:latest + batdetect2 1.3.1",
        "Held-out clean grid_v2 overview plus detector metadata",
        True,
        True,
        False,
        Path("outputs/agent_runs/p6e5_batdetect2_metadata_assisted_heldout/evaluation"),
        "Held-out unconstrained VLM refinement.",
    ),
    ExperimentSpec(
        "heldout_p6e5_policy_b",
        "held_out",
        "p6e5_heldout10",
        "P6E.5 Policy B anchored validator",
        "qwen3.6:latest + batdetect2 1.3.1",
        "Held-out metadata-assisted predictions with frozen Policy B",
        True,
        True,
        True,
        Path("outputs/agent_runs/p6e5_policy_b_anchored_validator_heldout/evaluation"),
        "Held-out result; Policy B did not improve aggregate performance.",
    ),
)


CASE_SPECS = (
    ("OP_016", "dense short-call sequence", "rep_fixed_qwen3_6_grid_v2", "Fixed VLM misses the dense sequence."),
    ("OP_016", "dense short-call sequence", "rep_tiled_0p5_qwen3_6", "Tiling improves recall but retains many false positives."),
    ("OP_016", "dense short-call sequence", "target_op016_tiled_0p25_qwen3_6", "Smaller tiles do not resolve the case."),
    ("OP_016", "dense short-call sequence", "rep_batdetect2_proposal_only", "Detector timing recovers six calls with no false positives."),
    ("OP_016", "dense short-call sequence", "rep_batdetect2_metadata_assisted_qwen3_6", "Unconstrained refinement shifts good detector boxes."),
    ("OP_016", "dense short-call sequence", "rep_p6e2_full_proposal_preserving", "Proposal preservation recovers detector timing."),
    ("OP_016", "dense short-call sequence", "rep_p6e4_policy_b_anchored", "Anchored policy retains the recovered detector timing."),
    ("OP_004", "useful VLM duration expansion", "rep_batdetect2_metadata_assisted_qwen3_6", "VLM expansion improves an under-wide proposal."),
    ("OP_004", "useful VLM duration expansion", "rep_p6e2_full_proposal_preserving", "Over-conservative preservation removes a useful refinement."),
    ("OP_004", "useful VLM duration expansion", "rep_p6e4_policy_b_anchored", "Anchored expansion restores the useful refinement."),
    ("OP_045", "source proposal extent failure", "rep_fixed_qwen3_6_grid_v2", "Fixed-view VLM handles this simple partial clip."),
    ("OP_045", "source proposal extent failure", "rep_pcen_qwen3_6", "PCEN regresses a clip solved by the fixed-view baseline."),
    ("OP_045", "source proposal extent failure", "rep_batdetect2_proposal_only", "Detector proposals are too short for the annotation standard."),
    ("OP_045", "source proposal extent failure", "rep_p6e4_policy_b_anchored", "Preservation cannot repair a bad source extent."),
    ("OP_032", "held-out useful expansion reverted", "heldout_p6e5_proposal_only", "Under-wide proposals produce no temporal matches."),
    ("OP_032", "held-out useful expansion reverted", "heldout_p6e5_metadata_assisted", "VLM expansion recovers two events."),
    ("OP_032", "held-out useful expansion reverted", "heldout_p6e5_policy_b", "Policy B reverts one useful expansion."),
    ("OP_042", "held-out harmful rigid shift", "heldout_p6e5_proposal_only", "Detector proposals match all five events."),
    ("OP_042", "held-out harmful rigid shift", "heldout_p6e5_metadata_assisted", "VLM shifts one good proposal out of match."),
    ("OP_042", "held-out harmful rigid shift", "heldout_p6e5_policy_b", "Policy B successfully restores the shifted event."),
)


def load_aggregate_metrics(path: Path, experiment_id: str = "") -> dict[str, Any]:
    """Load a canonical evaluator summary with explicit missing-field errors."""
    if not path.is_file():
        label = f" for {experiment_id}" if experiment_id else ""
        raise FileNotFoundError(f"Aggregate summary missing{label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_AGGREGATE_FIELDS - payload.keys())
    if missing:
        raise ValueError(
            f"Aggregate summary {path} is missing fields: {', '.join(missing)}"
        )
    return payload


def consolidated_row(spec: ExperimentSpec, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "experiment_group": spec.experiment_group,
        "clip_scope": spec.clip_scope,
        "clip_count": metrics["clip_count"],
        "method": spec.method,
        "model": spec.model,
        "input_condition": spec.input_condition,
        "uses_batdetect2_proposals": spec.uses_batdetect2_proposals,
        "uses_vlm": spec.uses_vlm,
        "uses_validator": spec.uses_validator,
        "TP": metrics["total_tp"],
        "FP": metrics["total_fp"],
        "FN": metrics["total_fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "F1": metrics["f1"],
        "mean_time_iou": metrics["mean_time_iou"],
        "mean_frequency_iou": metrics["mean_frequency_iou"],
        "mean_box_iou": metrics["mean_box_iou"],
        "box_iou_ge_0_3": metrics["strict_box_iou_0_3_count"],
        "box_iou_ge_0_5": metrics["strict_box_iou_0_5_count"],
        "notes": spec.notes,
    }


def load_per_clip_metrics(evaluation_dir: Path) -> dict[str, dict[str, str]]:
    path = evaluation_dir / "per_clip_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Per-clip metrics missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def build_case_highlights(
    case_specs: tuple[tuple[str, str, str, str], ...],
    experiment_map: dict[str, ExperimentSpec],
) -> list[dict[str, Any]]:
    """Build report-ready case rows from canonical per-clip evaluator CSVs."""
    cache: dict[str, dict[str, dict[str, str]]] = {}
    rows: list[dict[str, Any]] = []
    for clip_id, issue, experiment_id, interpretation in case_specs:
        if experiment_id not in experiment_map:
            raise KeyError(f"Unknown case-highlight experiment: {experiment_id}")
        spec = experiment_map[experiment_id]
        if experiment_id not in cache:
            cache[experiment_id] = load_per_clip_metrics(spec.evaluation_dir)
        if clip_id not in cache[experiment_id]:
            raise ValueError(f"Clip {clip_id} is absent from {experiment_id} per-clip metrics")
        metric = cache[experiment_id][clip_id]
        rows.append(
            {
                "clip_id": clip_id,
                "key_issue": issue,
                "method": spec.method,
                "TP": int(metric["tp"]),
                "FP": int(metric["fp"]),
                "FN": int(metric["fn"]),
                "F1": float(metric["f1"]),
                "interpretation": interpretation,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_metrics_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rank | Method | TP | FP | FN | Precision | Recall | F1 | Time IoU | Frequency IoU | Box IoU |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(sorted(rows, key=lambda item: item["F1"], reverse=True), start=1):
        lines.append(
            f"| {rank} | {row['method']} | {row['TP']} | {row['FP']} | {row['FN']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['F1']:.4f} | "
            f"{row['mean_time_iou']:.4f} | {row['mean_frequency_iou']:.4f} | "
            f"{row['mean_box_iou']:.4f} |"
        )
    return lines


def write_report(path: Path, consolidated: list[dict[str, Any]]) -> None:
    representative = [
        row for row in consolidated if row["experiment_group"] == "representative_six"
    ]
    held_out = [row for row in consolidated if row["experiment_group"] == "held_out"]
    targeted = [
        row for row in consolidated if row["experiment_group"] == "targeted_single_clip"
    ]
    lines = [
        "# P6 Single-Agent Tool-Use Ablation Summary",
        "",
        "## Scope",
        "",
        "This report consolidates completed single-agent, preprocessing, detector-tool, and deterministic-validator experiments from canonical evaluator artifacts. Representative-six, held-out-ten, and OP_016-only results are kept as separate comparison scopes.",
        "",
        "## A. Main Results",
        "",
        "### Representative Six",
        "",
        *_format_metrics_table(representative),
        "",
        "### Targeted OP_016 Result",
        "",
        *_format_metrics_table(targeted),
        "",
        "The 0.25 s tiled result is a single-clip diagnostic and must not be ranked against six-clip aggregates.",
        "",
        "### Held-Out Validation",
        "",
        *_format_metrics_table(held_out),
        "",
        "## B. Main Findings",
        "",
        "- The 0.5 s tiled condition improves representative-six recall from 0.6061 to 0.7576, but mean box IoU falls from 0.3757 to 0.3429. Tiling helps candidate recovery more than precise localisation.",
        "- PCEN does not improve aggregate F1 over the fixed baseline and regresses clean cases such as OP_045.",
        "- BatDetect2 proposals provide strong timing priors, most clearly on OP_016.",
        "- Unconstrained VLM refinement can damage accurate proposals and performs below proposal-only in both representative and held-out comparisons.",
        "- Deterministic validation repairs specific harmful shifts, including OP_016 and held-out OP_042.",
        "- Policy B improves the development subset but does not improve aggregate held-out performance and harms held-out OP_032.",
        "- Proposal-only remains the strongest held-out baseline with F1 0.8293.",
        "",
        "## C. Key Case Studies",
        "",
        "- **OP_016:** Dense short-call sequence. Detector timing recovers six calls; unconstrained VLM refinement damages five good proposals; preservation restores them. The left-boundary event remains missing.",
        "- **OP_045:** Detector proposals are too short relative to the annotation standard. Preservation cannot repair a source extent failure.",
        "- **OP_004:** Moderate VLM duration expansion is useful, but over-conservative preservation removes that gain. Policy B recovers it in the development subset.",
        "- **OP_032:** Held-out detector boxes are under-wide. VLM expansion recovers two events, while Policy B wrongly reverts one.",
        "- **OP_042:** Held-out VLM refinement rigidly shifts a good proposal. Policy B correctly restores it.",
        "",
        "## D. Dissertation-Ready Interpretation",
        "",
        "Reliable agentic bioacoustic annotation requires constrained tool use rather than unconditional trust in either the detector or the VLM. BatDetect2 supplies useful candidate timing, while the VLM can add missing events and refine under-wide extents. Yet unconstrained refinement can destroy correct proposals, and fixed deterministic thresholds can suppress useful changes out of sample.",
        "",
        "The evidence therefore supports provenance-aware verification, normalized deviation checks, and explicit separation of proposal acceptance, rejection, expansion, and new-event creation. Future systems should combine duration-normalized validation with evidence-based extent expansion and, only after deterministic baselines are stable, a critic or referee mechanism.",
        "",
        "## E. Recommendation",
        "",
        "Do not expand the current Policy B rule to all 45 clips. Freeze the present single-agent/tool-use findings as the completed P6 evidence base. The next stage should prioritize dissertation write-up and presentation preparation. A multi-agent critic prototype should be treated as a carefully scoped follow-up using the documented OP_016, OP_045, OP_004, OP_032, and OP_042 failure modes, not as a replacement for the frozen baselines.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    consolidated = [
        consolidated_row(
            spec,
            load_aggregate_metrics(
                spec.evaluation_dir / "aggregate_summary.json", spec.experiment_id
            ),
        )
        for spec in EXPERIMENTS
    ]
    experiment_map = {spec.experiment_id: spec for spec in EXPERIMENTS}
    highlights = build_case_highlights(CASE_SPECS, experiment_map)
    metrics_path = args.output_dir / "p6_consolidated_metrics.csv"
    cases_path = args.output_dir / "p6_case_highlights.csv"
    report_path = args.output_dir / "p6_single_agent_tool_use_summary_report.md"
    write_csv(metrics_path, consolidated, CONSOLIDATED_FIELDS)
    write_csv(cases_path, highlights, CASE_FIELDS)
    write_report(report_path, consolidated)
    print(f"Experiments consolidated: {len(consolidated)}")
    print(f"Case highlights: {len(highlights)}")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Case CSV: {cases_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
