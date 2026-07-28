"""Evaluate deterministic central proposal selection baselines for Stage 2.

The Stage 1/2 event windows are 0.300 s crops centred on the target event. This
script uses that known window geometry, but never uses the GT box or species
label for proposal selection. It selects from real BatDetect2 proposal files and
then evaluates the selected proposals against the target GT box.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.prepare_stage2_sample_level_batdetect2_proposals import (  # noqa: E402
    ALLOWED_SPECIES,
    DEFAULT_STAGE1_MANIFEST,
    DEFAULT_V2_MANIFEST,
    WindowSpec,
    box_iou,
    build_window_specs,
    frequency_iou,
    gt_box,
    load_csv_rows,
    proposal_box,
    temporal_iou,
)
from scripts.inference.run_stage2_joint_proposal_constrained_pilot80 import (  # noqa: E402
    select_balanced_subset,
)


DEFAULT_PROPOSAL_DIR = (
    REPO_ROOT / "outputs/tool_outputs/batdetect2_multispecies_stage1_windows/proposals"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_classification/"
    "stage2_central_proposal_selection_baseline"
)
CENTRE_SECONDS = 0.150
CENTRE_TOLERANCE_SECONDS = 0.020
RULES = (
    "highest_score",
    "nearest_to_centre",
    "centre_then_score",
    "top3_centre_candidates",
)


def proposal_centre(proposal: dict[str, Any]) -> float:
    return (
        float(proposal["start_time_seconds"]) + float(proposal["end_time_seconds"])
    ) / 2


def centre_distance(proposal: dict[str, Any], centre_seconds: float = CENTRE_SECONDS) -> float:
    return abs(proposal_centre(proposal) - centre_seconds)


def select_proposals(
    proposals: list[dict[str, Any]],
    rule: str,
    centre_seconds: float = CENTRE_SECONDS,
    centre_tolerance_seconds: float = CENTRE_TOLERANCE_SECONDS,
) -> list[dict[str, Any]]:
    """Select proposal(s) using deterministic centre/score rules."""

    if rule not in RULES:
        raise ValueError(f"Unknown selection rule: {rule}")
    if not proposals:
        return []
    if rule == "highest_score":
        return [max(proposals, key=lambda p: (float(p["det_prob"]), -centre_distance(p, centre_seconds)))]
    if rule == "nearest_to_centre":
        return [min(proposals, key=lambda p: (centre_distance(p, centre_seconds), -float(p["det_prob"])))]
    if rule == "centre_then_score":
        centred = [
            p
            for p in proposals
            if centre_distance(p, centre_seconds) <= centre_tolerance_seconds
        ]
        if centred:
            return [max(centred, key=lambda p: (float(p["det_prob"]), -centre_distance(p, centre_seconds)))]
        return select_proposals(proposals, "nearest_to_centre", centre_seconds, centre_tolerance_seconds)
    sorted_by_centre = sorted(
        proposals,
        key=lambda p: (centre_distance(p, centre_seconds), -float(p["det_prob"])),
    )
    return sorted_by_centre[:3]


def load_proposals(proposal_dir: Path, anonymous_sample_id: str) -> list[dict[str, Any]]:
    path = proposal_dir / f"{anonymous_sample_id}_batdetect2_proposals.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    return events if isinstance(events, list) else []


def best_match(
    gt: dict[str, Any],
    selected: list[dict[str, Any]],
    protocol: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float]:
    if protocol == "temporal_iou_0p3":
        threshold = 0.3
        scored = [(temporal_iou(gt, proposal_box(p)), p) for p in selected]
        scored = [item for item in scored if item[0] >= threshold]
        if not scored:
            return None, selected, 0.0
        score, matched = max(scored, key=lambda item: (item[0], float(item[1]["det_prob"])))
        return matched, [p for p in selected if p is not matched], score
    if protocol == "temporal_iou_0p1":
        threshold = 0.1
        scored = [(temporal_iou(gt, proposal_box(p)), p) for p in selected]
        scored = [item for item in scored if item[0] >= threshold]
        if not scored:
            return None, selected, 0.0
        score, matched = max(scored, key=lambda item: (item[0], float(item[1]["det_prob"])))
        return matched, [p for p in selected if p is not matched], score
    if protocol == "start_time_10ms":
        scored = [
            (abs(float(p["start_time_seconds"]) - float(gt["start_time"])), p)
            for p in selected
        ]
        scored = [item for item in scored if item[0] <= 0.010]
        if not scored:
            return None, selected, 0.0
        distance, matched = min(scored, key=lambda item: (item[0], -float(item[1]["det_prob"])))
        return matched, [p for p in selected if p is not matched], distance
    raise ValueError(protocol)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def evaluate_rule(
    *,
    specs: list[WindowSpec],
    proposal_dir: Path,
    rule: str,
    clip_scope: str,
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    per_species: dict[str, Counter[str]] = defaultdict(Counter)
    protocol_counts: dict[str, Counter[str]] = defaultdict(Counter)
    matched_iou_values: dict[str, list[float]] = {"time": [], "frequency": [], "box": []}
    samples_with_proposals = 0
    for spec in specs:
        proposals = load_proposals(proposal_dir, spec.anonymous_sample_id)
        if proposals:
            samples_with_proposals += 1
        selected = select_proposals(proposals, rule)
        gt = gt_box(spec)
        match_03, unmatched_03, _score = best_match(gt, selected, "temporal_iou_0p3")
        if match_03:
            per_species[spec.species]["TP"] += 1
            pbox = proposal_box(match_03)
            matched_iou_values["time"].append(temporal_iou(gt, pbox))
            matched_iou_values["frequency"].append(frequency_iou(gt, pbox))
            matched_iou_values["box"].append(box_iou(gt, pbox))
        else:
            per_species[spec.species]["FN"] += 1
            missed_rows.append(
                {
                    "clip_scope": clip_scope,
                    "rule": rule,
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "species": spec.species,
                    "proposal_count": len(proposals),
                    "selected_count": len(selected),
                    "gt_start_time": gt["start_time"],
                    "gt_end_time": gt["end_time"],
                }
            )
        for proposal in unmatched_03:
            fp_rows.append(
                {
                    "clip_scope": clip_scope,
                    "rule": rule,
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "species": spec.species,
                    "proposal_id": proposal["proposal_id"],
                    "start_time": proposal["start_time_seconds"],
                    "end_time": proposal["end_time_seconds"],
                    "low_freq": proposal["low_frequency_hz"],
                    "high_freq": proposal["high_frequency_hz"],
                    "det_prob": proposal["det_prob"],
                    "centre_distance_seconds": centre_distance(proposal),
                }
            )
        for proposal in selected:
            pbox = proposal_box(proposal)
            selected_rows.append(
                {
                    "clip_scope": clip_scope,
                    "rule": rule,
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "species": spec.species,
                    "proposal_id": proposal["proposal_id"],
                    "proposal_count": len(proposals),
                    "selected_count": len(selected),
                    "start_time": proposal["start_time_seconds"],
                    "end_time": proposal["end_time_seconds"],
                    "low_freq": proposal["low_frequency_hz"],
                    "high_freq": proposal["high_frequency_hz"],
                    "det_prob": proposal["det_prob"],
                    "centre_distance_seconds": centre_distance(proposal),
                    "time_iou": temporal_iou(gt, pbox),
                    "frequency_iou": frequency_iou(gt, pbox),
                    "box_iou": box_iou(gt, pbox),
                    "matched_iou_0p3": str(proposal is match_03).lower(),
                }
            )
        for protocol in ("temporal_iou_0p3", "temporal_iou_0p1", "start_time_10ms"):
            match, unmatched, _ = best_match(gt, selected, protocol)
            protocol_counts[protocol]["TP" if match else "FN"] += 1
            protocol_counts[protocol]["FP"] += len(unmatched)

    metric_rows: list[dict[str, Any]] = []
    for protocol in ("temporal_iou_0p3", "temporal_iou_0p1", "start_time_10ms"):
        counts = protocol_counts[protocol]
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        metric_rows.append(
            {
                "clip_scope": clip_scope,
                "rule": rule,
                "protocol": protocol,
                "sample_count": len(specs),
                "samples_with_proposals": samples_with_proposals,
                "proposal_missing_rate": 1 - safe_div(samples_with_proposals, len(specs)),
                "selected_proposal_count": len(selected_rows),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1(precision, recall),
                "mean_time_iou": (
                    sum(matched_iou_values["time"]) / len(matched_iou_values["time"])
                    if protocol == "temporal_iou_0p3" and matched_iou_values["time"]
                    else ""
                ),
                "mean_frequency_iou": (
                    sum(matched_iou_values["frequency"]) / len(matched_iou_values["frequency"])
                    if protocol == "temporal_iou_0p3" and matched_iou_values["frequency"]
                    else ""
                ),
                "mean_box_iou": (
                    sum(matched_iou_values["box"]) / len(matched_iou_values["box"])
                    if protocol == "temporal_iou_0p3" and matched_iou_values["box"]
                    else ""
                ),
            }
        )
    species_rows: list[dict[str, Any]] = []
    for species in ALLOWED_SPECIES:
        counts = per_species[species]
        support = counts["TP"] + counts["FN"]
        species_rows.append(
            {
                "clip_scope": clip_scope,
                "rule": rule,
                "species": species,
                "support": support,
                "TP": counts["TP"],
                "FN": counts["FN"],
                "recall_iou_0p3": safe_div(counts["TP"], support),
            }
        )
    return {
        "metric_rows": metric_rows,
        "species_rows": species_rows,
        "selected_rows": selected_rows,
        "missed_rows": missed_rows,
        "false_positive_rows": fp_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def report_lines(metric_rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    def row_for(scope: str, rule: str, protocol: str) -> dict[str, Any]:
        for row in metric_rows:
            if row["clip_scope"] == scope and row["rule"] == rule and row["protocol"] == protocol:
                return row
        return {}

    proposal_only = load_json_if_exists(
        REPO_ROOT
        / "outputs/analysis_reports/multispecies_classification/"
        "qwen3_6_stage2b_true_proposal_constrained_pilot80/"
        "proposal_only_pilot80_comparison.json"
    )
    stage2b = load_json_if_exists(
        REPO_ROOT
        / "outputs/analysis_reports/multispecies_classification/"
        "qwen3_6_stage2b_true_proposal_constrained_pilot80/aggregate_summary.json"
    )
    lines = [
        "# Stage 2 Central Proposal Selection Baseline",
        "",
        "## Scope",
        "",
        "This no-inference baseline selects from real sample-level BatDetect2 proposals using only detector confidence and the known local window centre at `0.150 s`. It does not use GT boxes, GT labels, VLM predictions, or diagnostic overlays for selection.",
        "",
        "## Pilot80 IoU >= 0.3 Results",
        "",
        "| Rule | TP | FP | FN | Precision | Recall | F1 | Mean time IoU | Mean freq IoU | Mean box IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best_rule = ""
    best_f1 = -1.0
    for rule in RULES:
        row = row_for("pilot80", rule, "temporal_iou_0p3")
        if not row:
            continue
        if float(row["F1"]) > best_f1:
            best_f1 = float(row["F1"])
            best_rule = rule
        lines.append(
            f"| {rule} | {row['TP']} | {row['FP']} | {row['FN']} | "
            f"{float(row['precision']):.3f} | {float(row['recall']):.3f} | "
            f"{float(row['F1']):.3f} | {float(row['mean_time_iou']):.3f} | "
            f"{float(row['mean_frequency_iou']):.3f} | {float(row['mean_box_iou']):.3f} |"
        )
    lines.extend(["", "## Pilot80 Comparison Context", ""])
    if proposal_only:
        row = proposal_only["temporal_iou_0p3"]
        lines.append(
            f"- Raw proposal-only pilot80: F1 `{row['F1']:.3f}`, TP `{row['TP']}`, FP `{row['FP']}`, FN `{row['FN']}`."
        )
    if stage2b:
        row = stage2b["temporal_iou_0p3"]
        lines.append(
            f"- Stage 2B VLM proposal filtering: F1 `{row['F1']:.3f}`, TP `{row['TP']}`, FP `{row['FP']}`, FN `{row['FN']}`."
        )
    if best_rule:
        best = row_for("pilot80", best_rule, "temporal_iou_0p3")
        lines.append(
            f"- Best deterministic rule on pilot80: `{best_rule}` with F1 `{float(best['F1']):.3f}`."
        )
    lines.extend(
        [
            "",
            "## Full240 Check",
            "",
            "| Rule | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rule in RULES:
        row = row_for("full240", rule, "temporal_iou_0p3")
        if row:
            lines.append(
                f"| {rule} | {row['TP']} | {row['FP']} | {row['FN']} | "
                f"{float(row['precision']):.3f} | {float(row['recall']):.3f} | "
                f"{float(row['F1']):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The centre-based rules test whether proposal selection, rather than proposal quality, is the main localisation bottleneck. A strong centre rule means the target call is often present in the detector proposals and can be recovered without VLM filtering. A weak centre rule means either detector proposal quality is insufficient or the crop contains multiple plausible centre-near calls.",
            "",
            f"Use `{best_rule or 'the best pilot80 rule'}` as the deterministic proposal-selection baseline for the next Stage 2C run. If a VLM condition cannot beat this rule while preserving recall, the VLM is not adding reliable localisation value.",
        ]
    )
    return lines


def run_analysis(
    *,
    stage1_manifest: Path,
    v2_manifest: Path,
    proposal_dir: Path,
    output_dir: Path,
    include_full240: bool,
) -> None:
    stage1_rows = load_csv_rows(stage1_manifest)
    v2_rows = load_csv_rows(v2_manifest)
    scopes = [("pilot80", select_balanced_subset(stage1_rows, 10))]
    if include_full240:
        scopes.append(("full240", stage1_rows))
    metric_rows: list[dict[str, Any]] = []
    species_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    for clip_scope, rows in scopes:
        specs = build_window_specs(stage1_rows=rows, v2_rows=v2_rows)
        for rule in RULES:
            result = evaluate_rule(
                specs=specs,
                proposal_dir=proposal_dir,
                rule=rule,
                clip_scope=clip_scope,
            )
            metric_rows.extend(result["metric_rows"])
            species_rows.extend(result["species_rows"])
            selected_rows.extend(result["selected_rows"])
            missed_rows.extend(result["missed_rows"])
            fp_rows.extend(result["false_positive_rows"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "central_proposal_metrics.csv", metric_rows)
    write_csv(output_dir / "per_species_central_proposal_metrics.csv", species_rows)
    write_csv(output_dir / "selected_proposals.csv", selected_rows)
    write_csv(output_dir / "missed_events.csv", missed_rows)
    write_csv(output_dir / "false_positive_selected_proposals.csv", fp_rows)
    (output_dir / "central_proposal_selection_report.md").write_text(
        "\n".join(report_lines(metric_rows, output_dir)) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-manifest", type=Path, default=DEFAULT_STAGE1_MANIFEST)
    parser.add_argument("--v2-manifest", type=Path, default=DEFAULT_V2_MANIFEST)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--full240", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        stage1_manifest=args.stage1_manifest,
        v2_manifest=args.v2_manifest,
        proposal_dir=args.proposal_dir,
        output_dir=args.output_dir,
        include_full240=args.full240,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
