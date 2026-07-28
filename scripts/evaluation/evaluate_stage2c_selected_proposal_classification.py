"""Evaluate Stage 2C selected-proposal species classification."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.prepare_stage2_sample_level_batdetect2_proposals import (  # noqa: E402
    DEFAULT_STAGE1_MANIFEST,
    DEFAULT_V2_MANIFEST,
    ALLOWED_SPECIES,
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


CONDITION_NAME = "qwen3_6_stage2c_nearest_centre_proposal_classification_pilot80"
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/agent_runs/multispecies_classification" / CONDITION_NAME
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/analysis_reports/multispecies_classification" / CONDITION_NAME


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def selected_proposal_from_prediction(row: dict[str, str]) -> dict[str, Any] | None:
    if row.get("selected_proposal_available") != "true":
        return None
    try:
        return {
            "proposal_id": row["selected_proposal_id"],
            "start_time_seconds": float(row["selected_start_time"]),
            "end_time_seconds": float(row["selected_end_time"]),
            "low_frequency_hz": float(row["selected_low_freq"]),
            "high_frequency_hz": float(row["selected_high_freq"]),
            "det_prob": float(row["selected_det_prob"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def protocol_match(gt: dict[str, Any], proposal: dict[str, Any] | None, protocol: str) -> bool:
    if proposal is None:
        return False
    pbox = proposal_box(proposal)
    if protocol == "temporal_iou_0p3":
        return temporal_iou(gt, pbox) >= 0.3
    if protocol == "temporal_iou_0p1":
        return temporal_iou(gt, pbox) >= 0.1
    if protocol == "start_time_10ms":
        return abs(float(pbox["start_time"]) - float(gt["start_time"])) <= 0.010
    raise ValueError(protocol)


def per_species_classification(matched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for species in ALLOWED_SPECIES:
        tp = sum(row["true_species"] == species and row["predicted_species"] == species for row in matched_rows)
        fp = sum(row["true_species"] != species and row["predicted_species"] == species for row in matched_rows)
        fn = sum(row["true_species"] == species and row["predicted_species"] != species for row in matched_rows)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        rows.append(
            {
                "species": species,
                "matched_support": sum(row["true_species"] == species for row in matched_rows),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1(precision, recall),
            }
        )
    return rows


def confusion_matrix(matched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for true_species in ALLOWED_SPECIES:
        row: dict[str, Any] = {"true_species": true_species}
        for predicted_species in ALLOWED_SPECIES:
            row[predicted_species] = sum(
                item["true_species"] == true_species
                and item["predicted_species"] == predicted_species
                for item in matched_rows
            )
        rows.append(row)
    return rows


def evaluate(
    *,
    run_dir: Path,
    output_dir: Path,
    stage1_manifest: Path,
    v2_manifest: Path,
) -> dict[str, Any]:
    predictions = {row["anonymous_sample_id"]: row for row in read_csv(run_dir / "parsed_predictions.csv")}
    subset_manifest = run_dir / "pilot80_subset_manifest.csv"
    subset_rows = (
        read_csv(subset_manifest)
        if subset_manifest.is_file()
        else select_balanced_subset(load_csv_rows(stage1_manifest), 10)
    )
    specs = build_window_specs(stage1_rows=subset_rows, v2_rows=load_csv_rows(v2_manifest))
    matched_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    invalid_label_count = 0
    parse_success_count = 0
    selected_available_count = 0
    ious = {"time": [], "frequency": [], "box": []}
    protocol_counts: dict[str, Counter[str]] = {
        "temporal_iou_0p3": Counter(),
        "temporal_iou_0p1": Counter(),
        "start_time_10ms": Counter(),
    }
    for spec in specs:
        pred = predictions.get(spec.anonymous_sample_id, {})
        proposal = selected_proposal_from_prediction(pred)
        gt = gt_box(spec)
        selected_available = proposal is not None
        selected_available_count += 1 if selected_available else 0
        parse_success = pred.get("parse_status") == "success"
        parse_success_count += 1 if parse_success else 0
        predicted_species = pred.get("predicted_species", "")
        if parse_success and predicted_species not in ALLOWED_SPECIES:
            invalid_label_count += 1
        for protocol in protocol_counts:
            matched = protocol_match(gt, proposal, protocol)
            protocol_counts[protocol]["TP" if matched else "FN"] += 1
            protocol_counts[protocol]["FP"] += 0 if matched else (1 if selected_available else 0)
        match_03 = protocol_match(gt, proposal, "temporal_iou_0p3")
        if match_03 and proposal is not None:
            pbox = proposal_box(proposal)
            time_iou = temporal_iou(gt, pbox)
            freq_iou = frequency_iou(gt, pbox)
            bx_iou = box_iou(gt, pbox)
            ious["time"].append(time_iou)
            ious["frequency"].append(freq_iou)
            ious["box"].append(bx_iou)
            matched_rows.append(
                {
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "true_species": spec.species,
                    "predicted_species": predicted_species,
                    "species_correct": str(parse_success and predicted_species == spec.species).lower(),
                    "parse_status": pred.get("parse_status", "missing"),
                    "proposal_id": proposal["proposal_id"],
                    "confidence": pred.get("confidence", ""),
                    "time_iou": time_iou,
                    "frequency_iou": freq_iou,
                    "box_iou": bx_iou,
                }
            )
        else:
            missed_rows.append(
                {
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "true_species": spec.species,
                    "selected_proposal_available": str(selected_available).lower(),
                    "parse_status": pred.get("parse_status", "missing"),
                    "reason": "selected_proposal_not_matched_iou_0p3",
                }
            )
            if selected_available and proposal is not None:
                fp_rows.append(
                    {
                        "sample_id": spec.sample_id,
                        "anonymous_sample_id": spec.anonymous_sample_id,
                        "true_species": spec.species,
                        "proposal_id": proposal["proposal_id"],
                        "start_time": proposal["start_time_seconds"],
                        "end_time": proposal["end_time_seconds"],
                    }
                )
        sample_rows.append(
            {
                "sample_id": spec.sample_id,
                "anonymous_sample_id": spec.anonymous_sample_id,
                "true_species": spec.species,
                "parse_status": pred.get("parse_status", "missing"),
                "selected_proposal_available": str(selected_available).lower(),
                "matched_iou_0p3": str(match_03).lower(),
                "predicted_species": predicted_species,
            }
        )

    protocol_metrics: dict[str, dict[str, Any]] = {}
    for protocol, counts in protocol_counts.items():
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        protocol_metrics[protocol] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "F1": f1(precision, recall),
        }
    class_rows = per_species_classification(matched_rows)
    species_correct = sum(row["species_correct"] == "true" for row in matched_rows)
    class_acc = safe_div(species_correct, len(matched_rows))
    macro_f1 = mean(row["F1"] for row in class_rows) if class_rows else 0.0
    balanced_acc = mean(row["recall"] for row in class_rows) if class_rows else 0.0
    total_predictions = selected_available_count
    joint_correct = species_correct
    joint_precision = safe_div(joint_correct, total_predictions)
    joint_recall = safe_div(joint_correct, len(specs))
    joint_rows = []
    for species in ALLOWED_SPECIES:
        support = sum(spec.species == species for spec in specs)
        correct = sum(
            row["true_species"] == species and row["species_correct"] == "true"
            for row in matched_rows
        )
        joint_rows.append(
            {
                "species": species,
                "support": support,
                "joint_correct": correct,
                "joint_recall": safe_div(correct, support),
            }
        )
    summary = {
        "condition": run_dir.name,
        "sample_count": len(specs),
        "selected_proposal_available_count": selected_available_count,
        "parse_success": parse_success_count,
        "parse_success_rate": safe_div(parse_success_count, len(specs)),
        "invalid_species_label_count": invalid_label_count,
        "total_predictions": total_predictions,
        "temporal_iou_0p3": protocol_metrics["temporal_iou_0p3"],
        "temporal_iou_0p1": protocol_metrics["temporal_iou_0p1"],
        "start_time_10ms": protocol_metrics["start_time_10ms"],
        "mean_time_iou": mean(ious["time"]) if ious["time"] else 0.0,
        "mean_frequency_iou": mean(ious["frequency"]) if ious["frequency"] else 0.0,
        "mean_box_iou": mean(ious["box"]) if ious["box"] else 0.0,
        "classification_on_matched": {
            "matched_count": len(matched_rows),
            "species_accuracy": class_acc,
            "macro_F1": macro_f1,
            "balanced_accuracy": balanced_acc,
        },
        "joint": {
            "joint_correct": joint_correct,
            "precision": joint_precision,
            "recall": joint_recall,
            "F1": f1(joint_precision, joint_recall),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "sample_level_results.csv", sample_rows)
    write_csv(output_dir / "matched_selected_proposals.csv", matched_rows)
    write_csv(output_dir / "missed_events.csv", missed_rows)
    write_csv(output_dir / "false_positive_selected_proposals.csv", fp_rows)
    write_csv(output_dir / "classification_per_species_metrics.csv", class_rows)
    write_csv(output_dir / "classification_confusion_matrix.csv", confusion_matrix(matched_rows))
    write_csv(output_dir / "joint_per_species_recall.csv", joint_rows)
    write_report(output_dir / "stage2c_selected_proposal_classification_report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    m = summary["temporal_iou_0p3"]
    cls = summary["classification_on_matched"]
    joint = summary["joint"]
    scope_label = "Full240" if summary["sample_count"] > 80 else "Pilot80"
    lines = [
        f"# Stage 2C Nearest-Centre Proposal Classification {scope_label}",
        "",
        "## Design",
        "",
        "This condition preserves deterministic `nearest_to_centre` BatDetect2 proposal coordinates and asks qwen3.6 only to classify the selected proposal. The model is not allowed to refine, redraw, add, or remove detections.",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Selected proposals available: {summary['selected_proposal_available_count']}",
        f"- Parse success: {summary['parse_success']}/{summary['sample_count']} ({summary['parse_success_rate']:.3f})",
        "",
        "## Localisation",
        "",
        "| Protocol | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in ("temporal_iou_0p3", "temporal_iou_0p1", "start_time_10ms"):
        row = summary[protocol]
        lines.append(
            f"| {protocol} | {row['TP']} | {row['FP']} | {row['FN']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['F1']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Mean time IoU: {summary['mean_time_iou']:.3f}",
            f"- Mean frequency IoU: {summary['mean_frequency_iou']:.3f}",
            f"- Mean 2D box IoU: {summary['mean_box_iou']:.3f}",
            "",
            "## Classification on Matched Selected Proposals",
            "",
            f"- Matched selected proposals: {cls['matched_count']}",
            f"- Species accuracy: {cls['species_accuracy']:.3f}",
            f"- Macro-F1: {cls['macro_F1']:.3f}",
            f"- Balanced accuracy: {cls['balanced_accuracy']:.3f}",
            "",
            "## Joint Metrics",
            "",
            f"- Joint correct: {joint['joint_correct']}",
            f"- Joint precision: {joint['precision']:.3f}",
            f"- Joint recall: {joint['recall']:.3f}",
            f"- Joint F1: {joint['F1']:.3f}",
            "",
            "## Interpretation",
            "",
            "Localisation is fixed by deterministic proposal selection, so remaining joint-task failure is dominated by species classification. If matched-proposal species accuracy remains low, the bottleneck is acoustic label discrimination rather than target localisation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage1-manifest", type=Path, default=DEFAULT_STAGE1_MANIFEST)
    parser.add_argument("--v2-manifest", type=Path, default=DEFAULT_V2_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        stage1_manifest=args.stage1_manifest,
        v2_manifest=args.v2_manifest,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
