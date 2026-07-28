"""Evaluate Stage 2 pilot80 joint localisation and classification predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.run_stage1a_multispecies_classification import ALLOWED_LABELS
from scripts.inference.run_stage2_joint_proposal_constrained_pilot80 import CONDITION_NAME


DEFAULT_RUN_DIR = Path("outputs/agent_runs/multispecies_classification") / CONDITION_NAME
DEFAULT_OUTPUT_DIR = Path("outputs/analysis_reports/multispecies_classification") / CONDITION_NAME


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return safe_div(inter, union)


def box_iou(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    time_inter = max(0.0, min(pred["end_time"], gt["event_end_time"]) - max(pred["start_time"], gt["event_start_time"]))
    freq_inter = max(0.0, min(pred["high_freq"], gt["event_high_freq"]) - max(pred["low_freq"], gt["event_low_freq"]))
    inter = time_inter * freq_inter
    pred_area = max(0.0, pred["end_time"] - pred["start_time"]) * max(0.0, pred["high_freq"] - pred["low_freq"])
    gt_area = max(0.0, gt["event_end_time"] - gt["event_start_time"]) * max(0.0, gt["event_high_freq"] - gt["event_low_freq"])
    return safe_div(inter, pred_area + gt_area - inter)


def parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def valid_detection(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    start = parse_float(raw.get("start_time"))
    end = parse_float(raw.get("end_time"))
    low = parse_float(raw.get("low_freq"))
    high = parse_float(raw.get("high_freq"))
    conf = parse_float(raw.get("confidence"))
    species = raw.get("predicted_species")
    if start is None or end is None or low is None or high is None:
        return None, "non_numeric_geometry"
    if start >= end or low >= high or low < 0:
        return None, "invalid_geometry"
    if conf is None or not 0 <= conf <= 1:
        return None, "invalid_confidence"
    if species not in ALLOWED_LABELS:
        return None, "invalid_species_label"
    return {
        "proposal_id": str(raw.get("proposal_id") or ""),
        "decision": str(raw.get("decision") or ""),
        "start_time": start,
        "end_time": end,
        "low_freq": low,
        "high_freq": high,
        "predicted_species": species,
        "confidence": conf,
    }, ""


def gt_timing(row: dict[str, str]) -> tuple[float, float, str]:
    """Use local-window GT timing when available, otherwise source timing."""

    if row.get("local_gt_start_time") not in (None, "") and row.get("local_gt_end_time") not in (None, ""):
        return float(row["local_gt_start_time"]), float(row["local_gt_end_time"]), "local_window"
    return float(row["event_start_time"]), float(row["event_end_time"]), "source_time"


def greedy_match(gt: dict[str, Any], predictions: list[dict[str, Any]], threshold: float) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    eligible = []
    for pred in predictions:
        tiou = interval_iou(pred["start_time"], pred["end_time"], gt["event_start_time"], gt["event_end_time"])
        eligible.append((tiou, pred["confidence"], pred))
    eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if eligible and eligible[0][0] >= threshold:
        matched = eligible[0][2]
        return matched, [pred for pred in predictions if pred is not matched]
    return None, predictions


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(run_dir: Path, output_dir: Path, condition_name: str | None = None) -> dict[str, Any]:
    subset_rows = load_csv(run_dir / "pilot80_subset_manifest.csv")
    prediction_rows = {row["anonymous_sample_id"]: row for row in load_csv(run_dir / "parsed_predictions.csv")}
    matched_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    total_predictions = 0
    valid_predictions = 0
    parse_success = 0
    proposal_status_counts: Counter[str] = Counter()

    for gt_row in subset_rows:
        anon_id = gt_row["anonymous_sample_id"]
        pred_row = prediction_rows.get(anon_id, {})
        proposal_status_counts[pred_row.get("proposal_status", "missing_prediction_row")] += 1
        gt_start, gt_end, coordinate_frame = gt_timing(gt_row)
        gt = {
            "event_start_time": gt_start,
            "event_end_time": gt_end,
            "event_low_freq": float(gt_row["event_low_freq"]),
            "event_high_freq": float(gt_row["event_high_freq"]),
            "species": gt_row["species"],
        }
        raw_predictions = []
        if pred_row.get("parse_status") == "success":
            parse_success += 1
            try:
                raw_predictions = json.loads(pred_row.get("detections_json") or "[]")
            except json.JSONDecodeError:
                raw_predictions = []
        total_predictions += len(raw_predictions)
        predictions: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_predictions, start=1):
            parsed, reason = valid_detection(raw)
            if parsed is None:
                invalid_rows.append({"anonymous_sample_id": anon_id, "detection_index": index, "reason": reason})
            else:
                parsed["detection_index"] = index
                predictions.append(parsed)
        valid_predictions += len(predictions)

        match, unmatched = greedy_match(gt, predictions, threshold=0.3)
        matched_for_sample = 0
        if match is not None:
            matched_for_sample = 1
            tiou = interval_iou(match["start_time"], match["end_time"], gt["event_start_time"], gt["event_end_time"])
            fiou = interval_iou(match["low_freq"], match["high_freq"], gt["event_low_freq"], gt["event_high_freq"])
            biou = box_iou(match, gt)
            matched_rows.append(
                {
                    "sample_id": gt_row["sample_id"],
                    "anonymous_sample_id": anon_id,
                    "true_species": gt["species"],
                    "predicted_species": match["predicted_species"],
                    "species_correct": str(match["predicted_species"] == gt["species"]).lower(),
                    "confidence": match["confidence"],
                    "proposal_id": match.get("proposal_id", ""),
                    "decision": match.get("decision", ""),
                    "time_iou": tiou,
                    "frequency_iou": fiou,
                    "box_iou": biou,
                    "start_time_error": match["start_time"] - gt["event_start_time"],
                    "end_time_error": match["end_time"] - gt["event_end_time"],
                    "low_frequency_error": match["low_freq"] - gt["event_low_freq"],
                    "high_frequency_error": match["high_freq"] - gt["event_high_freq"],
                }
            )
        else:
            missed_rows.append(
                {
                    "sample_id": gt_row["sample_id"],
                    "anonymous_sample_id": anon_id,
                    "true_species": gt["species"],
                    "event_start_time": gt["event_start_time"],
                    "event_end_time": gt["event_end_time"],
                    "event_low_freq": gt["event_low_freq"],
                    "event_high_freq": gt["event_high_freq"],
                }
            )
        for pred in unmatched:
            false_positive_rows.append(
                {
                    "sample_id": gt_row["sample_id"],
                    "anonymous_sample_id": anon_id,
                    "predicted_species": pred["predicted_species"],
                    "confidence": pred["confidence"],
                    "start_time": pred["start_time"],
                    "end_time": pred["end_time"],
                    "low_freq": pred["low_freq"],
                    "high_freq": pred["high_freq"],
                }
            )
        sample_rows.append(
            {
                "sample_id": gt_row["sample_id"],
                "anonymous_sample_id": anon_id,
                "species": gt["species"],
                "parse_status": pred_row.get("parse_status", "missing"),
                "proposal_status": pred_row.get("proposal_status", ""),
                "prediction_count": len(raw_predictions),
                "valid_prediction_count": len(predictions),
                "matched": matched_for_sample,
                "false_positives": len(unmatched),
                "missed": 1 - matched_for_sample,
            }
        )

    gt_count = len(subset_rows)
    tp = len(matched_rows)
    fp = len(false_positive_rows)
    fn = len(missed_rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    joint_correct = sum(row["species_correct"] == "true" for row in matched_rows)
    joint_precision = safe_div(joint_correct, tp + fp)
    joint_recall = safe_div(joint_correct, gt_count)
    # Threshold variants.
    all_valid_by_sample: dict[str, list[dict[str, Any]]] = {}
    pred_rows = prediction_rows
    for gt_row in subset_rows:
        raw = json.loads(pred_rows.get(gt_row["anonymous_sample_id"], {}).get("detections_json") or "[]") if pred_rows.get(gt_row["anonymous_sample_id"], {}).get("parse_status") == "success" else []
        all_valid_by_sample[gt_row["anonymous_sample_id"]] = [p for p, reason in (valid_detection(item) for item in raw) if p is not None]

    def threshold_counts(mode: str) -> tuple[int, int, int]:
        mode_tp = mode_fp = mode_fn = 0
        for gt_row in subset_rows:
            gt_start, gt_end, _coordinate_frame = gt_timing(gt_row)
            gt = {
                "event_start_time": gt_start,
                "event_end_time": gt_end,
            }
            preds = list(all_valid_by_sample[gt_row["anonymous_sample_id"]])
            matched = None
            if mode == "iou_0p1":
                matched, preds = greedy_match(gt, preds, 0.1)  # type: ignore[arg-type]
            elif mode == "start_10ms":
                preds.sort(key=lambda p: (abs(p["start_time"] - gt["event_start_time"]), -p["confidence"]))
                if preds and abs(preds[0]["start_time"] - gt["event_start_time"]) <= 0.010:
                    matched = preds.pop(0)
            else:
                raise ValueError(mode)
            mode_tp += 1 if matched else 0
            mode_fn += 0 if matched else 1
            mode_fp += len(preds)
        return mode_tp, mode_fp, mode_fn

    iou01_tp, iou01_fp, iou01_fn = threshold_counts("iou_0p1")
    ms_tp, ms_fp, ms_fn = threshold_counts("start_10ms")

    # Classification on matched.
    class_rows = []
    for species in ALLOWED_LABELS:
        stp = sum(r["true_species"] == species and r["predicted_species"] == species for r in matched_rows)
        sfp = sum(r["true_species"] != species and r["predicted_species"] == species for r in matched_rows)
        sfn = sum(r["true_species"] == species and r["predicted_species"] != species for r in matched_rows)
        sprec = safe_div(stp, stp + sfp)
        srec = safe_div(stp, stp + sfn)
        class_rows.append(
            {
                "species": species,
                "matched_support": sum(r["true_species"] == species for r in matched_rows),
                "TP": stp,
                "FP": sfp,
                "FN": sfn,
                "precision": sprec,
                "recall": srec,
                "F1": f1(sprec, srec),
            }
        )
    class_acc = safe_div(joint_correct, len(matched_rows))
    macro_f1 = mean(row["F1"] for row in class_rows) if class_rows else 0.0
    bal_acc = mean(row["recall"] for row in class_rows) if class_rows else 0.0
    confusion_rows = []
    for true_species in ALLOWED_LABELS:
        out = {"true_species": true_species}
        for pred_species in ALLOWED_LABELS:
            out[pred_species] = sum(r["true_species"] == true_species and r["predicted_species"] == pred_species for r in matched_rows)
        confusion_rows.append(out)
    joint_recall_rows = []
    for species in ALLOWED_LABELS:
        support = sum(row["species"] == species for row in subset_rows)
        correct = sum(row["true_species"] == species and row["species_correct"] == "true" for row in matched_rows)
        joint_recall_rows.append({"species": species, "support": support, "joint_correct": correct, "joint_recall": safe_div(correct, support)})

    condition = condition_name or run_dir.name
    summary = {
        "condition": condition,
        "sample_count": gt_count,
        "parse_success": parse_success,
        "parse_success_rate": safe_div(parse_success, gt_count),
        "proposal_status_counts": dict(proposal_status_counts),
        "coordinate_frame": gt_timing(subset_rows[0])[2] if subset_rows else "unknown",
        "samples_with_available_proposals": proposal_status_counts.get("available", 0),
        "total_input_proposals": sum(int(row.get("proposal_count") or 0) for row in subset_rows),
        "total_predictions": total_predictions,
        "valid_predictions": valid_predictions,
        "invalid_bbox_count": len([r for r in invalid_rows if r["reason"] != "invalid_species_label"]),
        "invalid_species_label_count": len([r for r in invalid_rows if r["reason"] == "invalid_species_label"]),
        "temporal_iou_0p3": {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1(precision, recall)},
        "temporal_iou_0p1": {"TP": iou01_tp, "FP": iou01_fp, "FN": iou01_fn, "precision": safe_div(iou01_tp, iou01_tp + iou01_fp), "recall": safe_div(iou01_tp, iou01_tp + iou01_fn), "F1": f1(safe_div(iou01_tp, iou01_tp + iou01_fp), safe_div(iou01_tp, iou01_tp + iou01_fn))},
        "start_time_10ms": {"TP": ms_tp, "FP": ms_fp, "FN": ms_fn, "precision": safe_div(ms_tp, ms_tp + ms_fp), "recall": safe_div(ms_tp, ms_tp + ms_fn), "F1": f1(safe_div(ms_tp, ms_tp + ms_fp), safe_div(ms_tp, ms_tp + ms_fn))},
        "mean_time_iou": mean(float(r["time_iou"]) for r in matched_rows) if matched_rows else 0.0,
        "mean_frequency_iou": mean(float(r["frequency_iou"]) for r in matched_rows) if matched_rows else 0.0,
        "mean_box_iou": mean(float(r["box_iou"]) for r in matched_rows) if matched_rows else 0.0,
        "classification_on_matched": {"matched_count": len(matched_rows), "species_accuracy": class_acc, "macro_F1": macro_f1, "balanced_accuracy": bal_acc},
        "joint": {"joint_correct": joint_correct, "precision": joint_precision, "recall": joint_recall, "F1": f1(joint_precision, joint_recall)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "sample_level_results.csv", sample_rows)
    write_csv(output_dir / "matched_detections.csv", matched_rows)
    write_csv(output_dir / "false_positives.csv", false_positive_rows)
    write_csv(output_dir / "missed_events.csv", missed_rows)
    write_csv(output_dir / "invalid_detections.csv", invalid_rows, ["anonymous_sample_id", "detection_index", "reason"])
    write_csv(output_dir / "classification_per_species_metrics.csv", class_rows)
    write_csv(output_dir / "classification_confusion_matrix.csv", confusion_rows)
    write_csv(output_dir / "joint_per_species_recall.csv", joint_recall_rows)

    proposal_note = (
        "This run used real sample-level BatDetect2 proposal files as candidate metadata."
        if proposal_status_counts.get("available", 0)
        else "No sample-level proposal files were available, so this run is a scaffold without detector proposals."
    )
    report = f"""# Stage 2 Joint Localisation + Species Classification Pilot80

## Design

- Condition: `{condition}`
- Model: `qwen3.6:latest`
- Subset: deterministic first 10 samples per species from the Stage 1 manifest (`80` total).
- Input images: clean `centred_crop_image_path` images only, with no GT box marker and no human-review overlays.
- Proposal metadata: `{dict(proposal_status_counts)}`.
- Coordinate frame for evaluation: `{summary['coordinate_frame']}`.
- Total input proposals in subset: `{summary['total_input_proposals']}`.

{proposal_note}

## Localisation Metrics

| Protocol | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Temporal IoU >= 0.3 | {tp} | {fp} | {fn} | {precision:.3f} | {recall:.3f} | {f1(precision, recall):.3f} |
| Temporal IoU >= 0.1 | {iou01_tp} | {iou01_fp} | {iou01_fn} | {summary['temporal_iou_0p1']['precision']:.3f} | {summary['temporal_iou_0p1']['recall']:.3f} | {summary['temporal_iou_0p1']['F1']:.3f} |
| 10 ms start-time | {ms_tp} | {ms_fp} | {ms_fn} | {summary['start_time_10ms']['precision']:.3f} | {summary['start_time_10ms']['recall']:.3f} | {summary['start_time_10ms']['F1']:.3f} |

- Mean time IoU: `{summary['mean_time_iou']:.3f}`
- Mean frequency IoU: `{summary['mean_frequency_iou']:.3f}`
- Mean 2D box IoU: `{summary['mean_box_iou']:.3f}`

## Classification on Matched Detections

- Matched detections: `{len(matched_rows)}`
- Species accuracy on matched events: `{class_acc:.3f}`
- Macro-F1 on matched events: `{macro_f1:.3f}`
- Balanced accuracy on matched events: `{bal_acc:.3f}`

## Joint Metrics

- Joint correct detections: `{joint_correct}`
- Joint precision: `{joint_precision:.3f}`
- Joint recall: `{joint_recall:.3f}`
- Joint F1: `{f1(joint_precision, joint_recall):.3f}`

## Interpretation

This pilot should be interpreted in light of proposal availability and coordinate frame. When real sample-level proposals are available, localisation performance reflects the model's proposal verification/refinement behaviour rather than free-form detection alone.

The comparison with Stage 1C should focus on matched detections: if localisation succeeds but species accuracy remains low, species discrimination remains the bottleneck. If localisation itself fails, the joint task is failing before species classification can be fairly assessed.

The full single-agent proposal-constrained localisation and BatDetect2 proposal-only results were produced on the Ozimops full45 benchmark, not this multi-species event-crop pilot, so they are workflow references rather than directly comparable metrics.
"""
    (output_dir / "stage2_joint_pilot80_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-name", default=None)
    args = parser.parse_args()
    summary = evaluate(args.run_dir, args.output_dir, condition_name=args.condition_name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
