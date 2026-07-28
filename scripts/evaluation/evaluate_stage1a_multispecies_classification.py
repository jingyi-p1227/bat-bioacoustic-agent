"""Evaluate multi-species Stage 1 classification predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.run_stage1a_multispecies_classification import (  # noqa: E402
    ALLOWED_LABELS,
    CONDITION_NAME,
)


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_stage1_gt_event_classification_dataset/"
    "stage1_manifest.csv"
)
DEFAULT_PREDICTION_CSV = (
    REPO_ROOT
    / "outputs/agent_runs/multispecies_classification/"
    / CONDITION_NAME
    / "parsed_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_classification/"
    / CONDITION_NAME
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def join_predictions(
    manifest_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_anon = {row["anonymous_sample_id"]: row for row in prediction_rows}
    joined: list[dict[str, Any]] = []
    for row in manifest_rows:
        pred = by_anon.get(row["anonymous_sample_id"], {})
        predicted = pred.get("predicted_species", "")
        parse_status = pred.get("parse_status", "missing")
        joined.append(
            {
                **row,
                "true_species": row["species"],
                "predicted_species": predicted,
                "parse_status": parse_status,
                "confidence": pred.get("confidence", ""),
                "correct": str(parse_status == "success" and predicted == row["species"]).lower(),
                "invalid_label": str(
                    parse_status == "success" and predicted not in ALLOWED_LABELS
                ).lower(),
                "parse_error": pred.get("parse_error", ""),
            }
        )
    return joined


def per_species_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = []
    for label in ALLOWED_LABELS:
        tp = sum(row["true_species"] == label and row["predicted_species"] == label for row in rows)
        fp = sum(row["true_species"] != label and row["predicted_species"] == label for row in rows)
        fn = sum(row["true_species"] == label and row["predicted_species"] != label for row in rows)
        support = sum(row["true_species"] == label for row in rows)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        metrics.append(
            {
                "species": label,
                "support": support,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1_score(precision, recall),
            }
        )
    return metrics


def confusion_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    for true_label in ALLOWED_LABELS:
        out = {"true_species": true_label}
        for pred_label in ALLOWED_LABELS:
            out[pred_label] = sum(
                row["true_species"] == true_label
                and row["predicted_species"] == pred_label
                for row in rows
            )
        out["parse_failed_or_invalid"] = sum(
            row["true_species"] == true_label
            and (
                row["parse_status"] != "success"
                or row["predicted_species"] not in ALLOWED_LABELS
            )
            for row in rows
        )
        matrix.append(out)
    return matrix


def confidence_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["parse_status"] != "success" or not row.get("confidence"):
            continue
        key = "correct" if row["correct"] == "true" else "incorrect"
        groups[key].append(float(row["confidence"]))
    output = []
    for key in ("correct", "incorrect"):
        values = groups.get(key, [])
        output.append(
            {
                "group": key,
                "count": len(values),
                "mean_confidence": mean(values) if values else 0.0,
                "min_confidence": min(values) if values else 0.0,
                "max_confidence": max(values) if values else 0.0,
            }
        )
    return output


def aggregate_metrics(rows: list[dict[str, Any]], species_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parse_success = sum(row["parse_status"] == "success" for row in rows)
    invalid_label = sum(row["invalid_label"] == "true" for row in rows)
    correct = sum(row["correct"] == "true" for row in rows)
    recalls = [float(row["recall"]) for row in species_rows]
    f1_values = [float(row["F1"]) for row in species_rows]
    return {
        "sample_count": total,
        "parse_success_count": parse_success,
        "parse_failure_count": total - parse_success,
        "parse_success_rate": safe_div(parse_success, total),
        "invalid_label_count": invalid_label,
        "overall_accuracy": safe_div(correct, total),
        "macro_F1": mean(f1_values) if f1_values else 0.0,
        "balanced_accuracy": mean(recalls) if recalls else 0.0,
    }


def main_confusions(rows: list[dict[str, Any]], top_k: int = 10) -> list[tuple[tuple[str, str], int]]:
    counts = Counter(
        (row["true_species"], row["predicted_species"])
        for row in rows
        if row["parse_status"] == "success"
        and row["predicted_species"] in ALLOWED_LABELS
        and row["true_species"] != row["predicted_species"]
    )
    return counts.most_common(top_k)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    *,
    aggregate: dict[str, Any],
    species_rows: list[dict[str, Any]],
    confusions: list[tuple[tuple[str, str], int]],
    condition_name: str,
    input_note: str,
) -> None:
    ranked = sorted(species_rows, key=lambda row: float(row["F1"]), reverse=True)
    rhino_rows = [row for row in species_rows if row["species"].startswith("Rhinolophus")]
    myotis_rows = [row for row in species_rows if row["species"].startswith("Myotis")]
    lines = [
        "# Multi-Species Stage 1 Classification Metrics",
        "",
        f"Condition: `{condition_name}`",
        "",
        input_note,
        "",
        "## Aggregate Metrics",
        "",
        f"- Samples: `{aggregate['sample_count']}`",
        f"- Parse success: `{aggregate['parse_success_count']}/{aggregate['sample_count']}` (`{aggregate['parse_success_rate']:.3f}`)",
        f"- Invalid label count: `{aggregate['invalid_label_count']}`",
        f"- Overall accuracy: `{aggregate['overall_accuracy']:.3f}`",
        f"- Macro-F1: `{aggregate['macro_F1']:.3f}`",
        f"- Balanced accuracy: `{aggregate['balanced_accuracy']:.3f}`",
        "",
        "## Per-Species Metrics",
        "",
        "| Species | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in species_rows:
        lines.append(
            f"| {row['species']} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['F1']):.3f} | {row['support']} |"
        )
    lines.extend(["", "## Main Confusions", ""])
    if confusions:
        for (true_label, pred_label), count in confusions:
            lines.append(f"- `{true_label}` -> `{pred_label}`: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Easiest species by F1: `{ranked[0]['species']}` (`F1={float(ranked[0]['F1']):.3f}`).",
            f"- Hardest species by F1: `{ranked[-1]['species']}` (`F1={float(ranked[-1]['F1']):.3f}`).",
            f"- Rhinolophus mean F1: `{mean(float(row['F1']) for row in rhino_rows):.3f}`.",
            f"- Myotis mean F1: `{mean(float(row['F1']) for row in myotis_rows):.3f}`.",
            "- Inspect the confusion matrix to determine whether Myotis species are mutually confused.",
            "- Inspect `Plecotus auritus` recall/F1 for the expected challenging-species behaviour.",
            "- Treat `Ozimops petersi` separately from UK species in interpretation because it comes from the Australia benchmark anchor.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-name", default=CONDITION_NAME)
    parser.add_argument(
        "--input-note",
        default=(
            "Model-facing images used `centred_crop_image_path` only. GT diagnostic "
            "overlays, GT-box marker images, species cards, Walters guidance, and "
            "image exemplars were not used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv(args.manifest)
    prediction_rows = read_csv(args.predictions)
    rows = join_predictions(manifest_rows, prediction_rows)
    species_rows = per_species_metrics(rows)
    aggregate = aggregate_metrics(rows, species_rows)
    matrix_rows = confusion_matrix(rows)
    confidence_rows = confidence_summary(rows)
    failure_rows = [
        row
        for row in rows
        if row["parse_status"] != "success"
        or row["predicted_species"] != row["true_species"]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", rows)
    write_csv(args.output_dir / "confusion_matrix.csv", matrix_rows)
    write_csv(args.output_dir / "per_species_metrics.csv", species_rows)
    write_csv(args.output_dir / "failure_cases.csv", failure_rows)
    write_csv(args.output_dir / "confidence_analysis.csv", confidence_rows)
    write_summary(
        args.output_dir / "metrics_summary.md",
        aggregate=aggregate,
        species_rows=species_rows,
        confusions=main_confusions(rows),
        condition_name=args.condition_name,
        input_note=args.input_note,
    )
    print(f"Evaluated {len(rows)} sample(s)")
    print(f"Accuracy={aggregate['overall_accuracy']:.3f} Macro-F1={aggregate['macro_F1']:.3f}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
