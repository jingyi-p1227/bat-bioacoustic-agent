"""Pydantic Evals orchestration wrapper for existing annotation metrics.

This script does not run the agent and does not replace ``evaluation.py``.
It uses Pydantic Evals to organize one prediction-vs-ground-truth case, while
delegating all domain-specific matching and metric computation to
``evaluation.evaluate_by_mode``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evaluation import evaluate_by_mode


DEFAULT_OUTPUT_DIR = Path("outputs/pydantic_evals")


def load_existing_prediction(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return the existing prediction artifact/config without running a model."""
    return dict(inputs)


def metric_score(metrics: dict[str, Any]) -> float:
    """Choose a simple scalar score for the Pydantic Evals report."""
    if "mean_iou" in metrics:
        return float(metrics["mean_iou"])
    if "mean_temporal_iou" in metrics:
        return float(metrics["mean_temporal_iou"])

    ground_truth_count = float(metrics.get("ground_truth_count") or 0)
    if ground_truth_count == 0:
        return 0.0
    return float(metrics.get("matched_events") or 0) / ground_truth_count


@dataclass
class ExistingAnnotationMetricsEvaluator(Evaluator):
    """Evaluate existing JSON artifacts using the local metric engine."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        metrics = evaluate_by_mode(
            predicted_path=ctx.output["prediction_path"],
            ground_truth_path=ctx.expected_output["ground_truth_path"],
            iou_threshold=float(ctx.output["iou_threshold"]),
            mode=ctx.output["mode"],
        )
        return metric_score(metrics)


def safe_case_name(case_name: str) -> str:
    """Convert a case name to a filesystem-safe stem."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name.strip())
    return safe_name.strip("._") or "pydantic_eval"


def build_case(
    *,
    prediction_path: Path,
    ground_truth_path: Path,
    mode: str,
    iou_threshold: float,
    case_name: str,
) -> Case:
    """Build one CLI-driven Pydantic Evals case."""
    return Case(
        name=case_name,
        inputs={
            "prediction_path": str(prediction_path),
            "mode": mode,
            "iou_threshold": iou_threshold,
        },
        expected_output={
            "ground_truth_path": str(ground_truth_path),
        },
        metadata={
            "case_name": case_name,
            "notes": "Existing prediction JSON evaluated against existing ground truth JSON.",
        },
    )


def save_metrics_json(
    metrics: dict[str, Any],
    *,
    output_path: Path,
    case_name: str,
    mode: str,
    score: float,
) -> Path:
    """Save the deterministic metric dictionary plus orchestration metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_name": case_name,
        "mode": mode,
        "score": score,
        "metrics": metrics,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return output_path


def save_summary_csv(
    metrics: dict[str, Any],
    *,
    output_path: Path,
    case_name: str,
    mode: str,
    score: float,
) -> Path:
    """Save a flat one-row summary CSV for quick inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "case_name": case_name,
        "mode": mode,
        "score": score,
        **metrics,
    }
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a thin Pydantic Evals wrapper around evaluation.py."
    )
    parser.add_argument("--pred", required=True, help="Prediction EventResult JSON path.")
    parser.add_argument("--gt", required=True, help="Ground-truth EventResult JSON path.")
    parser.add_argument(
        "--mode",
        choices=["box2d", "temporal"],
        default="box2d",
        help="Evaluation mode passed through to evaluation.py.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum IoU threshold for event matching.",
    )
    parser.add_argument(
        "--case-name",
        default="local_annotation_eval",
        help="Name for the Pydantic Evals case and output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for deterministic metrics outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.pred)
    ground_truth_path = Path(args.gt)
    file_stem = safe_case_name(args.case_name)

    case = build_case(
        prediction_path=prediction_path,
        ground_truth_path=ground_truth_path,
        mode=args.mode,
        iou_threshold=args.iou_threshold,
        case_name=args.case_name,
    )
    dataset = Dataset(name="Local annotation evals", cases=[case])
    dataset.add_evaluator(ExistingAnnotationMetricsEvaluator())

    report = dataset.evaluate_sync(load_existing_prediction)
    report.print(
        include_input=True,
        include_expected_output=True,
        include_output=True,
        include_metadata=True,
        include_durations=False,
    )

    metrics = evaluate_by_mode(
        predicted_path=prediction_path,
        ground_truth_path=ground_truth_path,
        iou_threshold=args.iou_threshold,
        mode=args.mode,
    )
    score = metric_score(metrics)
    metrics_path = save_metrics_json(
        metrics,
        output_path=args.output_dir / f"{file_stem}_metrics.json",
        case_name=args.case_name,
        mode=args.mode,
        score=score,
    )
    summary_path = save_summary_csv(
        metrics,
        output_path=args.output_dir / f"{file_stem}_summary.csv",
        case_name=args.case_name,
        mode=args.mode,
        score=score,
    )

    print(f"Saved metrics JSON to {metrics_path}")
    print(f"Saved summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
