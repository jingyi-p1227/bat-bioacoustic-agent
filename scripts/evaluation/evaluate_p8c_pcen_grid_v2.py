"""Evaluate P8C against frozen grid_v2 and existing PCEN full-45 runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
EVALUATION_SCRIPT_DIR = REPO_ROOT / "scripts/evaluation"
for path in (SRC_ROOT, EVALUATION_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_multi_protocol_detection import (  # noqa: E402
    DEFAULT_EVAL_DIR,
    FrozenRun,
    all_clip_ids,
    evaluate_runs,
    write_csv,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/analysis_reports/p8c_pcen_grid_v2_full45"


def p8c_runs(eval_dir: Path) -> list[FrozenRun]:
    full_45 = all_clip_ids(eval_dir)
    return [
        FrozenRun(
            "p5_qwen_grid_v2_full",
            "full_45",
            "full_45",
            "standard_grid_v2",
            "qwen3.6:latest",
            REPO_ROOT / "outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2",
            full_45,
            "Frozen standard dB spectrogram plus grid_v2 baseline.",
        ),
        FrozenRun(
            "p8b_pcen_full45",
            "full_45",
            "full_45",
            "PCEN_existing",
            "qwen3.6:latest",
            REPO_ROOT / "outputs/agent_runs/p8b_pcen_qwen3_6_full45/predictions",
            full_45,
            "Existing P8B PCEN full-45 run; generation code used grid_v2 styling.",
        ),
        FrozenRun(
            "p8c_pcen_grid_v2_full45",
            "full_45",
            "full_45",
            "PCEN_grid_v2_explicit",
            "qwen3.6:latest",
            REPO_ROOT / "outputs/agent_runs/p8c_pcen_grid_v2_qwen3_6_full45/predictions",
            full_45,
            "New P8C explicit PCEN plus grid_v2 confirmatory run.",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = p8c_runs(args.eval_dir)
    experiment_rows, case_rows, pair_rows = evaluate_runs(runs, args.eval_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "p8c_condition_summary.csv", experiment_rows)
    write_csv(args.output_dir / "p8c_case_level_results.csv", case_rows)
    write_csv(args.output_dir / "p8c_matched_pair_box_quality.csv", pair_rows)
    write_csv(
        args.output_dir / "p8c_grid_pcen_comparison.csv",
        [
            {
                "experiment_id": row["experiment_id"],
                "method": row["method"],
                "model": row["model"],
                "protocol": row["protocol"],
                "clip_count": row["clip_count"],
                "predicted_count": row["predicted_count"],
                "ground_truth_count": row["ground_truth_count"],
                "TP": row["TP"],
                "FP": row["FP"],
                "FN": row["FN"],
                "precision": row["precision"],
                "recall": row["recall"],
                "F1": row["F1"],
                "mean_time_iou": row["mean_time_iou"],
                "mean_frequency_iou": row["mean_frequency_iou"],
                "mean_box_iou": row["mean_box_iou"],
                "box_iou_ge_0_3": row["box_iou_ge_0_3"],
                "box_iou_ge_0_5": row["box_iou_ge_0_5"],
                "parse_success_count": row["parse_success_count"],
                "parse_failure_count": row["parse_failure_count"],
                "notes": row["notes"],
            }
            for row in experiment_rows
        ],
    )
    print(f"Wrote P8C evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
