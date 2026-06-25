"""Summarize Prompt V2 model smoke-test runs without rerunning models."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("outputs/agent_runs/model_smoke_test_comparison.csv")
DEFAULT_CLIP_IDS = ["OP_001", "OP_010", "OP_045", "OP_003", "OP_004", "OP_016"]
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")


@dataclass(frozen=True)
class SmokeRun:
    run_name: str
    run_dir: Path
    grid_style: str = ""
    setting: str = ""
    evaluation_dir_name: str = "evaluation"


SUMMARY_FIELDS = [
    "model_name",
    "run_name",
    "grid_style",
    "setting",
    "clips_run",
    "parse_success_count",
    "parse_failure_count",
    "total_gt",
    "total_predictions",
    "TP",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "mean_time_iou",
    "mean_frequency_iou",
    "mean_box_iou",
    "box_iou_gte_0_3_count",
    "box_iou_gte_0_5_count",
]


def parse_run_specs(values: list[str]) -> list[SmokeRun]:
    """Parse run specs as run_name:grid_style:setting:evaluation_dir=path."""
    runs: list[SmokeRun] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Run spec must be run_name=path, got: {value}")
        run_label, run_dir = value.split("=", 1)
        label_parts = run_label.split(":")
        run_name = label_parts[0]
        grid_style = label_parts[1] if len(label_parts) > 1 else ""
        setting = label_parts[2] if len(label_parts) > 2 else ""
        evaluation_dir_name = label_parts[3] if len(label_parts) > 3 else "evaluation"
        run_name = run_name.strip()
        if not run_name:
            raise ValueError(f"Run name cannot be empty in spec: {value}")
        runs.append(
            SmokeRun(
                run_name=run_name,
                run_dir=Path(run_dir),
                grid_style=grid_style.strip(),
                setting=setting.strip(),
                evaluation_dir_name=evaluation_dir_name.strip() or "evaluation",
            )
        )
    return runs


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def prediction_parse_status(prediction_path: Path) -> tuple[str | None, str | None]:
    """Return model name and parse status for a prediction artifact."""
    payload = load_json(prediction_path)
    model_name = payload.get("model_name")
    parse_status = payload.get("parse_status")
    if parse_status is None:
        parse_status = "success" if isinstance(payload.get("events"), list) else "failed"
    return (
        str(model_name) if model_name is not None else None,
        str(parse_status),
    )


def summarize_run(run: SmokeRun, clip_ids: list[str]) -> dict[str, Any]:
    """Combine parse status and evaluation aggregate for one smoke-test run."""
    success_count = 0
    failure_count = 0
    model_name = ""
    for clip_id in clip_ids:
        prediction_path = run.run_dir / f"{clip_id}_predictions.json"
        error_path = run.run_dir / f"{clip_id}_parse_error.txt"
        try:
            current_model, parse_status = prediction_parse_status(prediction_path)
            if current_model and not model_name:
                model_name = current_model
            if parse_status == "failed" or error_path.exists():
                failure_count += 1
            else:
                success_count += 1
        except Exception:
            failure_count += 1

    aggregate = load_json(run.run_dir / run.evaluation_dir_name / "aggregate_summary.json")
    return {
        "model_name": model_name,
        "run_name": run.run_name,
        "grid_style": run.grid_style,
        "setting": run.setting,
        "clips_run": len(clip_ids),
        "parse_success_count": success_count,
        "parse_failure_count": failure_count,
        "total_gt": aggregate["total_ground_truth_events"],
        "total_predictions": aggregate["total_predictions"],
        "TP": aggregate["total_tp"],
        "FP": aggregate["total_fp"],
        "FN": aggregate["total_fn"],
        "precision": aggregate["precision"],
        "recall": aggregate["recall"],
        "F1": aggregate["f1"],
        "mean_time_iou": aggregate["mean_time_iou"],
        "mean_frequency_iou": aggregate["mean_frequency_iou"],
        "mean_box_iou": aggregate["mean_box_iou"],
        "box_iou_gte_0_3_count": aggregate["strict_box_iou_0_3_count"],
        "box_iou_gte_0_5_count": aggregate["strict_box_iou_0_5_count"],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def resolve_all_clip_ids(eval_dir: str | Path) -> list[str]:
    """Return all evaluation clip ids from audio/*.wav in stable order."""
    audio_dir = Path(eval_dir) / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Evaluation audio directory not found: {audio_dir}")
    clip_ids = [path.stem for path in sorted(audio_dir.glob("*.wav"))]
    if not clip_ids:
        raise ValueError(f"No WAV files found in {audio_dir}")
    return clip_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Prompt V2 alternative-model smoke tests."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as run_name=run_dir. Repeat once per model.",
    )
    parser.add_argument(
        "--clip-list",
        default=",".join(DEFAULT_CLIP_IDS),
        help="Comma-separated clip ids included in the smoke test.",
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Summarize every WAV clip in <eval-dir>/audio.",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_ids = resolve_all_clip_ids(args.eval_dir) if args.all else parse_clip_ids(args.clip_list)
    rows = [summarize_run(run, clip_ids) for run in parse_run_specs(args.run)]
    write_summary_csv(args.output_file, rows)
    print(f"Saved comparison summary to {args.output_file}")
    for row in rows:
        print(
            f"{row['run_name']}: model={row['model_name']} "
            f"success={row['parse_success_count']} failed={row['parse_failure_count']} "
            f"F1={float(row['F1']):.3f}"
        )


if __name__ == "__main__":
    main()
