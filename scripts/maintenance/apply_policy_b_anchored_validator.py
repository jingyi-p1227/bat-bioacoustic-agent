"""Apply frozen P6E.4 Policy B to an arbitrary clip list in shadow mode."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.maintenance.apply_timing_preserving_validator as timing
from scripts.maintenance.apply_timing_rule_ablation_validator import (
    constrain_clip_payload,
    write_policy_decision_summary,
)


POLICY_NAME = "policy_b_anchored_expansion"
DEFAULT_PROPOSAL_DIR = Path("outputs/tool_outputs/batdetect2_proposals/p6e5_heldout")
DEFAULT_PREDICTION_DIR = Path(
    "outputs/agent_runs/p6e5_batdetect2_metadata_assisted_heldout/predictions"
)
DEFAULT_AUDIT_CSV = Path(
    "outputs/agent_runs/p6e5_batdetect2_metadata_assisted_heldout/"
    "proposal_deviation_analysis/proposal_deviation_events.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/agent_runs/p6e5_policy_b_anchored_validator_heldout"
)
DEFAULT_CLIP_LIST_FILE = Path(
    "outputs/evaluation_sets/ozimops_petersi_v1/p6e5_heldout_clip_list.txt"
)
METHOD_EVALUATIONS = {
    "proposal-only": Path(
        "outputs/agent_runs/p6e5_batdetect2_proposal_only_heldout/evaluation"
    ),
    "unconstrained metadata-assisted qwen3.6": Path(
        "outputs/agent_runs/p6e5_batdetect2_metadata_assisted_heldout/evaluation"
    ),
    "Policy B anchored validator": DEFAULT_OUTPUT_DIR / "evaluation",
}


def load_clip_list(path: Path) -> list[str]:
    """Read a stable one-id-per-line clip list."""
    if not path.is_file():
        raise FileNotFoundError(f"Clip list not found: {path}")
    clip_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    clip_ids = [clip_id for clip_id in clip_ids if clip_id and not clip_id.startswith("#")]
    if not clip_ids:
        raise ValueError(f"Clip list is empty: {path}")
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError(f"Clip list contains duplicate ids: {path}")
    return clip_ids


def output_prediction_path(output_dir: Path, clip_id: str) -> Path:
    return output_dir / "predictions" / f"{clip_id}_predictions.json"


def build_method_summary(
    method_evaluations: dict[str, Path],
) -> list[dict[str, object]]:
    """Read canonical aggregate summaries for the held-out comparison."""
    rows: list[dict[str, object]] = []
    for method, evaluation_dir in method_evaluations.items():
        payload = json.loads(
            (evaluation_dir / "aggregate_summary.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "method": method,
                "clip_count": payload["clip_count"],
                "prediction_count": payload["total_predictions"],
                "TP": payload["total_tp"],
                "FP": payload["total_fp"],
                "FN": payload["total_fn"],
                "precision": payload["precision"],
                "recall": payload["recall"],
                "F1": payload["f1"],
                "mean_time_iou": payload["mean_time_iou"],
                "mean_frequency_iou": payload["mean_frequency_iou"],
                "mean_box_iou": payload["mean_box_iou"],
                "box_iou_gte_0_3": payload["strict_box_iou_0_3_count"],
                "box_iou_gte_0_5": payload["strict_box_iou_0_5_count"],
            }
        )
    return rows


def write_method_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("At least one method summary row is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_policy_b(
    *,
    proposal_dir: Path,
    prediction_dir: Path,
    audit_csv: Path,
    output_dir: Path,
    clip_ids: list[str],
    overwrite: bool,
) -> tuple[list[Path], Path]:
    """Write direct Policy B outputs without changing P6E.4 artifacts."""
    audit_rows = timing.load_audit_rows(audit_csv)
    paths: list[Path] = []
    payloads: list[dict] = []
    for clip_id in clip_ids:
        source_prediction_path = prediction_dir / f"{clip_id}_predictions.json"
        proposal_path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
        output_path = output_prediction_path(output_dir, clip_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {output_path}. Use --overwrite.")
        prediction_payload = json.loads(source_prediction_path.read_text(encoding="utf-8"))
        proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        constrained = constrain_clip_payload(
            clip_id=clip_id,
            prediction_payload=prediction_payload,
            proposal_payload=proposal_payload,
            audit_rows=audit_rows,
            policy_name=POLICY_NAME,
        )
        constrained["source_prediction_path"] = source_prediction_path.as_posix()
        constrained["proposal_metadata_path"] = proposal_path.as_posix()
        output_path.write_text(
            json.dumps(constrained, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths.append(output_path)
        payloads.append(constrained)
    summary_path = output_dir / "validation_decision_summary.csv"
    write_policy_decision_summary(summary_path, payloads)
    return paths, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-list-file", type=Path, default=DEFAULT_CLIP_LIST_FILE)
    parser.add_argument("--clip-list", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only:
        summary_path = args.output_dir / "p6e5_heldout_summary.csv"
        method_evaluations = {
            **METHOD_EVALUATIONS,
            "Policy B anchored validator": args.output_dir / "evaluation",
        }
        write_method_summary(summary_path, build_method_summary(method_evaluations))
        print(f"Held-out summary: {summary_path}")
        return
    clip_ids = (
        timing.parse_clip_ids(args.clip_list)
        if args.clip_list
        else load_clip_list(args.clip_list_file)
    )
    paths, summary_path = apply_policy_b(
        proposal_dir=args.proposal_dir,
        prediction_dir=args.prediction_dir,
        audit_csv=args.audit_csv,
        output_dir=args.output_dir,
        clip_ids=clip_ids,
        overwrite=args.overwrite,
    )
    print(f"Created {len(paths)} Policy B prediction files.")
    print(f"Decision summary: {summary_path}")


if __name__ == "__main__":
    main()
