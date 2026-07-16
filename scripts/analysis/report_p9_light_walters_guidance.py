"""Create P9-light reports from existing evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_audio_agent.experiments.p9_light import (  # noqa: E402
    OPTIONAL_AGENT_CONDITION,
    TARGET_CLIPS,
    default_paths,
    load_json,
    walters_prompt_insert,
)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def condition_label(condition: str) -> str:
    labels = {
        "proposal_only": "BatDetect2 proposal-only",
        "agent_proposals_only": "Agent + proposals",
        "agent_methodological_literature": "Agent + methodological literature",
        "agent_walters_guidance": "Agent + Walters checklist",
        OPTIONAL_AGENT_CONDITION: "Agent + annotation memory + Walters checklist",
    }
    return labels.get(condition, condition)


def rows_by_protocol(rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["protocol"] == protocol]


def metric_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Condition | Pred | GT | TP | FP | FN | Precision | Recall | F1 | Mean time IoU | Mean freq IoU | Mean box IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {condition_label(row['condition'])} | {row['predicted_count']} | {row['ground_truth_count']} | "
            f"{row['TP']} | {row['FP']} | {row['FN']} | {fnum(row['precision']):.3f} | "
            f"{fnum(row['recall']):.3f} | {fnum(row['F1']):.3f} | {fnum(row.get('mean_time_iou')):.3f} | "
            f"{fnum(row.get('mean_frequency_iou')):.3f} | {fnum(row.get('mean_box_iou')):.3f} |"
        )
    return lines


def best_delta_cases(case_rows: list[dict[str, Any]], protocol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_clip: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in case_rows:
        if row["protocol"] == protocol:
            by_clip[row["clip_id"]][row["condition"]] = row
    deltas = []
    for clip_id, conds in by_clip.items():
        base = conds.get("agent_proposals_only")
        walters = conds.get("agent_walters_guidance")
        if base and walters:
            deltas.append(
                {
                    "clip_id": clip_id,
                    "base_f1": fnum(base["F1"]),
                    "walters_f1": fnum(walters["F1"]),
                    "delta_f1": fnum(walters["F1"]) - fnum(base["F1"]),
                    "base_fp": int(float(base["FP"])),
                    "walters_fp": int(float(walters["FP"])),
                    "base_fn": int(float(base["FN"])),
                    "walters_fn": int(float(walters["FN"])),
                }
            )
    deltas.sort(key=lambda row: row["delta_f1"], reverse=True)
    return deltas[:5], list(reversed(deltas[-5:])) if deltas else []


def write_guidance_card_markdown(paths) -> None:
    card = load_json(paths.walters_card_path)
    lines = [
        "# P9-light Walters-Style Generic Acoustic-Parameter Guidance",
        "",
        "This card is a generic annotation checklist, not an *Ozimops petersi* prior.",
        "",
        "Safety flags:",
        "",
        f"- status: `{card['status']}`",
        f"- not species-specific: `{card['not_species_specific']}`",
        f"- not OP prior: `{card['not_op_prior']}`",
        f"- no numeric species ranges: `{card['no_numeric_species_ranges']}`",
        f"- no European numeric transfer: `{card['no_european_numeric_transfer']}`",
        "",
        "Prompt insert:",
        "",
        "```text",
        walters_prompt_insert(card),
        "```",
    ]
    (paths.analysis_dir / "p9_light_walters_guidance_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    paths = default_paths(REPO_ROOT)
    summary_rows = read_csv(paths.analysis_dir / "p9_light_condition_summary.csv")
    case_rows = read_csv(paths.analysis_dir / "p9_light_case_level_results.csv")
    parse_rows = read_csv(paths.analysis_dir / "p9_light_parse_summary.csv")
    write_guidance_card_markdown(paths)

    temporal_03 = rows_by_protocol(summary_rows, "temporal_iou_0.3")
    temporal_01 = rows_by_protocol(summary_rows, "temporal_iou_0.1")
    start_10 = rows_by_protocol(summary_rows, "start_time_proximity_10ms")
    improved, degraded = best_delta_cases(case_rows, "temporal_iou_0.3")

    report_lines = [
        "# P9-light Walters-Style Generic Acoustic-Parameter Guidance Report",
        "",
        "P9-light replaces the blocked OP-specific acoustic-reference experiment with a generic acoustic-parameter checklist inspired by Walters et al. 2012. It does not use OP-specific numeric ranges or European species-specific numeric ranges.",
        "",
        "## Target Set",
        "",
        ", ".join(f"`{clip}`" for clip in TARGET_CLIPS),
        "",
        "## Parse Summary",
        "",
        "| Condition | Clips | Model calls | Parse success | Parse failure | Retries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in parse_rows:
        report_lines.append(
            f"| {condition_label(row['condition'])} | {row['clip_count']} | {row['model_calls']} | "
            f"{row['parse_success_count']} | {row['parse_failure_count']} | {row['retry_count']} |"
        )
    report_lines.extend(["", "## Temporal IoU >= 0.3", "", *metric_table(temporal_03)])
    report_lines.extend(["", "## Temporal IoU >= 0.1", "", *metric_table(temporal_01)])
    report_lines.extend(["", "## Start-Time Proximity <= 10 ms", "", *metric_table(start_10)])
    report_lines.extend(
        [
            "",
            "## Walters Guidance Case Deltas",
            "",
            "Deltas compare `agent_walters_guidance` against `agent_proposals_only` under temporal IoU >= 0.3.",
            "",
            "### Largest improvements",
            "",
            "| Clip | Baseline F1 | Walters F1 | Delta F1 | Baseline FP/FN | Walters FP/FN |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in improved:
        report_lines.append(
            f"| {row['clip_id']} | {row['base_f1']:.3f} | {row['walters_f1']:.3f} | {row['delta_f1']:.3f} | "
            f"{row['base_fp']}/{row['base_fn']} | {row['walters_fp']}/{row['walters_fn']} |"
        )
    report_lines.extend(
        [
            "",
            "### Largest degradations",
            "",
            "| Clip | Baseline F1 | Walters F1 | Delta F1 | Baseline FP/FN | Walters FP/FN |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in degraded:
        report_lines.append(
            f"| {row['clip_id']} | {row['base_f1']:.3f} | {row['walters_f1']:.3f} | {row['delta_f1']:.3f} | "
            f"{row['base_fp']}/{row['base_fn']} | {row['walters_fp']}/{row['walters_fn']} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation Prompts",
            "",
            "- Compare no reference with methodological literature and Walters-style guidance.",
            "- Check whether Walters guidance improves 10 ms onset-level detection.",
            "- Check whether gains are from fewer FP, fewer FN, or better matched-pair box quality.",
            "- Treat any OP-specific prior as still blocked unless Santiago provides a safe external source.",
        ]
    )
    (paths.analysis_dir / "p9_light_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    supervisor_lines = [
        "# P9-light Supervisor Summary",
        "",
        "The original OP-specific acoustic-reference experiment remains blocked because all locally available *Ozimops petersi* annotations overlap with the frozen 45-clip evaluation set.",
        "",
        "P9-light instead tests whether a Walters-style generic acoustic-parameter checklist helps bounding-box annotation. It uses no OP-specific numeric ranges and no European species-specific numeric ranges. The checklist simply reminds the model to inspect onset, offset, duration, low/high frequency extent, bandwidth, ridge shape and artefact rejection.",
        "",
        "See `p9_light_report.md` and `p9_light_condition_summary.csv` for the measured comparison against no-reference, methodological-literature, and BatDetect2 proposal-only conditions.",
        "",
        "If Walters-style guidance helps, the next safe step is to ask Santiago for an independent OP-specific acoustic source. If it does not help, the project should avoid adding more prompt text and focus on detector proposals, validation, or human review.",
    ]
    (paths.analysis_dir / "p9_light_supervisor_summary.md").write_text("\n".join(supervisor_lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(paths.analysis_dir / "p9_light_report.md"), "supervisor_summary": str(paths.analysis_dir / "p9_light_supervisor_summary.md")}, indent=2))


if __name__ == "__main__":
    main()
