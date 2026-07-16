"""Create P8 supervisor-facing summaries from P8A/P8B outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
P8A_DIR = REPO_ROOT / "outputs/analysis_reports/p8a_multi_protocol_detection"
P8B_RUN_DIR = REPO_ROOT / "outputs/agent_runs/p8b_pcen_qwen3_6_full45"
P8_SUMMARY_DIR = REPO_ROOT / "outputs/analysis_reports/p8_10ms_iou_sensitivity_and_pcen"
GRID_EXPERIMENT_ID = "p5_qwen_grid_v2_full"
PCEN_EXPERIMENT_ID = "p8b_pcen_full45"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str) -> float:
    return float(value) if value not in {"", None} else 0.0


def p8_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["experiment_id"] in {GRID_EXPERIMENT_ID, PCEN_EXPERIMENT_ID}]


def metric_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "experiment_id": row["experiment_id"],
        "condition": "standard_grid_v2" if row["experiment_id"] == GRID_EXPERIMENT_ID else "PCEN_grid_v2",
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
    }


def paired_protocol_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {(row["experiment_id"], row["protocol"]): row for row in rows}
    deltas = []
    for protocol in ("temporal_iou_0.1", "temporal_iou_0.3", "start_time_proximity_10ms"):
        grid = by_key[(GRID_EXPERIMENT_ID, protocol)]
        pcen = by_key[(PCEN_EXPERIMENT_ID, protocol)]
        deltas.append(
            {
                "protocol": protocol,
                "grid_F1": grid["F1"],
                "pcen_F1": pcen["F1"],
                "delta_F1": as_float(pcen["F1"]) - as_float(grid["F1"]),
                "grid_FP": grid["FP"],
                "pcen_FP": pcen["FP"],
                "delta_FP": int(float(pcen["FP"])) - int(float(grid["FP"])),
                "grid_FN": grid["FN"],
                "pcen_FN": pcen["FN"],
                "delta_FN": int(float(pcen["FN"])) - int(float(grid["FN"])),
                "grid_mean_box_iou": grid["mean_box_iou"],
                "pcen_mean_box_iou": pcen["mean_box_iou"],
                "delta_mean_box_iou": as_float(pcen["mean_box_iou"]) - as_float(grid["mean_box_iou"]),
            }
        )
    return deltas


def case_deltas(case_rows: list[dict[str, str]], protocol: str = "temporal_iou_0.3") -> list[dict[str, Any]]:
    rows = [row for row in case_rows if row["protocol"] == protocol and row["experiment_id"] in {GRID_EXPERIMENT_ID, PCEN_EXPERIMENT_ID}]
    by_clip = {(row["experiment_id"], row["clip_id"]): row for row in rows}
    clips = sorted({row["clip_id"] for row in rows})
    output = []
    for clip_id in clips:
        grid = by_clip.get((GRID_EXPERIMENT_ID, clip_id))
        pcen = by_clip.get((PCEN_EXPERIMENT_ID, clip_id))
        if not grid or not pcen:
            continue
        output.append(
            {
                "clip_id": clip_id,
                "grid_F1": grid["F1"],
                "pcen_F1": pcen["F1"],
                "delta_F1": as_float(pcen["F1"]) - as_float(grid["F1"]),
                "grid_TP": grid["TP"],
                "pcen_TP": pcen["TP"],
                "grid_FP": grid["FP"],
                "pcen_FP": pcen["FP"],
                "grid_FN": grid["FN"],
                "pcen_FN": pcen["FN"],
            }
        )
    return output


def select_diagnostic_shortlist(deltas: list[dict[str, Any]]) -> list[str]:
    sorted_deltas = sorted(deltas, key=lambda row: row["delta_F1"])
    shortlist = {"OP_045"}
    shortlist.update(row["clip_id"] for row in sorted_deltas[:3])
    shortlist.update(row["clip_id"] for row in sorted_deltas[-3:])
    unchanged = [row["clip_id"] for row in deltas if abs(row["delta_F1"]) < 1e-9]
    shortlist.update(unchanged[:3])
    return sorted(shortlist)


def write_reports(experiment_rows: list[dict[str, str]], case_rows: list[dict[str, str]]) -> None:
    selected = p8_rows(experiment_rows)
    condition_rows = [metric_row(row) for row in selected]
    write_csv(P8B_RUN_DIR / "p8b_condition_summary.csv", condition_rows)
    write_csv(P8B_RUN_DIR / "p8b_protocol_comparison.csv", paired_protocol_deltas(selected))
    write_csv(
        P8B_RUN_DIR / "p8b_case_level_results.csv",
        [row for row in case_rows if row["experiment_id"] in {GRID_EXPERIMENT_ID, PCEN_EXPERIMENT_ID}],
    )
    matched_rows = read_csv(P8A_DIR / "matched_pair_box_quality.csv")
    write_csv(
        P8B_RUN_DIR / "p8b_matched_pair_box_quality.csv",
        [row for row in matched_rows if row["experiment_id"] in {GRID_EXPERIMENT_ID, PCEN_EXPERIMENT_ID}],
    )

    deltas = paired_protocol_deltas(selected)
    case_delta_rows = case_deltas(case_rows)
    shortlist = select_diagnostic_shortlist(case_delta_rows)
    write_csv(P8_SUMMARY_DIR / "p8_main_results_table.csv", condition_rows)

    grid_03 = next(row for row in selected if row["experiment_id"] == GRID_EXPERIMENT_ID and row["protocol"] == "temporal_iou_0.3")
    pcen_03 = next(row for row in selected if row["experiment_id"] == PCEN_EXPERIMENT_ID and row["protocol"] == "temporal_iou_0.3")
    proximity_grid = next(row for row in selected if row["experiment_id"] == GRID_EXPERIMENT_ID and row["protocol"] == "start_time_proximity_10ms")
    proximity_pcen = next(row for row in selected if row["experiment_id"] == PCEN_EXPERIMENT_ID and row["protocol"] == "start_time_proximity_10ms")

    p8b_report = [
        "# P8B Full-45 PCEN Confirmatory Report",
        "",
        "P8B ran `qwen3.6:latest` on all 45 clean PCEN spectrogram inputs using the frozen Prompt V2 schema and the HPC Ollama endpoint. All 45 clips parsed successfully.",
        "",
        "## Main comparison under frozen temporal IoU >= 0.3",
        "",
        "| Condition | TP | FP | FN | Precision | Recall | F1 | Mean box IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| standard grid_v2 | {grid_03['TP']} | {grid_03['FP']} | {grid_03['FN']} | {float(grid_03['precision']):.3f} | {float(grid_03['recall']):.3f} | {float(grid_03['F1']):.3f} | {float(grid_03['mean_box_iou']):.3f} |",
        f"| PCEN grid_v2 | {pcen_03['TP']} | {pcen_03['FP']} | {pcen_03['FN']} | {float(pcen_03['precision']):.3f} | {float(pcen_03['recall']):.3f} | {float(pcen_03['F1']):.3f} | {float(pcen_03['mean_box_iou']):.3f} |",
        "",
        "## Protocol sensitivity",
        "",
        "| Protocol | grid F1 | PCEN F1 | delta F1 | delta FP | delta FN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in deltas:
        p8b_report.append(
            f"| {row['protocol']} | {float(row['grid_F1']):.3f} | {float(row['pcen_F1']):.3f} | {row['delta_F1']:.3f} | {row['delta_FP']} | {row['delta_FN']} |"
        )
    p8b_report.extend(
        [
            "",
            "## Diagnostic shortlist",
            "",
            "Recommended overlays to inspect: " + ", ".join(f"`{clip}`" for clip in shortlist) + ".",
            "",
            "## Interpretation",
            "",
            "PCEN did not improve the full-set confirmatory result under the frozen temporal IoU >= 0.3 protocol. It increased false positives, slightly increased false negatives, and reduced mean box IoU relative to the standard grid_v2 baseline. Under the 10 ms start-time proximity view, PCEN recovered more onset-near events, but this came with poorer box quality and does not overturn the frozen-protocol conclusion.",
        ]
    )
    (P8B_RUN_DIR / "p8b_full45_pcen_confirmatory_report.md").write_text("\n".join(p8b_report) + "\n", encoding="utf-8")

    supervisor = [
        "# P8 Supervisor Summary: 10 ms / IoU Sensitivity and PCEN Confirmatory Run",
        "",
        "## What changed",
        "",
        "P8A re-evaluated frozen predictions under temporal IoU >= 0.1, temporal IoU >= 0.3, and BatDetect2-style start-time proximity <= 10 ms. P8B ran a new full-45 PCEN condition with `qwen3.6:latest` on the HPC Ollama endpoint.",
        "",
        "## Main result",
        "",
        f"Under the frozen temporal IoU >= 0.3 protocol, standard grid_v2 achieved F1={float(grid_03['F1']):.3f}, while full-45 PCEN achieved F1={float(pcen_03['F1']):.3f}. PCEN therefore did not confirm as an improvement over the existing fixed-view baseline.",
        "",
        f"Under the 10 ms start-time proximity protocol, standard grid_v2 achieved F1={float(proximity_grid['F1']):.3f}, while PCEN achieved F1={float(proximity_pcen['F1']):.3f}. This indicates that PCEN can improve onset-near detection counts, but its lower mean box IoU and weaker frozen-protocol F1 mean it should be treated as a sensitivity finding rather than a replacement for grid_v2.",
        "",
        "## Practical conclusion",
        "",
        "PCEN should not replace the current grid_v2 input representation for the dissertation baseline. The more useful direction remains constrained proposal/tool use and validation rather than another global visual preprocessing condition.",
        "",
        "## Key files",
        "",
        f"- P8A metrics: `{(P8A_DIR / 'experiment_level_protocol_summary.csv').relative_to(REPO_ROOT)}`",
        f"- P8B parse summary: `{(P8B_RUN_DIR / 'p8b_parse_summary.csv').relative_to(REPO_ROOT)}`",
        f"- P8B diagnostic figures: `{(P8B_RUN_DIR / 'diagnostic_figures').relative_to(REPO_ROOT)}`",
        f"- Main table: `{(P8_SUMMARY_DIR / 'p8_main_results_table.csv').relative_to(REPO_ROOT)}`",
    ]
    P8_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (P8_SUMMARY_DIR / "p8_supervisor_summary.md").write_text("\n".join(supervisor) + "\n", encoding="utf-8")


def main() -> None:
    experiment_rows = read_csv(P8A_DIR / "experiment_level_protocol_summary.csv")
    case_rows = read_csv(P8A_DIR / "case_level_protocol_results.csv")
    write_reports(experiment_rows, case_rows)
    print(f"Wrote P8 summary outputs to {P8_SUMMARY_DIR}")
    print(f"Wrote P8B report to {P8B_RUN_DIR / 'p8b_full45_pcen_confirmatory_report.md'}")


if __name__ == "__main__":
    main()
