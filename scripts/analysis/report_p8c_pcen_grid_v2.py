"""Create the P8C PCEN+grid_v2 comparison report."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "outputs/analysis_reports/p8c_pcen_grid_v2_full45"
P8C_RUN_DIR = REPO_ROOT / "outputs/agent_runs/p8c_pcen_grid_v2_qwen3_6_full45"
P8B_RUN_DIR = REPO_ROOT / "outputs/agent_runs/p8b_pcen_qwen3_6_full45"
P8B_INPUT_DIR = REPO_ROOT / "outputs/agent_inputs/p8b_pcen_full45"
P8C_INPUT_DIR = REPO_ROOT / "outputs/agent_inputs/p8c_pcen_grid_v2_full45"
GRID_ID = "p5_qwen_grid_v2_full"
P8B_ID = "p8b_pcen_full45"
P8C_ID = "p8c_pcen_grid_v2_full45"
PROTOCOL_ORDER = ("temporal_iou_0.3", "temporal_iou_0.1", "start_time_proximity_10ms")


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


def as_float(value: str | int | float) -> float:
    return float(value) if value not in {"", None} else 0.0


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["experiment_id"], row["protocol"]): row for row in rows}


def parse_summary(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    rows = read_csv(path)
    success = sum(1 for row in rows if row.get("parse_status") == "success")
    return success, len(rows) - success


def same_file_content(left: Path, right: Path) -> bool:
    if not left.is_file() or not right.is_file():
        return False
    return left.read_bytes() == right.read_bytes()


def reproducibility_counts() -> tuple[int, int, int]:
    clips = [
        path.name.split("_pcen_grid_v2.png")[0]
        for path in sorted(P8C_INPUT_DIR.glob("OP_*_pcen_grid_v2.png"))
    ]
    identical_images = sum(
        1
        for clip_id in clips
        if same_file_content(
            P8B_INPUT_DIR / f"{clip_id}_pcen_grid_v2.png",
            P8C_INPUT_DIR / f"{clip_id}_pcen_grid_v2.png",
        )
    )
    identical_raw = sum(
        1
        for clip_id in clips
        if same_file_content(
            P8B_RUN_DIR / "raw_responses" / f"{clip_id}_raw_response.txt",
            P8C_RUN_DIR / "raw_responses" / f"{clip_id}_raw_response.txt",
        )
    )
    return len(clips), identical_images, identical_raw


def metric_cells(row: dict[str, str]) -> str:
    return (
        f"{row['TP']} / {row['FP']} / {row['FN']} | "
        f"{as_float(row['precision']):.3f} | {as_float(row['recall']):.3f} | "
        f"{as_float(row['F1']):.3f} | {as_float(row['mean_time_iou']):.3f} | "
        f"{as_float(row['mean_frequency_iou']):.3f} | {as_float(row['mean_box_iou']):.3f}"
    )


def delta_row(rows_by_key: dict[tuple[str, str], dict[str, str]], protocol: str) -> dict[str, Any]:
    grid = rows_by_key[(GRID_ID, protocol)]
    p8b = rows_by_key[(P8B_ID, protocol)]
    p8c = rows_by_key[(P8C_ID, protocol)]
    return {
        "protocol": protocol,
        "grid_v2_F1": grid["F1"],
        "p8b_pcen_F1": p8b["F1"],
        "p8c_pcen_grid_v2_F1": p8c["F1"],
        "p8c_minus_grid_v2_F1": as_float(p8c["F1"]) - as_float(grid["F1"]),
        "p8c_minus_p8b_F1": as_float(p8c["F1"]) - as_float(p8b["F1"]),
        "grid_v2_mean_box_iou": grid["mean_box_iou"],
        "p8b_pcen_mean_box_iou": p8b["mean_box_iou"],
        "p8c_pcen_grid_v2_mean_box_iou": p8c["mean_box_iou"],
    }


def write_report(rows: list[dict[str, str]]) -> Path:
    rows_by_key = index_rows(rows)
    delta_rows = [delta_row(rows_by_key, protocol) for protocol in PROTOCOL_ORDER]
    write_csv(ANALYSIS_DIR / "p8c_protocol_deltas.csv", delta_rows)

    parse_success, parse_failure = parse_summary(P8C_RUN_DIR / "p8c_parse_summary.csv")
    clip_count, identical_images, identical_raw = reproducibility_counts()
    grid_03 = rows_by_key[(GRID_ID, "temporal_iou_0.3")]
    p8b_03 = rows_by_key[(P8B_ID, "temporal_iou_0.3")]
    p8c_03 = rows_by_key[(P8C_ID, "temporal_iou_0.3")]

    lines = [
        "# P8C PCEN + grid_v2 Full-45 Confirmatory Report",
        "",
        "P8C ran `qwen3.6:latest` on 45 clean PCEN spectrogram images with the existing `grid_v2` axis/grid styling. The run uses the frozen Prompt V2 JSON schema and evaluates outputs with the same P8 multi-protocol detector metrics.",
        "",
        "## Reproducibility note",
        "",
        "The existing PCEN generator records `GRID_STYLE = \"grid_v2\"`, and the prior P8B image suffix is `_pcen_grid_v2.png`. Therefore P8C should be interpreted as an explicit, separately saved confirmatory run of PCEN+grid_v2 rather than proof that an earlier PCEN-only condition lacked grid lines.",
        "",
        f"Post-run audit: {identical_images}/{clip_count} P8C input images are byte-identical to the corresponding P8B PCEN images, and {identical_raw}/{clip_count} raw model responses are byte-identical. Parsed prediction JSON files differ only where run metadata such as timestamps and paths differ.",
        "",
        "## Parse Summary",
        "",
        f"- Parse successes: {parse_success}",
        f"- Parse failures: {parse_failure}",
        f"- Run directory: `{P8C_RUN_DIR.relative_to(REPO_ROOT)}`",
        "",
        "## Main comparison under temporal IoU >= 0.3",
        "",
        "| Condition | TP / FP / FN | Precision | Recall | F1 | Mean time IoU | Mean frequency IoU | Mean box IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| frozen grid_v2 | {metric_cells(grid_03)} |",
        f"| existing P8B PCEN | {metric_cells(p8b_03)} |",
        f"| new P8C PCEN + grid_v2 | {metric_cells(p8c_03)} |",
        "",
        "## Protocol sensitivity",
        "",
        "| Protocol | grid_v2 F1 | P8B PCEN F1 | P8C PCEN+grid_v2 F1 | P8C - grid_v2 | P8C - P8B |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in delta_rows:
        lines.append(
            f"| {row['protocol']} | {as_float(row['grid_v2_F1']):.3f} | {as_float(row['p8b_pcen_F1']):.3f} | {as_float(row['p8c_pcen_grid_v2_F1']):.3f} | {row['p8c_minus_grid_v2_F1']:.3f} | {row['p8c_minus_p8b_F1']:.3f} |"
        )

    p8c_f1 = as_float(p8c_03["F1"])
    grid_f1 = as_float(grid_03["F1"])
    p8b_f1 = as_float(p8b_03["F1"])
    keep_recommendation = (
        "P8C improves over both the frozen grid_v2 baseline and existing P8B PCEN under the frozen IoU>=0.3 protocol."
        if p8c_f1 > max(grid_f1, p8b_f1)
        else "P8C does not outperform the frozen grid_v2 baseline under the frozen IoU>=0.3 protocol, so it should not replace grid_v2 as the dissertation baseline."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            keep_recommendation,
            "",
            "The most important comparison is temporal IoU >= 0.3, because it is the frozen strong-labelling protocol. The looser IoU>=0.1 and 10 ms start-time proximity rows are sensitivity checks rather than replacement headline metrics.",
            "",
            "## Key files",
            "",
            f"- Condition summary: `{(ANALYSIS_DIR / 'p8c_condition_summary.csv').relative_to(REPO_ROOT)}`",
            f"- Case-level results: `{(ANALYSIS_DIR / 'p8c_case_level_results.csv').relative_to(REPO_ROOT)}`",
            f"- Matched-pair quality: `{(ANALYSIS_DIR / 'p8c_matched_pair_box_quality.csv').relative_to(REPO_ROOT)}`",
            f"- Diagnostics: `{(P8C_RUN_DIR / 'diagnostic_figures').relative_to(REPO_ROOT)}`",
            "",
        ]
    )
    path = ANALYSIS_DIR / "p8c_pcen_grid_v2_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    rows = read_csv(ANALYSIS_DIR / "p8c_condition_summary.csv")
    report_path = write_report(rows)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
