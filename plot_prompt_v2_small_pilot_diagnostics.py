"""Create Prompt V2 pilot diagnostic overlays and a failure-analysis report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from main import make_spectrogram, to_decibels


DEFAULT_PRED_DIR = Path("outputs/agent_runs/prompt_v2_small_pilot")
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_EVALUATION_DIR = DEFAULT_PRED_DIR / "evaluation"
DEFAULT_FIGURE_DIR = DEFAULT_EVALUATION_DIR / "diagnostic_figures"
DEFAULT_CLIP_IDS = ["OP_001", "OP_010", "OP_045", "OP_003", "OP_004", "OP_016"]
DEFAULT_MIN_DB = -130.0
DEFAULT_MAX_FREQ_HZ = 120_000.0

CLIP_ROLES = {
    "OP_001": "canonical multi-event",
    "OP_010": "dense multi-event / separation",
    "OP_045": "simple clean / partial-final-clip",
    "OP_003": "right-truncated boundary",
    "OP_004": "left-truncated boundary",
    "OP_016": "dense boundary-stress",
}


def diagnostic_output_path(output_dir: str | Path, clip_id: str) -> Path:
    """Return the stable output path for one diagnostic figure."""
    return Path(output_dir) / f"{clip_id}_diagnostic_overlay.png"


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Load one evaluation CSV, including a header-only empty report."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_evaluation_csvs(evaluation_dir: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load all status and metric CSV files needed by the diagnostics."""
    evaluation_dir = Path(evaluation_dir)
    return {
        "per_clip": load_csv_rows(evaluation_dir / "per_clip_metrics.csv"),
        "matched": load_csv_rows(evaluation_dir / "matched_events.csv"),
        "unmatched": load_csv_rows(evaluation_dir / "unmatched_predictions.csv"),
        "missed": load_csv_rows(evaluation_dir / "missed_ground_truth_events.csv"),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def resolve_prediction_path(pred_dir: Path, clip_id: str) -> Path:
    """Accept existing plural files and merged tiled singular files."""
    candidates = [
        pred_dir / f"{clip_id}_predictions.json",
        pred_dir / f"{clip_id}_prediction.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Prediction JSON not found for {clip_id}: "
        + ", ".join(str(path) for path in candidates)
    )


def read_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def spectrogram_image(spec: np.ndarray, min_db: float) -> np.ndarray:
    spec_db = to_decibels(spec)
    spec_db = np.clip(spec_db, min_db, 0)
    return (spec_db - min_db) / -min_db


def rows_for_clip(rows: list[dict[str, str]], clip_id: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("clip_id") == clip_id]


def draw_box(
    ax,
    *,
    start: float,
    end: float,
    low_hz: float,
    high_hz: float,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    """Draw one time-frequency box and a compact ID label."""
    width = end - start
    height_khz = (high_hz - low_hz) / 1000
    if width <= 0 or height_khz <= 0:
        return
    low_khz = low_hz / 1000
    ax.add_patch(
        Rectangle(
            (start, low_khz),
            width,
            height_khz,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    )
    ax.text(
        start,
        low_khz + height_khz,
        label,
        color=color,
        fontsize=6.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
    )


def plot_diagnostic_clip(
    *,
    clip_id: str,
    pred_dir: Path,
    eval_dir: Path,
    evaluation_rows: dict[str, list[dict[str, str]]],
    output_dir: Path,
    min_db: float = DEFAULT_MIN_DB,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> Path:
    """Plot GT and predictions with their evaluation status."""
    audio_path = eval_dir / "audio" / f"{clip_id}.wav"
    gt_payload = load_json(
        eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"
    )
    prediction_payload = load_json(resolve_prediction_path(pred_dir, clip_id))
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    clip_metrics_rows = rows_for_clip(evaluation_rows["per_clip"], clip_id)
    if len(clip_metrics_rows) != 1:
        raise ValueError(f"Expected one per-clip metric row for {clip_id}")
    metrics = clip_metrics_rows[0]
    matched_rows = rows_for_clip(evaluation_rows["matched"], clip_id)
    unmatched_rows = rows_for_clip(evaluation_rows["unmatched"], clip_id)
    missed_rows = rows_for_clip(evaluation_rows["missed"], clip_id)
    matched_prediction_ids = {row["prediction_id"] for row in matched_rows}
    matched_gt_ids = {row["ground_truth_event_id"] for row in matched_rows}
    unmatched_prediction_ids = {row["prediction_id"] for row in unmatched_rows}
    missed_gt_ids = {row["ground_truth_event_id"] for row in missed_rows}

    audio, sample_rate = read_mono_audio(audio_path)
    spec, stft = make_spectrogram(audio, sample_rate)
    image = spectrogram_image(spec, min_db)
    extent = list(stft.extent(len(audio)))
    extent[2] /= 1000
    extent[3] /= 1000

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )
    duration = len(audio) / sample_rate
    ax.set_xlim(0, duration)
    ax.set_ylim(0, min(max_freq_hz, sample_rate / 2) / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(
        f"{clip_id} diagnostic | TP={metrics['tp']} FP={metrics['fp']} "
        f"FN={metrics['fn']}"
    )
    ax.grid(color="white", linewidth=0.3, alpha=0.2)

    for index, event in enumerate(gt_payload.get("events", []), start=1):
        event_id = str(event["event_id"])
        missed = event_id in missed_gt_ids
        color = "red" if missed else "lime"
        linestyle = "--" if missed else "-"
        status = "miss" if missed else "match"
        draw_box(
            ax,
            start=float(event["start_time"]),
            end=float(event["end_time"]),
            low_hz=float(event["low_frequency"]),
            high_hz=float(event["high_frequency"]),
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=f"G{index}:{status}",
        )
        if event_id not in matched_gt_ids and event_id not in missed_gt_ids:
            raise ValueError(f"GT event {event_id} has no evaluation status")

    for index, event in enumerate(prediction_payload.get("events", []), start=1):
        event_id = str(event["event_id"])
        unmatched = event_id in unmatched_prediction_ids
        color = "orange" if unmatched else "cyan"
        linestyle = ":" if unmatched else "-"
        status = "FP" if unmatched else "match"
        draw_box(
            ax,
            start=float(event["start_time_seconds"]),
            end=float(event["end_time_seconds"]),
            low_hz=float(event["low_frequency_hz"]),
            high_hz=float(event["high_frequency_hz"]),
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=f"P{index}:{status}",
        )
        if event_id not in matched_prediction_ids and event_id not in unmatched_prediction_ids:
            raise ValueError(f"Prediction {event_id} has no evaluation status")

    ax.legend(
        handles=[
            Line2D([0], [0], color="lime", linewidth=2, label="Matched GT"),
            Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Missed GT"),
            Line2D([0], [0], color="cyan", linewidth=2, label="Matched prediction"),
            Line2D(
                [0],
                [0],
                color="orange",
                linestyle=":",
                linewidth=2,
                label="Unmatched prediction",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = diagnostic_output_path(output_dir, clip_id)
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def category_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for category in row.get("failure_categories", "").split(";"):
            if category:
                counts[category] += 1
    return counts


def fmt(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def write_failure_analysis(
    *,
    output_path: Path,
    aggregate: dict[str, Any],
    evaluation_rows: dict[str, list[dict[str, str]]],
    figure_dir: Path,
    run_name: str = "prompt_v2_small_pilot",
) -> Path:
    """Write a report grounded in the saved evaluation outputs."""
    per_clip = evaluation_rows["per_clip"]
    missed_counts = category_counts(evaluation_rows["missed"])
    unmatched_counts = category_counts(evaluation_rows["unmatched"])
    matched_counts = category_counts(evaluation_rows["matched"])

    metrics_by_clip = {row["clip_id"]: row for row in per_clip}

    lines = [
        f"# Prompt V2 Diagnostic Failure Analysis: {run_name}",
        "",
        "## Aggregate Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Pilot clips | {aggregate['clip_count']} |",
        f"| Ground-truth events | {aggregate['total_ground_truth_events']} |",
        f"| Predictions | {aggregate['total_predictions']} |",
        f"| TP / FP / FN | {aggregate['total_tp']} / {aggregate['total_fp']} / {aggregate['total_fn']} |",
        f"| Precision | {fmt(aggregate['precision'])} |",
        f"| Recall | {fmt(aggregate['recall'])} |",
        f"| F1 | {fmt(aggregate['f1'])} |",
        f"| Mean temporal IoU | {fmt(aggregate['mean_time_iou'])} |",
        f"| Mean frequency IoU | {fmt(aggregate['mean_frequency_iou'])} |",
        f"| Mean box IoU | {fmt(aggregate['mean_box_iou'])} |",
        f"| Matched pairs with box IoU >= 0.3 | {aggregate['strict_box_iou_0_3_count']} |",
        f"| Matched pairs with box IoU >= 0.5 | {aggregate['strict_box_iou_0_5_count']} |",
        "",
        f"This run produced {aggregate['total_predictions']} predictions for "
        f"{aggregate['total_ground_truth_events']} GT events, with "
        f"{aggregate['total_tp']} temporal matches under the frozen evaluation "
        "protocol. Use the overlays to separate timing misses from frequency-box "
        "localisation failures.",
        "",
        "## Per-Clip Summary",
        "",
        "| clip_id | role | GT | pred | TP | FP | FN | F1 | mean time IoU | mean freq IoU | mean box IoU | figure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for clip_id in DEFAULT_CLIP_IDS:
        if clip_id not in metrics_by_clip:
            continue
        row = metrics_by_clip[clip_id]
        figure_path = diagnostic_output_path(figure_dir, clip_id).as_posix()
        lines.append(
            f"| `{clip_id}` | {CLIP_ROLES[clip_id]} | {row['num_ground_truth_events']} "
            f"| {row['num_predictions']} | {row['tp']} | {row['fp']} | {row['fn']} "
            f"| {fmt(row['f1'])} | {fmt(row['mean_time_iou'])} "
            f"| {fmt(row['mean_frequency_iou'])} | {fmt(row['mean_box_iou'])} "
            f"| `{figure_path}` |"
        )

    lines.extend(["", "## Representative Clip Interpretation", ""])
    for clip_id in DEFAULT_CLIP_IDS:
        if clip_id not in metrics_by_clip:
            continue
        row = metrics_by_clip[clip_id]
        lines.extend(
            [
                f"### {clip_id}: {CLIP_ROLES[clip_id]}",
                "",
                (
                    f"GT={row['num_ground_truth_events']}, "
                    f"pred={row['num_predictions']}, TP={row['tp']}, "
                    f"FP={row['fp']}, FN={row['fn']}, F1={fmt(row['f1'])}, "
                    f"mean temporal IoU={fmt(row['mean_time_iou'])}, "
                    f"mean frequency IoU={fmt(row['mean_frequency_iou'])}, "
                    f"mean box IoU={fmt(row['mean_box_iou'])}."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Main Observed Failure Modes",
            "",
            f"- **`missed_call` ({missed_counts['missed_call']}):** Recall was low "
            "outside OP_045, especially for dense and boundary clips.",
            f"- **`false_positive` ({unmatched_counts['false_positive']}):** Many "
            "predictions were shifted away from the short GT calls and therefore "
            "failed the temporal IoU threshold.",
            f"- **`poor_frequency_localisation` ({matched_counts['poor_frequency_localisation']} explicit suggestion):** "
            "Even temporally matched pairs often used the wrong frequency band. "
            "Aggregate mean frequency IoU was only "
            f"{fmt(aggregate['mean_frequency_iou'])}.",
            f"- **`boundary_truncation_error` ({missed_counts['boundary_truncation_error']}):** "
            "All truncated GT events in the pilot were missed.",
            "- **Possible `merged_calls`:** No merge is confirmed automatically by "
            "the one-to-one CSV output. Broad or regularly spaced predictions should "
            "be checked visually for boxes that span more than one visible call.",
            f"- **Possible over-wide/under-wide boxes:** "
            f"`over_wide_frequency_box` was suggested {matched_counts['over_wide_frequency_box']} times and "
            f"`under_wide_frequency_box` {matched_counts['under_wide_frequency_box']} time(s). "
            "These are heuristic suggestions and require visual confirmation.",
            "",
            "## Implications for Prompt V3 or Workflow Changes",
            "",
            "1. Add a stronger left-to-right scan instruction tied to visible pulse "
            "centres rather than regular time spacing.",
            "2. Emphasise that the main harmonic in this set is generally around "
            "the visible 30-40 kHz call band; the model repeatedly placed boxes too "
            "low in frequency.",
            "3. Add explicit boundary checks: inspect the first and last 50 ms before "
            "finishing the event list.",
            "4. Consider a two-pass workflow: first identify temporal call centres, "
            "then refine tight frequency bounds for each candidate.",
            "5. Use OP_045 as a positive demonstration and OP_003/OP_004/OP_016 as "
            "counterexamples for boundary and dense-call handling.",
            "6. Do not proceed to the full 45-clip run until a revised prompt or "
            "workflow improves the representative boundary and dense clips.",
            "",
            "## Diagnostic Legend",
            "",
            "- Lime solid: matched GT.",
            "- Red dashed: missed GT.",
            "- Cyan solid: matched prediction.",
            "- Orange dotted: unmatched prediction.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def resolve_all_clip_ids(evaluation_rows: dict[str, list[dict[str, str]]]) -> list[str]:
    """Return every evaluated clip id in stable order from per-clip metrics."""
    clip_ids = [
        row["clip_id"]
        for row in evaluation_rows["per_clip"]
        if row.get("clip_id")
    ]
    if not clip_ids:
        raise ValueError("No clip_id values found in per_clip_metrics.csv")
    return sorted(dict.fromkeys(clip_ids))


def resolve_eval_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the evaluation output directory from either CLI spelling."""
    return args.eval_output_dir or args.evaluation_dir or DEFAULT_EVALUATION_DIR


def resolve_output_dir(args: argparse.Namespace, eval_output_dir: Path) -> Path:
    """Default diagnostic figures to the selected evaluation output directory."""
    return args.output_dir or eval_output_dir / "diagnostic_figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Prompt V2 small-pilot diagnostics and write failure analysis."
    )
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        help="Backward-compatible name for --eval-output-dir.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        help="Directory containing aggregate_summary.json and evaluation CSV files.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--run-name",
        default="prompt_v2_small_pilot",
        help="Label used in the generated failure-analysis report.",
    )
    parser.add_argument(
        "--clip-list",
        default=",".join(DEFAULT_CLIP_IDS),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every clip listed in per_clip_metrics.csv.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Only write diagnostic overlay images; do not update failure_analysis.md.",
    )
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ_HZ)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_output_dir = resolve_eval_output_dir(args)
    output_dir = resolve_output_dir(args, eval_output_dir)
    evaluation_rows = load_evaluation_csvs(eval_output_dir)
    clip_ids = (
        resolve_all_clip_ids(evaluation_rows)
        if args.all
        else parse_clip_ids(args.clip_list)
    )
    aggregate = load_json(eval_output_dir / "aggregate_summary.json")

    saved_paths = [
        plot_diagnostic_clip(
            clip_id=clip_id,
            pred_dir=args.pred_dir,
            eval_dir=args.eval_dir,
            evaluation_rows=evaluation_rows,
            output_dir=output_dir,
            min_db=args.min_db,
            max_freq_hz=args.max_freq,
        )
        for clip_id in clip_ids
    ]
    report_path = None
    if not args.skip_report:
        report_path = write_failure_analysis(
            output_path=eval_output_dir / "failure_analysis.md",
            aggregate=aggregate,
            evaluation_rows=evaluation_rows,
            figure_dir=output_dir,
            run_name=args.run_name,
        )

    print(f"Saved {len(saved_paths)} diagnostic figure(s):")
    for path in saved_paths:
        print(path)
    if report_path is not None:
        print(f"Saved failure analysis to {report_path}")


if __name__ == "__main__":
    main()
