"""Evaluate existing Prompt V2 pilot predictions against clip-level ground truth."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PRED_DIR = Path("outputs/agent_runs/prompt_v2_small_pilot")
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_OUTPUT_DIR = DEFAULT_PRED_DIR / "evaluation"
DEFAULT_CLIP_IDS = ["OP_001", "OP_010", "OP_045", "OP_003", "OP_004", "OP_016"]
DEFAULT_TIME_IOU_THRESHOLD = 0.3


@dataclass(frozen=True)
class EventBox:
    event_id: str
    start_time: float
    end_time: float
    low_frequency: float
    high_frequency: float
    label: str
    confidence: float | None = None
    truncation_side: str = "none"


@dataclass(frozen=True)
class Match:
    prediction_index: int
    ground_truth_index: int
    time_iou: float


def interval_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return intersection-over-union for one-dimensional intervals."""
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(0.0, end_a - start_a) + max(0.0, end_b - start_b) - intersection
    return intersection / union if union > 0 else 0.0


def temporal_iou(prediction: EventBox, ground_truth: EventBox) -> float:
    return interval_iou(
        prediction.start_time,
        prediction.end_time,
        ground_truth.start_time,
        ground_truth.end_time,
    )


def frequency_iou(prediction: EventBox, ground_truth: EventBox) -> float:
    return interval_iou(
        prediction.low_frequency,
        prediction.high_frequency,
        ground_truth.low_frequency,
        ground_truth.high_frequency,
    )


def box_iou(prediction: EventBox, ground_truth: EventBox) -> float:
    """Return IoU over the two-dimensional time-frequency rectangle."""
    time_overlap = max(
        0.0,
        min(prediction.end_time, ground_truth.end_time)
        - max(prediction.start_time, ground_truth.start_time),
    )
    frequency_overlap = max(
        0.0,
        min(prediction.high_frequency, ground_truth.high_frequency)
        - max(prediction.low_frequency, ground_truth.low_frequency),
    )
    intersection = time_overlap * frequency_overlap
    prediction_area = max(0.0, prediction.end_time - prediction.start_time) * max(
        0.0, prediction.high_frequency - prediction.low_frequency
    )
    ground_truth_area = max(
        0.0, ground_truth.end_time - ground_truth.start_time
    ) * max(0.0, ground_truth.high_frequency - ground_truth.low_frequency)
    union = prediction_area + ground_truth_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_temporal_matches(
    predictions: list[EventBox],
    ground_truth: list[EventBox],
    time_iou_threshold: float = DEFAULT_TIME_IOU_THRESHOLD,
) -> list[Match]:
    """Match predictions by confidence, choosing the best unmatched temporal IoU."""
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: (
            predictions[index].confidence is not None,
            predictions[index].confidence
            if predictions[index].confidence is not None
            else 0.0,
            -index,
        ),
        reverse=True,
    )
    unmatched_ground_truth = set(range(len(ground_truth)))
    matches: list[Match] = []

    for prediction_index in prediction_order:
        candidates = [
            (
                temporal_iou(
                    predictions[prediction_index],
                    ground_truth[ground_truth_index],
                ),
                ground_truth_index,
            )
            for ground_truth_index in unmatched_ground_truth
        ]
        if not candidates:
            continue
        best_iou, best_ground_truth_index = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        if best_iou < time_iou_threshold:
            continue
        matches.append(
            Match(
                prediction_index=prediction_index,
                ground_truth_index=best_ground_truth_index,
                time_iou=best_iou,
            )
        )
        unmatched_ground_truth.remove(best_ground_truth_index)

    return matches


def prediction_event(row: dict[str, Any], index: int) -> EventBox:
    return EventBox(
        event_id=str(row.get("event_id") or f"prediction_{index + 1}"),
        start_time=float(row["start_time_seconds"]),
        end_time=float(row["end_time_seconds"]),
        low_frequency=float(row["low_frequency_hz"]),
        high_frequency=float(row["high_frequency_hz"]),
        label=str(row.get("label") or ""),
        confidence=(
            float(row["confidence"])
            if row.get("confidence") is not None
            else None
        ),
    )


def ground_truth_event(row: dict[str, Any], index: int) -> EventBox:
    return EventBox(
        event_id=str(row.get("event_id") or f"ground_truth_{index + 1}"),
        start_time=float(row["start_time"]),
        end_time=float(row["end_time"]),
        low_frequency=float(row["low_frequency"]),
        high_frequency=float(row["high_frequency"]),
        label=str(row.get("label") or ""),
        truncation_side=str(row.get("truncation_side") or "none"),
    )


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


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def matched_failure_categories(
    prediction: EventBox,
    ground_truth: EventBox,
    *,
    time_score: float,
    frequency_score: float,
    box_score: float,
) -> list[str]:
    """Suggest obvious localisation failures without replacing manual review."""
    categories: list[str] = []
    if frequency_score < 0.5:
        prediction_bandwidth = prediction.high_frequency - prediction.low_frequency
        ground_truth_bandwidth = (
            ground_truth.high_frequency - ground_truth.low_frequency
        )
        if prediction_bandwidth > ground_truth_bandwidth:
            categories.append("over_wide_frequency_box")
        else:
            categories.append("under_wide_frequency_box")
    if box_score < 0.3 and time_score >= 0.5:
        categories.append("poor_frequency_localisation")
    return categories


def evaluate_clip(
    *,
    clip_id: str,
    prediction_payload: dict[str, Any],
    ground_truth_payload: dict[str, Any],
    time_iou_threshold: float = DEFAULT_TIME_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate one clip and return metrics plus detailed event rows."""
    predictions = [
        prediction_event(row, index)
        for index, row in enumerate(prediction_payload.get("events", []))
    ]
    ground_truth = [
        ground_truth_event(row, index)
        for index, row in enumerate(ground_truth_payload.get("events", []))
    ]
    matches = greedy_temporal_matches(
        predictions,
        ground_truth,
        time_iou_threshold=time_iou_threshold,
    )
    matched_prediction_indices = {match.prediction_index for match in matches}
    matched_ground_truth_indices = {match.ground_truth_index for match in matches}

    matched_rows: list[dict[str, Any]] = []
    for match in matches:
        prediction = predictions[match.prediction_index]
        truth = ground_truth[match.ground_truth_index]
        frequency_score = frequency_iou(prediction, truth)
        box_score = box_iou(prediction, truth)
        categories = matched_failure_categories(
            prediction,
            truth,
            time_score=match.time_iou,
            frequency_score=frequency_score,
            box_score=box_score,
        )
        matched_rows.append(
            {
                "clip_id": clip_id,
                "prediction_id": prediction.event_id,
                "ground_truth_event_id": truth.event_id,
                "time_iou": match.time_iou,
                "frequency_iou": frequency_score,
                "box_iou": box_score,
                "start_time_error": prediction.start_time - truth.start_time,
                "end_time_error": prediction.end_time - truth.end_time,
                "low_frequency_error": (
                    prediction.low_frequency - truth.low_frequency
                ),
                "high_frequency_error": (
                    prediction.high_frequency - truth.high_frequency
                ),
                "predicted_label": prediction.label,
                "ground_truth_label": truth.label,
                "confidence": prediction.confidence,
                "truncation_side": truth.truncation_side,
                "failure_categories": ";".join(categories),
            }
        )

    unmatched_prediction_rows = [
        {
            "clip_id": clip_id,
            "prediction_id": prediction.event_id,
            "start_time_seconds": prediction.start_time,
            "end_time_seconds": prediction.end_time,
            "low_frequency_hz": prediction.low_frequency,
            "high_frequency_hz": prediction.high_frequency,
            "predicted_label": prediction.label,
            "confidence": prediction.confidence,
            "failure_categories": "false_positive",
        }
        for index, prediction in enumerate(predictions)
        if index not in matched_prediction_indices
    ]
    missed_ground_truth_rows = [
        {
            "clip_id": clip_id,
            "ground_truth_event_id": truth.event_id,
            "start_time": truth.start_time,
            "end_time": truth.end_time,
            "low_frequency": truth.low_frequency,
            "high_frequency": truth.high_frequency,
            "ground_truth_label": truth.label,
            "truncation_side": truth.truncation_side,
            "failure_categories": (
                "missed_call;boundary_truncation_error"
                if truth.truncation_side != "none"
                else "missed_call"
            ),
        }
        for index, truth in enumerate(ground_truth)
        if index not in matched_ground_truth_indices
    ]

    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    metrics = {
        "clip_id": clip_id,
        "num_ground_truth_events": len(ground_truth),
        "num_predictions": len(predictions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mean_time_iou": mean([row["time_iou"] for row in matched_rows]),
        "mean_frequency_iou": mean(
            [row["frequency_iou"] for row in matched_rows]
        ),
        "mean_box_iou": mean([row["box_iou"] for row in matched_rows]),
        "num_truncated_events": sum(
            truth.truncation_side != "none" for truth in ground_truth
        ),
    }
    return {
        "metrics": metrics,
        "matched_events": matched_rows,
        "unmatched_predictions": unmatched_prediction_rows,
        "missed_ground_truth_events": missed_ground_truth_rows,
    }


def aggregate_results(clip_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts across clips and means across all matched pairs."""
    metrics = [result["metrics"] for result in clip_results]
    matched_rows = [
        row for result in clip_results for row in result["matched_events"]
    ]
    total_tp = sum(row["tp"] for row in metrics)
    total_fp = sum(row["fp"] for row in metrics)
    total_fn = sum(row["fn"] for row in metrics)
    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)
    return {
        "clip_count": len(clip_results),
        "time_iou_threshold": DEFAULT_TIME_IOU_THRESHOLD,
        "total_ground_truth_events": sum(
            row["num_ground_truth_events"] for row in metrics
        ),
        "total_predictions": sum(row["num_predictions"] for row in metrics),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mean_time_iou": mean([row["time_iou"] for row in matched_rows]),
        "mean_frequency_iou": mean(
            [row["frequency_iou"] for row in matched_rows]
        ),
        "mean_box_iou": mean([row["box_iou"] for row in matched_rows]),
        "strict_box_iou_0_3_count": sum(
            row["box_iou"] >= 0.3 for row in matched_rows
        ),
        "strict_box_iou_0_5_count": sum(
            row["box_iou"] >= 0.5 for row in matched_rows
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_notes(
    path: Path,
    *,
    aggregate: dict[str, Any],
    missed_rows: list[dict[str, Any]],
    unmatched_rows: list[dict[str, Any]],
) -> None:
    """Write a concise template for later manual failure review."""
    path.write_text(
        "\n".join(
            [
                "# Prompt V2 Small Pilot Failure Notes",
                "",
                "## Run Summary",
                "",
                f"- Matched events: {aggregate['total_tp']}",
                f"- Missed ground-truth events: {len(missed_rows)}",
                f"- Unmatched predictions: {len(unmatched_rows)}",
                "",
                "## Suggested Categories",
                "",
                "- Unmatched ground truth: `missed_call`.",
                "- Unmatched prediction: `false_positive`.",
                "- Missed truncated ground truth: `boundary_truncation_error`.",
                "- Low frequency IoU: inspect `over_wide_frequency_box` or "
                "`under_wide_frequency_box`.",
                "- Low box IoU with adequate time IoU: "
                "`poor_frequency_localisation`.",
                "",
                "## Manual Review Notes",
                "",
                "| clip_id | event_id | observation | confirmed_categories | notes |",
                "| --- | --- | --- | --- | --- |",
                "|  |  |  |  |  |",
                "",
            ]
        ),
        encoding="utf-8",
    )


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


def run_evaluation(
    *,
    pred_dir: Path,
    eval_dir: Path,
    clip_ids: list[str],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run and persist the deterministic small-pilot evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_results = []
    for clip_id in clip_ids:
        prediction_payload = load_json(resolve_prediction_path(pred_dir, clip_id))
        ground_truth_payload = load_json(
            eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"
        )
        clip_results.append(
            evaluate_clip(
                clip_id=clip_id,
                prediction_payload=prediction_payload,
                ground_truth_payload=ground_truth_payload,
            )
        )

    aggregate = aggregate_results(clip_results)
    per_clip_rows = [result["metrics"] for result in clip_results]
    matched_rows = [
        row for result in clip_results for row in result["matched_events"]
    ]
    unmatched_rows = [
        row for result in clip_results for row in result["unmatched_predictions"]
    ]
    missed_rows = [
        row
        for result in clip_results
        for row in result["missed_ground_truth_events"]
    ]

    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "per_clip_metrics.csv",
        per_clip_rows,
        list(per_clip_rows[0]),
    )
    write_csv(
        output_dir / "matched_events.csv",
        matched_rows,
        [
            "clip_id",
            "prediction_id",
            "ground_truth_event_id",
            "time_iou",
            "frequency_iou",
            "box_iou",
            "start_time_error",
            "end_time_error",
            "low_frequency_error",
            "high_frequency_error",
            "predicted_label",
            "ground_truth_label",
            "confidence",
            "truncation_side",
            "failure_categories",
        ],
    )
    write_csv(
        output_dir / "unmatched_predictions.csv",
        unmatched_rows,
        [
            "clip_id",
            "prediction_id",
            "start_time_seconds",
            "end_time_seconds",
            "low_frequency_hz",
            "high_frequency_hz",
            "predicted_label",
            "confidence",
            "failure_categories",
        ],
    )
    write_csv(
        output_dir / "missed_ground_truth_events.csv",
        missed_rows,
        [
            "clip_id",
            "ground_truth_event_id",
            "start_time",
            "end_time",
            "low_frequency",
            "high_frequency",
            "ground_truth_label",
            "truncation_side",
            "failure_categories",
        ],
    )
    write_failure_notes(
        output_dir / "failure_notes_template.md",
        aggregate=aggregate,
        missed_rows=missed_rows,
        unmatched_rows=unmatched_rows,
    )
    return aggregate, clip_results


def print_table(rows: list[dict[str, Any]], fields: list[str]) -> None:
    display_rows = [[str(row.get(field, "")) for field in fields] for row in rows]
    widths = [
        max(len(field), *(len(row[index]) for row in display_rows))
        for index, field in enumerate(fields)
    ]
    print(" | ".join(field.ljust(widths[index]) for index, field in enumerate(fields)))
    print("-+-".join("-" * width for width in widths))
    for row in display_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing Prompt V2 small-pilot predictions."
    )
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--clip-list",
        default=",".join(DEFAULT_CLIP_IDS),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every WAV clip in <eval-dir>/audio.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate, clip_results = run_evaluation(
        pred_dir=args.pred_dir,
        eval_dir=args.eval_dir,
        clip_ids=(
            resolve_all_clip_ids(args.eval_dir)
            if args.all
            else parse_clip_ids(args.clip_list)
        ),
        output_dir=args.output_dir,
    )
    per_clip_rows = [result["metrics"] for result in clip_results]
    missed_rows = [
        row
        for result in clip_results
        for row in result["missed_ground_truth_events"]
    ]
    unmatched_rows = [
        row for result in clip_results for row in result["unmatched_predictions"]
    ]

    print("Aggregate summary:")
    print(json.dumps(aggregate, indent=2))
    print("\nPer-clip metrics:")
    print_table(
        per_clip_rows,
        ["clip_id", "num_ground_truth_events", "num_predictions", "tp", "fp", "fn",
         "precision", "recall", "f1", "mean_time_iou", "mean_frequency_iou",
         "mean_box_iou", "num_truncated_events"],
    )
    print("\nMissed ground-truth events:")
    if missed_rows:
        print_table(
            missed_rows,
            ["clip_id", "ground_truth_event_id", "start_time", "end_time",
             "truncation_side", "failure_categories"],
        )
    else:
        print("None")
    print("\nUnmatched predictions:")
    if unmatched_rows:
        print_table(
            unmatched_rows,
            ["clip_id", "prediction_id", "start_time_seconds", "end_time_seconds",
             "confidence", "failure_categories"],
        )
    else:
        print("None")


if __name__ == "__main__":
    main()
