"""Simple box-level evaluation for bioacoustic event annotations.

This script compares predicted time-frequency boxes against ground-truth boxes.
It intentionally stays local and lightweight: no external classifiers, no model
calls, and no interactive UI.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


OUTPUT_DIR = Path("outputs")
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.csv"


@dataclass
class Box:
    """A time-frequency event box."""

    event_id: str
    start_time_seconds: float
    end_time_seconds: float
    low_frequency_hz: float
    high_frequency_hz: float
    label: str = ""
    confidence: float | None = None


@dataclass
class TemporalInterval:
    """A temporal-only event interval."""

    event_id: str
    start_time_seconds: float
    end_time_seconds: float
    label: str = ""
    confidence: float | None = None


def _float_value(row: dict, key: str) -> float:
    """Read a required numeric field from a JSON/CSV row."""
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required field: {key}")
    return float(value)


def box_from_row(row: dict, fallback_event_id: str) -> Box:
    """Convert a JSON object or CSV row into a Box."""
    confidence = row.get("confidence")
    return Box(
        event_id=str(row.get("event_id") or fallback_event_id),
        start_time_seconds=_float_value(row, "start_time_seconds"),
        end_time_seconds=_float_value(row, "end_time_seconds"),
        low_frequency_hz=_float_value(row, "low_frequency_hz"),
        high_frequency_hz=_float_value(row, "high_frequency_hz"),
        label=str(row.get("label") or ""),
        confidence=float(confidence) if confidence not in (None, "") else None,
    )


def temporal_interval_from_row(row: dict, fallback_event_id: str) -> TemporalInterval:
    """Convert a JSON object or CSV row into a temporal-only interval."""
    confidence = row.get("confidence")
    return TemporalInterval(
        event_id=str(row.get("event_id") or fallback_event_id),
        start_time_seconds=_float_value(row, "start_time_seconds"),
        end_time_seconds=_float_value(row, "end_time_seconds"),
        label=str(row.get("label") or ""),
        confidence=float(confidence) if confidence not in (None, "") else None,
    )


def load_json_boxes(path: Path) -> list[Box]:
    """Load boxes from an EventResult-style JSON file.

    Expected shape:
        {"audio_path": "...", "events": [{...}, ...]}

    For convenience, this also accepts a top-level list of event dictionaries.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("events", [])

    return [box_from_row(row, f"event_{index + 1}") for index, row in enumerate(rows)]


def load_json_temporal_intervals(path: Path) -> list[TemporalInterval]:
    """Load temporal intervals from EventResult or TemporalEventResult JSON.

    Expected shape:
        {"audio_path": "...", "events": [{...}, ...]}

    For convenience, this also accepts a top-level list of event dictionaries.
    Frequency fields are ignored even when present.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("events", [])

    return [
        temporal_interval_from_row(row, f"event_{index + 1}")
        for index, row in enumerate(rows)
    ]


def load_csv_boxes(path: Path) -> list[Box]:
    """Load boxes from a CSV file with the standard event box columns."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return [box_from_row(row, f"event_{index + 1}") for index, row in enumerate(rows)]


def load_csv_temporal_intervals(path: Path) -> list[TemporalInterval]:
    """Load temporal-only intervals from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return [
        temporal_interval_from_row(row, f"event_{index + 1}")
        for index, row in enumerate(rows)
    ]


def load_boxes(path: str | Path) -> list[Box]:
    """Load boxes from JSON or CSV."""
    box_path = Path(path)
    suffix = box_path.suffix.lower()

    if suffix == ".json":
        return load_json_boxes(box_path)
    if suffix == ".csv":
        return load_csv_boxes(box_path)

    raise ValueError(f"Unsupported annotation format: {box_path.suffix}")


def load_temporal_intervals(path: str | Path) -> list[TemporalInterval]:
    """Load temporal intervals from JSON or CSV."""
    interval_path = Path(path)
    suffix = interval_path.suffix.lower()

    if suffix == ".json":
        return load_json_temporal_intervals(interval_path)
    if suffix == ".csv":
        return load_csv_temporal_intervals(interval_path)

    raise ValueError(f"Unsupported annotation format: {interval_path.suffix}")


def time_overlap(predicted: Box, ground_truth: Box) -> float:
    """Return overlap duration in seconds."""
    start = max(predicted.start_time_seconds, ground_truth.start_time_seconds)
    end = min(predicted.end_time_seconds, ground_truth.end_time_seconds)
    return max(0.0, end - start)


def temporal_overlap(predicted: TemporalInterval, ground_truth: TemporalInterval) -> float:
    """Return temporal overlap duration in seconds."""
    start = max(predicted.start_time_seconds, ground_truth.start_time_seconds)
    end = min(predicted.end_time_seconds, ground_truth.end_time_seconds)
    return max(0.0, end - start)


def temporal_duration(interval: TemporalInterval) -> float:
    """Return temporal interval duration in seconds."""
    return max(0.0, interval.end_time_seconds - interval.start_time_seconds)


def temporal_iou(predicted: TemporalInterval, ground_truth: TemporalInterval) -> float:
    """Return temporal intersection-over-union."""
    intersection = temporal_overlap(predicted, ground_truth)
    union = temporal_duration(predicted) + temporal_duration(ground_truth) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def frequency_overlap(predicted: Box, ground_truth: Box) -> float:
    """Return overlap bandwidth in Hz."""
    low = max(predicted.low_frequency_hz, ground_truth.low_frequency_hz)
    high = min(predicted.high_frequency_hz, ground_truth.high_frequency_hz)
    return max(0.0, high - low)


def box_area(box: Box) -> float:
    """Return box area in seconds * Hz."""
    duration = max(0.0, box.end_time_seconds - box.start_time_seconds)
    bandwidth = max(0.0, box.high_frequency_hz - box.low_frequency_hz)
    return duration * bandwidth


def box_iou(predicted: Box, ground_truth: Box) -> float:
    """Return 2D time-frequency intersection-over-union."""
    intersection = time_overlap(predicted, ground_truth) * frequency_overlap(
        predicted, ground_truth
    )
    union = box_area(predicted) + box_area(ground_truth) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_boxes(
    predicted_boxes: list[Box],
    ground_truth_boxes: list[Box],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedily match predictions to ground truth by descending IoU."""
    candidates = []
    for pred_index, predicted in enumerate(predicted_boxes):
        for gt_index, ground_truth in enumerate(ground_truth_boxes):
            iou = box_iou(predicted, ground_truth)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gt_index))

    matches = []
    used_predictions = set()
    used_ground_truth = set()

    for iou, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in used_predictions or gt_index in used_ground_truth:
            continue
        matches.append((pred_index, gt_index, iou))
        used_predictions.add(pred_index)
        used_ground_truth.add(gt_index)

    return matches


def match_temporal_intervals(
    predicted_intervals: list[TemporalInterval],
    ground_truth_intervals: list[TemporalInterval],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedily match predictions to ground truth by descending temporal IoU."""
    candidates = []
    for pred_index, predicted in enumerate(predicted_intervals):
        for gt_index, ground_truth in enumerate(ground_truth_intervals):
            iou = temporal_iou(predicted, ground_truth)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gt_index))

    matches = []
    used_predictions = set()
    used_ground_truth = set()

    for iou, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in used_predictions or gt_index in used_ground_truth:
            continue
        matches.append((pred_index, gt_index, iou))
        used_predictions.add(pred_index)
        used_ground_truth.add(gt_index)

    return matches


def evaluate(
    predicted_path: str | Path,
    ground_truth_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate predictions against ground truth and return one summary row."""
    predicted_boxes = load_boxes(predicted_path)
    ground_truth_boxes = load_boxes(ground_truth_path)
    matches = match_boxes(predicted_boxes, ground_truth_boxes, iou_threshold)

    matched_time_overlaps = [
        time_overlap(predicted_boxes[pred_index], ground_truth_boxes[gt_index])
        for pred_index, gt_index, _ in matches
    ]
    matched_frequency_overlaps = [
        frequency_overlap(predicted_boxes[pred_index], ground_truth_boxes[gt_index])
        for pred_index, gt_index, _ in matches
    ]
    matched_ious = [iou for _, _, iou in matches]

    matched_count = len(matches)
    false_positives = len(predicted_boxes) - matched_count
    missed_events = len(ground_truth_boxes) - matched_count

    return {
        "predicted_path": str(predicted_path),
        "ground_truth_path": str(ground_truth_path),
        "iou_threshold": iou_threshold,
        "predicted_count": len(predicted_boxes),
        "ground_truth_count": len(ground_truth_boxes),
        "matched_events": matched_count,
        "false_positives": false_positives,
        "missed_events": missed_events,
        "mean_time_overlap_seconds": mean(matched_time_overlaps),
        "mean_frequency_overlap_hz": mean(matched_frequency_overlaps),
        "mean_iou": mean(matched_ious),
    }


def evaluate_temporal(
    predicted_path: str | Path,
    ground_truth_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate temporal-only predictions against ground truth."""
    predicted_intervals = load_temporal_intervals(predicted_path)
    ground_truth_intervals = load_temporal_intervals(ground_truth_path)
    matches = match_temporal_intervals(
        predicted_intervals,
        ground_truth_intervals,
        iou_threshold,
    )
    matched_ious = [iou for _, _, iou in matches]

    matched_count = len(matches)
    false_positives = len(predicted_intervals) - matched_count
    missed_events = len(ground_truth_intervals) - matched_count
    precision = safe_divide(matched_count, matched_count + false_positives)
    recall = safe_divide(matched_count, matched_count + missed_events)
    f1 = harmonic_mean(precision, recall)

    return {
        "prediction_path": str(predicted_path),
        "ground_truth_path": str(ground_truth_path),
        "iou_threshold": iou_threshold,
        "predicted_count": len(predicted_intervals),
        "ground_truth_count": len(ground_truth_intervals),
        "matched_events": matched_count,
        "false_positives": false_positives,
        "missed_events": missed_events,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_temporal_iou": mean(matched_ious),
    }


def evaluate_by_mode(
    predicted_path: str | Path,
    ground_truth_path: str | Path,
    iou_threshold: float = 0.5,
    mode: str = "box2d",
) -> dict:
    """Evaluate predictions using either 2D boxes or temporal-only intervals."""
    if mode == "box2d":
        return evaluate(predicted_path, ground_truth_path, iou_threshold)
    if mode == "temporal":
        return evaluate_temporal(predicted_path, ground_truth_path, iou_threshold)
    raise ValueError(f"Unsupported evaluation mode: {mode}")


def mean(values: list[float]) -> float:
    """Return a simple mean, or 0.0 when there are no matched boxes."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def harmonic_mean(precision: float, recall: float) -> float:
    """Return F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def save_summary(row: dict, output_path: str | Path = SUMMARY_PATH) -> Path:
    """Save one evaluation summary row as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare predicted bioacoustic boxes with ground-truth boxes."
    )
    parser.add_argument(
        "predicted_json",
        nargs="?",
        help="Path to predicted EventResult JSON.",
    )
    parser.add_argument(
        "ground_truth",
        nargs="?",
        help="Path to ground-truth JSON or CSV.",
    )
    parser.add_argument(
        "--pred",
        dest="predicted_json_flag",
        help="Path to predicted EventResult JSON.",
    )
    parser.add_argument(
        "--gt",
        dest="ground_truth_flag",
        help="Path to ground-truth JSON or CSV.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum IoU for a predicted event to count as matched.",
    )
    parser.add_argument(
        "--mode",
        choices=["box2d", "temporal"],
        default="box2d",
        help="Evaluation mode. box2d uses time-frequency boxes; temporal uses start/end time only.",
    )
    parser.add_argument(
        "--output",
        default=str(SUMMARY_PATH),
        help="Path for the summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predicted_json = args.predicted_json_flag or args.predicted_json
    ground_truth = args.ground_truth_flag or args.ground_truth
    if not predicted_json or not ground_truth:
        raise SystemExit("Provide prediction and ground truth paths, either positionally or with --pred and --gt.")

    summary = evaluate(
        predicted_json,
        ground_truth,
        iou_threshold=args.iou_threshold,
    ) if args.mode == "box2d" else evaluate_temporal(
        predicted_json,
        ground_truth,
        iou_threshold=args.iou_threshold,
    )
    output_path = save_summary(summary, args.output)
    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
