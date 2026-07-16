"""Event-level matching protocols for bat-call localisation experiments.

This module is intentionally independent from the legacy root-level evaluator.
It preserves the established confidence-ordered one-to-one matching behaviour
while making matching protocols explicit for P8 sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any, Iterable


class MatchingProtocol(StrEnum):
    """Supported event matching protocols."""

    TEMPORAL_IOU_0_1 = "temporal_iou_0.1"
    TEMPORAL_IOU_0_3 = "temporal_iou_0.3"
    START_TIME_PROXIMITY_10MS = "start_time_proximity_10ms"


@dataclass(frozen=True)
class EventBox:
    """A single time-frequency event box."""

    event_id: str
    start_time: float
    end_time: float
    low_frequency: float | None = None
    high_frequency: float | None = None
    label: str | None = None
    confidence: float | None = None
    source_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0

    @property
    def center_time(self) -> float:
        return (self.start_time + self.end_time) / 2.0


@dataclass(frozen=True)
class MatchedPair:
    """A matched prediction and ground-truth event."""

    prediction: EventBox
    ground_truth: EventBox
    match_score: float
    protocol: MatchingProtocol
    rank: int

    @property
    def temporal_iou(self) -> float:
        return interval_iou(self.prediction.start_time, self.prediction.end_time, self.ground_truth.start_time, self.ground_truth.end_time)

    @property
    def frequency_iou(self) -> float:
        return frequency_iou(self.prediction, self.ground_truth)

    @property
    def box_iou(self) -> float:
        return box_iou(self.prediction, self.ground_truth)

    @property
    def start_time_error_ms(self) -> float:
        return (self.prediction.start_time - self.ground_truth.start_time) * 1000.0

    @property
    def end_time_error_ms(self) -> float:
        return (self.prediction.end_time - self.ground_truth.end_time) * 1000.0

    @property
    def center_time_error_ms(self) -> float:
        return (self.prediction.center_time - self.ground_truth.center_time) * 1000.0

    @property
    def duration_error_ms(self) -> float:
        return self.prediction.duration_ms - self.ground_truth.duration_ms


@dataclass(frozen=True)
class ClipEvaluation:
    """Evaluation result for one clip under one protocol."""

    clip_id: str
    protocol: MatchingProtocol
    predicted_count: int
    ground_truth_count: int
    matched: list[MatchedPair]
    unmatched_predictions: list[EventBox]
    missed_ground_truth: list[EventBox]
    parse_status: str = "success"
    parse_error: str = ""

    @property
    def tp(self) -> int:
        return len(self.matched)

    @property
    def fp(self) -> int:
        return len(self.unmatched_predictions)

    @property
    def fn(self) -> int:
        return len(self.missed_ground_truth)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return one-dimensional temporal IoU."""

    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def frequency_iou(prediction: EventBox, ground_truth: EventBox) -> float:
    """Return one-dimensional frequency IoU if both boxes have frequency bounds."""

    if (
        prediction.low_frequency is None
        or prediction.high_frequency is None
        or ground_truth.low_frequency is None
        or ground_truth.high_frequency is None
    ):
        return 0.0
    return interval_iou(
        prediction.low_frequency,
        prediction.high_frequency,
        ground_truth.low_frequency,
        ground_truth.high_frequency,
    )


def box_iou(prediction: EventBox, ground_truth: EventBox) -> float:
    """Return 2D time-frequency box IoU."""

    if (
        prediction.low_frequency is None
        or prediction.high_frequency is None
        or ground_truth.low_frequency is None
        or ground_truth.high_frequency is None
    ):
        return 0.0

    time_intersection = max(
        0.0,
        min(prediction.end_time, ground_truth.end_time)
        - max(prediction.start_time, ground_truth.start_time),
    )
    freq_intersection = max(
        0.0,
        min(prediction.high_frequency, ground_truth.high_frequency)
        - max(prediction.low_frequency, ground_truth.low_frequency),
    )
    intersection = time_intersection * freq_intersection
    pred_area = max(0.0, prediction.end_time - prediction.start_time) * max(
        0.0, (prediction.high_frequency or 0.0) - (prediction.low_frequency or 0.0)
    )
    gt_area = max(0.0, ground_truth.end_time - ground_truth.start_time) * max(
        0.0, (ground_truth.high_frequency or 0.0) - (ground_truth.low_frequency or 0.0)
    )
    union = pred_area + gt_area - intersection
    return intersection / union if union > 0 else 0.0


def is_valid_event(event: EventBox) -> bool:
    """Return whether event geometry can be evaluated."""

    if not (isfinite(event.start_time) and isfinite(event.end_time)):
        return False
    if event.start_time >= event.end_time:
        return False
    if event.low_frequency is not None and event.high_frequency is not None:
        return isfinite(event.low_frequency) and isfinite(event.high_frequency) and event.low_frequency < event.high_frequency
    return True


def confidence_sort_key(event: EventBox) -> tuple[int, float, int]:
    """Sort key matching the legacy evaluator's confidence ordering."""

    confidence = event.confidence if event.confidence is not None else float("-inf")
    return (1 if event.confidence is not None else 0, confidence, -event.source_index)


def match_score(
    prediction: EventBox,
    ground_truth: EventBox,
    protocol: MatchingProtocol,
    proximity_seconds: float = 0.010,
) -> float | None:
    """Return protocol-specific score if eligible, otherwise None."""

    if protocol == MatchingProtocol.TEMPORAL_IOU_0_1:
        score = interval_iou(prediction.start_time, prediction.end_time, ground_truth.start_time, ground_truth.end_time)
        return score if score >= 0.1 else None
    if protocol == MatchingProtocol.TEMPORAL_IOU_0_3:
        score = interval_iou(prediction.start_time, prediction.end_time, ground_truth.start_time, ground_truth.end_time)
        return score if score >= 0.3 else None
    if protocol == MatchingProtocol.START_TIME_PROXIMITY_10MS:
        distance = abs(prediction.start_time - ground_truth.start_time)
        return -distance if distance <= proximity_seconds else None
    raise ValueError(f"Unsupported matching protocol: {protocol}")


def match_events(
    predictions: Iterable[EventBox],
    ground_truth: Iterable[EventBox],
    protocol: MatchingProtocol,
    proximity_seconds: float = 0.010,
) -> tuple[list[MatchedPair], list[EventBox], list[EventBox]]:
    """Greedily match predictions to GT under the selected protocol."""

    pred_list = list(predictions)
    gt_list = list(ground_truth)
    matched_gt_indices: set[int] = set()
    matched_pred_indices: set[int] = set()
    pairs: list[MatchedPair] = []

    sorted_predictions = sorted(enumerate(pred_list), key=lambda item: confidence_sort_key(item[1]), reverse=True)
    for pred_index, prediction in sorted_predictions:
        candidates: list[tuple[float, int]] = []
        for gt_index, gt_event in enumerate(gt_list):
            if gt_index in matched_gt_indices:
                continue
            score = match_score(prediction, gt_event, protocol, proximity_seconds=proximity_seconds)
            if score is not None:
                candidates.append((score, gt_index))
        if not candidates:
            continue
        best_score, best_gt_index = max(candidates, key=lambda item: (item[0], -item[1]))
        matched_gt_indices.add(best_gt_index)
        matched_pred_indices.add(pred_index)
        pairs.append(
            MatchedPair(
                prediction=prediction,
                ground_truth=gt_list[best_gt_index],
                match_score=best_score,
                protocol=protocol,
                rank=len(pairs),
            )
        )

    unmatched_predictions = [pred for index, pred in enumerate(pred_list) if index not in matched_pred_indices]
    missed_ground_truth = [gt for index, gt in enumerate(gt_list) if index not in matched_gt_indices]
    return pairs, unmatched_predictions, missed_ground_truth


def evaluate_clip(
    clip_id: str,
    predictions: Iterable[EventBox],
    ground_truth: Iterable[EventBox],
    protocol: MatchingProtocol,
    proximity_seconds: float = 0.010,
    parse_status: str = "success",
    parse_error: str = "",
) -> ClipEvaluation:
    """Evaluate one clip under one protocol."""

    valid_predictions = [event for event in predictions if is_valid_event(event)]
    valid_ground_truth = [event for event in ground_truth if is_valid_event(event)]
    if parse_status != "success":
        valid_predictions = []
    matched, unmatched_predictions, missed_ground_truth = match_events(
        valid_predictions,
        valid_ground_truth,
        protocol,
        proximity_seconds=proximity_seconds,
    )
    return ClipEvaluation(
        clip_id=clip_id,
        protocol=protocol,
        predicted_count=len(valid_predictions),
        ground_truth_count=len(valid_ground_truth),
        matched=matched,
        unmatched_predictions=unmatched_predictions,
        missed_ground_truth=missed_ground_truth,
        parse_status=parse_status,
        parse_error=parse_error,
    )


def mean(values: Iterable[float]) -> float:
    vals = [value for value in values if isfinite(value)]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_clip_evaluations(evaluations: Iterable[ClipEvaluation]) -> dict[str, float | int]:
    """Aggregate clip evaluations by pooled TP/FP/FN counts."""

    evals = list(evaluations)
    tp = sum(item.tp for item in evals)
    fp = sum(item.fp for item in evals)
    fn = sum(item.fn for item in evals)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pairs = [pair for item in evals for pair in item.matched]
    return {
        "clip_count": len(evals),
        "predicted_count": sum(item.predicted_count for item in evals),
        "ground_truth_count": sum(item.ground_truth_count for item in evals),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_time_iou": mean(pair.temporal_iou for pair in pairs),
        "mean_frequency_iou": mean(pair.frequency_iou for pair in pairs),
        "mean_box_iou": mean(pair.box_iou for pair in pairs),
        "box_iou_ge_0_3": sum(1 for pair in pairs if pair.box_iou >= 0.3),
        "box_iou_ge_0_5": sum(1 for pair in pairs if pair.box_iou >= 0.5),
        "parse_success_count": sum(1 for item in evals if item.parse_status == "success"),
        "parse_failure_count": sum(1 for item in evals if item.parse_status != "success"),
    }

