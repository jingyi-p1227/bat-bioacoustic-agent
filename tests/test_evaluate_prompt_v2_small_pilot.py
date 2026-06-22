import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_prompt_v2_small_pilot import (
    EventBox,
    box_iou,
    evaluate_clip,
    frequency_iou,
    greedy_temporal_matches,
    temporal_iou,
)


def event(
    event_id: str,
    start: float,
    end: float,
    low: float = 20.0,
    high: float = 40.0,
    confidence: float | None = None,
) -> EventBox:
    return EventBox(
        event_id=event_id,
        start_time=start,
        end_time=end,
        low_frequency=low,
        high_frequency=high,
        label="Bat",
        confidence=confidence,
    )


def test_temporal_iou() -> None:
    assert temporal_iou(event("p", 0.0, 2.0), event("g", 1.0, 3.0)) == pytest.approx(
        1 / 3
    )


def test_frequency_iou() -> None:
    prediction = event("p", 0.0, 1.0, low=20.0, high=50.0)
    ground_truth = event("g", 0.0, 1.0, low=30.0, high=60.0)

    assert frequency_iou(prediction, ground_truth) == pytest.approx(20 / 40)


def test_box_iou() -> None:
    prediction = event("p", 0.0, 2.0, low=0.0, high=2.0)
    ground_truth = event("g", 1.0, 3.0, low=1.0, high=3.0)

    assert box_iou(prediction, ground_truth) == pytest.approx(1 / 7)


def test_greedy_matching_is_one_to_one_and_confidence_ordered() -> None:
    predictions = [
        event("low", 0.0, 1.0, confidence=0.2),
        event("high", 0.0, 1.0, confidence=0.9),
    ]
    ground_truth = [event("gt", 0.0, 1.0)]

    matches = greedy_temporal_matches(predictions, ground_truth, 0.3)

    assert len(matches) == 1
    assert matches[0].prediction_index == 1
    assert matches[0].ground_truth_index == 0


def test_evaluate_clip_calculates_expected_counts() -> None:
    prediction_payload = {
        "events": [
            {
                "event_id": "pred_1",
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.0,
                "low_frequency_hz": 20.0,
                "high_frequency_hz": 40.0,
                "label": "Bat",
                "confidence": 0.9,
            },
            {
                "event_id": "pred_fp",
                "start_time_seconds": 3.0,
                "end_time_seconds": 4.0,
                "low_frequency_hz": 20.0,
                "high_frequency_hz": 40.0,
                "label": "Bat",
                "confidence": 0.8,
            },
        ]
    }
    ground_truth_payload = {
        "events": [
            {
                "event_id": "gt_1",
                "start_time": 0.0,
                "end_time": 1.0,
                "low_frequency": 20.0,
                "high_frequency": 40.0,
                "label": "Bat",
                "truncation_side": "none",
            },
            {
                "event_id": "gt_missed",
                "start_time": 5.0,
                "end_time": 6.0,
                "low_frequency": 20.0,
                "high_frequency": 40.0,
                "label": "Bat",
                "truncation_side": "left",
            },
        ]
    }

    result = evaluate_clip(
        clip_id="OP_TEST",
        prediction_payload=prediction_payload,
        ground_truth_payload=ground_truth_payload,
    )
    metrics = result["metrics"]

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["mean_time_iou"] == pytest.approx(1.0)
    assert metrics["mean_frequency_iou"] == pytest.approx(1.0)
    assert metrics["mean_box_iou"] == pytest.approx(1.0)
    assert metrics["num_truncated_events"] == 1
    assert result["unmatched_predictions"][0]["failure_categories"] == "false_positive"
    assert (
        result["missed_ground_truth_events"][0]["failure_categories"]
        == "missed_call;boundary_truncation_error"
    )
