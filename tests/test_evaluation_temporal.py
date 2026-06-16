import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import evaluate, evaluate_by_mode, evaluate_temporal, temporal_iou


def test_temporal_iou_uses_temporal_union() -> None:
    predicted = type(
        "Interval",
        (),
        {"start_time_seconds": 0.0, "end_time_seconds": 2.0},
    )()
    ground_truth = type(
        "Interval",
        (),
        {"start_time_seconds": 1.0, "end_time_seconds": 3.0},
    )()

    assert temporal_iou(predicted, ground_truth) == 1 / 3


def test_pseudo_petersi_self_evaluation_temporal_mode_is_perfect() -> None:
    path = Path("ground_truth/pseudo_petersi_001_ground_truth.json")

    result = evaluate_temporal(path, path, iou_threshold=0.5)

    assert result["prediction_path"] == str(path)
    assert result["ground_truth_path"] == str(path)
    assert result["iou_threshold"] == 0.5
    assert result["predicted_count"] == 19
    assert result["ground_truth_count"] == 19
    assert result["matched_events"] == 19
    assert result["false_positives"] == 0
    assert result["missed_events"] == 0
    assert result["precision"] == 1
    assert result["recall"] == 1
    assert result["f1"] == 1
    assert result["mean_temporal_iou"] == 1


def test_temporal_mode_ignores_frequency_fields_in_eventresult(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted.json"
    ground_truth = tmp_path / "ground_truth.json"
    predicted.write_text(
        """
        {
          "audio_path": "audio/example.wav",
          "events": [
            {
              "event_id": "pred_1",
              "start_time_seconds": 1.0,
              "end_time_seconds": 2.0,
              "low_frequency_hz": 1000,
              "high_frequency_hz": 2000,
              "label": "event",
              "confidence": 0.8,
              "evidence": "temporal match",
              "tools_used": [],
              "human_review_needed": false,
              "review_reason": ""
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    ground_truth.write_text(
        """
        {
          "audio_path": "audio/example.wav",
          "events": [
            {
              "event_id": "gt_1",
              "start_time_seconds": 1.0,
              "end_time_seconds": 2.0,
              "low_frequency_hz": 90000,
              "high_frequency_hz": 100000,
              "label": "event",
              "confidence": 1.0,
              "evidence": "same time, different frequency",
              "tools_used": [],
              "human_review_needed": false,
              "review_reason": ""
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    temporal_result = evaluate_by_mode(predicted, ground_truth, mode="temporal")
    box_result = evaluate(predicted, ground_truth)

    assert temporal_result["matched_events"] == 1
    assert temporal_result["mean_temporal_iou"] == 1
    assert box_result["matched_events"] == 0
