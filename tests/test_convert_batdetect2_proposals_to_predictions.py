import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convert_batdetect2_proposals_to_predictions import (
    convert_proposal_event,
    convert_proposal_payload,
    prediction_output_path,
)


def proposal(end_time: float = 0.12) -> dict:
    return {
        "proposal_id": "bd2_001",
        "start_time_seconds": 0.1,
        "end_time_seconds": end_time,
        "low_frequency_hz": 30000,
        "high_frequency_hz": 38000,
        "det_prob": 0.82,
        "class_prob": 0.61,
        "label": "Nyctalus leisleri",
        "source": "batdetect2",
    }


def test_proposal_to_prediction_maps_confidence_and_generic_label() -> None:
    event = convert_proposal_event(proposal(), clip_duration_seconds=1.0)

    assert event["event_id"] == "bd2_001"
    assert event["label"] == "bat_call"
    assert event["confidence"] == 0.82
    assert event["evidence"] == (
        "Candidate event supplied by BatDetect2 proposal metadata."
    )
    assert event["human_review_needed"] is False


def test_proposal_conversion_preserves_detector_metadata() -> None:
    event = convert_proposal_event(proposal(), clip_duration_seconds=1.0)

    assert event["proposal_id"] == "bd2_001"
    assert event["det_prob"] == 0.82
    assert event["class_prob"] == 0.61
    assert event["original_label"] == "Nyctalus leisleri"
    assert event["proposal_source"] == "batdetect2"


def test_suspicious_end_time_is_clipped_and_marked_for_review() -> None:
    event = convert_proposal_event(proposal(end_time=1.004), clip_duration_seconds=1.0)

    assert event["end_time_seconds"] == 1.0
    assert event["original_end_time_seconds"] == 1.004
    assert event["clipped_to_clip_bounds"] is True
    assert event["human_review_needed"] is True
    assert "clip end" in event["review_reason"]


def test_payload_conversion_retains_threshold_and_events() -> None:
    payload = convert_proposal_payload(
        {
            "clip_id": "OP_016",
            "proposal_source": "batdetect2",
            "proposal_threshold": 0.3,
            "events": [proposal()],
        },
        clip_duration_seconds=1.0,
    )

    assert payload["clip_id"] == "OP_016"
    assert payload["model_name"] == "batdetect2"
    assert payload["proposal_threshold"] == 0.3
    assert len(payload["events"]) == 1


def test_prediction_output_filename() -> None:
    output_dir = Path("outputs/run/predictions")

    assert prediction_output_path(output_dir, "OP_001") == (
        output_dir / "OP_001_prediction.json"
    )

