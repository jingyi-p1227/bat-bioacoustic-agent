import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_prompt_v2_batdetect2_assisted_pilot as assisted


def proposal_payload() -> dict:
    return {
        "clip_id": "OP_016",
        "proposal_source": "batdetect2",
        "proposal_threshold": 0.3,
        "events": [
            {
                "proposal_id": "bd2_002",
                "start_time_seconds": 0.3,
                "end_time_seconds": 0.31,
                "low_frequency_hz": 31000,
                "high_frequency_hz": 36000,
                "det_prob": 0.7,
                "class_prob": 0.4,
                "label": "Pipistrellus nathusii",
            },
            {
                "proposal_id": "bd2_001",
                "start_time_seconds": 0.1,
                "end_time_seconds": 0.112,
                "low_frequency_hz": 32000,
                "high_frequency_hz": 38000,
                "det_prob": 0.8,
                "class_prob": 0.5,
                "label": "Nyctalus leisleri",
            },
        ],
    }


def valid_prediction() -> dict:
    return {
        "clip_id": "OP_016",
        "events": [
            {
                "event_id": "pred_001",
                "start_time_seconds": 0.101,
                "end_time_seconds": 0.111,
                "low_frequency_hz": 32500,
                "high_frequency_hz": 37500,
                "label": "bat_call",
                "confidence": 0.9,
                "evidence": "Visible call aligned with detector hint.",
                "human_review_needed": False,
                "review_reason": "",
                "used_proposal_id": "bd2_001",
                "proposal_source": "batdetect2",
                "refinement_note": "Tightened frequency bounds.",
            }
        ],
    }


def test_proposal_metadata_is_sorted_and_records_duration() -> None:
    rows = assisted.format_proposal_metadata(proposal_payload())

    assert [row["proposal_id"] for row in rows] == ["bd2_001", "bd2_002"]
    assert rows[0]["duration_ms"] == pytest.approx(12.0)
    assert rows[0]["original_label"] == "Nyctalus leisleri"


def test_prompt_marks_proposals_as_hints_and_taxonomy_as_unreliable() -> None:
    message = assisted.build_assisted_user_message(
        clip_id="OP_016",
        clip_duration_seconds=1.0,
        proposal_rows=assisted.format_proposal_metadata(proposal_payload()),
    )

    assert "hints, not labels or ground truth" in message
    assert "UK taxonomy labels are unreliable" in message
    assert "add any visible calls missing" in message
    assert '"proposal_id": "bd2_001"' in message


def test_parse_prediction_validates_and_preserves_provenance() -> None:
    parsed = assisted.parse_assisted_prediction(
        f"```json\n{json.dumps(valid_prediction())}\n```",
        expected_clip_id="OP_016",
        clip_duration_seconds=1.0,
    )

    event = parsed["events"][0]
    assert event["used_proposal_id"] == "bd2_001"
    assert event["proposal_source"] == "batdetect2"
    assert event["refinement_note"] == "Tightened frequency bounds."


def test_prediction_rejects_missing_provenance() -> None:
    payload = valid_prediction()
    del payload["events"][0]["used_proposal_id"]

    with pytest.raises(ValueError, match="used_proposal_id"):
        assisted.validate_assisted_prediction(
            payload,
            expected_clip_id="OP_016",
            clip_duration_seconds=1.0,
        )


def test_prediction_rejects_invalid_geometry() -> None:
    payload = valid_prediction()
    payload["events"][0]["end_time_seconds"] = 1.1

    with pytest.raises(ValueError, match="invalid time geometry"):
        assisted.validate_assisted_prediction(
            payload,
            expected_clip_id="OP_016",
            clip_duration_seconds=1.0,
        )


def test_load_proposal_payload_validates_clip_id(tmp_path: Path) -> None:
    path = tmp_path / "OP_016_batdetect2_proposals.json"
    path.write_text(json.dumps({**proposal_payload(), "clip_id": "OP_001"}))

    with pytest.raises(ValueError, match="does not match"):
        assisted.load_proposal_payload(tmp_path, "OP_016")
