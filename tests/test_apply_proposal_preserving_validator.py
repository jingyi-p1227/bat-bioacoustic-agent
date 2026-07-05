import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import apply_proposal_preserving_validator as validator


def proposal(proposal_id: str = "bd2_001", det_prob: float = 0.8) -> dict:
    return {
        "proposal_id": proposal_id,
        "start_time_seconds": 0.1,
        "end_time_seconds": 0.11,
        "low_frequency_hz": 30000,
        "high_frequency_hz": 40000,
        "det_prob": det_prob,
        "class_prob": 0.5,
        "label": "UK taxonomy label",
    }


def vlm_event(event_id: str = "pred_001", used_id: str = "bd2_001") -> dict:
    return {
        "event_id": event_id,
        "start_time_seconds": 0.105,
        "end_time_seconds": 0.115,
        "low_frequency_hz": 32000,
        "high_frequency_hz": 39000,
        "label": "bat_call",
        "confidence": 0.9,
        "evidence": "Visible event.",
        "human_review_needed": False,
        "review_reason": "",
        "used_proposal_id": used_id,
        "proposal_source": "batdetect2" if used_id else "",
        "refinement_note": "",
    }


def audit_row(**overrides: str) -> dict[str, str]:
    row = {
        "time_iou_between_proposal_and_prediction": "0.35",
        "unsupported_geometry_change": "True",
        "delta_start_ms": "5",
        "delta_end_ms": "5",
        "duration_ratio": "1.0",
        "frequency_shift_large": "False",
    }
    row.update(overrides)
    return row


def payloads(events: list[dict], proposals: list[dict]) -> tuple[dict, dict]:
    prediction_payload = {
        "clip_id": "OP_016",
        "clip_duration_seconds": 1.0,
        "prompt_version": "prompt_v2_bat_strong_label",
        "model_name": "qwen3.6:latest",
        "input_image_path": "clean.png",
        "events": events,
    }
    proposal_payload = {
        "clip_id": "OP_016",
        "proposal_threshold": 0.3,
        "events": proposals,
    }
    return prediction_payload, proposal_payload


def test_validator_rule_triggers_on_low_time_iou() -> None:
    preserve, reason = validator.should_preserve_proposal_geometry(
        audit_row(unsupported_geometry_change="False")
    )

    assert preserve is True
    assert "time IoU" in reason


def test_geometry_replacement_and_provenance() -> None:
    event = validator.preserve_linked_event(
        vlm_event(), proposal(), audit_row(), 1.0
    )

    assert event["start_time_seconds"] == 0.1
    assert event["end_time_seconds"] == 0.11
    assert event["validation_decision"] == "preserved_proposal_geometry"
    assert event["source_proposal_id"] == "bd2_001"
    assert event["original_vlm_geometry"]["start_time_seconds"] == 0.105
    assert event["det_prob"] == 0.8
    assert event["human_review_needed"] is True


def test_safe_geometry_keeps_vlm_coordinates() -> None:
    safe_audit = audit_row(
        time_iou_between_proposal_and_prediction="0.7",
        unsupported_geometry_change="False",
    )
    event = validator.preserve_linked_event(vlm_event(), proposal(), safe_audit, 1.0)

    assert event["start_time_seconds"] == 0.105
    assert event["validation_decision"] == "kept_vlm_geometry"
    assert "original_vlm_geometry" not in event


def test_new_vlm_event_is_retained_and_marked_for_review() -> None:
    event = validator.keep_new_vlm_event(vlm_event(used_id=""))

    assert event["validation_decision"] == "kept_new_vlm_event"
    assert event["source_proposal_id"] == ""
    assert event["human_review_needed"] is True


def test_high_confidence_rejected_proposal_is_restored() -> None:
    prediction_payload, proposal_payload = payloads([], [proposal(det_prob=0.8)])
    constrained = validator.constrain_clip_payload(
        clip_id="OP_016",
        prediction_payload=prediction_payload,
        proposal_payload=proposal_payload,
        audit_rows={},
    )

    event = constrained["events"][0]
    assert event["event_id"] == "restored_bd2_001"
    assert event["validation_decision"] == "restored_high_confidence_proposal"
    assert event["confidence"] == 0.8
    assert event["human_review_needed"] is True


def test_low_confidence_rejected_proposal_is_not_restored() -> None:
    prediction_payload, proposal_payload = payloads([], [proposal(det_prob=0.69)])
    constrained = validator.constrain_clip_payload(
        clip_id="OP_016",
        prediction_payload=prediction_payload,
        proposal_payload=proposal_payload,
        audit_rows={},
    )

    assert constrained["events"] == []


def test_output_event_is_evaluator_compatible() -> None:
    event = validator.restore_high_confidence_proposal(proposal(), 1.0)

    validator.validate_evaluator_compatible_event(event, 1.0)
    assert event["label"] == "bat_call"
    assert all(field in event for field in validator.GEOMETRY_FIELDS)


def test_decision_count_summary() -> None:
    rows = validator.count_decisions(
        [
            {
                "clip_id": "OP_001",
                "events": [
                    {"validation_decision": "kept_vlm_geometry"},
                    {"validation_decision": "preserved_proposal_geometry"},
                ],
            },
            {
                "clip_id": "OP_016",
                "events": [
                    {"validation_decision": "preserved_proposal_geometry"},
                ],
            },
        ]
    )

    assert rows[-1]["clip_id"] == "ALL"
    assert rows[-1]["kept_vlm_geometry"] == 1
    assert rows[-1]["preserved_proposal_geometry"] == 2
    assert rows[-1]["total_events"] == 3
