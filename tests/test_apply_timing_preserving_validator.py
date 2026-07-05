import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import apply_timing_preserving_validator as validator


def proposal() -> dict:
    return {
        "proposal_id": "bd2_001",
        "start_time_seconds": 0.1,
        "end_time_seconds": 0.11,
        "low_frequency_hz": 30000,
        "high_frequency_hz": 40000,
        "det_prob": 0.8,
        "class_prob": 0.5,
        "label": "UK taxonomy label",
    }


def event() -> dict:
    return {
        "event_id": "pred_001",
        "start_time_seconds": 0.105,
        "end_time_seconds": 0.115,
        "low_frequency_hz": 32000,
        "high_frequency_hz": 39000,
        "label": "bat_call",
        "confidence": 0.9,
        "evidence": "Visible event.",
        "human_review_needed": False,
        "review_reason": "",
        "used_proposal_id": "bd2_001",
        "proposal_source": "batdetect2",
        "refinement_note": "",
    }


def audit(**overrides: str) -> dict[str, str]:
    row = {
        "time_iou_between_proposal_and_prediction": "0.35",
        "delta_start_ms": "5",
        "delta_end_ms": "5",
        "duration_ratio": "1.0",
        "frequency_shift_large": "False",
    }
    row.update(overrides)
    return row


def test_timing_only_replacement_keeps_vlm_frequency() -> None:
    output = validator.apply_linked_event(event(), proposal(), audit(), 1.0)

    assert output["start_time_seconds"] == 0.1
    assert output["end_time_seconds"] == 0.11
    assert output["low_frequency_hz"] == 32000
    assert output["high_frequency_hz"] == 39000
    assert output["timing_decision"] == "preserved_proposal_timing"
    assert output["frequency_decision"] == "kept_vlm_frequency"


def test_frequency_only_replacement_keeps_vlm_timing() -> None:
    output = validator.apply_linked_event(
        event(),
        proposal(),
        audit(
            time_iou_between_proposal_and_prediction="0.7",
            frequency_shift_large="True",
        ),
        1.0,
    )

    assert output["start_time_seconds"] == 0.105
    assert output["end_time_seconds"] == 0.115
    assert output["low_frequency_hz"] == 30000
    assert output["high_frequency_hz"] == 40000
    assert output["timing_decision"] == "kept_vlm_timing"
    assert output["frequency_decision"] == "preserved_proposal_frequency"


def test_independent_decisions_can_preserve_both() -> None:
    output = validator.apply_linked_event(
        event(), proposal(), audit(frequency_shift_large="True"), 1.0
    )

    assert output["validation_decision"] == (
        "preserved_proposal_timing_and_frequency"
    )
    assert output["original_vlm_geometry"]["start_time_seconds"] == 0.105
    assert output["source_proposal_id"] == "bd2_001"
    assert output["det_prob"] == 0.8


def test_new_event_provenance_and_review() -> None:
    new_event = {**event(), "used_proposal_id": "", "proposal_source": ""}
    output = validator.keep_new_vlm_event(new_event)

    assert output["validation_decision"] == "kept_new_vlm_event"
    assert output["timing_decision"] == "not_applicable_new_event"
    assert output["frequency_decision"] == "not_applicable_new_event"
    assert output["human_review_needed"] is True


def test_restored_proposal_has_independent_decision_fields() -> None:
    output = validator.restore_rejected_proposal(proposal(), 1.0)

    assert output["timing_decision"] == "restored_proposal_timing"
    assert output["frequency_decision"] == "restored_proposal_frequency"
    assert output["original_vlm_geometry"] is None


def test_output_schema_is_evaluator_compatible() -> None:
    output = validator.apply_linked_event(event(), proposal(), audit(), 1.0)

    validator.validate_output_event(output, 1.0)
    assert output["label"] == "bat_call"


def test_decision_count_summary_tracks_axes_independently() -> None:
    payloads = [
        {
            "clip_id": "OP_016",
            "events": [
                validator.apply_linked_event(event(), proposal(), audit(), 1.0),
                validator.apply_linked_event(
                    {**event(), "event_id": "pred_002"},
                    proposal(),
                    audit(
                        time_iou_between_proposal_and_prediction="0.7",
                        frequency_shift_large="True",
                    ),
                    1.0,
                ),
            ],
        }
    ]
    rows = validator.count_decisions(payloads)
    aggregate = rows[-1]

    assert aggregate["timing__preserved_proposal_timing"] == 1
    assert aggregate["timing__kept_vlm_timing"] == 1
    assert aggregate["frequency__preserved_proposal_frequency"] == 1
    assert aggregate["frequency__kept_vlm_frequency"] == 1
    assert aggregate["total_events"] == 2
