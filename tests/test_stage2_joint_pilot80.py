import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_stage2_joint_pilot80 import (
    box_iou,
    greedy_match,
    gt_timing,
    interval_iou,
    valid_detection,
)
from scripts.inference.run_stage1a_multispecies_classification import ALLOWED_LABELS
from scripts.inference.run_stage2_joint_proposal_constrained_pilot80 import (
    build_system_prompt,
    build_user_message,
    compact_proposal,
    parse_joint_payload,
    parse_joint_response,
    select_balanced_subset,
)


def test_select_balanced_subset_first_n_per_species() -> None:
    rows = []
    for species in ALLOWED_LABELS:
        for index in range(12):
            rows.append({"species": species, "anonymous_sample_id": f"{species}_{index}"})

    selected = select_balanced_subset(rows, per_species=10)

    assert len(selected) == 80
    for species in ALLOWED_LABELS:
        assert sum(row["species"] == species for row in selected) == 10


def test_stage2_prompt_does_not_include_true_species_in_user_message() -> None:
    row = {
        "anonymous_sample_id": "sample_000001",
        "species": "Ozimops petersi",
    }

    proposals = [
        {
            "proposal_id": "bd2_001",
            "start_time_seconds": 0.1,
            "end_time_seconds": 0.12,
            "low_frequency_hz": 20_000,
            "high_frequency_hz": 40_000,
            "det_prob": 0.9,
            "label": "Ozimops petersi",
        }
    ]
    message = build_user_message(row, proposals, "available")

    assert "sample_000001" in message
    assert "Ozimops petersi" not in message
    assert "available" in message
    assert "bd2_001" in message


def test_stage2_system_prompt_contains_allowed_labels_and_schema() -> None:
    prompt = build_system_prompt()

    assert "joint localisation and species classification" in prompt
    assert "detections" in prompt
    for label in ALLOWED_LABELS:
        assert label in prompt


def test_parse_joint_response_validates_species_and_confidence() -> None:
    payload = {
        "detections": [
            {
                "start_time": 0.1,
                "end_time": 0.2,
                "low_freq": 20000,
                "high_freq": 40000,
                "predicted_species": "Pipistrellus pipistrellus",
                "confidence": 0.7,
                "proposal_id": "bd2_001",
                "decision": "retain",
                "reasoning_brief": "proposal matches visible call",
            }
        ],
        "rejected_proposals": [{"proposal_id": "bd2_002", "reason": "noise"}],
    }

    parsed = parse_joint_response(json.dumps(payload))
    detections, rejected = parse_joint_payload(json.dumps(payload))

    assert parsed[0]["predicted_species"] == "Pipistrellus pipistrellus"
    assert detections[0]["proposal_id"] == "bd2_001"
    assert rejected[0]["proposal_id"] == "bd2_002"
    with pytest.raises(ValueError, match="invalid predicted_species"):
        bad = dict(payload)
        bad["detections"] = [dict(payload["detections"][0], predicted_species="Unknown bat")]
        parse_joint_response(json.dumps(bad))


def test_compact_proposal_removes_detector_species_label() -> None:
    proposal = {
        "proposal_id": "bd2_001",
        "start_time_seconds": 0.1,
        "end_time_seconds": 0.12,
        "low_frequency_hz": 20_000,
        "high_frequency_hz": 40_000,
        "det_prob": 0.9,
        "label": "Pipistrellus pipistrellus",
    }

    compact = compact_proposal(proposal)

    assert compact["proposal_id"] == "bd2_001"
    assert compact["duration_ms"] == 20.0
    assert "label" not in compact


def test_gt_timing_prefers_local_window_fields() -> None:
    row = {
        "event_start_time": "9.0",
        "event_end_time": "9.1",
        "local_gt_start_time": "0.150",
        "local_gt_end_time": "0.170",
    }

    start, end, frame = gt_timing(row)

    assert start == 0.150
    assert end == 0.170
    assert frame == "local_window"


def test_stage2_iou_and_validation_helpers() -> None:
    assert interval_iou(0.0, 1.0, 0.5, 1.5) == pytest.approx(1 / 3)
    pred = {
        "start_time": 0.0,
        "end_time": 1.0,
        "low_freq": 10.0,
        "high_freq": 20.0,
        "predicted_species": "Ozimops petersi",
        "confidence": 0.9,
    }
    gt = {
        "event_start_time": 0.5,
        "event_end_time": 1.5,
        "event_low_freq": 15.0,
        "event_high_freq": 25.0,
    }
    assert box_iou(pred, gt) == pytest.approx(25 / 175)
    parsed, reason = valid_detection(pred)
    assert parsed is not None
    assert reason == ""


def test_greedy_match_prefers_highest_iou() -> None:
    gt = {"event_start_time": 0.0, "event_end_time": 0.1}
    poor = {"start_time": 0.0, "end_time": 0.3, "confidence": 0.9}
    good = {"start_time": 0.0, "end_time": 0.1, "confidence": 0.5}

    matched, remaining = greedy_match(gt, [poor, good], threshold=0.3)

    assert matched is good
    assert remaining == [poor]
