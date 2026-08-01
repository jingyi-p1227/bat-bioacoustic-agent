import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.maintenance.apply_timing_rule_ablation_validator as ablation


def audit(**overrides: str) -> dict[str, str]:
    row = {
        "time_iou_between_proposal_and_prediction": "0.3",
        "delta_start_ms": "12",
        "delta_end_ms": "11",
        "duration_ratio": "0.95",
        "frequency_shift_large": "False",
    }
    row.update(overrides)
    return row


def proposal(start: float = 0.1, end: float = 0.106, det_prob: float = 0.8) -> dict:
    return {
        "proposal_id": "bd2_001",
        "start_time_seconds": start,
        "end_time_seconds": end,
        "low_frequency_hz": 30000,
        "high_frequency_hz": 40000,
        "det_prob": det_prob,
        "class_prob": 0.5,
        "label": "UK label",
    }


def event() -> dict:
    return {
        "event_id": "pred_001",
        "start_time_seconds": 0.105,
        "end_time_seconds": 0.12,
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


def test_rigid_translation_is_preserved() -> None:
    classification, _ = ablation.classify_policy_b_timing(audit())

    assert classification == "preserve_near_rigid_translation"


def test_anchored_moderate_expansion_is_kept() -> None:
    classification, _ = ablation.classify_policy_b_timing(
        audit(delta_start_ms="5.5", delta_end_ms="13.3", duration_ratio="1.64")
    )

    assert classification == "keep_anchored_moderate_expansion"


@pytest.mark.parametrize("target_ms", [8.0, 10.0, 12.0])
def test_short_proposal_expands_to_target_duration(target_ms: float) -> None:
    output = ablation.apply_short_proposal_prior(
        event(), proposal(), 1.0, target_ms
    )

    assert (output["end_time_seconds"] - output["start_time_seconds"]) * 1000 == pytest.approx(
        target_ms
    )
    assert output["duration_prior_applied"] is True
    assert output["human_review_needed"] is True
    assert output["source_proposal_id"] == "bd2_001"


def test_duration_expansion_clips_to_clip_boundary() -> None:
    start, end = ablation.expand_interval_around_center(0.995, 1.0, 12.0, 1.0)

    assert start == pytest.approx(0.9915)
    assert end == 1.0
    assert end - start < 0.012


def test_policy_output_directory_generation() -> None:
    assert ablation.policy_output_dir(
        Path("outputs/run"), "policy_b_anchored_expansion"
    ) == Path("outputs/run/policy_b_anchored_expansion")


def test_summary_csv_writing(tmp_path: Path) -> None:
    row = {
        "policy_name": "policy_a_p6e3_baseline",
        "clip_scope": "representative6",
        "prediction_count": 31,
    }
    path = tmp_path / "summary.csv"
    ablation.write_summary_csv(path, [row])

    with path.open(newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["policy_name"] == "policy_a_p6e3_baseline"
    assert loaded[0]["prediction_count"] == "31"
