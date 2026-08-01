import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.maintenance.analyze_proposal_deviations as analysis


def proposal(proposal_id: str = "bd2_001", start: float = 0.1) -> dict:
    return {
        "proposal_id": proposal_id,
        "start_time_seconds": start,
        "end_time_seconds": start + 0.01,
        "low_frequency_hz": 30000,
        "high_frequency_hz": 40000,
    }


def prediction(start: float = 0.105, used_id: str = "bd2_001") -> dict:
    return {
        "event_id": "pred_001",
        "used_proposal_id": used_id,
        "start_time_seconds": start,
        "end_time_seconds": start + 0.01,
        "low_frequency_hz": 31000,
        "high_frequency_hz": 39000,
    }


def test_linking_prefers_explicit_proposal_id() -> None:
    proposals = [proposal("bd2_001", 0.1), proposal("bd2_002", 0.5)]
    linked, method = analysis.link_prediction_to_proposal(
        prediction(start=0.5, used_id="bd2_001"), proposals
    )

    assert linked is proposals[0]
    assert method == "used_proposal_id"


def test_linking_falls_back_to_temporal_overlap_then_center() -> None:
    proposals = [proposal("bd2_001", 0.1), proposal("bd2_002", 0.5)]
    linked, method = analysis.link_prediction_to_proposal(
        prediction(start=0.502, used_id=""), proposals
    )

    assert linked is proposals[1]
    assert method == "maximum_temporal_overlap"


def test_delta_and_duration_ratio_calculation() -> None:
    record = analysis.build_deviation_record(
        clip_id="OP_016",
        prediction={**prediction(start=0.105), "end_time_seconds": 0.125},
        proposal=proposal(),
        link_method="used_proposal_id",
        proposal_matched=True,
        prediction_matched=False,
    )

    assert record.delta_start_ms == pytest.approx(5.0)
    assert record.delta_end_ms == pytest.approx(15.0)
    assert record.proposal_duration_ms == pytest.approx(10.0)
    assert record.predicted_duration_ms == pytest.approx(20.0)
    assert record.duration_ratio == pytest.approx(2.0)


def test_flags_trigger_only_beyond_boundaries() -> None:
    flags = analysis.calculate_flags(
        delta_start_ms=10.01,
        delta_end_ms=0,
        duration_ratio=1.0,
        source_time_iou=0.49,
        delta_low_frequency_hz=0,
        delta_high_frequency_hz=10001,
    )

    assert flags.large_start_shift is True
    assert flags.large_end_shift is False
    assert flags.low_time_iou_with_source_proposal is True
    assert flags.frequency_shift_large is True
    assert flags.unsupported_geometry_change is True


@pytest.mark.parametrize(
    ("proposal_matched", "prediction_matched", "expected"),
    [
        (True, True, "both_matched"),
        (True, False, "proposal_was_good_but_prediction_broke_match"),
        (False, True, "prediction_improved_bad_proposal"),
        (False, False, "both_failed"),
    ],
)
def test_outcome_classification(
    proposal_matched: bool, prediction_matched: bool, expected: str
) -> None:
    assert analysis.classify_outcome(proposal_matched, prediction_matched) == expected


def test_decision_preserves_good_proposal_broken_by_prediction() -> None:
    flags = analysis.calculate_flags(
        delta_start_ms=5,
        delta_end_ms=5,
        duration_ratio=1,
        source_time_iou=0.3,
        delta_low_frequency_hz=0,
        delta_high_frequency_hz=0,
    )
    decision = analysis.choose_validator_decision(
        linked=True,
        flags=flags,
        outcome="proposal_was_good_but_prediction_broke_match",
    )

    assert decision.action == "preserve_original_proposal_geometry"


def test_pydantic_flags_reject_inconsistent_aggregate() -> None:
    with pytest.raises(ValueError, match="must equal"):
        analysis.ProposalDeviationFlags(
            large_start_shift=True,
            large_end_shift=False,
            duration_expansion=False,
            duration_shrinkage=False,
            low_time_iou_with_source_proposal=False,
            frequency_shift_large=False,
            unsupported_geometry_change=False,
        )


def test_clip_summary_and_csv_writing(tmp_path: Path) -> None:
    record = analysis.build_deviation_record(
        clip_id="OP_016",
        prediction=prediction(),
        proposal=proposal(),
        link_method="used_proposal_id",
        proposal_matched=True,
        prediction_matched=True,
    )
    summary = analysis.build_clip_summary("OP_016", [record], 1, 1)
    output = tmp_path / "summary.csv"
    analysis.write_csv(output, [summary])

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["clip_id"] == "OP_016"
    assert rows[0]["linked_prediction_count"] == "1"
