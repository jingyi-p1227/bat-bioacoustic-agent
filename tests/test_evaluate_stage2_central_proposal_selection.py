from __future__ import annotations

import pytest

from scripts.analysis.evaluate_stage2_central_proposal_selection import (
    best_match,
    centre_distance,
    select_proposals,
)


def proposal(
    proposal_id: str,
    start: float,
    end: float,
    det_prob: float,
    low: float = 20_000,
    high: float = 40_000,
) -> dict:
    return {
        "proposal_id": proposal_id,
        "start_time_seconds": start,
        "end_time_seconds": end,
        "low_frequency_hz": low,
        "high_frequency_hz": high,
        "det_prob": det_prob,
    }


def test_highest_score_selects_max_det_prob() -> None:
    proposals = [
        proposal("bd2_001", 0.14, 0.16, 0.5),
        proposal("bd2_002", 0.01, 0.03, 0.9),
    ]

    selected = select_proposals(proposals, "highest_score")

    assert selected[0]["proposal_id"] == "bd2_002"


def test_nearest_to_centre_uses_temporal_centre() -> None:
    proposals = [
        proposal("bd2_001", 0.00, 0.02, 0.9),
        proposal("bd2_002", 0.145, 0.155, 0.4),
    ]

    selected = select_proposals(proposals, "nearest_to_centre")

    assert selected[0]["proposal_id"] == "bd2_002"
    assert centre_distance(selected[0]) == pytest.approx(0.0)


def test_centre_then_score_prefers_centred_high_score_candidate() -> None:
    proposals = [
        proposal("bd2_001", 0.140, 0.160, 0.5),
        proposal("bd2_002", 0.145, 0.165, 0.8),
        proposal("bd2_003", 0.02, 0.04, 0.99),
    ]

    selected = select_proposals(proposals, "centre_then_score")

    assert selected[0]["proposal_id"] == "bd2_002"


def test_top3_centre_candidates_keeps_three_nearest() -> None:
    proposals = [
        proposal("bd2_001", 0.00, 0.02, 0.9),
        proposal("bd2_002", 0.145, 0.155, 0.4),
        proposal("bd2_003", 0.13, 0.15, 0.7),
        proposal("bd2_004", 0.16, 0.18, 0.8),
    ]

    selected = select_proposals(proposals, "top3_centre_candidates")

    assert [item["proposal_id"] for item in selected] == ["bd2_002", "bd2_003", "bd2_004"]


def test_best_match_counts_unmatched_selected_as_false_positive() -> None:
    gt = {
        "start_time": 0.145,
        "end_time": 0.165,
        "low_freq": 20_000,
        "high_freq": 40_000,
    }
    good = proposal("bd2_001", 0.145, 0.165, 0.8)
    poor = proposal("bd2_002", 0.01, 0.02, 0.9)

    matched, unmatched, score = best_match(gt, [good, poor], "temporal_iou_0p3")

    assert matched is good
    assert unmatched == [poor]
    assert score == pytest.approx(1.0)
