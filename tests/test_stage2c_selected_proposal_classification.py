from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.evaluation.evaluate_stage2c_selected_proposal_classification import (
    protocol_match,
    selected_proposal_from_prediction,
)
from scripts.inference.run_stage2c_selected_proposal_classification import (
    build_user_message,
    selected_proposal_index,
)


def test_selected_proposal_index_filters_nearest_centre_scope(tmp_path: Path) -> None:
    path = tmp_path / "selected.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_scope",
                "rule",
                "anonymous_sample_id",
                "proposal_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip_scope": "pilot80",
                "rule": "nearest_to_centre",
                "anonymous_sample_id": "sample_000001",
                "proposal_id": "bd2_001",
            }
        )
        writer.writerow(
            {
                "clip_scope": "full240",
                "rule": "highest_score",
                "anonymous_sample_id": "sample_000001",
                "proposal_id": "bd2_999",
            }
        )
        writer.writerow(
            {
                "clip_scope": "full240",
                "rule": "nearest_to_centre",
                "anonymous_sample_id": "sample_000001",
                "proposal_id": "bd2_240",
            }
        )

    indexed = selected_proposal_index(path)
    full_indexed = selected_proposal_index(path, "full240")

    assert indexed["sample_000001"]["proposal_id"] == "bd2_001"
    assert full_indexed["sample_000001"]["proposal_id"] == "bd2_240"


def test_stage2c_user_message_omits_true_species_and_preserves_coordinates() -> None:
    row = {
        "anonymous_sample_id": "sample_000001",
        "species": "Ozimops petersi",
    }
    proposal = {
        "proposal_id": "bd2_001",
        "start_time": "0.140",
        "end_time": "0.160",
        "low_freq": "20000",
        "high_freq": "40000",
        "det_prob": "0.9",
    }

    message = build_user_message(row, proposal)

    assert "Ozimops petersi" not in message
    assert "bd2_001" in message
    assert "0.14" in message
    assert "Do not alter" in message


def test_selected_proposal_from_prediction_uses_preserved_geometry() -> None:
    row = {
        "selected_proposal_available": "true",
        "selected_proposal_id": "bd2_001",
        "selected_start_time": "0.140",
        "selected_end_time": "0.160",
        "selected_low_freq": "20000",
        "selected_high_freq": "40000",
        "selected_det_prob": "0.9",
    }

    proposal = selected_proposal_from_prediction(row)

    assert proposal is not None
    assert proposal["start_time_seconds"] == 0.140
    assert proposal["end_time_seconds"] == 0.160


def test_protocol_match_uses_selected_proposal_coordinates() -> None:
    gt = {
        "start_time": 0.145,
        "end_time": 0.165,
        "low_freq": 20000,
        "high_freq": 40000,
    }
    proposal = {
        "proposal_id": "bd2_001",
        "start_time_seconds": 0.145,
        "end_time_seconds": 0.165,
        "low_frequency_hz": 20000,
        "high_frequency_hz": 40000,
        "det_prob": 0.9,
    }

    assert protocol_match(gt, proposal, "temporal_iou_0p3") is True
    assert protocol_match(gt, proposal, "start_time_10ms") is True
