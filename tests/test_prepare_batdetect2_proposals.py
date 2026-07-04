import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_batdetect2_proposals import (
    ProposalSummaryRow,
    convert_raw_event,
    convert_raw_payload,
    write_proposal_summary,
)


def raw_event(
    start: float,
    end: float,
    det_prob: float = 0.8,
) -> dict:
    return {
        "class": "Nyctalus leisleri",
        "class_prob": 0.6,
        "det_prob": det_prob,
        "start_time": start,
        "end_time": end,
        "low_freq": 30000,
        "high_freq": 38000,
    }


def test_convert_raw_event_to_proposal_schema() -> None:
    proposal = convert_raw_event(raw_event(0.1, 0.11), "bd2_001")

    assert proposal == {
        "proposal_id": "bd2_001",
        "start_time_seconds": 0.1,
        "end_time_seconds": 0.11,
        "low_frequency_hz": 30000.0,
        "high_frequency_hz": 38000.0,
        "det_prob": 0.8,
        "class_prob": 0.6,
        "label": "Nyctalus leisleri",
        "source": "batdetect2",
    }


def test_convert_raw_event_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="start_time"):
        convert_raw_event(raw_event(0.2, 0.1), "bd2_001")


def test_convert_payload_sorts_and_numbers_filtered_proposals() -> None:
    output, summary = convert_raw_payload(
        clip_id="OP_016",
        payload={
            "annotation": [
                raw_event(0.5, 0.51, 0.9),
                raw_event(0.1, 0.11, 0.8),
                raw_event(0.2, 0.21, 0.2),
            ]
        },
        min_det_prob=0.3,
    )

    assert [event["start_time_seconds"] for event in output["events"]] == [0.1, 0.5]
    assert [event["proposal_id"] for event in output["events"]] == ["bd2_001", "bd2_002"]
    assert summary.proposal_count == 2
    assert "filtered_below_threshold=1" in summary.notes


def test_write_proposal_summary_can_be_read_back(tmp_path: Path) -> None:
    path = tmp_path / "proposal_summary.csv"
    row = ProposalSummaryRow(
        clip_id="OP_016",
        proposal_count=7,
        mean_det_prob=0.7,
        min_start_time=0.1,
        max_end_time=0.9,
        notes="synthetic test",
    )

    write_proposal_summary([row], path)

    with path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))[0]
    assert saved["clip_id"] == "OP_016"
    assert saved["proposal_count"] == "7"
    assert saved["mean_det_prob"] == "0.7"

