from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "analysis"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_single_agent_full45_localisation import (  # noqa: E402
    is_valid_bbox,
    prediction_quality_counts,
)


def test_is_valid_bbox_accepts_prompt_v2_fields() -> None:
    assert is_valid_bbox(
        {
            "start_time_seconds": 0.1,
            "end_time_seconds": 0.2,
            "low_frequency_hz": 20000,
            "high_frequency_hz": 60000,
        }
    )


def test_is_valid_bbox_rejects_invalid_geometry() -> None:
    assert not is_valid_bbox(
        {
            "start_time_seconds": 0.2,
            "end_time_seconds": 0.1,
            "low_frequency_hz": 20000,
            "high_frequency_hz": 60000,
        }
    )
    assert not is_valid_bbox(
        {
            "start_time_seconds": 0.1,
            "end_time_seconds": 0.2,
            "low_frequency_hz": 60000,
            "high_frequency_hz": 20000,
        }
    )


def test_prediction_quality_counts_tracks_parse_and_invalid_boxes(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "OP_001_predictions.json").write_text(
        json.dumps(
            {
                "clip_id": "OP_001",
                "parse_status": "success",
                "events": [
                    {
                        "start_time_seconds": 0.1,
                        "end_time_seconds": 0.2,
                        "low_frequency_hz": 20000,
                        "high_frequency_hz": 60000,
                    },
                    {
                        "start_time_seconds": 0.3,
                        "end_time_seconds": 0.2,
                        "low_frequency_hz": 20000,
                        "high_frequency_hz": 60000,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (prediction_dir / "OP_002_predictions.json").write_text(
        json.dumps({"clip_id": "OP_002", "parse_status": "failure", "events": []}),
        encoding="utf-8",
    )

    counts = prediction_quality_counts(prediction_dir, ("OP_001", "OP_002", "OP_003"))

    assert counts["prediction_files_found"] == 2
    assert counts["parse_success_count"] == 1
    assert counts["parse_failure_count"] == 2
    assert counts["raw_event_count"] == 2
    assert counts["invalid_bbox_count"] == 1
