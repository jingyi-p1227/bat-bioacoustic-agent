import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_prompt_v2_adaptive_zoom import (
    accepted_zoom_windows,
    composite_output_path,
    overview_output_path,
    parse_view_plan,
    validate_zoom_request,
)


def test_parse_view_plan_validates_basic_schema() -> None:
    raw = json.dumps(
        {
            "clip_id": "OP_001",
            "preferred_grid": "grid_v2",
            "zoom_needed": True,
            "zoom_requests": [],
            "reason": "Need a closer view.",
        }
    )

    plan = parse_view_plan(raw, expected_clip_id="OP_001")

    assert plan["preferred_grid"] == "grid_v2"
    assert plan["zoom_needed"] is True


def test_validate_zoom_request_clips_to_bounds() -> None:
    window = validate_zoom_request(
        {
            "zoom_id": "zoom_001",
            "start_time_seconds": -0.5,
            "end_time_seconds": 2.0,
            "low_frequency_hz": -100,
            "high_frequency_hz": 200000,
            "reason": "Bounds test.",
        },
        index=0,
        clip_duration_seconds=1.0,
        max_frequency_hz=120000,
    )

    assert window is not None
    assert window.start_time_seconds == 0.0
    assert window.end_time_seconds == 1.0
    assert window.low_frequency_hz == 0.0
    assert window.high_frequency_hz == 120000


def test_validate_zoom_request_rejects_degenerate_window() -> None:
    assert (
        validate_zoom_request(
            {
                "zoom_id": "zoom_001",
                "start_time_seconds": 0.5,
                "end_time_seconds": 0.5,
                "low_frequency_hz": 20000,
                "high_frequency_hz": 60000,
                "reason": "Bad time.",
            },
            index=0,
            clip_duration_seconds=1.0,
            max_frequency_hz=120000,
        )
        is None
    )


def test_accepted_zoom_windows_limits_to_three_and_counts_rejections() -> None:
    plan = {
        "zoom_requests": [
            {
                "zoom_id": f"zoom_{index:03d}",
                "start_time_seconds": 0.0,
                "end_time_seconds": 0.2,
                "low_frequency_hz": 20000,
                "high_frequency_hz": 60000,
                "reason": "Valid.",
            }
            for index in range(1, 5)
        ]
    }

    accepted, rejected = accepted_zoom_windows(
        plan,
        clip_duration_seconds=1.0,
        max_frequency_hz=120000,
    )

    assert len(accepted) == 3
    assert rejected == 1


def test_adaptive_output_paths() -> None:
    output_dir = Path("outputs/adaptive")

    assert overview_output_path(output_dir, "OP_016") == (
        output_dir / "OP_016_overview_grid_v2.png"
    )
    assert composite_output_path(output_dir, "OP_016") == (
        output_dir / "OP_016_composite.png"
    )
