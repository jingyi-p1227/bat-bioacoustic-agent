import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_prompt_v2_gated_adaptive_zoom import (
    GatedClipResult,
    accepted_gated_zoom_windows,
    parse_args,
    parse_gated_view_plan,
    validate_zoom_request,
    write_view_plan_summary,
)


def gated_plan(**overrides):
    payload = {
        "clip_id": "OP_016",
        "overview_sufficient": False,
        "zoom_needed": True,
        "gating_reasons": ["dense_adjacent_calls"],
        "zoom_requests": [
            {
                "zoom_id": "zoom_001",
                "start_time_seconds": 0.0,
                "end_time_seconds": 0.25,
                "low_frequency_hz": 20000,
                "high_frequency_hz": 60000,
                "reason": "Dense calls.",
            }
        ],
        "reason": "Dense.",
    }
    payload.update(overrides)
    return payload


def test_parse_gated_view_plan_accepts_valid_plan() -> None:
    plan = parse_gated_view_plan(
        json.dumps(gated_plan()),
        expected_clip_id="OP_016",
    )

    assert plan["gating_reasons"] == ["dense_adjacent_calls"]
    assert plan["zoom_needed"] is True


def test_overview_sufficient_forces_zero_zoom() -> None:
    plan = parse_gated_view_plan(
        json.dumps(
            gated_plan(
                overview_sufficient=True,
                zoom_needed=True,
                gating_reasons=["dense_adjacent_calls"],
            )
        ),
        expected_clip_id="OP_016",
    )

    assert plan["zoom_needed"] is False
    assert plan["zoom_requests"] == []
    assert plan["gating_reasons"] == []


def test_zoom_needed_requires_valid_gating_reason() -> None:
    with pytest.raises(ValueError, match="valid gating reason"):
        parse_gated_view_plan(
            json.dumps(gated_plan(gating_reasons=["just_to_be_safe"])),
            expected_clip_id="OP_016",
        )


def test_max_zoom_two_unless_dense_or_boundary() -> None:
    requests = [
        {
            "zoom_id": f"zoom_{index:03d}",
            "start_time_seconds": 0.0,
            "end_time_seconds": 0.2,
            "low_frequency_hz": 20000,
            "high_frequency_hz": 60000,
            "reason": "Valid.",
        }
        for index in range(1, 4)
    ]
    plan = gated_plan(
        gating_reasons=["unclear_frequency_bounds"],
        zoom_requests=requests,
    )

    accepted, rejected = accepted_gated_zoom_windows(
        plan,
        clip_duration_seconds=1.0,
        max_frequency_hz=120000,
    )

    assert len(accepted) == 2
    assert rejected == 1


def test_dense_or_boundary_allows_three_zooms() -> None:
    requests = [
        {
            "zoom_id": f"zoom_{index:03d}",
            "start_time_seconds": 0.0,
            "end_time_seconds": 0.2,
            "low_frequency_hz": 20000,
            "high_frequency_hz": 60000,
            "reason": "Valid.",
        }
        for index in range(1, 4)
    ]
    plan = gated_plan(
        gating_reasons=["boundary_truncated_calls"],
        zoom_requests=requests,
    )

    accepted, rejected = accepted_gated_zoom_windows(
        plan,
        clip_duration_seconds=1.0,
        max_frequency_hz=120000,
    )

    assert len(accepted) == 3
    assert rejected == 0


def test_validate_zoom_request_clips_bounds() -> None:
    window = validate_zoom_request(
        {
            "zoom_id": "zoom_001",
            "start_time_seconds": -1,
            "end_time_seconds": 2,
            "low_frequency_hz": -100,
            "high_frequency_hz": 999999,
            "reason": "Bounds.",
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


def test_full_set_cli_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_prompt_v2_gated_adaptive_zoom.py",
            "--input-dir",
            "inputs",
            "--generated-input-dir",
            "generated",
            "--all",
            "--overview-only",
        ],
    )

    args = parse_args()

    assert args.overview_dir == Path("inputs")
    assert args.adaptive_input_dir == Path("generated")
    assert args.all is True
    assert args.overview_only is True


def test_view_plan_summary_records_zoom_disabled(tmp_path: Path) -> None:
    path = write_view_plan_summary(
        tmp_path,
        [
            GatedClipResult(
                clip_id="OP_016",
                plan_parse_success=True,
                final_parse_status="success",
                predicted_event_count=2,
                overview_sufficient=False,
                zoom_needed=True,
                gating_reasons=["dense_adjacent_calls"],
                requested_zoom_count=2,
                accepted_zoom_count=0,
                rejected_zoom_count=2,
                zoom_disabled=True,
            )
        ],
    )

    text = path.read_text(encoding="utf-8")

    assert "zoom_disabled" in text
    assert "True" in text
