import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_prompt_v2_tiled_pilot as tiled
from merge_tiled_predictions import ManifestTile


def make_tile(tmp_path: Path, clip_id: str = "OP_016") -> ManifestTile:
    image_path = tmp_path / "OP_016_tile_002_start_0p400000_end_0p900000.png"
    image_path.write_bytes(b"synthetic image")
    return ManifestTile(
        clip_id=clip_id,
        tile_id="tile_0p5_overlap_0p1_tile_002",
        tile_setting="tile_0p5_overlap_0p1",
        tile_start_seconds=0.4,
        tile_end_seconds=0.9,
        image_path=image_path,
    )


def valid_response() -> str:
    return json.dumps(
        {
            "clip_id": "OP_016",
            "events": [
                {
                    "event_id": "tile_pred_001",
                    "start_time_seconds": 0.48,
                    "end_time_seconds": 0.50,
                    "low_frequency_hz": 30000,
                    "high_frequency_hz": 36000,
                    "label": "Ozimops petersi",
                    "confidence": 0.8,
                    "evidence": "Visible call.",
                    "human_review_needed": False,
                    "review_reason": "",
                }
            ],
        }
    )


def test_tile_user_message_requires_original_coordinates(tmp_path: Path) -> None:
    message = tiled.build_tile_user_message(
        tile=make_tile(tmp_path),
        clip_duration_seconds=1.0,
    )

    assert '"tile_start_seconds": 0.4' in message
    assert '"tile_end_seconds": 0.9' in message
    assert '"time_coordinate_frame": "original_clip"' in message
    assert "do not restart time at 0" in message


def test_select_tiles_groups_by_requested_clip_and_sets_duration(tmp_path: Path) -> None:
    first = make_tile(tmp_path)
    second = ManifestTile(
        clip_id="OP_016",
        tile_id="tile_0p5_overlap_0p1_tile_003",
        tile_setting="tile_0p5_overlap_0p1",
        tile_start_seconds=0.8,
        tile_end_seconds=1.0,
        image_path=tmp_path / "tile_003.png",
    )

    selected = tiled.select_tiles([second, first], ["OP_016"])

    assert [tile.tile_id for tile, _ in selected] == [
        "tile_0p5_overlap_0p1_tile_002",
        "tile_0p5_overlap_0p1_tile_003",
    ]
    assert all(duration == 1.0 for _, duration in selected)


def test_run_tile_writes_manifest_compatible_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile = make_tile(tmp_path)
    output_dir = tmp_path / "predictions"
    monkeypatch.setattr(tiled, "call_ollama_generate", lambda **kwargs: valid_response())

    result = tiled.run_tile(
        tile=tile,
        clip_duration_seconds=1.0,
        prompt_text="Prompt",
        output_dir=output_dir,
        model_name="qwen3.6:latest",
        timeout=1,
        num_predict=100,
        overwrite=False,
    )

    payload = json.loads(result.prediction_path.read_text(encoding="utf-8"))
    assert result.parse_status == "success"
    assert result.predicted_event_count == 1
    assert result.prediction_path.name == f"{tile.image_path.stem}.json"
    assert payload["clip_id"] == "OP_016"
    assert payload["tile_id"] == tile.tile_id
    assert payload["time_coordinate_frame"] == "original_clip"
    assert payload["events"][0]["start_time_seconds"] == 0.48


def test_run_tile_records_parse_failure_without_backend_call_in_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tile = make_tile(tmp_path)
    monkeypatch.setattr(tiled, "call_ollama_generate", lambda **kwargs: "invalid")

    result = tiled.run_tile(
        tile=tile,
        clip_duration_seconds=1.0,
        prompt_text="Prompt",
        output_dir=tmp_path / "predictions",
        model_name="qwen3.6:latest",
        timeout=1,
        num_predict=100,
        overwrite=False,
    )

    payload = json.loads(result.prediction_path.read_text(encoding="utf-8"))
    assert result.parse_status == "failed"
    assert result.parse_error_path is not None
    assert payload["events"] == []
    assert payload["parse_status"] == "failed"

