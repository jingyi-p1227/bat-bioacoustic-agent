import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merge_tiled_predictions import (
    ManifestTile,
    box_iou,
    group_tiles_by_clip,
    merge_clip_predictions,
    run_merge,
    select_clip_groups,
)


def make_tile(
    clip_id: str,
    index: int,
    start: float,
    end: float,
    root: Path,
) -> ManifestTile:
    stem = (
        f"{clip_id}_tile_{index:03d}_start_{start:.6f}_end_{end:.6f}"
    ).replace(".", "p")
    return ManifestTile(
        clip_id=clip_id,
        tile_id=f"tile_0p5_overlap_0p1_tile_{index:03d}",
        tile_setting="tile_0p5_overlap_0p1",
        tile_start_seconds=start,
        tile_end_seconds=end,
        image_path=root / f"{stem}.png",
    )


def event(
    event_id: str,
    start: float,
    end: float,
    low: float,
    high: float,
    confidence: float,
) -> dict:
    return {
        "event_id": event_id,
        "start_time_seconds": start,
        "end_time_seconds": end,
        "low_frequency_hz": low,
        "high_frequency_hz": high,
        "label": "Ozimops petersi",
        "confidence": confidence,
        "evidence": "synthetic test event",
        "human_review_needed": False,
        "review_reason": "",
    }


def write_prediction(path: Path, clip_id: str, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "clip_id": clip_id,
                "prompt_version": "prompt_v2_bat_strong_label",
                "model_name": "test-model",
                "backend": "test",
                "events": events,
            }
        ),
        encoding="utf-8",
    )


def test_group_tiles_by_clip_orders_each_clip_temporally(tmp_path: Path) -> None:
    tiles = [
        make_tile("OP_002", 1, 0.0, 0.5, tmp_path),
        make_tile("OP_001", 2, 0.4, 0.9, tmp_path),
        make_tile("OP_001", 1, 0.0, 0.5, tmp_path),
    ]

    groups = group_tiles_by_clip(tiles)

    assert set(groups) == {"OP_001", "OP_002"}
    assert [tile.tile_start_seconds for tile in groups["OP_001"]] == [0.0, 0.4]


def test_select_clip_groups_restricts_merge_to_target_clip(tmp_path: Path) -> None:
    groups = group_tiles_by_clip(
        [
            make_tile("OP_001", 1, 0.0, 0.5, tmp_path),
            make_tile("OP_016", 1, 0.0, 0.5, tmp_path),
        ]
    )

    selected = select_clip_groups(groups, ["OP_016"])

    assert list(selected) == ["OP_016"]


def test_box_iou_distinguishes_duplicate_and_non_overlapping_events() -> None:
    first = event("a", 0.42, 0.46, 30_000, 35_000, 0.9)
    duplicate = event("b", 0.421, 0.461, 30_100, 35_100, 0.6)
    separate = event("c", 0.7, 0.72, 30_000, 35_000, 0.8)

    assert box_iou(first, duplicate) > 0.5
    assert box_iou(first, separate) == 0.0


def test_merge_drops_invalid_and_out_of_tile_events_and_applies_nms(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "predictions"
    tiles = [
        make_tile("OP_016", 1, 0.0, 0.5, tmp_path),
        make_tile("OP_016", 2, 0.4, 0.9, tmp_path),
    ]
    write_prediction(
        prediction_dir / f"{tiles[0].image_path.stem}.json",
        "OP_016",
        [
            event("high_confidence", 0.42, 0.46, 30_000, 35_000, 0.9),
            event("crosses_tile", 0.48, 0.55, 31_000, 36_000, 0.7),
        ],
    )
    write_prediction(
        prediction_dir / f"{tiles[1].image_path.stem}.json",
        "OP_016",
        [
            event("lower_duplicate", 0.421, 0.461, 30_100, 35_100, 0.6),
            event("separate", 0.7, 0.72, 30_000, 35_000, 0.8),
            event("invalid", 0.8, 0.75, 30_000, 35_000, 0.5),
            event("outside_tile", 0.1, 0.2, 30_000, 35_000, 0.5),
        ],
    )

    result = merge_clip_predictions(
        clip_id="OP_016",
        tiles=tiles,
        prediction_dir=prediction_dir,
    )

    assert result.payload is not None
    merged = result.payload["events"]
    assert len(merged) == 3
    assert {row["source_event_id"] for row in merged} == {
        "high_confidence",
        "crosses_tile",
        "separate",
    }
    high_confidence = next(
        row for row in merged if row["source_event_id"] == "high_confidence"
    )
    crossing = next(row for row in merged if row["source_event_id"] == "crosses_tile")
    assert len(high_confidence["merged_duplicate_provenance"]) == 1
    assert crossing["crosses_source_tile_boundary"] is True
    assert result.summary.raw_tile_events == 6
    assert result.summary.duplicates_removed == 1
    assert result.summary.invalid_events_dropped == 1
    assert result.summary.out_of_tile_events_dropped == 1


def test_merge_clips_prediction_to_original_clip_duration(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    tile = make_tile("OP_045", 1, 0.4, 0.826466, tmp_path)
    write_prediction(
        prediction_dir / f"{tile.image_path.stem}.json",
        "OP_045",
        [event("clip_end", 0.8, 0.9, 30_000, 35_000, 0.9)],
    )

    result = merge_clip_predictions(
        clip_id="OP_045",
        tiles=[tile],
        prediction_dir=prediction_dir,
    )

    assert result.payload is not None
    assert result.payload["events"][0]["end_time_seconds"] == 0.826466


def test_run_merge_writes_clip_schema_and_summary(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tile_manifest.csv"
    prediction_dir = tmp_path / "predictions"
    output_dir = tmp_path / "merged_predictions"
    summary_path = tmp_path / "merge_summary.csv"
    image_path = tmp_path / "OP_001_tile_001_start_0p000000_end_0p500000.png"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "tile_id",
                "tile_start_seconds",
                "tile_end_seconds",
                "image_path",
                "original_audio_path",
                "tile_duration",
                "overlap",
                "grid_style",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "OP_001",
                "tile_id": "tile_0p5_overlap_0p1_tile_001",
                "tile_start_seconds": 0.0,
                "tile_end_seconds": 0.5,
                "image_path": image_path.name,
                "original_audio_path": "audio/OP_001.wav",
                "tile_duration": 0.5,
                "overlap": 0.1,
                "grid_style": "grid_v2",
            }
        )
    write_prediction(
        prediction_dir / f"{image_path.stem}.json",
        "OP_001",
        [event("valid", 0.1, 0.12, 30_000, 35_000, 0.9)],
    )

    summaries = run_merge(
        manifest_path=manifest_path,
        prediction_dir=prediction_dir,
        output_dir=output_dir,
        summary_path=summary_path,
        tile_setting="tile_0p5_overlap_0p1",
        nms_iou_threshold=0.5,
        drop_fully_outside_tile=True,
        overwrite=False,
        project_root=tmp_path,
    )

    merged_path = output_dir / "OP_001_prediction.json"
    payload = json.loads(merged_path.read_text(encoding="utf-8"))
    assert payload["clip_id"] == "OP_001"
    assert isinstance(payload["events"], list)
    assert payload["events"][0]["event_id"] == "OP_001_merged_001"
    assert summaries[0].merged_events == 1
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["clip_id"] == "OP_001"
    assert summary_rows[0]["merged_events"] == "1"
