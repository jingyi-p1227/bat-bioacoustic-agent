import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_prep.prepare_tiled_spectrogram_inputs import (
    TileManifestRow,
    generate_tile_windows,
    tile_image_name,
    write_tile_manifest,
)


def test_generate_tile_windows_with_overlap_and_partial_final_window() -> None:
    windows = generate_tile_windows(
        total_samples=1000,
        sample_rate_hz=1000,
        tile_duration_seconds=0.5,
        overlap_seconds=0.1,
    )

    assert [(window.start_sample, window.end_sample) for window in windows] == [
        (0, 500),
        (400, 900),
        (800, 1000),
    ]


def test_tile_windows_stay_inside_source_clip_coordinates() -> None:
    windows = generate_tile_windows(
        total_samples=827,
        sample_rate_hz=1000,
        tile_duration_seconds=0.25,
        overlap_seconds=0.05,
    )

    assert len(windows) == 4
    assert windows[0].start_seconds == 0
    assert windows[-1].end_seconds == pytest.approx(0.827)
    assert all(0 <= window.start_sample < window.end_sample <= 827 for window in windows)


def test_tile_image_name_contains_clip_tile_and_time_bounds() -> None:
    window = generate_tile_windows(
        total_samples=1000,
        sample_rate_hz=1000,
        tile_duration_seconds=0.5,
        overlap_seconds=0.1,
    )[1]

    assert tile_image_name("OP_016", window) == (
        "OP_016_tile_002_start_0p400000_end_0p900000.png"
    )


def test_write_tile_manifest_can_be_read_back(tmp_path: Path) -> None:
    manifest_path = tmp_path / "nested/tile_manifest.csv"
    row = TileManifestRow(
        clip_id="OP_016",
        tile_id="tile_0p5_overlap_0p1_tile_001",
        tile_start_seconds=0.0,
        tile_end_seconds=0.5,
        image_path="outputs/tiles/OP_016_tile_001.png",
        original_audio_path="outputs/eval/audio/OP_016.wav",
        tile_duration=0.5,
        overlap=0.1,
        grid_style="grid_v2",
    )

    write_tile_manifest([row], manifest_path)

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))
    assert saved_rows[0]["clip_id"] == "OP_016"
    assert saved_rows[0]["tile_start_seconds"] == "0.0"
    assert saved_rows[0]["grid_style"] == "grid_v2"


def test_generate_tile_windows_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_seconds"):
        generate_tile_windows(
            total_samples=1000,
            sample_rate_hz=1000,
            tile_duration_seconds=0.5,
            overlap_seconds=0.5,
        )

