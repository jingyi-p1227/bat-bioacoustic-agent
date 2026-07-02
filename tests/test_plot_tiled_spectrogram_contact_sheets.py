import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_tiled_spectrogram_contact_sheets import (
    CoverageSummaryRow,
    TileManifestEntry,
    contact_sheet_path,
    summarize_gt_coverage,
    visible_event_portion,
    write_coverage_summary,
)


def make_entry(start: float, end: float, index: int) -> TileManifestEntry:
    return TileManifestEntry(
        clip_id="OP_TEST",
        tile_id=f"tile_0p5_overlap_0p1_tile_{index:03d}",
        tile_setting="tile_0p5_overlap_0p1",
        tile_start_seconds=start,
        tile_end_seconds=end,
        image_path=Path(f"tile_{index}.png"),
        original_audio_path=Path("audio.wav"),
        overlap=0.1,
        grid_style="grid_v2",
    )


def test_contact_sheet_paths_keep_gt_diagnostics_separate(tmp_path: Path) -> None:
    clean = contact_sheet_path(
        tmp_path, "OP_016", "tile_0p5_overlap_0p1", gt_diagnostic=False
    )
    diagnostic = contact_sheet_path(
        tmp_path, "OP_016", "tile_0p5_overlap_0p1", gt_diagnostic=True
    )

    assert clean == (
        tmp_path
        / "contact_sheets/OP_016_tile_0p5_overlap_0p1_clean_contact_sheet.png"
    )
    assert diagnostic == (
        tmp_path
        / "contact_sheets_gt_diagnostic_only/"
        "OP_016_tile_0p5_overlap_0p1_gt_diagnostic_contact_sheet.png"
    )


def test_visible_event_portion_clips_to_tile_and_marks_truncation() -> None:
    event = {"start_time": 0.35, "end_time": 0.45}

    portion = visible_event_portion(event, 0.4, 0.9)

    assert portion is not None
    assert portion["start_time"] == 0.4
    assert portion["end_time"] == 0.45
    assert portion["visible_fraction"] == pytest.approx(0.5)
    assert portion["tile_truncated_left"] is True
    assert portion["tile_truncated_right"] is False


def test_coverage_summary_counts_visibility_and_internal_boundary_crossing() -> None:
    entries = [make_entry(0.0, 0.5, 1), make_entry(0.4, 0.9, 2)]
    events = [
        {"start_time": 0.1, "end_time": 0.2},
        {"start_time": 0.48, "end_time": 0.52},
        {"start_time": 0.85, "end_time": 0.89},
    ]

    row = summarize_gt_coverage(
        clip_id="OP_TEST",
        tile_setting="tile_0p5_overlap_0p1",
        entries=entries,
        events=events,
    )

    assert row.total_gt_events == 3
    assert row.gt_events_visible_in_at_least_one_tile == 3
    assert row.gt_events_crossing_tile_boundary == 1
    assert row.max_visible_fraction_per_gt_event_min == 1.0


def test_write_coverage_summary_can_be_read_back(tmp_path: Path) -> None:
    path = tmp_path / "tile_gt_coverage_summary.csv"
    row = CoverageSummaryRow(
        clip_id="OP_TEST",
        tile_setting="tile_0p5_overlap_0p1",
        total_gt_events=3,
        gt_events_visible_in_at_least_one_tile=3,
        gt_events_crossing_tile_boundary=1,
        max_visible_fraction_per_gt_event_min=1.0,
        notes="All covered.",
    )

    write_coverage_summary([row], path)

    with path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["clip_id"] == "OP_TEST"
    assert saved[0]["gt_events_crossing_tile_boundary"] == "1"
    assert saved[0]["max_visible_fraction_per_gt_event_min"] == "1.0"
