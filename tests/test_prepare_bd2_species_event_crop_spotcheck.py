from pathlib import Path

from scripts.analysis.prepare_bd2_species_event_crop_spotcheck import (
    CropBounds,
    EventRecord,
    clean_crop_path,
    crop_bounds_for_event,
    diagnostic_crop_path,
    select_representative_events,
    two_panel_path,
    valid_box,
)
from scripts.analysis.prepare_bd2_species_visual_spotcheck import CandidateRow


def make_event(index: int) -> EventRecord:
    return EventRecord(
        event_uuid=f"event_{index}",
        event_rank=index,
        start_time=float(index) / 10,
        low_frequency=30_000.0,
        end_time=float(index) / 10 + 0.01,
        high_frequency=40_000.0,
        tag_values=("Echolocation", "Species"),
    )


def make_candidate() -> CandidateRow:
    return CandidateRow(
        species="Myotis daubentonii",
        example_index=2,
        dataset="uk",
        source="bat_conservation_ireland",
        recording_path="myotis_daubentonii_123.wav",
        event_count=10,
        duration_seconds=2.0,
        quality_hint="unmarked",
        difficulty_prior="likely_harder",
    )


def test_select_representative_events_first_middle_last() -> None:
    events = [make_event(index) for index in range(1, 11)]
    selected = select_representative_events(events, max_events=3)
    assert [event.event_rank for event in selected] == [1, 5, 10]


def test_crop_bounds_are_clamped_to_audio_and_nyquist() -> None:
    event = EventRecord(
        event_uuid="edge",
        event_rank=1,
        start_time=0.02,
        low_frequency=5_000,
        end_time=0.08,
        high_frequency=125_000,
        tag_values=(),
    )
    bounds = crop_bounds_for_event(event, duration_seconds=1.0, nyquist_hz=128_000)
    assert bounds == CropBounds(0.0, 0.18, 0.0, 128_000)


def test_valid_box_rejects_invalid_geometry() -> None:
    assert valid_box([0.1, 30_000, 0.2, 40_000])
    assert not valid_box([0.2, 30_000, 0.1, 40_000])
    assert not valid_box([0.1, 40_000, 0.2, 30_000])
    assert not valid_box(["bad", 40_000, 0.2, 30_000])


def test_event_crop_output_paths_are_separated() -> None:
    candidate = make_candidate()
    event = make_event(3)
    clean = clean_crop_path(Path("out"), candidate, event)
    panel = two_panel_path(Path("out"), candidate, event)
    diagnostic = diagnostic_crop_path(Path("out"), candidate, event)
    assert clean.as_posix().endswith("clean_event_crops/02_myotis_daubentonii_123_event_003_clean_crop.png")
    assert "two_panel_previews" in panel.as_posix()
    assert "gt_crop_overlays_human_review_only" in diagnostic.as_posix()
    assert len({clean, panel, diagnostic}) == 3
