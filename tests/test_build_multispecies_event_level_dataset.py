from pathlib import Path

import csv
import pytest
import numpy as np
from PIL import Image

from scripts.analysis.build_multispecies_event_level_dataset import (
    EventSampleInput,
    compute_padded_window,
    compute_centered_time_window,
    padded_audio_segment,
    read_event_samples_from_v1_manifest,
    read_event_samples,
    sample_id_for,
    source_recording_id,
    species_counts,
)


def make_sample() -> EventSampleInput:
    return EventSampleInput(
        species="Myotis daubentonii",
        example_index=4,
        source_dataset="uk",
        source="bat_conservation_ireland",
        source_recording="example.wav",
        event_index=12,
        event_uuid="event-12",
        event_start_time=0.45,
        event_end_time=0.47,
        event_low_freq=30_000,
        event_high_freq=60_000,
        audio_path=Path("audio/example.wav"),
        audio_duration_seconds=1.0,
        sample_rate_hz=256_000,
        candidate_event_count=12,
        quality_hint="unmarked",
        difficulty_prior="likely_harder",
    )


def test_centered_time_window_keeps_fixed_width_when_possible() -> None:
    window = compute_centered_time_window(
        event_start=0.45,
        event_end=0.47,
        audio_duration=1.0,
        half_context_seconds=0.15,
    )

    assert window.centered
    assert window.start_seconds == 0.31
    assert window.end_seconds == 0.61


def test_centered_time_window_shifts_at_left_boundary() -> None:
    window = compute_centered_time_window(
        event_start=0.01,
        event_end=0.03,
        audio_duration=1.0,
        half_context_seconds=0.15,
    )

    assert not window.centered
    assert window.start_seconds == 0.0
    assert window.end_seconds == 0.3


def test_padded_window_preserves_center_fraction_at_audio_boundary() -> None:
    window = compute_padded_window(
        event_start=0.01,
        event_end=0.03,
        audio_duration=1.0,
        half_context_seconds=0.15,
    )

    assert window.requested_start_time == -0.13
    assert window.requested_end_time == 0.17
    assert window.actual_audio_start_time == 0.0
    assert window.left_padding_seconds == 0.13
    assert window.right_padding_seconds == 0.0
    assert window.target_center_x_fraction == 0.5
    assert window.target_centered_pass


def test_padded_audio_segment_adds_left_and_right_silence() -> None:
    audio = np.ones(10, dtype=np.float32)
    window = compute_padded_window(
        event_start=0.05,
        event_end=0.05,
        audio_duration=1.0,
        half_context_seconds=0.15,
    )

    segment = padded_audio_segment(audio=audio, sample_rate=10, padded_window=window)

    assert len(segment) == 3
    assert segment[0] == 0
    assert segment[1] == 1
    assert segment[2] == 1


def test_sample_id_and_split_group_are_stable() -> None:
    sample = make_sample()

    assert sample_id_for(sample) == "myotis_daubentonii_rec04_event012"
    assert (
        source_recording_id(sample)
        == "uk_bat_conservation_ireland_example"
    )


def test_read_event_samples_filters_to_valid_written_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "review.csv"
    csv_path.write_text(
        "\n".join(
            [
                "species,example_index,dataset,source,recording_path,source_event_rank,event_uuid,event_start_time_seconds,event_end_time_seconds,event_low_frequency_hz,event_high_frequency_hz,audio_path,audio_duration_seconds,sample_rate_hz,candidate_event_count,quality_hint,difficulty_prior,audio_exists,crop_written",
                "Myotis daubentonii,1,uk,source,a.wav,2,evt,0.1,0.2,30000,60000,a.wav,1.0,256000,5,unmarked,likely_harder,true,true",
                "Myotis daubentonii,2,uk,source,b.wav,2,evt,0.2,0.1,30000,60000,b.wav,1.0,256000,5,unmarked,likely_harder,true,true",
                "Unknown species,1,uk,source,c.wav,2,evt,0.1,0.2,30000,60000,c.wav,1.0,256000,5,unmarked,unknown,true,true",
                "Myotis daubentonii,3,uk,source,d.wav,2,evt,0.1,0.2,30000,60000,d.wav,1.0,256000,5,unmarked,likely_harder,false,true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_event_samples(csv_path)

    assert len(rows) == 1
    assert rows[0].source_recording == "a.wav"


def test_read_event_samples_from_v1_manifest_preserves_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "sample_id,species,source_dataset,source,source_recording,event_index,event_uuid,event_start_time,event_end_time,event_low_freq,event_high_freq,original_audio_path,audio_duration_seconds,sample_rate_hz,candidate_event_count,quality_hint,difficulty_prior",
                "myotis_daubentonii_rec04_event012,Myotis daubentonii,uk,source,a.wav,12,evt,0.1,0.2,30000,60000,a.wav,1.0,256000,5,unmarked,likely_harder",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = read_event_samples_from_v1_manifest(manifest)

    assert len(rows) == 1
    assert rows[0].example_index == 4
    assert rows[0].event_index == 12


def test_species_counts_groups_recordings() -> None:
    sample = make_sample()
    rows = [
        {
            "species": sample.species,
            "source_recording_id": source_recording_id(sample),
            "source_dataset": "uk",
            "event_density": "medium",
            "difficulty_prior": "likely_harder",
        },
        {
            "species": sample.species,
            "source_recording_id": source_recording_id(sample),
            "source_dataset": "uk",
            "event_density": "medium",
            "difficulty_prior": "likely_harder",
        },
    ]

    counts = species_counts(rows)

    assert counts[0]["sample_count"] == 2
    assert counts[0]["recording_count"] == 1


def test_v2_generated_dataset_integrity_if_present() -> None:
    manifest_path = Path(
        "outputs/analysis_reports/multispecies_event_level_dataset_v2_centred/"
        "multispecies_event_dataset_manifest.csv"
    )
    if not manifest_path.exists():
        pytest.skip("V2 generated dataset is not present")

    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))

    assert rows
    assert all(row["target_centered_pass"] == "true" for row in rows)
    assert all(row["split_group"] and row["source_recording_id"] for row in rows)
    assert all("gt_diagnostic" not in row["image_path"] for row in rows)
    counts = {row["species"]: 0 for row in rows}
    for row in rows:
        counts[row["species"]] += 1
        with Image.open(row["image_path"]) as image:
            assert image.size == (800, 600)
    assert set(counts.values()) == {30}
