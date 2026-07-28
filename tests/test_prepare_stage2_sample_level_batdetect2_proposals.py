from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.analysis.prepare_stage2_sample_level_batdetect2_proposals import (
    WINDOW_SECONDS,
    WindowSpec,
    build_window_specs,
    convert_raw_payload,
    evaluate_proposals,
    export_audio_windows,
    padded_audio_window,
)


def spec(**overrides: object) -> WindowSpec:
    values = {
        "sample_id": "species_rec01_event001",
        "anonymous_sample_id": "sample_000001",
        "species": "Rhinolophus hipposideros",
        "source_dataset": "uk",
        "source_recording": "example.wav",
        "source_recording_id": "uk_source_example",
        "split_group": "uk_source_example",
        "event_index": 1,
        "event_start_time": 0.02,
        "event_end_time": 0.04,
        "event_low_freq": 90_000.0,
        "event_high_freq": 110_000.0,
        "requested_start_time": -0.13,
        "requested_end_time": 0.17,
        "actual_audio_start_time": 0.0,
        "actual_audio_end_time": 0.17,
        "left_padding_seconds": 0.13,
        "right_padding_seconds": 0.0,
        "sample_rate_hz": 1000,
        "audio_duration_seconds": 1.0,
        "original_audio_path": Path("audio.wav"),
    }
    values.update(overrides)
    return WindowSpec(**values)  # type: ignore[arg-type]


def test_window_spec_converts_gt_to_local_coordinates() -> None:
    item = spec(event_start_time=0.02, event_end_time=0.04, requested_start_time=-0.13)

    assert item.local_gt_start == pytest.approx(0.15)
    assert item.local_gt_end == pytest.approx(0.17)


def test_build_window_specs_uses_v2_window_and_audio_path(tmp_path: Path) -> None:
    audio_path = tmp_path / "example.wav"
    audio_path.write_bytes(b"placeholder")
    stage1_rows = [
        {
            "sample_id": "s1",
            "anonymous_sample_id": "sample_000001",
            "species": "Plecotus auritus",
            "source_dataset": "uk",
            "source_recording": "example.wav",
            "source_recording_id": "uk_source_example",
            "split_group": "uk_source_example",
            "event_index": "2",
            "event_start_time": "0.10",
            "event_end_time": "0.12",
            "event_low_freq": "30000",
            "event_high_freq": "50000",
        }
    ]
    v2_rows = [
        {
            "sample_id": "s1",
            "requested_start_time": "-0.04",
            "requested_end_time": "0.26",
            "actual_audio_start_time": "0",
            "actual_audio_end_time": "0.26",
            "left_padding_seconds": "0.04",
            "right_padding_seconds": "0",
            "sample_rate_hz": "1000",
            "audio_duration_seconds": "1.0",
            "original_audio_path": str(audio_path),
        }
    ]

    specs = build_window_specs(stage1_rows=stage1_rows, v2_rows=v2_rows)

    assert specs[0].original_audio_path == audio_path
    assert specs[0].local_gt_start == pytest.approx(0.14)
    assert specs[0].local_gt_end == pytest.approx(0.16)


def test_padded_audio_window_adds_silence_and_preserves_duration() -> None:
    audio = np.ones(200, dtype=np.float32)
    item = spec(
        actual_audio_start_time=0.0,
        actual_audio_end_time=0.17,
        left_padding_seconds=0.13,
        right_padding_seconds=0.0,
    )

    segment = padded_audio_window(audio, 1000, item)

    assert len(segment) == int(WINDOW_SECONDS * 1000)
    assert np.all(segment[:130] == 0)
    assert np.all(segment[130:] == 1)


def test_export_audio_windows_writes_readable_wav(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.ones(1000, dtype=np.float32), 1000)
    item = spec(original_audio_path=source)

    rows = export_audio_windows([item], tmp_path / "windows")

    output = tmp_path / "windows" / "sample_000001.wav"
    audio, sample_rate = sf.read(output)
    assert rows[0]["export_status"] == "success"
    assert sample_rate == 1000
    assert len(audio) == 300


def test_convert_raw_payload_filters_and_numbers_proposals() -> None:
    raw = {
        "annotation": [
            {
                "start_time": 0.15,
                "end_time": 0.17,
                "low_freq": 90_000,
                "high_freq": 110_000,
                "det_prob": 0.9,
                "class_prob": 0.2,
                "class": "Rhinolophus hipposideros",
            },
            {
                "start_time": 0.01,
                "end_time": 0.02,
                "low_freq": 20_000,
                "high_freq": 30_000,
                "det_prob": 0.1,
                "class_prob": 0.1,
            },
        ]
    }

    payload, summary = convert_raw_payload(
        sample_id="s1",
        anonymous_sample_id="sample_000001",
        raw_payload=raw,
        min_det_prob=0.3,
    )

    assert payload["events"][0]["proposal_id"] == "bd2_001"
    assert payload["events"][0]["start_time_seconds"] == pytest.approx(0.15)
    assert summary["proposal_count"] == 1
    assert summary["below_threshold_count"] == 1


def test_evaluate_proposals_reports_match_and_false_positive(tmp_path: Path) -> None:
    item = spec(
        requested_start_time=-0.13,
        event_start_time=0.02,
        event_end_time=0.04,
        event_low_freq=90_000,
        event_high_freq=110_000,
    )
    proposal_dir = tmp_path / "proposals"
    proposal_dir.mkdir()
    payload = {
        "events": [
            {
                "proposal_id": "bd2_001",
                "start_time_seconds": 0.15,
                "end_time_seconds": 0.17,
                "low_frequency_hz": 90_000,
                "high_frequency_hz": 110_000,
                "det_prob": 0.9,
            },
            {
                "proposal_id": "bd2_002",
                "start_time_seconds": 0.01,
                "end_time_seconds": 0.02,
                "low_frequency_hz": 20_000,
                "high_frequency_hz": 30_000,
                "det_prob": 0.5,
            },
        ]
    }
    (proposal_dir / "sample_000001_batdetect2_proposals.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = evaluate_proposals(specs=[item], proposal_dir=proposal_dir)

    assert result["aggregate"]["temporal_iou_0p3"]["TP"] == 1
    assert result["aggregate"]["temporal_iou_0p3"]["FP"] == 1
    assert result["aggregate"]["temporal_iou_0p3"]["FN"] == 0
    assert result["per_species_rows"][0]["proposal_recall_iou_0p3"] == 1.0
