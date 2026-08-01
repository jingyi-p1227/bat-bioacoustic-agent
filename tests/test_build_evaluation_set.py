import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_prep.build_evaluation_set import build_evaluation_set, load_project


def create_synthetic_dataset(root: Path) -> Path:
    dataset_dir = root / "dataset"
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True)

    sample_rate = 10
    audio = np.linspace(-0.5, 0.5, 25, dtype=np.float32)
    sf.write(audio_dir / "pseudo_petersi_001.wav", audio, sample_rate)

    payload = {
        "version": "1.1.0",
        "data": {
            "tags": [
                {"id": 0, "key": "soundevent:call_type", "value": "Echolocation"},
                {"id": 1, "key": "dwc:scientificName", "value": "Ozimops petersi"},
            ],
            "recordings": [
                {
                    "uuid": "recording-1",
                    "path": "pseudo_petersi_001.wav",
                    "duration": 2.5,
                    "channels": 1,
                    "samplerate": sample_rate,
                }
            ],
            "sound_events": [
                {
                    "uuid": "event-crossing",
                    "recording": "recording-1",
                    "geometry": {
                        "type": "BoundingBox",
                        "coordinates": [0.81234567, 20000, 1.23456789, 40000],
                    },
                },
                {
                    "uuid": "event-final",
                    "recording": "recording-1",
                    "geometry": {
                        "type": "BoundingBox",
                        "coordinates": [2.21234567, 25000, 2.45678912, 45000],
                    },
                },
            ],
            "sound_event_annotations": [
                {
                    "uuid": "annotation-1",
                    "sound_event": "event-crossing",
                    "tags": [0, 1],
                },
                {
                    "uuid": "annotation-2",
                    "sound_event": "event-final",
                    "tags": [0, 1],
                },
            ],
        },
    }
    (dataset_dir / "annotations.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return dataset_dir


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_loads_small_synthetic_annotation_json(tmp_path: Path) -> None:
    dataset_dir = create_synthetic_dataset(tmp_path)

    project = load_project(dataset_dir)

    assert len(project["recordings"]) == 1
    assert len(project["sound_events"]) == 2


def test_overlapping_events_use_clip_relative_times(tmp_path: Path) -> None:
    dataset_dir = create_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "evaluation_set"

    build_evaluation_set(dataset_dir=dataset_dir, output_dir=output_dir)

    first_clip = read_json(output_dir / "ground_truth/OP_001_ground_truth.json")
    second_clip = read_json(output_dir / "ground_truth/OP_002_ground_truth.json")
    first_event = first_clip["events"][0]
    second_event = second_clip["events"][0]

    assert first_event["event_id"] == "event-crossing"
    assert first_event["start_time"] == pytest.approx(0.812346)
    assert first_event["end_time"] == pytest.approx(1.0)
    assert second_event["start_time"] == pytest.approx(0.0)
    assert second_event["end_time"] == pytest.approx(0.234568)
    assert second_event["source_start_time"] == pytest.approx(0.812346)
    assert second_event["source_end_time"] == pytest.approx(1.234568)
    assert second_event["tags"][1]["value"] == "Ozimops petersi"


def test_event_truncation_metadata(tmp_path: Path) -> None:
    dataset_dir = create_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "evaluation_set"

    build_evaluation_set(dataset_dir=dataset_dir, output_dir=output_dir)

    first_clip = read_json(output_dir / "ground_truth/OP_001_ground_truth.json")
    second_clip = read_json(output_dir / "ground_truth/OP_002_ground_truth.json")
    final_clip = read_json(output_dir / "ground_truth/OP_003_ground_truth.json")

    right_truncated = first_clip["events"][0]
    left_truncated = second_clip["events"][0]
    non_truncated = final_clip["events"][0]

    assert right_truncated["is_truncated_by_clip_boundary"] is True
    assert right_truncated["truncation_side"] == "right"
    assert left_truncated["is_truncated_by_clip_boundary"] is True
    assert left_truncated["truncation_side"] == "left"
    assert non_truncated["is_truncated_by_clip_boundary"] is False
    assert non_truncated["truncation_side"] == "none"


def test_written_time_values_are_rounded_to_six_places(tmp_path: Path) -> None:
    dataset_dir = create_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "evaluation_set"

    build_evaluation_set(dataset_dir=dataset_dir, output_dir=output_dir)

    first_clip = read_json(output_dir / "ground_truth/OP_001_ground_truth.json")
    final_clip = read_json(output_dir / "ground_truth/OP_003_ground_truth.json")
    rows = read_manifest(output_dir / "manifest.csv")

    assert first_clip["events"][0]["start_time"] == 0.812346
    assert first_clip["events"][0]["source_end_time"] == 1.234568
    assert final_clip["events"][0]["start_time"] == 0.212346
    assert final_clip["events"][0]["end_time"] == 0.456789
    assert rows[-1]["source_end_time"] == "2.5"
    assert rows[-1]["clip_duration"] == "0.5"


def test_manifest_rows_are_portable_and_classified(tmp_path: Path) -> None:
    dataset_dir = create_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "evaluation_set"

    summary = build_evaluation_set(dataset_dir=dataset_dir, output_dir=output_dir)
    rows = read_manifest(output_dir / "manifest.csv")

    assert summary["clip_count"] == 3
    assert rows[0]["clip_id"] == "OP_001"
    assert rows[0]["clip_path"] == "audio/OP_001.wav"
    assert rows[0]["ground_truth_path"] == "ground_truth/OP_001_ground_truth.json"
    assert rows[0]["source_recording"] == "pseudo_petersi_001.wav"
    assert rows[0]["has_target_event"] == "true"
    assert rows[0]["num_gt_events"] == "1"
    assert rows[0]["event_density"] == "low"
    assert rows[0]["auto_scenario"] == "low_activity"
    assert not Path(rows[0]["clip_path"]).is_absolute()


def test_final_partial_clip_is_preserved(tmp_path: Path) -> None:
    dataset_dir = create_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "evaluation_set"

    build_evaluation_set(dataset_dir=dataset_dir, output_dir=output_dir)

    rows = read_manifest(output_dir / "manifest.csv")
    final_row = rows[-1]
    audio_info = sf.info(output_dir / final_row["clip_path"])

    assert float(final_row["source_start_time"]) == pytest.approx(2.0)
    assert float(final_row["source_end_time"]) == pytest.approx(2.5)
    assert float(final_row["clip_duration"]) == pytest.approx(0.5)
    assert audio_info.duration == pytest.approx(0.5)


def test_refuses_existing_output_directory_without_overwrite(tmp_path: Path) -> None:
    dataset_dir = create_synthetic_dataset(tmp_path)
    output_dir = tmp_path / "evaluation_set"
    build_evaluation_set(dataset_dir=dataset_dir, output_dir=output_dir)

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_evaluation_set(dataset_dir=dataset_dir, output_dir=output_dir)
