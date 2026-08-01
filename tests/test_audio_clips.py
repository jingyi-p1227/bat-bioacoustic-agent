import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_prep.split_audio_clips import split_audio_file


def write_test_wav(path: Path, sample_count: int, sample_rate: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.linspace(-0.5, 0.5, sample_count, dtype=np.float32)
    sf.write(str(path), audio, sample_rate)
    return path


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_fixed_duration_split_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = write_test_wav(Path("audio/source.wav"), sample_count=25, sample_rate=10)

    rows, _ = split_audio_file(source, clip_duration_seconds=1.0)

    assert len(rows) == 3
    assert [row.start_sample for row in rows] == [0, 10, 20]


def test_final_partial_clip_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = write_test_wav(Path("audio/source.wav"), sample_count=25, sample_rate=10)

    rows, _ = split_audio_file(source, clip_duration_seconds=1.0)

    assert rows[-1].start_sample == 20
    assert rows[-1].end_sample == 25
    assert rows[-1].clip_start_time_seconds == 2.0
    assert rows[-1].clip_end_time_seconds == 2.5


def test_metadata_preserves_original_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = write_test_wav(Path("audio/source.wav"), sample_count=12, sample_rate=4)

    rows, _ = split_audio_file(source, clip_duration_seconds=1.0)

    assert rows[0].clip_start_time_seconds == 0.0
    assert rows[0].clip_end_time_seconds == 1.0
    assert rows[1].clip_start_time_seconds == 1.0
    assert rows[1].clip_end_time_seconds == 2.0
    assert rows[2].clip_start_time_seconds == 2.0
    assert rows[2].clip_end_time_seconds == 3.0


def test_output_clip_paths_are_inside_source_clip_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = write_test_wav(Path("audio/source.wav"), sample_count=12, sample_rate=4)

    rows, _ = split_audio_file(source, clip_duration_seconds=1.0)

    for row in rows:
        assert row.output_clip_path.startswith("audio/clips/source/")
        assert Path(row.output_clip_path).exists()


def test_refuses_to_overwrite_existing_clips_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = write_test_wav(Path("audio/source.wav"), sample_count=12, sample_rate=4)
    _, manifest_path = split_audio_file(source, clip_duration_seconds=1.0)
    manifest_path.unlink()

    with pytest.raises(FileExistsError, match="existing clip"):
        split_audio_file(source, clip_duration_seconds=1.0)


def test_jsonl_manifest_can_be_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = write_test_wav(Path("audio/source.wav"), sample_count=12, sample_rate=4)

    rows, manifest_path = split_audio_file(source, clip_duration_seconds=1.0)
    manifest_rows = read_manifest(manifest_path)

    assert len(manifest_rows) == len(rows)
    assert manifest_rows[0]["source_audio_path"] == "audio/source.wav"
    assert manifest_rows[0]["clip_index"] == 0
    assert manifest_rows[0]["start_sample"] == 0
    assert manifest_rows[0]["end_sample"] == 4
    assert manifest_rows[0]["output_clip_path"] == "audio/clips/source/source_clip_0000.wav"
