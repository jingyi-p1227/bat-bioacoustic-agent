import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visualization.plot_evaluation_clip_ground_truth import resolve_clip_paths


def create_eval_clip(root: Path, clip_id: str = "OP_001") -> Path:
    eval_dir = root / "eval_set"
    audio_dir = eval_dir / "audio"
    ground_truth_dir = eval_dir / "ground_truth"
    audio_dir.mkdir(parents=True)
    ground_truth_dir.mkdir(parents=True)

    audio = np.zeros(100, dtype=np.float32)
    sf.write(audio_dir / f"{clip_id}.wav", audio, 100)
    ground_truth = {
        "clip_id": clip_id,
        "clip_path": f"audio/{clip_id}.wav",
        "events": [],
    }
    (ground_truth_dir / f"{clip_id}_ground_truth.json").write_text(
        json.dumps(ground_truth),
        encoding="utf-8",
    )
    return eval_dir


def test_resolve_clip_paths_finds_expected_audio_and_ground_truth(
    tmp_path: Path,
) -> None:
    eval_dir = create_eval_clip(tmp_path)

    audio_path, ground_truth_path = resolve_clip_paths(eval_dir, "OP_001")

    assert audio_path == eval_dir / "audio/OP_001.wav"
    assert ground_truth_path == eval_dir / "ground_truth/OP_001_ground_truth.json"


def test_resolve_clip_paths_raises_clear_error_for_invalid_clip_id(
    tmp_path: Path,
) -> None:
    eval_dir = create_eval_clip(tmp_path)

    with pytest.raises(FileNotFoundError, match="Clip id 'OP_999'"):
        resolve_clip_paths(eval_dir, "OP_999")
