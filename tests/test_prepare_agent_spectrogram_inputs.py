import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_agent_spectrogram_inputs import (
    resolve_audio_path,
    save_clean_spectrogram,
)


def create_audio_only_eval_dir(root: Path, clip_id: str = "OP_001") -> Path:
    eval_dir = root / "eval_set"
    audio_dir = eval_dir / "audio"
    audio_dir.mkdir(parents=True)

    sample_rate = 4000
    audio = np.zeros(sample_rate, dtype=np.float32)
    sf.write(audio_dir / f"{clip_id}.wav", audio, sample_rate)
    return eval_dir


def test_resolve_audio_path_finds_valid_clip(tmp_path: Path) -> None:
    eval_dir = create_audio_only_eval_dir(tmp_path)

    audio_path = resolve_audio_path(eval_dir, "OP_001")

    assert audio_path == eval_dir / "audio/OP_001.wav"


def test_resolve_audio_path_rejects_invalid_clip_id(tmp_path: Path) -> None:
    eval_dir = create_audio_only_eval_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match="clip id 'OP_999'"):
        resolve_audio_path(eval_dir, "OP_999")


def test_save_clean_spectrogram_creates_output_directory(tmp_path: Path) -> None:
    eval_dir = create_audio_only_eval_dir(tmp_path)
    output_dir = tmp_path / "nested/agent_inputs"

    output_path = save_clean_spectrogram(
        eval_dir=eval_dir,
        clip_id="OP_001",
        output_dir=output_dir,
        max_freq_hz=2000,
    )

    assert output_dir.is_dir()
    assert output_path == output_dir / "OP_001_spectrogram.png"
    assert output_path.is_file()
