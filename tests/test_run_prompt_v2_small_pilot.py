import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_prompt_v2_small_pilot as pilot


def valid_response(clip_id: str = "OP_001") -> str:
    return json.dumps(
        {
            "clip_id": clip_id,
            "events": [
                {
                    "event_id": "pred_001",
                    "start_time_seconds": 0.1,
                    "end_time_seconds": 0.2,
                    "low_frequency_hz": 30000,
                    "high_frequency_hz": 40000,
                    "label": "Ozimops petersi",
                    "confidence": 0.8,
                    "evidence": "Visible short call.",
                    "human_review_needed": False,
                    "review_reason": "",
                }
            ],
        }
    )


def create_runtime_files(root: Path, clip_id: str = "OP_001") -> tuple[Path, Path]:
    eval_dir = root / "eval"
    image_dir = root / "images"
    (eval_dir / "audio").mkdir(parents=True)
    image_dir.mkdir()
    sf.write(
        eval_dir / "audio" / f"{clip_id}.wav",
        np.zeros(4000, dtype=np.float32),
        4000,
    )
    (image_dir / f"{clip_id}_spectrogram.png").write_bytes(b"synthetic-image")
    return eval_dir, image_dir


def test_load_prompt_reads_prompt_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Annotate visible calls.", encoding="utf-8")

    assert pilot.load_prompt(prompt_path) == "Annotate visible calls."


def test_read_clip_duration_uses_wav_metadata(tmp_path: Path) -> None:
    eval_dir, _ = create_runtime_files(tmp_path)

    assert pilot.read_clip_duration(eval_dir, "OP_001") == pytest.approx(1.0)


def test_parse_prediction_handles_valid_json() -> None:
    prediction = pilot.parse_prediction(
        f"```json\n{valid_response()}\n```",
        expected_clip_id="OP_001",
    )

    assert prediction["clip_id"] == "OP_001"
    assert len(prediction["events"]) == 1


def test_invalid_json_records_parse_failure_and_keeps_raw_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_dir, image_dir = create_runtime_files(tmp_path)
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(
        pilot,
        "call_ollama_generate",
        lambda **kwargs: "not valid json",
    )

    result = pilot.run_clip(
        clip_id="OP_001",
        prompt_text="Prompt",
        eval_dir=eval_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        model_name="test-model",
        backend="ollama_generate",
        timeout=1,
        num_predict=100,
    )

    assert result.parse_status == "failed"
    assert result.output_json_path is None
    assert result.raw_response_path.read_text(encoding="utf-8") == "not valid json"
    assert result.parse_error_path is not None
    assert "JSONDecodeError" in result.parse_error_path.read_text(encoding="utf-8")
    assert not (output_dir / "OP_001_predictions.json").exists()
