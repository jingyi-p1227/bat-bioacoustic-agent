import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_model_smoke_tests import (
    SmokeRun,
    parse_run_specs,
    resolve_all_clip_ids,
    summarize_run,
    write_summary_csv,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def create_smoke_run(root: Path) -> Path:
    run_dir = root / "prompt_v2_smoke_test"
    write_json(
        run_dir / "OP_001_predictions.json",
        {
            "clip_id": "OP_001",
            "model_name": "test-model",
            "events": [{"event_id": "pred_001"}],
        },
    )
    write_json(
        run_dir / "OP_002_predictions.json",
        {
            "clip_id": "OP_002",
            "model_name": "test-model",
            "parse_status": "failed",
            "events": [],
        },
    )
    (run_dir / "OP_002_parse_error.txt").write_text("failed", encoding="utf-8")
    write_json(
        run_dir / "evaluation" / "aggregate_summary.json",
        {
            "total_ground_truth_events": 5,
            "total_predictions": 1,
            "total_tp": 1,
            "total_fp": 0,
            "total_fn": 4,
            "precision": 1.0,
            "recall": 0.2,
            "f1": 1 / 3,
            "mean_time_iou": 0.8,
            "mean_frequency_iou": 0.7,
            "mean_box_iou": 0.6,
            "strict_box_iou_0_3_count": 1,
            "strict_box_iou_0_5_count": 1,
        },
    )
    return run_dir


def test_parse_run_specs() -> None:
    runs = parse_run_specs(["gemma:grid_v1=outputs/gemma", "qwen=outputs/qwen"])

    assert runs == [
        SmokeRun("gemma", Path("outputs/gemma"), "grid_v1"),
        SmokeRun("qwen", Path("outputs/qwen")),
    ]


def test_summarize_run_counts_parse_status_and_metrics(tmp_path: Path) -> None:
    run_dir = create_smoke_run(tmp_path)

    summary = summarize_run(
        SmokeRun("prompt_v2_smoke_test", run_dir),
        ["OP_001", "OP_002"],
    )

    assert summary["model_name"] == "test-model"
    assert summary["grid_style"] == ""
    assert summary["parse_success_count"] == 1
    assert summary["parse_failure_count"] == 1
    assert summary["total_gt"] == 5
    assert summary["TP"] == 1
    assert summary["F1"] == 1 / 3


def test_write_summary_csv(tmp_path: Path) -> None:
    output_file = tmp_path / "summary.csv"
    row = summarize_run(
        SmokeRun("prompt_v2_smoke_test", create_smoke_run(tmp_path)),
        ["OP_001", "OP_002"],
    )

    write_summary_csv(output_file, [row])

    with output_file.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["run_name"] == "prompt_v2_smoke_test"
    assert rows[0]["parse_failure_count"] == "1"


def test_resolve_all_clip_ids(tmp_path: Path) -> None:
    audio_dir = tmp_path / "eval" / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "OP_002.wav").write_bytes(b"")
    (audio_dir / "OP_001.wav").write_bytes(b"")

    assert resolve_all_clip_ids(tmp_path / "eval") == ["OP_001", "OP_002"]
