import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_single_agent_tool_use_experiments import (
    ExperimentSpec,
    build_case_highlights,
    consolidated_row,
    load_aggregate_metrics,
)


def aggregate_payload() -> dict:
    return {
        "clip_count": 2,
        "total_predictions": 7,
        "total_tp": 5,
        "total_fp": 2,
        "total_fn": 1,
        "precision": 5 / 7,
        "recall": 5 / 6,
        "f1": 10 / 13,
        "mean_time_iou": 0.5,
        "mean_frequency_iou": 0.6,
        "mean_box_iou": 0.3,
        "strict_box_iou_0_3_count": 3,
        "strict_box_iou_0_5_count": 1,
    }


def spec(evaluation_dir: Path) -> ExperimentSpec:
    return ExperimentSpec(
        "test_experiment",
        "test_group",
        "test_scope",
        "Test method",
        "test-model",
        "test input",
        True,
        True,
        False,
        evaluation_dir,
        "test note",
    )


def test_metric_loading(tmp_path: Path) -> None:
    path = tmp_path / "aggregate_summary.json"
    path.write_text(json.dumps(aggregate_payload()), encoding="utf-8")

    assert load_aggregate_metrics(path)["total_tp"] == 5


def test_missing_summary_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing for experiment_x"):
        load_aggregate_metrics(tmp_path / "missing.json", "experiment_x")


def test_consolidated_row_construction(tmp_path: Path) -> None:
    row = consolidated_row(spec(tmp_path), aggregate_payload())

    assert row["experiment_id"] == "test_experiment"
    assert row["TP"] == 5
    assert row["uses_batdetect2_proposals"] is True
    assert row["F1"] == pytest.approx(10 / 13)


def test_case_highlight_construction(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    with (evaluation_dir / "per_clip_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_id", "tp", "fp", "fn", "f1"])
        writer.writeheader()
        writer.writerow({"clip_id": "OP_016", "tp": 6, "fp": 0, "fn": 1, "f1": 0.923})

    experiment = spec(evaluation_dir)
    rows = build_case_highlights(
        (("OP_016", "dense", experiment.experiment_id, "Useful case."),),
        {experiment.experiment_id: experiment},
    )

    assert rows[0]["clip_id"] == "OP_016"
    assert rows[0]["TP"] == 6
    assert rows[0]["interpretation"] == "Useful case."
