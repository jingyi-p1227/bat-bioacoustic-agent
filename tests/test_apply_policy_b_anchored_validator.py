import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apply_policy_b_anchored_validator import (
    build_method_summary,
    load_clip_list,
    output_prediction_path,
    write_method_summary,
)


def test_load_clip_list_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "clips.txt"
    path.write_text("# held out\nOP_009\n\nOP_015\n", encoding="utf-8")

    assert load_clip_list(path) == ["OP_009", "OP_015"]


def test_load_clip_list_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "clips.txt"
    path.write_text("OP_009\nOP_009\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_clip_list(path)


def test_output_prediction_path() -> None:
    assert output_prediction_path(Path("outputs/run"), "OP_009") == Path(
        "outputs/run/predictions/OP_009_predictions.json"
    )


def test_method_summary_uses_canonical_aggregate_fields(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    (evaluation_dir / "aggregate_summary.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )

    rows = build_method_summary({"method": evaluation_dir})
    output = tmp_path / "summary.csv"
    write_method_summary(output, rows)

    assert rows[0]["method"] == "method"
    assert rows[0]["TP"] == 5
    assert "box_iou_gte_0_5" in output.read_text(encoding="utf-8")
