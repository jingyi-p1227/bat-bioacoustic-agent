import csv
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visualization.plot_prompt_v2_small_pilot_diagnostics import (
    diagnostic_output_path,
    load_csv_rows,
    load_evaluation_csvs,
    resolve_all_clip_ids,
    resolve_eval_output_dir,
    resolve_output_dir,
    resolve_prediction_path,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_diagnostic_output_path() -> None:
    output_dir = Path("outputs/evaluation/diagnostic_figures")

    assert diagnostic_output_path(output_dir, "OP_001") == (
        output_dir / "OP_001_diagnostic_overlay.png"
    )


def test_resolve_prediction_path_accepts_merged_singular_name(tmp_path: Path) -> None:
    path = tmp_path / "OP_016_prediction.json"
    path.write_text("{}", encoding="utf-8")

    assert resolve_prediction_path(tmp_path, "OP_016") == path


def test_load_csv_rows_handles_header_only_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    write_csv(path, ["clip_id", "prediction_id"], [])

    assert load_csv_rows(path) == []


def test_load_evaluation_csvs(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "per_clip_metrics.csv",
        ["clip_id", "tp"],
        [{"clip_id": "OP_001", "tp": "2"}],
    )
    write_csv(
        tmp_path / "matched_events.csv",
        ["clip_id", "prediction_id"],
        [{"clip_id": "OP_001", "prediction_id": "pred_001"}],
    )
    write_csv(
        tmp_path / "unmatched_predictions.csv",
        ["clip_id", "prediction_id"],
        [],
    )
    write_csv(
        tmp_path / "missed_ground_truth_events.csv",
        ["clip_id", "ground_truth_event_id"],
        [],
    )

    rows = load_evaluation_csvs(tmp_path)

    assert rows["per_clip"][0]["clip_id"] == "OP_001"
    assert rows["matched"][0]["prediction_id"] == "pred_001"
    assert rows["unmatched"] == []
    assert rows["missed"] == []


def test_resolve_all_clip_ids_uses_per_clip_metrics() -> None:
    rows = {
        "per_clip": [
            {"clip_id": "OP_010"},
            {"clip_id": "OP_001"},
            {"clip_id": "OP_010"},
        ]
    }

    assert resolve_all_clip_ids(rows) == ["OP_001", "OP_010"]


def test_resolve_eval_output_dir_prefers_new_cli_name() -> None:
    args = Namespace(
        eval_output_dir=Path("outputs/run/evaluation"),
        evaluation_dir=Path("outputs/old/evaluation"),
    )

    assert resolve_eval_output_dir(args) == Path("outputs/run/evaluation")


def test_resolve_output_dir_defaults_under_eval_output_dir() -> None:
    args = Namespace(output_dir=None)
    eval_output_dir = Path("outputs/run/evaluation")

    assert resolve_output_dir(args, eval_output_dir) == (
        eval_output_dir / "diagnostic_figures"
    )
