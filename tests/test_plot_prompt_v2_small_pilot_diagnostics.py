import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_prompt_v2_small_pilot_diagnostics import (
    diagnostic_output_path,
    load_csv_rows,
    load_evaluation_csvs,
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
