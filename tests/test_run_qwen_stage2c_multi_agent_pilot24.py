import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference import run_qwen_stage2c_multi_agent_pilot24 as pilot


def test_select_pilot_samples_prefers_qwen_wrong_with_selected_proposals() -> None:
    rows = []
    selected = {}
    qwen = {}
    species = pilot.ALLOWED_LABELS[0]
    for index in range(5):
        anon = f"sample_{index:06d}"
        rows.append({"anonymous_sample_id": anon, "sample_id": f"s{index}", "species": species})
        selected[anon] = {"proposal_id": "bd2_001"}
        qwen[anon] = {
            "parse_status": "success",
            "predicted_species": species if index >= 3 else pilot.ALLOWED_LABELS[1],
        }
    chosen = pilot.select_pilot_samples(rows, selected, qwen, per_species=3)
    assert [row["anonymous_sample_id"] for row in chosen] == [
        "sample_000000",
        "sample_000001",
        "sample_000002",
    ]


def test_parse_agent2_rejects_invalid_review_decision() -> None:
    raw = json.dumps(
        {
            "review_decision": "maybe",
            "revised_species": pilot.ALLOWED_LABELS[0],
            "confidence": 0.5,
            "reasoning_brief": "x",
            "human_review_recommended": True,
        }
    )
    with pytest.raises(ValueError, match="review_decision"):
        pilot.parse_agent2(raw)


def test_parse_agent3_validates_forced_choice_species() -> None:
    raw = json.dumps(
        {
            "final_species": pilot.ALLOWED_LABELS[0],
            "confidence": 0.7,
            "review_status": "accepted",
            "human_review_recommended": False,
            "reasoning_brief": "consistent evidence",
        }
    )
    parsed = pilot.parse_agent3(raw)
    assert parsed["final_species"] == pilot.ALLOWED_LABELS[0]
    assert parsed["review_status"] == "accepted"


def test_aggregate_metrics_uses_all_allowed_species_for_macro_f1() -> None:
    rows = [
        {"true_species": pilot.ALLOWED_LABELS[0], "predicted": pilot.ALLOWED_LABELS[0]},
        {"true_species": pilot.ALLOWED_LABELS[1], "predicted": pilot.ALLOWED_LABELS[0]},
    ]
    metrics = pilot.aggregate_metrics(rows, "predicted")
    assert metrics["accuracy"] == 0.5
    assert 0.0 < metrics["macro_F1"] < 1.0


def test_confusion_matrix_has_allowed_species_columns() -> None:
    rows = [{"true_species": pilot.ALLOWED_LABELS[0], "predicted": pilot.ALLOWED_LABELS[1]}]
    matrix = pilot.confusion_matrix(rows, "predicted")
    assert len(matrix) == len(pilot.ALLOWED_LABELS)
    assert matrix[0][pilot.ALLOWED_LABELS[1]] == 1
