import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference import run_openrouter_multi_agent_stage2c_pilot24 as pilot


def test_assert_expected_model_rejects_substitution() -> None:
    try:
        pilot.assert_expected_model({"model": "other/model"})
    except RuntimeError as exc:
        assert "silent substitution" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_usage_row_from_response_records_provider_and_cost() -> None:
    row = pilot.usage_row_from_response(
        task="x",
        sample_id="s",
        anonymous_sample_id="sample_000001",
        agent="agent1",
        response_payload={
            "model": pilot.MODEL_NAME,
            "provider": "OpenAI",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        },
        latency=1.25,
        parse_status="success",
        parse_error="",
    )
    assert row["returned_model"] == pilot.MODEL_NAME
    assert row["provider"] == "OpenAI"
    assert row["estimated_cost_usd"] > 0


def test_final_prediction_row_preserves_selected_coordinates() -> None:
    row = {"sample_id": "s", "anonymous_sample_id": "sample_000001", "species": pilot.ALLOWED_LABELS[0]}
    proposal = {
        "proposal_id": "bd2_001",
        "start_time": "0.1",
        "end_time": "0.2",
        "low_freq": "10000",
        "high_freq": "50000",
    }
    out = pilot.final_prediction_row(
        row,
        proposal,
        {"predicted_species": pilot.ALLOWED_LABELS[0]},
        {"review_decision": "accept", "revised_species": pilot.ALLOWED_LABELS[0]},
        {"final_species": pilot.ALLOWED_LABELS[0], "review_status": "accepted"},
        ("success", "success", "success"),
        ("", "", ""),
    )
    assert out["selected_start_time"] == "0.1"
    assert out["selected_end_time"] == "0.2"
    assert out["final_correct"] == "true"


def test_comparison_metric_rows_includes_qwen_multi_agent() -> None:
    selected = [{"sample_id": "s", "anonymous_sample_id": "sample_000001", "species": pilot.ALLOWED_LABELS[0]}]
    qwen_single = {"sample_000001": {"predicted_species": pilot.ALLOWED_LABELS[1]}}
    qwen_multi = {"sample_000001": {"final_species": pilot.ALLOWED_LABELS[0]}}
    gpt_single = {"sample_000001": {"predicted_species": pilot.ALLOWED_LABELS[0]}}
    rows = pilot.comparison_metric_rows(selected, qwen_single, qwen_multi, gpt_single)
    assert rows[0]["qwen_multi_agent_correct"] == "true"
