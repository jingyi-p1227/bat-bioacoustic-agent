import csv
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_stage1a_multispecies_classification import (
    aggregate_metrics,
    confusion_matrix,
    join_predictions,
    per_species_metrics,
)
from scripts.inference.run_stage1a_multispecies_classification import (
    ALLOWED_LABELS,
    build_system_prompt,
    build_user_message,
    parse_classification,
    write_results,
    RunResult,
)
from scripts.inference.run_stage1b_multispecies_classification import (
    build_system_prompt as build_stage1b_system_prompt,
    build_user_message as build_stage1b_user_message,
)
from scripts.inference.run_stage1c_multispecies_classification import (
    VERIFIED_EVIDENCE_IDS,
    build_system_prompt as build_stage1c_system_prompt,
    build_user_message as build_stage1c_user_message,
    parse_guided_classification,
)


def test_stage1a_prompt_mentions_centred_target_and_allowed_labels() -> None:
    prompt = build_system_prompt()

    assert "horizontally centred" in prompt
    assert "Choose exactly one species label" in prompt
    for label in ALLOWED_LABELS:
        assert label in prompt


def test_stage1a_user_message_does_not_include_species_label() -> None:
    row = {
        "anonymous_sample_id": "sample_000001",
        "target_center_x_fraction": "0.5",
        "species": "Ozimops petersi",
    }

    message = build_user_message(row)

    assert "sample_000001" in message
    assert "Ozimops petersi" not in message
    assert not any(label in message for label in ALLOWED_LABELS)


def test_stage1b_prompt_mentions_box_marker_and_allowed_labels() -> None:
    prompt = build_stage1b_system_prompt()

    assert "neutral box" in prompt
    assert "boxed target call" in prompt
    assert "Choose exactly one species label" in prompt
    for label in ALLOWED_LABELS:
        assert label in prompt


def test_stage1b_user_message_does_not_include_species_label() -> None:
    row = {
        "anonymous_sample_id": "sample_000001",
        "species": "Ozimops petersi",
    }

    message = build_stage1b_user_message(row)

    assert "sample_000001" in message
    assert "gt_box_marker" in message
    assert "Ozimops petersi" not in message
    assert not any(label in message for label in ALLOWED_LABELS)


def test_stage1c_prompt_contains_compact_guidance_and_source_ids() -> None:
    prompt = build_stage1c_system_prompt()

    assert "compact acoustic guidance" in prompt.lower()
    assert "provisional diagnostic guidance" in prompt.lower()
    assert "guidance_used" in prompt
    for evidence_id in VERIFIED_EVIDENCE_IDS:
        assert evidence_id in prompt


def test_stage1c_user_message_does_not_include_species_label() -> None:
    row = {
        "anonymous_sample_id": "sample_000001",
        "species": "Ozimops petersi",
    }

    message = build_stage1c_user_message(row)

    assert "sample_000001" in message
    assert "Ozimops petersi" not in message
    assert not any(label in message for label in ALLOWED_LABELS)


def test_parse_guided_classification_requires_guidance_used() -> None:
    with pytest.raises(ValueError, match="guidance_used"):
        parse_guided_classification(
            """
            {
              "predicted_species": "Plecotus auritus",
              "confidence": 0.72,
              "reasoning_brief": "The boxed call is faint.",
              "visual_evidence": ["boxed target call"]
            }
            """
        )

    parsed = parse_guided_classification(
        """
        {
          "predicted_species": "Plecotus auritus",
          "confidence": 0.72,
          "reasoning_brief": "The boxed call is faint.",
          "visual_evidence": ["boxed target call"],
          "guidance_used": ["signal quality", "call shape"]
        }
        """
    )
    assert parsed["guidance_used"] == ["signal quality", "call shape"]


def test_parse_classification_accepts_valid_json() -> None:
    parsed = parse_classification(
        """
        {
          "predicted_species": "Plecotus auritus",
          "confidence": 0.72,
          "reasoning_brief": "The centred call has a broad, low-frequency shape.",
          "visual_evidence": ["centred target call", "visible frequency sweep"]
        }
        """
    )

    assert parsed["predicted_species"] == "Plecotus auritus"
    assert parsed["confidence"] == 0.72


def test_parse_classification_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="invalid predicted_species"):
        parse_classification(
            """
            {
              "predicted_species": "Unknown bat",
              "confidence": 0.3,
              "reasoning_brief": "uncertain",
              "visual_evidence": ["faint call"]
            }
            """
        )


def test_write_results_creates_parsed_and_failure_csvs(tmp_path: Path) -> None:
    results = [
        RunResult(
            sample_id="src_001",
            anonymous_sample_id="sample_000001",
            true_species="Ozimops petersi",
            parse_status="success",
            predicted_species="Ozimops petersi",
            confidence="0.8",
            raw_response_path=tmp_path / "raw.txt",
            parse_error="",
        ),
        RunResult(
            sample_id="src_002",
            anonymous_sample_id="sample_000002",
            true_species="Plecotus auritus",
            parse_status="failed",
            predicted_species="",
            confidence="",
            raw_response_path=tmp_path / "raw2.txt",
            parse_error="JSONDecodeError",
        ),
    ]

    write_results(tmp_path, results)

    with (tmp_path / "parsed_predictions.csv").open(newline="", encoding="utf-8") as handle:
        parsed_rows = list(csv.DictReader(handle))
    with (tmp_path / "parse_failures.csv").open(newline="", encoding="utf-8") as handle:
        failure_rows = list(csv.DictReader(handle))

    assert len(parsed_rows) == 2
    assert len(failure_rows) == 1
    assert failure_rows[0]["anonymous_sample_id"] == "sample_000002"


def test_stage1a_evaluator_aggregate_and_confusion_matrix() -> None:
    manifest_rows = [
        {"anonymous_sample_id": "sample_1", "species": "Ozimops petersi"},
        {"anonymous_sample_id": "sample_2", "species": "Ozimops petersi"},
        {"anonymous_sample_id": "sample_3", "species": "Plecotus auritus"},
    ]
    prediction_rows = [
        {
            "anonymous_sample_id": "sample_1",
            "parse_status": "success",
            "predicted_species": "Ozimops petersi",
            "confidence": "0.9",
            "parse_error": "",
        },
        {
            "anonymous_sample_id": "sample_2",
            "parse_status": "success",
            "predicted_species": "Plecotus auritus",
            "confidence": "0.6",
            "parse_error": "",
        },
        {
            "anonymous_sample_id": "sample_3",
            "parse_status": "failed",
            "predicted_species": "",
            "confidence": "",
            "parse_error": "bad json",
        },
    ]

    joined = join_predictions(manifest_rows, prediction_rows)
    species_rows = per_species_metrics(joined)
    aggregate = aggregate_metrics(joined, species_rows)
    matrix = confusion_matrix(joined)

    assert aggregate["sample_count"] == 3
    assert aggregate["parse_success_count"] == 2
    assert aggregate["overall_accuracy"] == pytest.approx(1 / 3)
    ozimops = next(row for row in species_rows if row["species"] == "Ozimops petersi")
    assert ozimops["TP"] == 1
    assert ozimops["FN"] == 1
    ozimops_matrix = next(row for row in matrix if row["true_species"] == "Ozimops petersi")
    assert ozimops_matrix["Ozimops petersi"] == 1
    assert ozimops_matrix["Plecotus auritus"] == 1
