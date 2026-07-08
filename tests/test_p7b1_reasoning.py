import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from build_event_characterisation_eval_dataset import build_dataset, inputs_for_condition
from event_characterisation_models import (
    GroundedEventInterpretation,
    GroundedLimitation,
    InterpretedEvent,
    RetrievedLiteratureEvidence,
    SequenceInterpretation,
)
from event_characterisation_retrieval import (
    retrieve_for_condition,
    safe_annotation_context,
    safe_literature_context,
)
from extract_event_characterisation_features import load_jsonl_records
from p7b1_reasoning_evaluators import (
    EvidenceScopeComplianceEvaluator,
    GroundedLimitationValidityEvaluator,
    RecommendedStrategyEvaluator,
    RetrievalRelevanceEvaluator,
    ReviewTriggerEvaluator,
    RiskFlagPrecisionEvaluator,
    RiskFlagRecallEvaluator,
    load_reasoning_references,
)
from run_p7b1_knowledge_grounded_reasoning import build_prompt, output_template


EVIDENCE_PATH = Path(
    "docs/literature_reference_library/verified_evidence_store.jsonl"
)


def case_by_id(clip_id: str):
    return next(case for case in build_dataset().cases if case.name == clip_id)


def output_from_case(clip_id: str) -> GroundedEventInterpretation:
    case = case_by_id(clip_id)
    expected = case.expected_output
    return GroundedEventInterpretation(
        clip_id=clip_id,
        interpreted_events=[
            InterpretedEvent(
                event_id=event.event_id,
                duration_ms=event.duration_ms,
                bandwidth_hz=event.bandwidth_hz,
                temporal_center_seconds=event.temporal_center_seconds,
                frequency_center_hz=event.frequency_center_hz,
                event_order=event.event_order,
                previous_inter_event_interval_ms=event.previous_inter_event_interval_ms,
                next_inter_event_interval_ms=event.next_inter_event_interval_ms,
                clip_relative_position=event.clip_relative_position,
                left_boundary_truncated=event.left_boundary_truncated,
                right_boundary_truncated=event.right_boundary_truncated,
                event_overlap=event.event_overlap,
                scientific_name=event.scientific_name,
                scientific_name_status=(
                    "direct_annotation"
                    if event.scientific_name_directly_annotated
                    else "not_provided"
                ),
                confidence=1.0,
            )
            for event in expected.events
        ],
        sequence_interpretation=SequenceInterpretation(
            event_count=expected.sequence.event_count,
            event_density_events_per_second=(
                expected.sequence.event_density_events_per_second
            ),
            event_density_category=expected.sequence.event_density_category,
        ),
        confidence=1.0,
        human_review_needed=False,
        review_reason="",
    )


def context(clip_id: str, output: GroundedEventInterpretation, condition: str):
    case = case_by_id(clip_id)
    return SimpleNamespace(
        inputs=inputs_for_condition(case.inputs, condition),
        expected_output=case.expected_output,
        output=output,
        metadata=case.metadata,
    )


def test_reasoning_enums_and_confidence_are_validated() -> None:
    output = output_from_case("OP_016")
    with pytest.raises(ValidationError):
        GroundedEventInterpretation.model_validate(
            {**output.model_dump(), "risk_flags": ["invented_risk"]}
        )
    with pytest.raises(ValidationError):
        GroundedEventInterpretation.model_validate(
            {**output.model_dump(), "reasoning_confidence": 1.1}
        )


def test_reasoning_references_load_all_frozen_cases() -> None:
    references = load_reasoning_references()
    assert set(references) == {
        "OP_001", "OP_003", "OP_004", "OP_010",
        "OP_016", "OP_045", "OP_032", "OP_042",
    }
    assert references["OP_016"].reference_type == "expert_project_derived"


def test_risk_strategy_and_review_evaluators() -> None:
    output = output_from_case("OP_016")
    output.risk_flags = ["dense_short_call_sequence", "harmful_rigid_shift"]
    output.recommended_strategy = "prefer_detector_geometry"
    output.human_review_needed = True
    output.review_reason = "Boundary truncation and conflicting tool geometry."
    ctx = context("OP_016", output, "baseline")

    assert RiskFlagPrecisionEvaluator().evaluate(ctx) == pytest.approx(1.0)
    assert RiskFlagRecallEvaluator().evaluate(ctx) == pytest.approx(0.5)
    assert RecommendedStrategyEvaluator().evaluate(ctx) is True
    assert ReviewTriggerEvaluator().evaluate(ctx) is True


def test_literature_validity_scope_and_retrieval_relevance() -> None:
    output = output_from_case("OP_016")
    records = load_jsonl_records(EVIDENCE_PATH, RetrievedLiteratureEvidence)
    evidence = next(
        item for item in records
        if item.evidence_id == "batdetect2_external_validation_required"
    )
    output.retrieved_literature_evidence = [evidence]
    output.grounded_limitations = [
        GroundedLimitation(
            limitation_id="limit-1",
            statement="Detector outputs require local validation.",
            evidence_ids=[evidence.evidence_id],
            scope="detector validation",
        )
    ]
    ctx = context("OP_016", output, "literature_only")

    assert GroundedLimitationValidityEvaluator().evaluate(ctx) == pytest.approx(1.0)
    assert EvidenceScopeComplianceEvaluator().evaluate(ctx) == pytest.approx(1.0)
    assert RetrievalRelevanceEvaluator().evaluate(ctx) == pytest.approx(1.0)


def test_prompt_never_contains_reasoning_reference_labels() -> None:
    case = case_by_id("OP_016")
    inputs = inputs_for_condition(case.inputs, "combined")
    annotation, literature, _ = retrieve_for_condition(
        condition="combined", inputs=inputs, expected=case.expected_output
    )
    system, user = build_prompt(
        inputs=inputs,
        expected=case.expected_output,
        annotation_context=[safe_annotation_context(item) for item in annotation],
        literature_context=[safe_literature_context(item) for item in literature],
    )
    prompt = f"{system}\n{user}"
    assert "reasoning_reference_labels" not in prompt
    assert "acceptable_risk_flags" not in prompt
    assert "unacceptable_risk_flags" not in prompt
    assert "review_expected" not in prompt
    assert "ground_truth" not in prompt.lower()


def test_condition_context_isolation_and_template_schema() -> None:
    case = case_by_id("OP_003")
    for condition, annotation_enabled, literature_enabled in (
        ("baseline", False, False),
        ("annotation_memory_only", True, False),
        ("literature_only", False, True),
        ("combined", True, True),
    ):
        inputs = inputs_for_condition(case.inputs, condition)
        annotation, literature, _ = retrieve_for_condition(
            condition=condition, inputs=inputs, expected=case.expected_output
        )
        assert bool(annotation) is annotation_enabled
        assert bool(literature) is literature_enabled
        template = output_template(
            expected=case.expected_output,
            annotation_context=[safe_annotation_context(item) for item in annotation],
            literature_context=[safe_literature_context(item) for item in literature],
        )
        parsed = GroundedEventInterpretation.model_validate(template)
        assert bool(parsed.retrieved_annotation_cases) is annotation_enabled
        assert bool(parsed.retrieved_literature_evidence) is literature_enabled


def test_verified_evidence_store_is_small_and_valid() -> None:
    records = load_jsonl_records(EVIDENCE_PATH, RetrievedLiteratureEvidence)
    assert 4 <= len(records) <= 8
    assert len({item.evidence_id for item in records}) == len(records)
    assert all("TODO" not in item.source_citation for item in records)

