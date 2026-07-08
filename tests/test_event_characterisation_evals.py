import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic_evals import Dataset

from build_event_characterisation_eval_dataset import (
    HELD_OUT_CLIP_IDS,
    REPRESENTATIVE_CLIP_IDS,
    build_dataset,
    build_expected_output,
    inputs_for_condition,
    save_dataset_snapshot,
)
from event_characterisation_evaluators import (
    AnnotationCaseIdExistsEvaluator,
    BandwidthErrorEvaluator,
    BoundaryStatusEvaluator,
    DurationErrorEvaluator,
    EventCountEvaluator,
    EventOrderEvaluator,
    HumanReviewRuleEvaluator,
    LiteratureEvidenceIdExistsEvaluator,
    OutputSchemaValidityEvaluator,
    ScientificNameGroundingEvaluator,
    UnsupportedBehaviourClaimEvaluator,
)
from event_characterisation_models import (
    EventBox,
    EventCharacterisationCaseMetadata,
    EventCharacterisationExpected,
    EventCharacterisationInput,
    ExpectedEventFeatures,
    ExpectedSequenceFeatures,
    ExploratoryHypothesis,
    GroundedEventInterpretation,
    InterpretedEvent,
    RetrievedAnnotationCase,
    RetrievedLiteratureEvidence,
    SequenceInterpretation,
)


EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
INPUT_DIR = Path("outputs/agent_inputs/prompt_v2_full_grid_v2")
MEMORY_PATH = Path("docs/annotation_example_library/annotation_memory.jsonl")


def evaluator_context(output: object, expected: object) -> SimpleNamespace:
    return SimpleNamespace(output=output, expected_output=expected)


def output_from_expected(
    expected: EventCharacterisationExpected,
) -> GroundedEventInterpretation:
    events = [
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
    ]
    return GroundedEventInterpretation(
        clip_id=expected.clip_id,
        interpreted_events=events,
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


def test_event_box_scientific_name_validation() -> None:
    with pytest.raises(ValidationError):
        EventBox(
            event_id="event",
            start_time_seconds=0.1,
            end_time_seconds=0.2,
            low_frequency_hz=20_000,
            high_frequency_hz=40_000,
            scientific_name_source="direct_annotation",
        )


def test_expected_output_construction_is_deterministic() -> None:
    ground_truth = {
        "clip_id": "TEST",
        "source_start_time": 2.0,
        "source_end_time": 3.0,
        "events": [
            {
                "event_id": "second",
                "start_time": 0.3,
                "end_time": 0.32,
                "low_frequency": 25_000,
                "high_frequency": 45_000,
                "truncation_side": "none",
                "tags": [],
            },
            {
                "event_id": "first",
                "start_time": 0.0,
                "end_time": 0.01,
                "low_frequency": 30_000,
                "high_frequency": 35_000,
                "truncation_side": "left",
                "tags": [
                    {
                        "key": "dwc:scientificName",
                        "value": "Ozimops petersi",
                    }
                ],
            },
        ],
    }
    expected = build_expected_output(
        ground_truth=ground_truth,
        clip_duration_seconds=1.0,
    )

    assert [event.event_id for event in expected.events] == ["first", "second"]
    assert expected.events[0].duration_ms == pytest.approx(10.0)
    assert expected.events[0].bandwidth_hz == pytest.approx(5_000.0)
    assert expected.events[0].next_inter_event_interval_ms == pytest.approx(290.0)
    assert expected.events[0].left_boundary_truncated is True
    assert expected.events[0].scientific_name == "Ozimops petersi"
    assert expected.events[1].scientific_name is None


def test_dataset_cases_and_condition_settings_are_reusable() -> None:
    dataset = build_dataset(eval_dir=EVAL_DIR, input_dir=INPUT_DIR)

    assert len(dataset.cases) == 8
    assert [case.name for case in dataset.cases[:6]] == list(
        REPRESENTATIVE_CLIP_IDS
    )
    assert [case.name for case in dataset.cases[6:]] == list(HELD_OUT_CLIP_IDS)
    assert all(
        "exploratory_hypotheses" not in case.expected_output.model_dump()
        for case in dataset.cases
    )

    original = dataset.cases[0].inputs
    combined = inputs_for_condition(original, "combined")
    assert original.annotation_memory_enabled is False
    assert original.literature_evidence_enabled is False
    assert combined.annotation_memory_enabled is True
    assert combined.literature_evidence_enabled is True
    assert combined.frozen_event_boxes == original.frozen_event_boxes


def test_dataset_snapshot_is_json_serialisable(tmp_path: Path) -> None:
    dataset = build_dataset(eval_dir=EVAL_DIR, input_dir=INPUT_DIR)
    save_dataset_snapshot(dataset, tmp_path)

    payload = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (tmp_path / "schema_version.json").read_text(encoding="utf-8")
    )
    assert len(payload["cases"]) == 8
    assert metadata["case_count"] == 8
    assert (tmp_path / "case_summary.csv").is_file()
    assert (tmp_path / "dataset_schema.json").is_file()

    typed_dataset = Dataset[
        EventCharacterisationInput,
        EventCharacterisationExpected,
        EventCharacterisationCaseMetadata,
    ].from_file(tmp_path / "dataset.json")
    assert len(typed_dataset.cases) == 8
    assert isinstance(typed_dataset.cases[0].inputs, EventCharacterisationInput)
    assert isinstance(
        typed_dataset.cases[0].expected_output, EventCharacterisationExpected
    )
    assert isinstance(
        typed_dataset.cases[0].metadata, EventCharacterisationCaseMetadata
    )


def test_deterministic_evaluator_result_types_and_exact_values() -> None:
    dataset = build_dataset(eval_dir=EVAL_DIR, input_dir=INPUT_DIR)
    expected = dataset.cases[0].expected_output
    output = output_from_expected(expected)
    ctx = evaluator_context(output, expected)

    assert OutputSchemaValidityEvaluator().evaluate(ctx) is True
    assert EventCountEvaluator().evaluate(ctx) is True
    assert DurationErrorEvaluator().evaluate(ctx) == pytest.approx(0.0)
    assert BandwidthErrorEvaluator().evaluate(ctx) == pytest.approx(0.0)
    assert EventOrderEvaluator().evaluate(ctx) is True
    assert BoundaryStatusEvaluator().evaluate(ctx) is True
    assert isinstance(DurationErrorEvaluator().evaluate(ctx), float)
    assert isinstance(EventCountEvaluator().evaluate(ctx), bool)


def test_annotation_case_id_exists_evaluator() -> None:
    dataset = build_dataset(eval_dir=EVAL_DIR, input_dir=INPUT_DIR)
    expected = dataset.cases[0].expected_output
    output = output_from_expected(expected)
    first_record = RetrievedAnnotationCase.model_validate_json(
        MEMORY_PATH.read_text(encoding="utf-8").splitlines()[0]
    )
    output.retrieved_annotation_cases = [first_record]
    ctx = evaluator_context(output, expected)
    evaluator = AnnotationCaseIdExistsEvaluator(memory_path=MEMORY_PATH)
    assert evaluator.evaluate(ctx) is True

    output.retrieved_annotation_cases = [
        first_record.model_copy(update={"case_id": "missing-case"})
    ]
    assert evaluator.evaluate(ctx) is False


def test_literature_evidence_id_exists_evaluator(tmp_path: Path) -> None:
    dataset = build_dataset(eval_dir=EVAL_DIR, input_dir=INPUT_DIR)
    expected = dataset.cases[0].expected_output
    output = output_from_expected(expected)
    evidence = RetrievedLiteratureEvidence(
        evidence_id="evidence-1",
        claim="Verified synthetic test claim.",
        scope="Test only.",
        limitations="Not runtime literature evidence.",
        source_id="test-source",
        source_citation="Synthetic test citation.",
        provenance={"verification_status": "synthetic_test"},
    )
    store = tmp_path / "evidence.jsonl"
    store.write_text(evidence.model_dump_json() + "\n", encoding="utf-8")
    evaluator = LiteratureEvidenceIdExistsEvaluator(evidence_path=store)

    output.retrieved_literature_evidence = [evidence]
    assert evaluator.evaluate(evaluator_context(output, expected)) is True
    output.retrieved_literature_evidence = [
        evidence.model_copy(update={"evidence_id": "missing-evidence"})
    ]
    assert evaluator.evaluate(evaluator_context(output, expected)) is False


def test_unsupported_behaviour_and_exploratory_review_rules() -> None:
    dataset = build_dataset(eval_dir=EVAL_DIR, input_dir=INPUT_DIR)
    expected = dataset.cases[0].expected_output
    output = output_from_expected(expected)
    output.interpreted_events[0].confirmed_interpretations = [
        "This is a search-phase call."
    ]
    ctx = evaluator_context(output, expected)
    assert UnsupportedBehaviourClaimEvaluator().evaluate(ctx) is False

    output.interpreted_events[0].confirmed_interpretations = []
    output.exploratory_hypotheses = [
        ExploratoryHypothesis(
            hypothesis_id="hypothesis-1",
            hypothesis_type="call_phase",
            event_id=output.interpreted_events[0].event_id,
            claim="Possible search-phase call.",
            confidence=0.4,
            evidence="Exploratory only.",
        )
    ]
    output.human_review_needed = True
    output.review_reason = "Behaviour is not ground-truth evaluable."
    assert UnsupportedBehaviourClaimEvaluator().evaluate(ctx) is True
    assert HumanReviewRuleEvaluator().evaluate(ctx) is True

    payload = output.model_dump()
    payload["human_review_needed"] = False
    with pytest.raises(ValidationError):
        GroundedEventInterpretation.model_validate(payload)


def test_missing_scientific_name_must_not_be_claimed_as_direct() -> None:
    expected = EventCharacterisationExpected(
        clip_id="missing-species",
        events=[
            ExpectedEventFeatures(
                event_id="event-1",
                duration_ms=10.0,
                bandwidth_hz=10_000.0,
                temporal_center_seconds=0.1,
                frequency_center_hz=30_000.0,
                event_order=1,
                previous_inter_event_interval_ms=None,
                next_inter_event_interval_ms=None,
                clip_relative_position=0.1,
                left_boundary_truncated=False,
                right_boundary_truncated=False,
                boundary_truncation_known=False,
                event_overlap=False,
                overlapping_event_ids=[],
                scientific_name=None,
                scientific_name_directly_annotated=False,
            )
        ],
        sequence=ExpectedSequenceFeatures(
            event_count=1,
            event_density_events_per_second=1.0,
            event_density_category="low",
        ),
    )
    output = output_from_expected(expected)
    output.interpreted_events[0].scientific_name = "Ozimops petersi"
    output.interpreted_events[0].scientific_name_status = "direct_annotation"
    evaluator = ScientificNameGroundingEvaluator()
    assert evaluator.evaluate(evaluator_context(output, expected)) is False

    output.interpreted_events[0].scientific_name_status = "prediction"
    assert evaluator.evaluate(evaluator_context(output, expected)) is True
