import json
from pathlib import Path

from pydantic_evals import Case, Dataset

from build_event_characterisation_eval_dataset import (
    build_dataset,
    inputs_for_condition,
)
from event_characterisation_evaluators import DETERMINISTIC_EVALUATORS
from event_characterisation_models import (
    GroundedEventInterpretation,
    InterpretedEvent,
    SequenceInterpretation,
)
from event_characterisation_retrieval import (
    RetrievalTrace,
    retrieve_for_condition,
    safe_annotation_context,
    write_retrieval_trace,
)
from run_event_characterisation_ablation import (
    build_prompt,
    dereference_json_schema,
    output_template,
    parse_prediction,
)


def case_by_id(clip_id: str) -> Case:
    return next(case for case in build_dataset().cases if case.name == clip_id)


def output_from_case(case: Case) -> GroundedEventInterpretation:
    expected = case.expected_output
    return GroundedEventInterpretation(
        clip_id=case.name,
        interpreted_events=[
            InterpretedEvent(
                event_id=event.event_id,
                duration_ms=event.duration_ms,
                bandwidth_hz=event.bandwidth_hz,
                temporal_center_seconds=event.temporal_center_seconds,
                frequency_center_hz=event.frequency_center_hz,
                event_order=event.event_order,
                previous_inter_event_interval_ms=(
                    event.previous_inter_event_interval_ms
                ),
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


def test_target_case_exclusion_and_annotation_ranking() -> None:
    case = case_by_id("OP_016")
    inputs = inputs_for_condition(case.inputs, "annotation_memory_only")
    records, evidence, trace = retrieve_for_condition(
        condition="annotation_memory_only",
        inputs=inputs,
        expected=case.expected_output,
    )

    assert len(records) == 2
    assert evidence == []
    assert trace.target_case_excluded is True
    assert "OP_016" not in [match.retrieved_id for match in trace.annotation_matches]
    assert trace.annotation_matches[0].retrieval_score >= trace.annotation_matches[1].retrieval_score
    assert trace.annotation_matches[0].retrieved_id == "OP_003"
    assert "boundary" in " ".join(trace.annotation_matches[0].match_reasons)


def test_evidence_retrieval_uses_only_verified_store() -> None:
    case = case_by_id("OP_032")
    inputs = inputs_for_condition(case.inputs, "literature_only")
    records, evidence, trace = retrieve_for_condition(
        condition="literature_only",
        inputs=inputs,
        expected=case.expected_output,
    )

    assert records == []
    assert evidence
    assert trace.literature_matches
    assert all(item.evidence_id for item in evidence)
    assert [item.evidence_id for item in evidence] == [
        item.retrieved_id for item in trace.literature_matches
    ]


def test_condition_isolation() -> None:
    case = case_by_id("OP_003")
    for condition, expected_annotation, expected_literature in (
        ("baseline", False, False),
        ("annotation_memory_only", True, False),
        ("literature_only", False, True),
        ("combined", True, True),
    ):
        inputs = inputs_for_condition(case.inputs, condition)
        records, evidence, trace = retrieve_for_condition(
            condition=condition,
            inputs=inputs,
            expected=case.expected_output,
        )
        assert inputs.annotation_memory_enabled is expected_annotation
        assert inputs.literature_evidence_enabled is expected_literature
        assert bool(records) is expected_annotation
        assert bool(evidence) is expected_literature
        assert trace.annotation_memory_enabled is expected_annotation
        assert trace.literature_evidence_enabled is expected_literature


def test_prompt_construction_has_no_gt_or_diagnostic_leakage() -> None:
    case = case_by_id("OP_016")
    inputs = inputs_for_condition(case.inputs, "annotation_memory_only")
    records, _, _ = retrieve_for_condition(
        condition="annotation_memory_only",
        inputs=inputs,
        expected=case.expected_output,
    )
    system, user = build_prompt(
        condition="annotation_memory_only",
        inputs=inputs,
        expected=case.expected_output,
        annotation_context=[safe_annotation_context(record) for record in records],
        literature_context=[],
    )
    combined = f"{system}\n{user}".lower()

    assert "immutable_deterministic_features" in combined
    assert "do not detect new events" in combined
    assert "gt_overlay" not in combined
    assert "diagnostic" not in combined
    assert "ground_truth" not in combined
    assert "outputs/evaluation_sets" not in combined
    assert "outputs/agent_runs" not in combined

    template = output_template(
        expected=case.expected_output,
        annotation_context=[safe_annotation_context(record) for record in records],
        literature_context=[],
    )
    assert GroundedEventInterpretation.model_validate(template).clip_id == "OP_016"

    schema = dereference_json_schema(
        GroundedEventInterpretation.model_json_schema()
    )
    assert "$defs" not in json.dumps(schema)
    assert "$ref" not in json.dumps(schema)


def test_prediction_parsing_and_trace_serialisation(tmp_path: Path) -> None:
    case = case_by_id("OP_001")
    output = output_from_case(case)
    raw = f"```json\n{output.model_dump_json()}\n```"
    parsed = parse_prediction(raw, "OP_001")
    assert parsed.clip_id == "OP_001"

    inputs = inputs_for_condition(case.inputs, "baseline")
    _, _, trace = retrieve_for_condition(
        condition="baseline",
        inputs=inputs,
        expected=case.expected_output,
    )
    path = tmp_path / "trace.json"
    write_retrieval_trace(trace, path)
    assert RetrievalTrace.model_validate_json(path.read_text()) == trace


def test_pydantic_evaluator_integration() -> None:
    case = case_by_id("OP_001")
    output = output_from_case(case)
    dataset = Dataset(
        name="integration",
        cases=[
            Case(
                name=case.name,
                inputs=case.inputs,
                expected_output=case.expected_output,
                metadata=case.metadata,
            )
        ],
        evaluators=[evaluator() for evaluator in DETERMINISTIC_EVALUATORS],
    )
    report = dataset.evaluate_sync(lambda _: json.loads(output.model_dump_json()))

    assert report.failures == []
    assert report.cases[0].evaluator_failures == []
    assert report.cases[0].assertions["OutputSchemaValidityEvaluator"].value is True
    assert report.cases[0].scores["DurationErrorEvaluator"].value == 0.0
