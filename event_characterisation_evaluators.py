"""Deterministic Pydantic Evals evaluators for event characterisation.

No evaluator calls a model. Numeric values are compared with the expected
features generated from frozen GT geometry. TODO(P7C): manual-rubric or
LLM-as-judge evaluators may later assess rationale quality, but must remain
separate from these ground-truth evaluators.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from event_characterisation_models import (
    EventCharacterisationExpected,
    EventCharacterisationInput,
    GroundedEventInterpretation,
    RetrievedAnnotationCase,
    RetrievedLiteratureEvidence,
)


DEFAULT_ANNOTATION_MEMORY = Path(
    "docs/annotation_example_library/annotation_memory.jsonl"
)
DEFAULT_LITERATURE_EVIDENCE = Path(
    "docs/literature_reference_library/verified_evidence_store.jsonl"
)
MISSING_VALUE_ERROR = 1_000_000_000.0
BEHAVIOUR_TERMS = (
    "search phase",
    "search-phase",
    "approach phase",
    "approach-phase",
    "feeding buzz",
    "behaviour",
    "behavior",
    "social call",
    "individual identity",
    "environment",
    "signal quality",
    "echo",
    "artifact",
    "artefact",
)


def _output(value: Any) -> GroundedEventInterpretation | None:
    try:
        return GroundedEventInterpretation.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        return None


def _expected(value: Any) -> EventCharacterisationExpected | None:
    if value is None:
        return None
    try:
        return EventCharacterisationExpected.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        return None


def _event_maps(ctx: EvaluatorContext) -> tuple[dict, dict] | None:
    output = _output(ctx.output)
    expected = _expected(ctx.expected_output)
    if output is None or expected is None:
        return None
    return (
        {event.event_id: event for event in output.interpreted_events},
        {event.event_id: event for event in expected.events},
    )


def _mean_event_error(
    ctx: EvaluatorContext,
    field: str,
) -> float:
    maps = _event_maps(ctx)
    if maps is None:
        return MISSING_VALUE_ERROR
    output_events, expected_events = maps
    if set(output_events) != set(expected_events):
        return MISSING_VALUE_ERROR
    if not expected_events:
        return 0.0
    return sum(
        abs(float(getattr(output_events[event_id], field)) - float(getattr(expected, field)))
        for event_id, expected in expected_events.items()
    ) / len(expected_events)


def _load_ids(path: Path, id_field: str, model: type) -> set[str]:
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = model.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Invalid store record at {path}:{line_number}") from exc
            identifiers.add(str(getattr(record, id_field)))
    return identifiers


class OutputSchemaValidityEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return _output(ctx.output) is not None


class EventCountEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        expected = _expected(ctx.expected_output)
        return bool(
            output is not None
            and expected is not None
            and len(output.interpreted_events) == expected.sequence.event_count
            and output.sequence_interpretation.event_count
            == expected.sequence.event_count
        )


class DurationErrorEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        return _mean_event_error(ctx, "duration_ms")


class BandwidthErrorEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        return _mean_event_error(ctx, "bandwidth_hz")


class TemporalCentreErrorEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        return _mean_event_error(ctx, "temporal_center_seconds")


class FrequencyCentreErrorEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        return _mean_event_error(ctx, "frequency_center_hz")


class EventOrderEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        maps = _event_maps(ctx)
        if maps is None:
            return False
        output_events, expected_events = maps
        return set(output_events) == set(expected_events) and all(
            output_events[event_id].event_order == expected.event_order
            for event_id, expected in expected_events.items()
        )


class InterEventIntervalEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        maps = _event_maps(ctx)
        if maps is None:
            return MISSING_VALUE_ERROR
        output_events, expected_events = maps
        if set(output_events) != set(expected_events):
            return MISSING_VALUE_ERROR
        differences: list[float] = []
        for event_id, expected in expected_events.items():
            output_event = output_events[event_id]
            for field in (
                "previous_inter_event_interval_ms",
                "next_inter_event_interval_ms",
            ):
                expected_value = getattr(expected, field)
                output_value = getattr(output_event, field)
                if expected_value is None or output_value is None:
                    if expected_value != output_value:
                        return MISSING_VALUE_ERROR
                    continue
                differences.append(abs(float(output_value) - float(expected_value)))
        return sum(differences) / len(differences) if differences else 0.0


class BoundaryStatusEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        maps = _event_maps(ctx)
        if maps is None:
            return False
        output_events, expected_events = maps
        return set(output_events) == set(expected_events) and all(
            (
                output_events[event_id].left_boundary_truncated,
                output_events[event_id].right_boundary_truncated,
            )
            == (
                expected.left_boundary_truncated,
                expected.right_boundary_truncated,
            )
            for event_id, expected in expected_events.items()
        )


class EventOverlapEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        maps = _event_maps(ctx)
        if maps is None:
            return False
        output_events, expected_events = maps
        return set(output_events) == set(expected_events) and all(
            output_events[event_id].event_overlap == expected.event_overlap
            for event_id, expected in expected_events.items()
        )


class SequenceFeatureEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        expected = _expected(ctx.expected_output)
        if output is None or expected is None:
            return False
        actual = output.sequence_interpretation
        target = expected.sequence
        return (
            actual.event_count == target.event_count
            and actual.event_density_category == target.event_density_category
            and abs(
                actual.event_density_events_per_second
                - target.event_density_events_per_second
            )
            <= 1e-9
        )


@dataclass
class AnnotationCaseIdExistsEvaluator(Evaluator):
    memory_path: Path = DEFAULT_ANNOTATION_MEMORY

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        if output is None:
            return False
        valid_ids = _load_ids(self.memory_path, "case_id", RetrievedAnnotationCase)
        return all(
            record.case_id in valid_ids for record in output.retrieved_annotation_cases
        )


@dataclass
class LiteratureEvidenceIdExistsEvaluator(Evaluator):
    evidence_path: Path = DEFAULT_LITERATURE_EVIDENCE

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        if output is None:
            return False
        valid_ids = _load_ids(
            self.evidence_path, "evidence_id", RetrievedLiteratureEvidence
        )
        return all(
            record.evidence_id in valid_ids
            for record in output.retrieved_literature_evidence
        )


class TargetCaseRetrievalExclusionEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        if output is None:
            return False
        try:
            inputs = EventCharacterisationInput.model_validate(ctx.inputs)
        except (ValidationError, TypeError, ValueError):
            return False
        return all(
            record.case_id != inputs.clip_id
            for record in output.retrieved_annotation_cases
        )


class ConditionIsolationEvaluator(Evaluator):
    """Ensure disabled stores cannot appear in the runtime output."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        if output is None:
            return False
        try:
            inputs = EventCharacterisationInput.model_validate(ctx.inputs)
        except (ValidationError, TypeError, ValueError):
            return False
        if not inputs.annotation_memory_enabled and output.retrieved_annotation_cases:
            return False
        if not inputs.literature_evidence_enabled and output.retrieved_literature_evidence:
            return False
        return True


class UnsupportedBehaviourClaimEvaluator(Evaluator):
    """Reject confirmed behaviour-like claims outside exploratory hypotheses."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        if output is None:
            return False
        confirmed_text = [
            statement
            for event in output.interpreted_events
            for statement in event.confirmed_interpretations
        ]
        confirmed_text.extend(output.sequence_interpretation.confirmed_interpretations)
        confirmed_text.extend(output.unsupported_claims)
        lowered = "\n".join(confirmed_text).lower()
        return not any(
            re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered)
            for term in BEHAVIOUR_TERMS
        )


class ScientificNameGroundingEvaluator(Evaluator):
    """Require direct names to match GT; allow explicit predictions."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        maps = _event_maps(ctx)
        if maps is None:
            return False
        output_events, expected_events = maps
        if set(output_events) != set(expected_events):
            return False
        for event_id, expected in expected_events.items():
            output_event = output_events[event_id]
            if output_event.scientific_name_status == "not_provided":
                continue
            if output_event.scientific_name_status == "prediction":
                if not output_event.scientific_name:
                    return False
                continue
            if not expected.scientific_name_directly_annotated:
                return False
            if output_event.scientific_name != expected.scientific_name:
                return False
        return True


class HumanReviewRuleEvaluator(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        if output is None:
            return False
        if output.exploratory_hypotheses:
            return output.human_review_needed and bool(output.review_reason.strip())
        return True


DETERMINISTIC_EVALUATORS: tuple[type[Evaluator], ...] = (
    OutputSchemaValidityEvaluator,
    EventCountEvaluator,
    DurationErrorEvaluator,
    BandwidthErrorEvaluator,
    TemporalCentreErrorEvaluator,
    FrequencyCentreErrorEvaluator,
    EventOrderEvaluator,
    InterEventIntervalEvaluator,
    BoundaryStatusEvaluator,
    EventOverlapEvaluator,
    SequenceFeatureEvaluator,
    AnnotationCaseIdExistsEvaluator,
    LiteratureEvidenceIdExistsEvaluator,
    TargetCaseRetrievalExclusionEvaluator,
    ConditionIsolationEvaluator,
    UnsupportedBehaviourClaimEvaluator,
    ScientificNameGroundingEvaluator,
    HumanReviewRuleEvaluator,
)


# TODO(P7C): add a separate manual rubric interface for rationale quality.
# TODO(P7C): add an LLM judge only after a human-scored calibration set exists.
