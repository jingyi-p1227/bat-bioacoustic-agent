"""Deterministic evaluators for P7B.1 knowledge-grounded reasoning.

Reference labels are loaded only by evaluation code. They are deliberately
absent from prompt construction and retrieval inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from event_characterisation_models import (
    EventCharacterisationInput,
    GroundedEventInterpretation,
    RecommendedStrategy,
    RetrievedLiteratureEvidence,
    RiskFlag,
)
from extract_event_characterisation_features import load_jsonl_records


DEFAULT_REASONING_REFERENCES = Path(
    "outputs/evaluation_sets/event_characterisation_v1/reasoning_reference_labels.json"
)
DEFAULT_EVIDENCE_STORE = Path(
    "docs/literature_reference_library/verified_evidence_store.jsonl"
)


class ReasoningReferenceLabel(BaseModel):
    """Project-derived scoring reference kept outside model-visible Cases."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    acceptable_risk_flags: list[RiskFlag]
    unacceptable_risk_flags: list[RiskFlag]
    acceptable_strategies: list[RecommendedStrategy] = Field(min_length=1)
    review_expected: bool
    review_reasons: list[str]
    acceptable_annotation_case_types: list[str]
    acceptable_literature_evidence_ids: list[str]
    provenance: list[str]
    reference_type: Literal["deterministic", "expert_project_derived"]


class ReasoningReferenceSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    description: str
    cases: list[ReasoningReferenceLabel]


def load_reasoning_references(
    path: Path = DEFAULT_REASONING_REFERENCES,
) -> dict[str, ReasoningReferenceLabel]:
    payload = ReasoningReferenceSet.model_validate_json(path.read_text(encoding="utf-8"))
    references = {item.clip_id: item for item in payload.cases}
    if len(references) != len(payload.cases):
        raise ValueError("reasoning reference labels contain duplicate clip_id values")
    return references


def _output(value: Any) -> GroundedEventInterpretation | None:
    try:
        return GroundedEventInterpretation.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        return None


def _inputs(value: Any) -> EventCharacterisationInput | None:
    try:
        return EventCharacterisationInput.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        return None


@dataclass
class _ReferenceEvaluator:
    reference_path: Path = DEFAULT_REASONING_REFERENCES

    def reference(self, ctx: EvaluatorContext) -> ReasoningReferenceLabel | None:
        inputs = _inputs(ctx.inputs)
        if inputs is None:
            return None
        return load_reasoning_references(self.reference_path).get(inputs.clip_id)


class RiskFlagPrecisionEvaluator(_ReferenceEvaluator, Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        output = _output(ctx.output)
        reference = self.reference(ctx)
        if output is None or reference is None:
            return 0.0
        predicted = set(output.risk_flags)
        acceptable = set(reference.acceptable_risk_flags)
        return len(predicted & acceptable) / len(predicted) if predicted else 0.0


class RiskFlagRecallEvaluator(_ReferenceEvaluator, Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        output = _output(ctx.output)
        reference = self.reference(ctx)
        if output is None or reference is None:
            return 0.0
        acceptable = set(reference.acceptable_risk_flags)
        if not acceptable:
            return 1.0
        return len(set(output.risk_flags) & acceptable) / len(acceptable)


class RecommendedStrategyEvaluator(_ReferenceEvaluator, Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        reference = self.reference(ctx)
        return bool(
            output is not None
            and reference is not None
            and output.recommended_strategy in reference.acceptable_strategies
        )


@dataclass
class GroundedLimitationValidityEvaluator(Evaluator):
    evidence_path: Path = DEFAULT_EVIDENCE_STORE

    def evaluate(self, ctx: EvaluatorContext) -> float:
        output = _output(ctx.output)
        inputs = _inputs(ctx.inputs)
        if output is None or inputs is None:
            return 0.0
        limitations = output.grounded_limitations
        if not limitations:
            return 1.0 if not inputs.literature_evidence_enabled else 0.0
        evidence = {
            item.evidence_id: item
            for item in load_jsonl_records(
                self.evidence_path, RetrievedLiteratureEvidence
            )
        }
        valid = 0
        for limitation in limitations:
            identifiers = set(limitation.evidence_ids)
            if (
                limitation.statement.strip()
                and limitation.scope.strip()
                and identifiers
                and identifiers <= set(evidence)
            ):
                valid += 1
        return valid / len(limitations)


class EvidenceScopeComplianceEvaluator(_ReferenceEvaluator, Evaluator):
    """Check citations against retrieved and manually acceptable evidence IDs."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        output = _output(ctx.output)
        inputs = _inputs(ctx.inputs)
        reference = self.reference(ctx)
        if output is None or inputs is None or reference is None:
            return 0.0
        limitations = output.grounded_limitations
        if not limitations:
            return 1.0 if not inputs.literature_evidence_enabled else 0.0
        retrieved = {
            item.evidence_id for item in output.retrieved_literature_evidence
        }
        acceptable = set(reference.acceptable_literature_evidence_ids)
        compliant = 0
        for limitation in limitations:
            identifiers = set(limitation.evidence_ids)
            if identifiers and identifiers <= retrieved and identifiers <= acceptable:
                compliant += 1
        return compliant / len(limitations)


class ReviewTriggerEvaluator(_ReferenceEvaluator, Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = _output(ctx.output)
        reference = self.reference(ctx)
        return bool(
            output is not None
            and reference is not None
            and output.human_review_needed == reference.review_expected
            and (not output.human_review_needed or output.review_reason.strip())
        )


class RetrievalRelevanceEvaluator(_ReferenceEvaluator, Evaluator):
    """Score retrieved records against manually curated acceptable categories."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        output = _output(ctx.output)
        inputs = _inputs(ctx.inputs)
        reference = self.reference(ctx)
        if output is None or inputs is None or reference is None:
            return 0.0
        judgements: list[bool] = []
        acceptable_types = set(reference.acceptable_annotation_case_types)
        for item in output.retrieved_annotation_cases:
            judgements.append(bool(set(item.case_type) & acceptable_types))
        acceptable_evidence = set(reference.acceptable_literature_evidence_ids)
        for item in output.retrieved_literature_evidence:
            judgements.append(item.evidence_id in acceptable_evidence)
        if not judgements:
            return 1.0 if not (
                inputs.annotation_memory_enabled or inputs.literature_evidence_enabled
            ) else 0.0
        return sum(judgements) / len(judgements)


REASONING_EVALUATORS: tuple[type[Evaluator], ...] = (
    RiskFlagPrecisionEvaluator,
    RiskFlagRecallEvaluator,
    RecommendedStrategyEvaluator,
    GroundedLimitationValidityEvaluator,
    EvidenceScopeComplianceEvaluator,
    ReviewTriggerEvaluator,
    RetrievalRelevanceEvaluator,
)
