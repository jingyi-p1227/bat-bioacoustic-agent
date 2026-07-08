"""Transparent retrieval for P7 event-characterisation conditions.

Retrieval is deterministic and rule based. Target cases are always excluded.
Prompt-facing annotation records omit evidence paths and provenance so GT and
diagnostic artifacts can never become model inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from event_characterisation_models import (
    EventCharacterisationExpected,
    EventCharacterisationInput,
    RetrievedAnnotationCase,
    RetrievedLiteratureEvidence,
)
from extract_event_characterisation_features import load_jsonl_records


DEFAULT_ANNOTATION_MEMORY = Path(
    "docs/annotation_example_library/annotation_memory.jsonl"
)
DEFAULT_LITERATURE_EVIDENCE = Path(
    "docs/literature_reference_library/verified_evidence_store.jsonl"
)
ConditionName = Literal[
    "baseline", "annotation_memory_only", "literature_only", "combined"
]
RETRIEVAL_STOPWORDS = {
    "call",
    "calls",
    "case",
    "clip",
    "complete",
    "event",
    "events",
    "multi",
}


class RetrievalMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieved_id: str
    retrieval_score: float = Field(ge=0.0)
    match_reasons: list[str]


class RetrievalTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    condition: ConditionName
    annotation_memory_enabled: bool
    literature_evidence_enabled: bool
    annotation_matches: list[RetrievalMatch]
    literature_matches: list[RetrievalMatch]
    annotation_store_version: str
    literature_store_version: str
    target_case_excluded: bool


def store_version(path: Path) -> str:
    """Return a stable short SHA-256 version for one source store."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def _tokens(values: list[str]) -> set[str]:
    text = " ".join(values).lower().replace("0.5", "half_second")
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.replace("_", " "))
        if len(token) > 2 and token not in RETRIEVAL_STOPWORDS
    }


def query_features(
    inputs: EventCharacterisationInput,
    expected: EventCharacterisationExpected,
) -> list[str]:
    """Create non-behavioural retrieval descriptors from deterministic inputs."""

    descriptors = [
        expected.sequence.event_density_category,
        "multi event" if expected.sequence.event_count > 1 else "single event",
    ]
    if expected.sequence.event_density_category == "high":
        descriptors.extend(["dense", "multi event"])
    if any(event.left_boundary_truncated for event in expected.events):
        descriptors.extend(["left boundary", "truncated"])
    if any(event.right_boundary_truncated for event in expected.events):
        descriptors.extend(["right boundary", "truncated"])
    if expected.events and (
        sum(event.duration_ms for event in expected.events) / len(expected.events) < 15
    ):
        descriptors.extend(["short call", "timing"])
    if inputs.clip_duration_seconds < 0.999:
        descriptors.extend(["partial clip", "clean case regression"])
    if any(event.event_overlap for event in expected.events):
        descriptors.append("overlap")
    return descriptors


def retrieve_annotation_memory(
    *,
    target_case_id: str,
    descriptors: list[str],
    memory_path: Path = DEFAULT_ANNOTATION_MEMORY,
    top_k: int = 2,
) -> tuple[list[RetrievedAnnotationCase], list[RetrievalMatch]]:
    """Rank annotation cases by transparent token overlap and scenario bonuses."""

    records = load_jsonl_records(memory_path, RetrievedAnnotationCase)
    query = _tokens(descriptors)
    ranked: list[tuple[float, str, RetrievedAnnotationCase, list[str]]] = []
    for record in records:
        if record.case_id == target_case_id:
            continue
        type_tokens = _tokens(record.case_type)
        observable_tokens = _tokens(record.observable_features)
        failure_tokens = _tokens(record.known_failure_modes)
        type_overlap = sorted(query & type_tokens)
        observable_overlap = sorted(query & observable_tokens)
        failure_overlap = sorted(query & failure_tokens)
        score = (
            3.0 * len(type_overlap)
            + 2.0 * len(observable_overlap)
            + len(failure_overlap)
        )
        reasons: list[str] = []
        if type_overlap:
            reasons.append(f"case_type overlap: {', '.join(type_overlap)}")
        if observable_overlap:
            reasons.append(
                f"observable overlap: {', '.join(observable_overlap)}"
            )
        if failure_overlap:
            reasons.append(
                f"failure-mode overlap: {', '.join(failure_overlap)}"
            )
        if not reasons:
            reasons.append("fallback case with no exact descriptor overlap")
        ranked.append((score, record.case_id, record, reasons))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[:top_k]
    return (
        [item[2] for item in selected],
        [
            RetrievalMatch(
                retrieved_id=item[1],
                retrieval_score=item[0],
                match_reasons=item[3],
            )
            for item in selected
        ],
    )


def retrieve_literature_evidence(
    *,
    descriptors: list[str],
    evidence_path: Path = DEFAULT_LITERATURE_EVIDENCE,
    top_k: int = 3,
) -> tuple[list[RetrievedLiteratureEvidence], list[RetrievalMatch]]:
    """Rank only records already present in the verified evidence store."""

    records = load_jsonl_records(evidence_path, RetrievedLiteratureEvidence)
    query = _tokens(descriptors + ["event characterisation", "tool validation"])
    ranked: list[
        tuple[float, str, RetrievedLiteratureEvidence, list[str]]
    ] = []
    for record in records:
        scope_tokens = _tokens([record.scope])
        claim_tokens = _tokens([record.claim])
        overlap = sorted(query & (scope_tokens | claim_tokens))
        score = 2.0 * len(query & scope_tokens) + len(query & claim_tokens)
        reasons = (
            [f"scope/claim overlap: {', '.join(overlap)}"]
            if overlap
            else ["fallback verified evidence with no exact descriptor overlap"]
        )
        ranked.append((score, record.evidence_id, record, reasons))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[:top_k]
    return (
        [item[2] for item in selected],
        [
            RetrievalMatch(
                retrieved_id=item[1],
                retrieval_score=item[0],
                match_reasons=item[3],
            )
            for item in selected
        ],
    )


def retrieve_for_condition(
    *,
    condition: ConditionName,
    inputs: EventCharacterisationInput,
    expected: EventCharacterisationExpected,
    memory_path: Path = DEFAULT_ANNOTATION_MEMORY,
    evidence_path: Path = DEFAULT_LITERATURE_EVIDENCE,
) -> tuple[
    list[RetrievedAnnotationCase],
    list[RetrievedLiteratureEvidence],
    RetrievalTrace,
]:
    """Retrieve condition-specific context and a serialisable trace."""

    descriptors = query_features(inputs, expected)
    annotation_records: list[RetrievedAnnotationCase] = []
    annotation_matches: list[RetrievalMatch] = []
    literature_records: list[RetrievedLiteratureEvidence] = []
    literature_matches: list[RetrievalMatch] = []
    if inputs.annotation_memory_enabled:
        annotation_records, annotation_matches = retrieve_annotation_memory(
            target_case_id=inputs.clip_id,
            descriptors=descriptors,
            memory_path=memory_path,
        )
    if inputs.literature_evidence_enabled:
        literature_records, literature_matches = retrieve_literature_evidence(
            descriptors=descriptors,
            evidence_path=evidence_path,
        )
    trace = RetrievalTrace(
        clip_id=inputs.clip_id,
        condition=condition,
        annotation_memory_enabled=inputs.annotation_memory_enabled,
        literature_evidence_enabled=inputs.literature_evidence_enabled,
        annotation_matches=annotation_matches,
        literature_matches=literature_matches,
        annotation_store_version=store_version(memory_path),
        literature_store_version=store_version(evidence_path),
        target_case_excluded=all(
            match.retrieved_id != inputs.clip_id for match in annotation_matches
        ),
    )
    return annotation_records, literature_records, trace


def safe_annotation_context(record: RetrievedAnnotationCase) -> dict:
    """Return prompt-safe memory fields, excluding paths and provenance."""

    return {
        "case_id": record.case_id,
        "case_type": record.case_type,
        "observable_features": record.observable_features,
        "known_failure_modes": record.known_failure_modes,
        "recommended_actions": record.recommended_actions,
        "anti_patterns": record.anti_patterns,
        "evidence_paths": [],
        "provenance": {
            "source": "annotation_memory",
            "leakage_policy": "target_excluded_paths_removed",
        },
    }


def safe_literature_context(record: RetrievedLiteratureEvidence) -> dict:
    """Return the complete verified literature record for grounded citation."""

    return record.model_dump(mode="json")


def write_retrieval_trace(trace: RetrievalTrace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trace.model_dump_json(indent=2) + "\n", encoding="utf-8")
