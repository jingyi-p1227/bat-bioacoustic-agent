import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from extract_event_characterisation_features import (
    RetrievedAnnotationCase,
    RetrievedLiteratureEvidence,
    SequenceCharacterisation,
    characterise_events,
    characterise_payload,
    load_jsonl_records,
)
from event_characterisation_models import (
    ExploratoryHypothesis,
    GroundedEventInterpretation,
    InterpretedEvent,
    SequenceInterpretation,
)


MEMORY_PATH = Path("docs/annotation_example_library/annotation_memory.jsonl")
EVIDENCE_PATH = Path(
    "docs/literature_reference_library/verified_evidence_store.jsonl"
)


def event(
    event_id: str,
    start: float,
    end: float,
    low: float = 20_000.0,
    high: float = 40_000.0,
    **extra: object,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "start_time": start,
        "end_time": end,
        "low_frequency": low,
        "high_frequency": high,
        **extra,
    }


def test_annotation_memory_records_validate() -> None:
    records = load_jsonl_records(MEMORY_PATH, RetrievedAnnotationCase)

    assert len(records) == 9
    assert {record.case_id for record in records} == {
        "OP_001",
        "OP_003",
        "OP_004",
        "OP_010",
        "OP_016",
        "OP_027",
        "OP_032",
        "OP_042",
        "OP_045",
    }
    assert all(record.evidence_paths for record in records)


def test_verified_evidence_store_contains_only_verified_records() -> None:
    records = load_jsonl_records(EVIDENCE_PATH, RetrievedLiteratureEvidence)

    assert 4 <= len(records) <= 8
    assert len({record.evidence_id for record in records}) == len(records)
    assert all(record.provenance.get("verified_url") for record in records)
    assert all(record.provenance.get("verified_on") for record in records)
    assert "TODO" not in EVIDENCE_PATH.read_text(encoding="utf-8")

    valid = RetrievedLiteratureEvidence.model_validate(
        {
            "evidence_id": "verified-example",
            "claim": "A claim checked directly against its source.",
            "scope": "Synthetic schema test only.",
            "limitations": "Not written to the runtime evidence store.",
            "source_id": "synthetic-source",
            "source_citation": "Synthetic citation for schema validation.",
            "provenance": {
                "verification_status": "synthetic_test",
                "source_card": "none",
            },
        }
    )
    assert valid.evidence_id == "verified-example"

    with pytest.raises(ValidationError):
        RetrievedLiteratureEvidence.model_validate(
            {
                "evidence_id": "missing-claim",
                "scope": "Synthetic schema test only.",
                "limitations": "Missing a required field.",
                "source_id": "synthetic-source",
                "source_citation": "Synthetic citation.",
                "provenance": {},
            }
        )


def test_duration_bandwidth_and_centers_are_deterministic() -> None:
    result = characterise_events(
        clip_id="clip",
        clip_duration_seconds=1.0,
        events=[event("event-1", 0.10, 0.15, 25_000.0, 45_000.0)],
    )
    item = result.events[0]

    assert item.duration_ms == pytest.approx(50.0)
    assert item.bandwidth_hz == pytest.approx(20_000.0)
    assert item.temporal_center_seconds == pytest.approx(0.125)
    assert item.frequency_center_hz == pytest.approx(35_000.0)
    assert item.clip_relative_position == pytest.approx(0.125)
    assert result.event_count == 1
    assert result.event_density_events_per_second == pytest.approx(1.0)
    assert result.event_density_category == "low"


def test_event_order_intervals_and_overlap() -> None:
    result = characterise_events(
        clip_id="clip",
        clip_duration_seconds=1.0,
        events=[
            event("middle", 0.20, 0.30),
            event("first", 0.00, 0.10),
            event("overlap", 0.25, 0.35),
        ],
    )

    assert [item.event_id for item in result.events] == [
        "first",
        "middle",
        "overlap",
    ]
    assert [item.event_order for item in result.events] == [1, 2, 3]
    assert result.events[0].next_inter_event_interval_ms == pytest.approx(100.0)
    assert result.events[1].previous_inter_event_interval_ms == pytest.approx(100.0)
    assert result.events[1].next_inter_event_interval_ms == pytest.approx(-50.0)
    assert result.events[2].previous_inter_event_interval_ms == pytest.approx(-50.0)
    assert result.events[1].event_overlap is True
    assert result.events[1].overlapping_event_ids == ["overlap"]
    assert result.events[2].overlapping_event_ids == ["middle"]


def test_boundary_truncation_uses_explicit_or_source_metadata() -> None:
    explicit = characterise_events(
        clip_id="explicit",
        clip_duration_seconds=1.0,
        events=[event("left", 0.0, 0.02, truncation_side="left")],
    ).events[0]
    assert explicit.touches_left_clip_boundary is True
    assert explicit.left_boundary_truncated is True
    assert explicit.right_boundary_truncated is False
    assert explicit.boundary_truncation_known is True
    assert explicit.boundary_truncation_basis == "explicit_metadata"

    source_derived = characterise_events(
        clip_id="source",
        clip_duration_seconds=1.0,
        clip_source_start_seconds=2.0,
        clip_source_end_seconds=3.0,
        events=[
            event(
                "right",
                0.98,
                1.0,
                source_start_time=2.98,
                source_end_time=3.01,
            )
        ],
    ).events[0]
    assert source_derived.right_boundary_truncated is True
    assert source_derived.boundary_truncation_basis == "source_time_comparison"

    unknown = characterise_events(
        clip_id="unknown",
        clip_duration_seconds=1.0,
        events=[event("touch", 0.0, 0.02)],
    ).events[0]
    assert unknown.touches_left_clip_boundary is True
    assert unknown.left_boundary_truncated is False
    assert unknown.boundary_truncation_known is False
    assert unknown.boundary_truncation_basis == "unknown"


def test_evaluation_set_payload_and_schema_serialisation() -> None:
    payload = {
        "clip_id": "OP_TEST",
        "source_start_time": 2.0,
        "source_end_time": 3.0,
        "events": [
            event(
                "event-1",
                0.2,
                0.21,
                label="Ozimops petersi",
                truncation_side="none",
            )
        ],
    }
    result = characterise_payload(payload)
    round_trip = SequenceCharacterisation.model_validate_json(
        result.model_dump_json()
    )

    assert round_trip == result
    assert round_trip.events[0].label == "Ozimops petersi"

    case = load_jsonl_records(MEMORY_PATH, RetrievedAnnotationCase)[0]
    event_result = result.events[0]
    interpretation = GroundedEventInterpretation(
        clip_id=result.clip_id,
        interpreted_events=[
            InterpretedEvent(
                event_id=event_result.event_id,
                duration_ms=event_result.duration_ms,
                bandwidth_hz=event_result.bandwidth_hz,
                temporal_center_seconds=event_result.temporal_center_seconds,
                frequency_center_hz=event_result.frequency_center_hz,
                event_order=event_result.event_order,
                previous_inter_event_interval_ms=(
                    event_result.previous_inter_event_interval_ms
                ),
                next_inter_event_interval_ms=(
                    event_result.next_inter_event_interval_ms
                ),
                clip_relative_position=event_result.clip_relative_position,
                left_boundary_truncated=event_result.left_boundary_truncated,
                right_boundary_truncated=event_result.right_boundary_truncated,
                event_overlap=event_result.event_overlap,
                scientific_name="Ozimops petersi",
                scientific_name_status="direct_annotation",
                confidence=1.0,
            )
        ],
        sequence_interpretation=SequenceInterpretation(
            event_count=1,
            event_density_events_per_second=1.0,
            event_density_category="low",
        ),
        retrieved_annotation_cases=[case],
        confidence=1.0,
        human_review_needed=False,
        review_reason="",
    )
    serialized = json.loads(interpretation.model_dump_json())
    assert serialized["interpreted_events"][0]["event_id"] == "event-1"
    assert serialized["exploratory_hypotheses"] == []


def test_behaviour_hypothesis_requires_human_review() -> None:
    result = characterise_events(
        clip_id="clip",
        clip_duration_seconds=1.0,
        events=[event("event-1", 0.1, 0.2)],
    )

    with pytest.raises(ValidationError, match="require human_review_needed"):
        GroundedEventInterpretation(
            clip_id="clip",
            interpreted_events=[],
            sequence_interpretation=SequenceInterpretation(
                event_count=0,
                event_density_events_per_second=0.0,
                event_density_category="zero",
            ),
            exploratory_hypotheses=[
                ExploratoryHypothesis(
                    hypothesis_id="hypothesis-1",
                    hypothesis_type="call_phase",
                    claim="Possible search-phase call",
                    confidence=0.5,
                    evidence="Exploratory only; no behavioural GT.",
                )
            ],
            confidence=0.5,
            human_review_needed=False,
            review_reason="",
        )

    accepted = GroundedEventInterpretation(
        clip_id="clip",
        interpreted_events=[],
        sequence_interpretation=SequenceInterpretation(
            event_count=0,
            event_density_events_per_second=0.0,
            event_density_category="zero",
        ),
        exploratory_hypotheses=[
            ExploratoryHypothesis(
                hypothesis_id="hypothesis-1",
                hypothesis_type="call_phase",
                claim="Possible search-phase call",
                confidence=0.5,
                evidence="Exploratory only; no behavioural GT.",
            )
        ],
        confidence=0.5,
        human_review_needed=True,
        review_reason="Behavioural hypotheses are not ground-truth evaluable.",
    )
    assert accepted.human_review_needed is True
