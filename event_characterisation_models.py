"""Pydantic models for deterministic event characterisation and later evals.

Expected outputs contain only directly annotated or deterministically derived
ground truth. Behaviour-related statements are permitted only as explicitly
exploratory hypotheses and never belong in expected outputs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


BehaviourHypothesisType = Literal[
    "behaviour",
    "call_phase",
    "social_call",
    "individual_identity",
    "environment",
    "signal_quality",
    "echo_or_artifact",
    "other",
]
TruncationBasis = Literal[
    "explicit_metadata", "source_time_comparison", "unknown"
]
DensityCategory = Literal["zero", "low", "medium", "high"]


class EventBox(BaseModel):
    """Frozen event geometry supplied to the characterisation task."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    start_time_seconds: float = Field(ge=0.0)
    end_time_seconds: float = Field(gt=0.0)
    low_frequency_hz: float = Field(ge=0.0)
    high_frequency_hz: float = Field(gt=0.0)
    scientific_name: str | None = None
    scientific_name_source: Literal["direct_annotation", "unavailable"]
    source_start_time_seconds: float | None = None
    source_end_time_seconds: float | None = None
    truncation_side: Literal["none", "left", "right", "both"] | None = None

    @model_validator(mode="after")
    def validate_box(self) -> "EventBox":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        if self.high_frequency_hz <= self.low_frequency_hz:
            raise ValueError("high_frequency_hz must be greater than low_frequency_hz")
        if self.scientific_name_source == "direct_annotation" and not self.scientific_name:
            raise ValueError(
                "direct_annotation scientific_name_source requires scientific_name"
            )
        if self.scientific_name_source == "unavailable" and self.scientific_name:
            raise ValueError(
                "scientific_name must be absent when its source is unavailable"
            )
        return self


class EventCharacterisationInput(BaseModel):
    """Input shared by all four later P7B retrieval conditions."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    clip_duration_seconds: float = Field(gt=0.0)
    frozen_event_boxes: list[EventBox]
    spectrogram_path: str
    annotation_memory_enabled: bool = False
    literature_evidence_enabled: bool = False


class ExpectedEventFeatures(BaseModel):
    """Scorable direct and deterministic features for one frozen event."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    duration_ms: float = Field(gt=0.0)
    bandwidth_hz: float = Field(gt=0.0)
    temporal_center_seconds: float = Field(ge=0.0)
    frequency_center_hz: float = Field(ge=0.0)
    event_order: int = Field(ge=1)
    previous_inter_event_interval_ms: float | None
    next_inter_event_interval_ms: float | None
    clip_relative_position: float = Field(ge=0.0, le=1.0)
    left_boundary_truncated: bool
    right_boundary_truncated: bool
    boundary_truncation_known: bool
    event_overlap: bool
    overlapping_event_ids: list[str]
    scientific_name: str | None
    scientific_name_directly_annotated: bool

    @model_validator(mode="after")
    def validate_scientific_name_source(self) -> "ExpectedEventFeatures":
        if self.scientific_name_directly_annotated and not self.scientific_name:
            raise ValueError(
                "directly annotated scientific name must have a non-empty value"
            )
        if not self.scientific_name_directly_annotated and self.scientific_name:
            raise ValueError(
                "scientific name must be absent when it is not directly annotated"
            )
        return self


class ExpectedSequenceFeatures(BaseModel):
    """Scorable deterministic features for a complete clip sequence."""

    model_config = ConfigDict(extra="forbid")

    event_count: int = Field(ge=0)
    event_density_events_per_second: float = Field(ge=0.0)
    event_density_category: DensityCategory


class EventCharacterisationExpected(BaseModel):
    """Expected output containing no behavioural or rationale targets."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    events: list[ExpectedEventFeatures]
    sequence: ExpectedSequenceFeatures

    @model_validator(mode="after")
    def validate_event_count(self) -> "EventCharacterisationExpected":
        if self.sequence.event_count != len(self.events):
            raise ValueError("sequence.event_count must equal len(events)")
        return self


class RetrievedAnnotationCase(BaseModel):
    """One validated record retrieved from annotation memory."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    case_type: list[str]
    observable_features: list[str]
    known_failure_modes: list[str]
    recommended_actions: list[str]
    anti_patterns: list[str]
    evidence_paths: list[str]
    provenance: dict[str, str]


class RetrievedLiteratureEvidence(BaseModel):
    """One claim backed by a verified literature source."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    claim: str
    scope: str
    limitations: str
    source_id: str
    source_citation: str
    provenance: dict[str, str]


class ExploratoryHypothesis(BaseModel):
    """A non-ground-truth hypothesis that always requires review."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    hypothesis_type: BehaviourHypothesisType
    claim: str
    event_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    status: Literal["exploratory_hypothesis"] = "exploratory_hypothesis"
    human_review_required: Literal[True] = True


class InterpretedEvent(BaseModel):
    """Runtime interpretation of one event, including deterministic values."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    duration_ms: float = Field(gt=0.0)
    bandwidth_hz: float = Field(gt=0.0)
    temporal_center_seconds: float = Field(ge=0.0)
    frequency_center_hz: float = Field(ge=0.0)
    event_order: int = Field(ge=1)
    previous_inter_event_interval_ms: float | None
    next_inter_event_interval_ms: float | None
    clip_relative_position: float = Field(ge=0.0, le=1.0)
    left_boundary_truncated: bool
    right_boundary_truncated: bool
    event_overlap: bool
    scientific_name: str | None = None
    scientific_name_status: Literal[
        "direct_annotation", "prediction", "not_provided"
    ] = "not_provided"
    confirmed_interpretations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_scientific_name_status(self) -> "InterpretedEvent":
        if self.scientific_name_status == "not_provided" and self.scientific_name:
            raise ValueError(
                "scientific_name must be absent when status is not_provided"
            )
        if self.scientific_name_status != "not_provided" and not self.scientific_name:
            raise ValueError(
                "scientific_name is required for direct_annotation or prediction"
            )
        return self


class SequenceInterpretation(BaseModel):
    """Runtime deterministic sequence interpretation."""

    model_config = ConfigDict(extra="forbid")

    event_count: int = Field(ge=0)
    event_density_events_per_second: float = Field(ge=0.0)
    event_density_category: DensityCategory
    confirmed_interpretations: list[str] = Field(default_factory=list)


class GroundedEventInterpretation(BaseModel):
    """Eval-compatible output with exploratory claims strictly isolated."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    interpreted_events: list[InterpretedEvent]
    sequence_interpretation: SequenceInterpretation
    retrieved_annotation_cases: list[RetrievedAnnotationCase] = Field(
        default_factory=list
    )
    retrieved_literature_evidence: list[RetrievedLiteratureEvidence] = Field(
        default_factory=list
    )
    exploratory_hypotheses: list[ExploratoryHypothesis] = Field(
        default_factory=list
    )
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    human_review_needed: bool
    review_reason: str

    @model_validator(mode="after")
    def validate_review_rule(self) -> "GroundedEventInterpretation":
        if self.exploratory_hypotheses and not self.human_review_needed:
            raise ValueError(
                "exploratory hypotheses require human_review_needed=true"
            )
        if self.human_review_needed and not self.review_reason.strip():
            raise ValueError("human review requires a non-empty review_reason")
        return self


class EventCharacterisationCaseMetadata(BaseModel):
    """Metadata for stratifying Pydantic Evals cases."""

    model_config = ConfigDict(extra="forbid")

    split: Literal["diagnostic_development", "held_out_validation"]
    scenario: str
    species: str | None
    event_count: int = Field(ge=0)
    boundary_case: bool
    representative_or_heldout: Literal["representative", "heldout"]
    ground_truth_fields_available: list[str]


class EventCharacterisation(BaseModel):
    """Internal deterministic geometry and neighbourhood features."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    label: str | None = None
    start_time_seconds: float = Field(ge=0.0)
    end_time_seconds: float = Field(gt=0.0)
    low_frequency_hz: float = Field(ge=0.0)
    high_frequency_hz: float = Field(gt=0.0)
    duration_ms: float = Field(gt=0.0)
    bandwidth_hz: float = Field(gt=0.0)
    temporal_center_seconds: float = Field(ge=0.0)
    frequency_center_hz: float = Field(ge=0.0)
    event_order: int = Field(ge=1)
    previous_inter_event_interval_ms: float | None
    next_inter_event_interval_ms: float | None
    clip_relative_position: float = Field(ge=0.0, le=1.0)
    touches_left_clip_boundary: bool
    touches_right_clip_boundary: bool
    left_boundary_truncated: bool
    right_boundary_truncated: bool
    boundary_truncation_known: bool
    boundary_truncation_basis: TruncationBasis
    event_overlap: bool
    overlapping_event_ids: list[str]

    @model_validator(mode="after")
    def validate_geometry(self) -> "EventCharacterisation":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        if self.high_frequency_hz <= self.low_frequency_hz:
            raise ValueError("high_frequency_hz must be greater than low_frequency_hz")
        return self


class SequenceCharacterisation(BaseModel):
    """Internal deterministic characterisation of all events in one clip."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    clip_duration_seconds: float = Field(gt=0.0)
    event_count: int = Field(ge=0)
    event_density_events_per_second: float = Field(ge=0.0)
    event_density_category: DensityCategory
    events: list[EventCharacterisation]

    @model_validator(mode="after")
    def validate_event_count(self) -> "SequenceCharacterisation":
        if self.event_count != len(self.events):
            raise ValueError("event_count must equal the number of events")
        return self
