from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from toy_audio_agent.evaluation.event_matching import MatchingProtocol
from toy_audio_agent.experiments.p9_light import (
    DISALLOWED_EVENT_FIELDS,
    OPTIONAL_AGENT_CONDITION,
    PROTOCOLS,
    TARGET_CLIPS,
    build_prompt,
    check_output_schema_fields,
    clip_descriptors,
    load_json,
    parse_prediction,
    proposal_prediction_events,
    retrieve_annotation_memory,
    validate_walters_card,
    walters_prompt_insert,
    write_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WALTERS_CARD = REPO_ROOT / "docs/acoustic_reference_library/walters_2012_generic_acoustic_parameter_guidance.json"


def test_walters_guidance_card_is_generic_not_op_prior() -> None:
    card = load_json(WALTERS_CARD)
    validate_walters_card(card)
    assert card["not_species_specific"] is True
    assert card["not_op_prior"] is True
    assert card["no_numeric_species_ranges"] is True
    assert card["no_european_numeric_transfer"] is True
    assert card["status"] == "usable_for_generic_guidance"


def test_walters_prompt_insert_has_no_numeric_species_ranges() -> None:
    text = walters_prompt_insert(load_json(WALTERS_CARD))
    forbidden_fragments = ["20 kHz", "40 kHz", "8 ms", "10 ms", "Ozimops petersi range"]
    assert all(fragment not in text for fragment in forbidden_fragments)
    assert "not an Ozimops petersi prior" in text
    assert "no expected numeric range" in text


def test_output_schema_rejects_disallowed_fields() -> None:
    payload = {
        "clip_id": "OP_001",
        "events": [
            {
                "event_id": "e1",
                "start_time": 0.1,
                "end_time": 0.2,
                "low_frequency": 20000.0,
                "high_frequency": 50000.0,
                "species": "Ozimops petersi",
            }
        ],
    }
    with pytest.raises(ValueError):
        check_output_schema_fields(payload)
    assert "species" in DISALLOWED_EVENT_FIELDS


def test_parse_prediction_accepts_simple_geometry_only_schema() -> None:
    raw = json.dumps(
        {
            "clip_id": "OP_001",
            "events": [
                {
                    "event_id": "e1",
                    "start_time": 0.1,
                    "end_time": 0.2,
                    "low_frequency": 20000.0,
                    "high_frequency": 50000.0,
                    "linked_proposal_id": "bd2_001",
                    "brief_reason": "fits proposal",
                }
            ],
        }
    )
    parsed = parse_prediction(raw, clip_id="OP_001", clip_duration=1.0)
    assert parsed.events[0].linked_proposal_id == "bd2_001"


def test_parse_failure_is_non_evaluable_by_status_convention() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_prediction("not json", clip_id="OP_001", clip_duration=1.0)


def test_all_protocols_include_p8_detection_views() -> None:
    assert PROTOCOLS == (
        MatchingProtocol.TEMPORAL_IOU_0_1,
        MatchingProtocol.TEMPORAL_IOU_0_3,
        MatchingProtocol.START_TIME_PROXIMITY_10MS,
    )


def test_target_clip_set_is_exactly_16_not_full45() -> None:
    assert len(TARGET_CLIPS) == 16
    assert TARGET_CLIPS == (
        "OP_001",
        "OP_003",
        "OP_004",
        "OP_010",
        "OP_016",
        "OP_045",
        "OP_009",
        "OP_015",
        "OP_018",
        "OP_020",
        "OP_025",
        "OP_027",
        "OP_032",
        "OP_036",
        "OP_041",
        "OP_042",
    )


def test_annotation_memory_retrieval_excludes_target_clip(tmp_path: Path) -> None:
    memory = tmp_path / "memory.jsonl"
    memory.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "OP_016",
                        "case_type": ["dense_short_call_sequence"],
                        "observable_features": ["dense"],
                        "known_failure_modes": [],
                        "recommended_actions": [],
                        "anti_patterns": [],
                        "evidence_paths": [],
                        "provenance": {},
                    }
                ),
                json.dumps(
                    {
                        "case_id": "OP_010",
                        "case_type": ["dense_multi_event"],
                        "observable_features": ["dense"],
                        "known_failure_modes": [],
                        "recommended_actions": [],
                        "anti_patterns": [],
                        "evidence_paths": [],
                        "provenance": {},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records, trace = retrieve_annotation_memory(
        memory,
        target_clip="OP_016",
        descriptors=["dense", "short call"],
    )
    assert all(record["case_id"] != "OP_016" for record in records)
    assert all(item["retrieved_id"] != "OP_016" for item in trace)


def test_prompt_condition_can_insert_walters_as_soft_guidance() -> None:
    card = load_json(WALTERS_CARD)
    rows = [
        {
            "proposal_id": "bd2_001",
            "start_time": 0.1,
            "end_time": 0.2,
            "duration_ms": 100.0,
            "low_frequency": 20000.0,
            "high_frequency": 50000.0,
            "det_prob": 0.9,
            "class_prob": 0.8,
            "original_label": "unknown",
        }
    ]
    _system, user, context = build_prompt(
        condition=OPTIONAL_AGENT_CONDITION,
        clip_id="OP_001",
        clip_duration=1.0,
        proposal_rows=rows,
        literature_records=[],
        annotation_records=[],
        walters_card=card,
    )
    assert "checklist" in user
    assert "no expected numeric range" in user
    assert context["walters_guidance"]


def test_proposal_conversion_uses_simple_schema() -> None:
    payload = {
        "events": [
            {
                "proposal_id": "bd2_001",
                "start_time_seconds": 0.1,
                "end_time_seconds": 0.2,
                "low_frequency_hz": 20000.0,
                "high_frequency_hz": 50000.0,
                "det_prob": 0.3,
                "class_prob": 0.5,
                "label": "metadata",
            }
        ]
    }
    events = proposal_prediction_events(payload)
    assert events == [
        {
            "event_id": "bd2_001",
            "start_time": 0.1,
            "end_time": 0.2,
            "low_frequency": 20000.0,
            "high_frequency": 50000.0,
            "linked_proposal_id": "bd2_001",
            "brief_reason": "BatDetect2 proposal-only baseline.",
        }
    ]


def test_clip_descriptors_are_proposal_only() -> None:
    descriptors = clip_descriptors(
        [
            {
                "start_time": 0.0,
                "end_time": 0.004,
                "duration_ms": 4.0,
                "proposal_id": "bd2_001",
            }
        ],
        clip_duration=1.0,
    )
    assert "left boundary" in descriptors
    assert "short call" in descriptors


def test_summary_csv_writer(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    write_csv(path, [{"condition": "agent_walters_guidance", "F1": 0.5}])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["condition"] == "agent_walters_guidance"
