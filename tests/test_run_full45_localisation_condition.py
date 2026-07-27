from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "inference"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_full45_localisation_condition import (  # noqa: E402
    build_condition_user_message,
    load_proposal_context,
    load_walters_context,
    retrieve_source_safe_annotation_examples,
)


def test_source_recording_safe_retrieval_excludes_target_source() -> None:
    source_recordings = {
        "OP_001": "source_a.wav",
        "OP_002": "source_a.wav",
        "OP_010": "source_b.wav",
        "OP_045": "source_c.wav",
    }
    memory = [
        {"case_id": "OP_001", "case_type": ["target"]},
        {"case_id": "OP_002", "case_type": ["same_source"]},
        {"case_id": "OP_010", "case_type": ["safe"], "evidence_paths": ["secret"]},
        {"case_id": "OP_045", "case_type": ["safe2"]},
    ]

    selected, trace = retrieve_source_safe_annotation_examples(
        clip_id="OP_001",
        source_recordings=source_recordings,
        memory_records=memory,
        top_k=2,
    )

    assert trace["source_recording_safe"]
    assert trace["target_case_excluded"]
    assert [row["case_id"] for row in selected] == ["OP_010", "OP_045"]
    assert "evidence_paths" not in selected[0]


def test_load_walters_context_uses_chunks_and_card_not_full_text(tmp_path: Path) -> None:
    card = tmp_path / "generic_card.md"
    card.write_text("compact checklist", encoding="utf-8")
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    (chunk_dir / "walters_2012_chunk_001.json").write_text(
        json.dumps(
            {
                "evidence_id": "walters_2012_chunk_001",
                "page_number": "3",
                "chunk_text": "generic acoustic parameter evidence",
                "detected_acoustic_concepts": ["duration"],
                "caution_note": "not an OP prior",
            }
        ),
        encoding="utf-8",
    )

    context = load_walters_context(card, chunk_dir)

    assert context["card_text"] == "compact checklist"
    assert context["evidence_chunks"][0]["evidence_id"] == "walters_2012_chunk_001"
    assert "full_text" not in json.dumps(context)


def test_load_proposal_context_formats_duration(tmp_path: Path) -> None:
    proposal_dir = tmp_path / "proposals"
    proposal_dir.mkdir()
    (proposal_dir / "OP_001_batdetect2_proposals.json").write_text(
        json.dumps(
            {
                "clip_id": "OP_001",
                "events": [
                    {
                        "proposal_id": "bd2_001",
                        "start_time_seconds": 0.1,
                        "end_time_seconds": 0.112,
                        "low_frequency_hz": 20000,
                        "high_frequency_hz": 60000,
                        "det_prob": 0.8,
                        "class_prob": 0.7,
                        "label": "metadata_only",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = load_proposal_context(proposal_dir, "OP_001")

    assert rows[0]["proposal_id"] == "bd2_001"
    assert rows[0]["duration_ms"] == 12.0
    assert rows[0]["original_label"] == "metadata_only"


def test_build_condition_user_message_names_generic_acoustic_cautions() -> None:
    message = build_condition_user_message(
        condition="walters_acoustic",
        clip_id="OP_001",
        clip_duration_seconds=1.0,
        condition_context={"evidence_chunks": []},
    )

    assert "Do not explain" in message
    assert "Ozimops petersi prior" in message
    assert "valid JSON" in message


def test_build_condition_user_message_names_conservative_proposal_timing() -> None:
    message = build_condition_user_message(
        condition="p14_best_stack_qwen3_6_proposal_constrained_conservative",
        clip_id="OP_001",
        clip_duration_seconds=1.0,
        condition_context=[{"proposal_id": "bd2_001"}],
    )

    assert "conservative final best-stack" in message
    assert "preserve proposal" in message
    assert "start_time_seconds" in message
    assert "unsupported onset/offset shifts" in message
