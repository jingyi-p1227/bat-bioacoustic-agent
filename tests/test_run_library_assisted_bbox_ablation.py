import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from event_characterisation_retrieval import RetrievalTrace, write_retrieval_trace
from run_library_assisted_bbox_ablation import (
    LibraryBBoxResponse,
    RunArtifact,
    build_prompt,
    format_proposals_for_prompt,
    metric_summary,
    normalise_clip_bounds,
    proposal_descriptors,
    retrieval_impact_row,
    retrieve_context,
    safe_annotation_context,
    split_for_clip,
    validate_response,
)
from event_characterisation_models import RetrievedAnnotationCase


def proposal_rows() -> list[dict]:
    return [
        {
            "proposal_id": "bd2_001",
            "start_time_seconds": 0.1,
            "end_time_seconds": 0.11,
            "duration_ms": 10.0,
            "low_frequency_hz": 30_000.0,
            "high_frequency_hz": 40_000.0,
            "det_prob": 0.8,
            "class_prob": 0.6,
            "original_label": "UK species",
        }
    ]


def valid_response() -> dict:
    return {
        "clip_id": "OP_016",
        "events": [
            {
                "event_id": "event_001",
                "start_time_seconds": 0.1,
                "end_time_seconds": 0.11,
                "low_frequency_hz": 30_000.0,
                "high_frequency_hz": 40_000.0,
                "label": "bat_call",
                "confidence": 0.8,
                "source_proposal_id": "bd2_001",
                "proposal_source": "batdetect2",
                "geometry_action": "preserve_proposal",
                "geometry_reason": "Visible evidence supports the proposal.",
                "retrieved_annotation_case_ids": [],
                "retrieved_literature_evidence_ids": [],
                "human_review_needed": False,
                "review_reason": "",
            }
        ],
        "rejected_proposals": [],
    }


def test_output_schema_and_geometry_action_enum() -> None:
    parsed = LibraryBBoxResponse.model_validate(valid_response())
    assert parsed.events[0].geometry_action == "preserve_proposal"

    invalid = valid_response()
    invalid["events"][0]["geometry_action"] = "reject_proposal"
    with pytest.raises(ValidationError):
        LibraryBBoxResponse.model_validate(invalid)


def test_clip_bound_normalisation_records_adjustment() -> None:
    payload = valid_response()
    payload["events"][0]["end_time_seconds"] = 1.0041
    normalised, adjustments = normalise_clip_bounds(payload, 1.0)
    assert normalised["events"][0]["end_time_seconds"] == 1.0
    assert len(adjustments) == 1


def test_validate_response_checks_proposal_and_retrieval_ids() -> None:
    parsed = validate_response(
        valid_response(),
        clip_id="OP_016",
        clip_duration_seconds=1.0,
        proposal_ids={"bd2_001"},
        annotation_ids=set(),
        literature_ids=set(),
    )
    assert len(parsed.events) == 1

    invalid = valid_response()
    invalid["events"][0]["retrieved_annotation_case_ids"] = ["OP_001"]
    with pytest.raises(ValueError, match="unsupplied annotation"):
        validate_response(
            invalid,
            clip_id="OP_016",
            clip_duration_seconds=1.0,
            proposal_ids={"bd2_001"},
            annotation_ids=set(),
            literature_ids=set(),
        )


def test_condition_isolation_and_target_case_exclusion() -> None:
    for condition, annotation_expected, literature_expected in (
        ("baseline", False, False),
        ("annotation_memory_only", True, False),
        ("literature_only", False, True),
        ("combined", True, True),
    ):
        annotation, literature, trace = retrieve_context(
            condition=condition,
            clip_id="OP_016",
            proposal_rows=proposal_rows(),
            clip_duration_seconds=1.0,
        )
        assert bool(annotation) is annotation_expected
        assert bool(literature) is literature_expected
        assert trace.target_case_excluded is True
        assert "OP_016" not in {item["case_id"] for item in annotation}


def test_safe_annotation_context_removes_artifacts_and_outcomes() -> None:
    record = RetrievedAnnotationCase(
        case_id="OP_X",
        case_type=["boundary_case"],
        observable_features=["ground-truth event geometry"],
        known_failure_modes=["previous model failed"],
        recommended_actions=["review boundary evidence"],
        anti_patterns=["blindly shift geometry"],
        evidence_paths=["outputs/diagnostic.png"],
        provenance={"metric": "F1=1.0"},
    )
    safe = safe_annotation_context(record)
    text = json.dumps(safe).lower()
    assert "ground-truth" not in text
    assert "previous model" not in text
    assert "diagnostic" not in text
    assert "f1=" not in text


def test_prompt_has_no_gt_diagnostic_or_metric_leakage() -> None:
    system, user = build_prompt(
        clip_id="OP_016",
        clip_duration_seconds=1.0,
        proposal_rows=proposal_rows(),
        annotation_context=[],
        literature_context=[],
    )
    text = f"{system}\n{user}".lower()
    assert "ground_truth" not in text
    assert "gt overlay" not in text
    assert "diagnostic" not in text
    assert "true positive" not in text
    assert "f1" not in text


def test_proposal_metadata_filter_and_descriptors() -> None:
    payload = {
        "events": [
            {
                "proposal_id": "low",
                "start_time_seconds": 0.0,
                "end_time_seconds": 0.01,
                "low_frequency_hz": 20_000,
                "high_frequency_hz": 30_000,
                "det_prob": 0.29,
                "class_prob": 0.5,
                "label": "x",
            },
            {
                "proposal_id": "keep",
                "start_time_seconds": 0.99,
                "end_time_seconds": 1.0,
                "low_frequency_hz": 20_000,
                "high_frequency_hz": 30_000,
                "det_prob": 0.30,
                "class_prob": 0.5,
                "label": "x",
            },
        ]
    }
    rows = format_proposals_for_prompt(payload)
    assert [row["proposal_id"] for row in rows] == ["keep"]
    descriptors = proposal_descriptors(rows, 1.0)
    assert "short proposal" in descriptors
    assert "right boundary" in descriptors


def test_trace_serialisation(tmp_path: Path) -> None:
    _, _, trace = retrieve_context(
        condition="combined",
        clip_id="OP_016",
        proposal_rows=proposal_rows(),
        clip_duration_seconds=1.0,
    )
    path = tmp_path / "trace.json"
    write_retrieval_trace(trace, path)
    loaded = RetrievalTrace.model_validate_json(path.read_text())
    assert loaded == trace
    assert loaded.annotation_store_version.startswith("sha256:")


def result(clip_id: str, tp: int, fp: int, fn: int, box: float) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = [
        {"time_iou": 0.8, "frequency_iou": 0.7, "box_iou": box}
        for _ in range(tp)
    ]
    return {
        "metrics": {
            "clip_id": clip_id,
            "num_ground_truth_events": tp + fn,
            "num_predictions": tp + fp,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_time_iou": 0.8 if tp else 0.0,
            "mean_frequency_iou": 0.7 if tp else 0.0,
            "mean_box_iou": box if tp else 0.0,
            "num_truncated_events": 0,
        },
        "matched_events": matched,
        "unmatched_predictions": [],
        "missed_ground_truth_events": [],
    }


def test_condition_summary_and_split_aggregation(tmp_path: Path) -> None:
    results = [result("OP_001", 2, 1, 0, 0.6), result("OP_032", 1, 0, 1, 0.5)]
    artifacts = [
        RunArtifact("baseline", clip, split_for_clip(clip), "success", tmp_path / "p", tmp_path / "r", None, tmp_path / "t", False)
        for clip in ("OP_001", "OP_032")
    ]
    payloads = {
        ("baseline", "OP_001"): {"events": [{"human_review_needed": False}] * 3},
        ("baseline", "OP_032"): {"events": [{"human_review_needed": True}]},
    }
    summary = metric_summary(
        condition="baseline", split="all", results=results,
        artifacts=artifacts, prediction_payloads=payloads,
    )
    assert summary["clip_count"] == 2
    assert summary["TP"] == 3
    assert summary["human_review_rate"] == pytest.approx(0.25)


def test_retrieval_impact_delta_calculation() -> None:
    baseline = {**result("OP_016", 1, 2, 3, 0.2)["metrics"], "condition": "baseline", "split": "representative", "parse_status": "success"}
    comparison = {**result("OP_016", 2, 1, 2, 0.4)["metrics"], "condition": "combined", "split": "representative", "parse_status": "success"}
    baseline_payload = {"events": [valid_response()["events"][0]]}
    changed_event = {**valid_response()["events"][0], "end_time_seconds": 0.12, "geometry_action": "refine_proposal"}
    row = retrieval_impact_row(baseline, comparison, baseline_payload, {"events": [changed_event]})
    assert row["delta_TP"] == 1
    assert row["delta_FP"] == -1
    assert row["retrieval_changed_geometry_decisions"] is True
    assert row["retrieval_improved_result"] is True
