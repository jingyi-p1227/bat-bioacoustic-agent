import json
from pathlib import Path

import pytest

from run_p7c_followup_checks import (
    ORACLE_CLIPS,
    build_oracle_manifest,
    delta_rows,
    ensure_one_retry_only,
    improvement_label,
    load_manifest_context,
    prompt_hash,
    write_oracle_report,
)


def test_one_retry_only_policy(tmp_path: Path) -> None:
    retry_dir = tmp_path / "retry"
    retry_dir.mkdir()
    ensure_one_retry_only(retry_dir)
    (retry_dir / "retry_raw_response.txt").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="one retry only"):
        ensure_one_retry_only(retry_dir)


def test_prompt_hash_is_stable() -> None:
    assert prompt_hash("system", "user") == prompt_hash("system", "user")
    assert prompt_hash("system", "user") != prompt_hash("system", "other")


def test_oracle_manifest_schema_target_exclusion_and_top_k(tmp_path: Path) -> None:
    manifest = build_oracle_manifest(tmp_path)
    assert (tmp_path / "oracle_retrieval_manifest.json").is_file()
    assert {row["target_clip"] for row in manifest["records"]} == set(ORACLE_CLIPS)
    assert manifest["annotation_store_version"].startswith("sha256:")
    assert manifest["literature_store_version"].startswith("sha256:")
    for row in manifest["records"]:
        assert row["target_clip"] not in row["selected_annotation_case_ids"]
        assert row["target_case_exclusion_holds"] is True
        assert len(row["selected_annotation_case_ids"]) <= 2
        assert len(row["selected_literature_evidence_ids"]) <= 2
        assert row["selected_annotation_case_ids"]
        assert row["selected_literature_evidence_ids"]


def test_oracle_context_contains_no_gt_diagnostic_or_metric_leakage(tmp_path: Path) -> None:
    manifest = build_oracle_manifest(tmp_path)
    annotation_context, literature_context, _ = load_manifest_context(
        manifest, "OP_032", include_literature=True
    )
    text = json.dumps(
        {
            "annotation_context": annotation_context,
            "literature_context": literature_context,
        },
        ensure_ascii=False,
    ).lower()
    forbidden = [
        "ground_truth",
        "gt_overlay",
        "diagnostic",
        "true positive",
        "false positive",
        "f1=",
        "iou",
        "outputs/",
    ]
    for term in forbidden:
        assert term not in text


def test_literature_context_can_be_excluded_for_oracle_annotation_only(tmp_path: Path) -> None:
    manifest = build_oracle_manifest(tmp_path)
    annotation_context, literature_context, record = load_manifest_context(
        manifest, "OP_045", include_literature=False
    )
    assert annotation_context
    assert literature_context == []
    assert record["target_clip"] == "OP_045"


def case_row(
    clip_id: str,
    condition: str,
    *,
    tp: int,
    fp: int,
    fn: int,
    f1: float,
    box: float,
    actions: str = "preserve_proposal:1",
) -> dict:
    return {
        "clip_id": clip_id,
        "condition": condition,
        "retrieval_type": "oracle" if condition.startswith("oracle") else "automatic",
        "parse_success": True,
        "selected_annotation_case_ids": "",
        "selected_literature_evidence_ids": "",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": 0.0,
        "recall": 0.0,
        "F1": f1,
        "mean_time_iou": 0.4,
        "mean_frequency_iou": 0.5,
        "mean_box_iou": box,
        "geometry_actions": actions,
        "human_review_needed": "0",
    }


def test_delta_calculation_and_improvement_labels() -> None:
    rows = []
    for clip_id in ORACLE_CLIPS:
        rows.extend(
            [
                case_row(clip_id, "baseline", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
                case_row(clip_id, "annotation_memory_only", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
                case_row(clip_id, "literature_only", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
                case_row(clip_id, "combined", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
                case_row(clip_id, "oracle_annotation_memory", tp=2, fp=0, fn=0, f1=1.0, box=0.4, actions="refine_proposal:1"),
                case_row(clip_id, "oracle_combined", tp=1, fp=2, fn=1, f1=0.4, box=0.1),
            ]
        )
    deltas = delta_rows(rows)
    assert len(deltas) == 8
    assert any(row["effect_label"] == "improvement" for row in deltas)
    assert any(row["effect_label"] == "degradation" for row in deltas)
    assert any(row["geometry_decision_changed"] for row in deltas)
    assert improvement_label(0.0, 0.0) == "neutral"


def test_oracle_report_aggregation(tmp_path: Path) -> None:
    manifest = build_oracle_manifest(tmp_path)
    rows = []
    for clip_id in ORACLE_CLIPS:
        rows.extend(
            [
                case_row(clip_id, "baseline", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
                case_row(clip_id, "annotation_memory_only", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
                case_row(clip_id, "literature_only", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
                case_row(clip_id, "combined", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
                case_row(clip_id, "oracle_annotation_memory", tp=2, fp=0, fn=0, f1=1.0, box=0.4),
                case_row(clip_id, "oracle_combined", tp=1, fp=1, fn=1, f1=0.5, box=0.2),
            ]
        )
    write_oracle_report(
        output_dir=tmp_path,
        case_rows=rows,
        deltas=delta_rows(rows),
        manifest=manifest,
        retry_status="success",
    )
    report = tmp_path / "p7c_oracle_retrieval_sensitivity_report.md"
    assert report.is_file()
    text = report.read_text(encoding="utf-8")
    assert "Formal P7C result" not in text  # exact phrase is not required
    assert "OP_032" in text
    assert "OP_045" in text
