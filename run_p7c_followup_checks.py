"""Run P7C diagnostic follow-up checks without changing frozen P7C outputs.

This script performs two limited checks:

1. exactly one technical retry for the malformed baseline/OP_032 response;
2. a two-clip oracle-retrieval sensitivity check using already existing
   annotation-memory and verified-literature records.

Ground truth is used only by the frozen evaluator after inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from evaluate_prompt_v2_small_pilot import run_evaluation
from event_characterisation_models import (
    RetrievedAnnotationCase,
    RetrievedLiteratureEvidence,
)
from event_characterisation_retrieval import (
    RetrievalMatch,
    store_version,
    write_retrieval_trace,
)
from extract_event_characterisation_features import load_jsonl_records
from plot_prompt_v2_small_pilot_diagnostics import (
    load_evaluation_csvs,
    plot_diagnostic_clip,
)
from run_library_assisted_bbox_ablation import (
    ANNOTATION_STORE,
    CLIP_IDS,
    CONDITIONS,
    DEFAULT_EVAL_DIR,
    DEFAULT_IMAGE_DIR,
    DEFAULT_OUTPUT_DIR,
    HELDOUT_CLIPS,
    LITERATURE_STORE,
    MODEL_NAME,
    OLLAMA_ENDPOINT,
    PROMPT_VERSION,
    PROPOSAL_THRESHOLD,
    REPRESENTATIVE_CLIPS,
    build_prompt,
    call_ollama,
    format_proposals_for_prompt,
    metric_summary,
    normalise_clip_bounds,
    proposal_dir_for_clip,
    retrieval_impact_row,
    safe_annotation_context,
    safe_literature_context,
    split_for_clip,
    validate_response,
    write_prediction,
)
from run_prompt_v2_batdetect2_assisted_pilot import load_proposal_payload
from run_prompt_v2_small_pilot import ollama_host, read_clip_duration


ORACLE_OUTPUT_DIR = Path("outputs/agent_runs/p7c_oracle_retrieval_sensitivity")
TECHNICAL_RETRY_DIR = (
    DEFAULT_OUTPUT_DIR / "technical_retries" / "baseline" / "OP_032"
)
TARGET_RETRY_CLIP = "OP_032"
ORACLE_CLIPS = ("OP_032", "OP_045")
ORACLE_CONDITIONS = ("oracle_annotation_memory", "oracle_combined")


@dataclass(frozen=True)
class ParseOutcome:
    parse_status: str
    prediction_path: Path
    raw_response_path: Path
    parse_error_path: Path | None
    parser_adjustments: list[str]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n\n{user}".encode("utf-8")).hexdigest()


def ensure_one_retry_only(retry_dir: Path) -> None:
    raw_path = retry_dir / "retry_raw_response.txt"
    if raw_path.exists():
        raise FileExistsError(
            f"Technical retry already exists at {raw_path}; one retry only."
        )


def require_qwen_available() -> str:
    if ollama_host().rstrip("/") != OLLAMA_ENDPOINT:
        raise RuntimeError(f"Set OLLAMA_HOST={OLLAMA_ENDPOINT}")
    result = subprocess.run(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    if MODEL_NAME not in result.stdout:
        raise RuntimeError(
            f"{MODEL_NAME} is not available through {OLLAMA_ENDPOINT}.\n"
            f"Available models:\n{result.stdout}"
        )
    return result.stdout


def parse_and_write_response(
    *,
    raw: str,
    clip_id: str,
    condition: str,
    output_dir: Path,
    eval_dir: Path,
    image_dir: Path,
    annotation_context: list[dict[str, Any]],
    literature_context: list[dict[str, Any]],
) -> ParseOutcome:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "retry_raw_response.txt"
    prediction_path = output_dir / f"{clip_id}_predictions.json"
    parsed_alias_path = output_dir / "parsed_prediction.json"
    error_path = output_dir / f"{clip_id}_parse_error.txt"
    raw_path.write_text(raw, encoding="utf-8")

    image_path = image_dir / f"{clip_id}_spectrogram.png"
    proposal_dir = proposal_dir_for_clip(clip_id)
    proposal_path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
    clip_duration = read_clip_duration(eval_dir, clip_id)
    proposal_rows = format_proposals_for_prompt(load_proposal_payload(proposal_dir, clip_id))
    try:
        payload = json.loads(raw)
        normalised_payload, parser_adjustments = normalise_clip_bounds(
            payload, clip_duration
        )
        response = validate_response(
            normalised_payload,
            clip_id=clip_id,
            clip_duration_seconds=clip_duration,
            proposal_ids={str(row["proposal_id"]) for row in proposal_rows},
            annotation_ids={str(row["case_id"]) for row in annotation_context},
            literature_ids={str(row["evidence_id"]) for row in literature_context},
        )
        for path in (prediction_path, parsed_alias_path):
            write_prediction(
                path,
                response=response,
                clip_id=clip_id,
                condition=condition,  # type: ignore[arg-type]
                image_path=image_path,
                proposal_path=proposal_path,
                clip_duration=clip_duration,
                parse_status="success",
                error="",
                parser_adjustments=parser_adjustments,
            )
        error_path.unlink(missing_ok=True)
        return ParseOutcome(
            "success", prediction_path, raw_path, None, parser_adjustments
        )
    except Exception as exc:
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        write_prediction(
            prediction_path,
            response=None,
            clip_id=clip_id,
            condition=condition,  # type: ignore[arg-type]
            image_path=image_path,
            proposal_path=proposal_path,
            clip_duration=clip_duration,
            parse_status="failure",
            error=f"{type(exc).__name__}: {exc}",
            parser_adjustments=[],
        )
        return ParseOutcome("failure", prediction_path, raw_path, error_path, [])


def technical_retry(
    *,
    output_dir: Path,
    eval_dir: Path,
    image_dir: Path,
    timeout: float,
    num_predict: int,
) -> dict[str, Any]:
    retry_dir = output_dir / "technical_retries" / "baseline" / TARGET_RETRY_CLIP
    ensure_one_retry_only(retry_dir)
    retry_dir.mkdir(parents=True, exist_ok=True)

    original_raw = (
        output_dir
        / "baseline"
        / "raw_responses"
        / f"{TARGET_RETRY_CLIP}_raw_response.txt"
    )
    original_prediction = (
        output_dir
        / "baseline"
        / "predictions"
        / f"{TARGET_RETRY_CLIP}_predictions.json"
    )
    original_error = (
        output_dir
        / "baseline"
        / "predictions"
        / f"{TARGET_RETRY_CLIP}_parse_error.txt"
    )
    shutil.copyfile(original_raw, retry_dir / "original_failed_raw_response.txt")
    reference = {
        "original_raw_response_path": original_raw.as_posix(),
        "original_raw_response_sha256": file_sha256(original_raw),
        "original_prediction_path": original_prediction.as_posix(),
        "original_parse_error_path": original_error.as_posix(),
        "original_parse_error": (
            original_error.read_text(encoding="utf-8") if original_error.is_file() else ""
        ),
    }
    (retry_dir / "original_failure_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )

    clip_duration = read_clip_duration(eval_dir, TARGET_RETRY_CLIP)
    proposal_rows = format_proposals_for_prompt(
        load_proposal_payload(proposal_dir_for_clip(TARGET_RETRY_CLIP), TARGET_RETRY_CLIP)
    )
    system, user = build_prompt(
        clip_id=TARGET_RETRY_CLIP,
        clip_duration_seconds=clip_duration,
        proposal_rows=proposal_rows,
        annotation_context=[],
        literature_context=[],
    )
    raw = call_ollama(
        image_path=image_dir / f"{TARGET_RETRY_CLIP}_spectrogram.png",
        system=system,
        user=user,
        timeout=timeout,
        num_predict=num_predict,
    )
    outcome = parse_and_write_response(
        raw=raw,
        clip_id=TARGET_RETRY_CLIP,
        condition="baseline",
        output_dir=retry_dir,
        eval_dir=eval_dir,
        image_dir=image_dir,
        annotation_context=[],
        literature_context=[],
    )
    metadata = {
        "clip_id": TARGET_RETRY_CLIP,
        "condition": "baseline",
        "retry_reason": "first-pass raw response was malformed/truncated JSON",
        "retry_policy": "one technical retry only",
        "model_name": MODEL_NAME,
        "ollama_endpoint": OLLAMA_ENDPOINT,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash_sha256": prompt_hash(system, user),
        "model_parameters": {
            "temperature": 0,
            "seed": 0,
            "num_predict": num_predict,
            "timeout": timeout,
        },
        "proposal_threshold": PROPOSAL_THRESHOLD,
        "parse_status": outcome.parse_status,
        "parser_adjustments": outcome.parser_adjustments,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (retry_dir / "retry_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (retry_dir / "parse_status.json").write_text(
        json.dumps(
            {
                "parse_status": outcome.parse_status,
                "parse_error_path": (
                    outcome.parse_error_path.as_posix()
                    if outcome.parse_error_path
                    else ""
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if outcome.parse_status == "success":
        aggregate, results = run_evaluation(
            pred_dir=retry_dir,
            eval_dir=eval_dir,
            clip_ids=[TARGET_RETRY_CLIP],
            output_dir=retry_dir / "evaluation",
        )
        (retry_dir / "evaluation_result.json").write_text(
            json.dumps({"aggregate": aggregate, "clip_result": results[0]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        regenerate_with_retry_tables(output_dir=output_dir, eval_dir=eval_dir, retry_prediction_path=outcome.prediction_path)
    return metadata


def copy_prediction_inputs(
    *,
    output_dir: Path,
    condition: str,
    target_dir: Path,
    retry_prediction_path: Path | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for clip_id in CLIP_IDS:
        source = (
            retry_prediction_path
            if condition == "baseline"
            and retry_prediction_path is not None
            and clip_id == TARGET_RETRY_CLIP
            else output_dir
            / condition
            / "predictions"
            / f"{clip_id}_predictions.json"
        )
        shutil.copyfile(source, target_dir / f"{clip_id}_predictions.json")


def load_prediction_payloads(pred_dir: Path, condition: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (condition, clip_id): json.loads(
            (pred_dir / f"{clip_id}_predictions.json").read_text(encoding="utf-8")
        )
        for clip_id in CLIP_IDS
    }


def regenerate_with_retry_tables(
    *, output_dir: Path, eval_dir: Path, retry_prediction_path: Path
) -> None:
    work_dir = output_dir / "technical_retries" / "with_retry_tables"
    prediction_root = work_dir / "prediction_inputs"
    evaluation_root = work_dir / "evaluations"
    all_case_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts: list[Any] = []

    class Artifact:
        def __init__(self, condition: str, clip_id: str, parse_status: str) -> None:
            self.condition = condition
            self.clip_id = clip_id
            self.parse_status = parse_status

    for condition in CONDITIONS:
        pred_dir = prediction_root / condition
        copy_prediction_inputs(
            output_dir=output_dir,
            condition=condition,
            target_dir=pred_dir,
            retry_prediction_path=retry_prediction_path,
        )
        _, results = run_evaluation(
            pred_dir=pred_dir,
            eval_dir=eval_dir,
            clip_ids=list(CLIP_IDS),
            output_dir=evaluation_root / condition,
        )
        payloads.update(load_prediction_payloads(pred_dir, condition))
        result_by_id = {item["metrics"]["clip_id"]: item for item in results}
        for clip_id in CLIP_IDS:
            payload = payloads[(condition, clip_id)]
            parse_status = str(payload.get("parse_status") or "failure")
            artifacts.append(Artifact(condition, clip_id, parse_status))
            metric = result_by_id[clip_id]["metrics"]
            all_case_rows.append(
                {
                    "condition": condition,
                    "clip_id": clip_id,
                    "split": split_for_clip(clip_id),
                    "parse_status": parse_status,
                    "prediction_count": metric["num_predictions"],
                    "tp": metric["tp"],
                    "fp": metric["fp"],
                    "fn": metric["fn"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "f1": metric["f1"],
                    "mean_time_iou": metric["mean_time_iou"],
                    "mean_frequency_iou": metric["mean_frequency_iou"],
                    "mean_box_iou": metric["mean_box_iou"],
                    "human_review_count": sum(
                        bool(event.get("human_review_needed"))
                        for event in payload["events"]
                    ),
                    "events_citing_annotation_memory": sum(
                        bool(event.get("retrieved_annotation_case_ids"))
                        for event in payload["events"]
                    ),
                    "events_citing_literature_evidence": sum(
                        bool(event.get("retrieved_literature_evidence_ids"))
                        for event in payload["events"]
                    ),
                    "retrieved_annotation_case_ids": "",
                    "retrieved_literature_evidence_ids": "",
                }
            )
        for split, ids in (
            ("all", set(CLIP_IDS)),
            ("representative", set(REPRESENTATIVE_CLIPS)),
            ("heldout", set(HELDOUT_CLIPS)),
        ):
            summaries.append(
                metric_summary(
                    condition=condition,
                    split=split,
                    results=[
                        item for item in results if item["metrics"]["clip_id"] in ids
                    ],
                    artifacts=[
                        item
                        for item in artifacts
                        if item.condition == condition and item.clip_id in ids
                    ],
                    prediction_payloads=payloads,
                )
            )
    case_lookup = {(row["condition"], row["clip_id"]): row for row in all_case_rows}
    impacts = [
        retrieval_impact_row(
            case_lookup[("baseline", clip_id)],
            case_lookup[(condition, clip_id)],
            payloads[("baseline", clip_id)],
            payloads[(condition, clip_id)],
        )
        for condition in CONDITIONS[1:]
        for clip_id in CLIP_IDS
    ]
    write_csv(output_dir / "p7c_case_level_results_with_retry.csv", all_case_rows)
    write_csv(output_dir / "p7c_condition_summary_with_retry.csv", summaries)
    write_csv(output_dir / "p7c_retrieval_impact_summary_with_retry.csv", impacts)


def by_id(records: list[Any], id_attr: str) -> dict[str, Any]:
    return {str(getattr(record, id_attr)): record for record in records}


def build_oracle_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "oracle_retrieval_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    annotation_records = by_id(
        load_jsonl_records(ANNOTATION_STORE, RetrievedAnnotationCase), "case_id"
    )
    evidence_records = by_id(
        load_jsonl_records(LITERATURE_STORE, RetrievedLiteratureEvidence),
        "evidence_id",
    )
    selections = {
        "OP_032": {
            "selected_annotation_case_ids": ["OP_004", "OP_045"],
            "selected_literature_evidence_ids": [
                "batdetect2_joint_detection_localisation",
                "batdetect2_external_validation_required",
            ],
            "human_selection_rationale": (
                "OP_004 covers useful anchored expansion and detector under-extension; "
                "OP_045 covers short source proposal extent failure. The BatDetect2 "
                "records constrain proposals as detector outputs that require validation."
            ),
            "selection_criteria": (
                "Non-target cases with abstract metadata about useful expansion, "
                "proposal under-extension, and avoiding rigid preservation."
            ),
        },
        "OP_045": {
            "selected_annotation_case_ids": ["OP_004", "OP_032"],
            "selected_literature_evidence_ids": [
                "batdetect2_joint_detection_localisation",
                "batdetect2_external_validation_required",
            ],
            "human_selection_rationale": (
                "OP_004 and OP_032 both describe useful evidence-supported expansion "
                "when detector proposals are under-wide or too conservative. The "
                "BatDetect2 records emphasise proposal status and validation limits."
            ),
            "selection_criteria": (
                "Non-target records about detector extent limitations and supported "
                "proposal modification, without coordinates or prior scores."
            ),
        },
    }
    for clip_id, row in selections.items():
        if clip_id in row["selected_annotation_case_ids"]:
            raise ValueError(f"Oracle manifest leaks target case {clip_id}")
        if len(row["selected_annotation_case_ids"]) > 2:
            raise ValueError("Oracle annotation top-k exceeds 2")
        if len(row["selected_literature_evidence_ids"]) > 2:
            raise ValueError("Oracle literature top-k exceeds 2")
        for case_id in row["selected_annotation_case_ids"]:
            if case_id not in annotation_records:
                raise ValueError(f"Unknown annotation case: {case_id}")
        for evidence_id in row["selected_literature_evidence_ids"]:
            if evidence_id not in evidence_records:
                raise ValueError(f"Unknown literature evidence: {evidence_id}")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "P7C oracle-retrieval sensitivity check; privileged diagnostic, not benchmark.",
        "annotation_store_path": ANNOTATION_STORE.as_posix(),
        "annotation_store_version": store_version(ANNOTATION_STORE),
        "literature_store_path": LITERATURE_STORE.as_posix(),
        "literature_store_version": store_version(LITERATURE_STORE),
        "records": [
            {
                "target_clip": clip_id,
                **row,
                "target_case_exclusion_holds": clip_id
                not in row["selected_annotation_case_ids"],
            }
            for clip_id, row in selections.items()
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest_context(
    manifest: dict[str, Any], clip_id: str, include_literature: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    record = next(row for row in manifest["records"] if row["target_clip"] == clip_id)
    annotation_records = by_id(
        load_jsonl_records(ANNOTATION_STORE, RetrievedAnnotationCase), "case_id"
    )
    evidence_records = by_id(
        load_jsonl_records(LITERATURE_STORE, RetrievedLiteratureEvidence),
        "evidence_id",
    )
    annotation_context = [
        safe_annotation_context(annotation_records[case_id])
        for case_id in record["selected_annotation_case_ids"]
    ]
    literature_context = (
        [
            safe_literature_context(evidence_records[evidence_id])
            for evidence_id in record["selected_literature_evidence_ids"]
        ]
        if include_literature
        else []
    )
    return annotation_context, literature_context, record


def run_oracle_clip(
    *,
    condition: str,
    clip_id: str,
    output_dir: Path,
    eval_dir: Path,
    image_dir: Path,
    manifest: dict[str, Any],
    timeout: float,
    num_predict: int,
) -> ParseOutcome:
    include_literature = condition == "oracle_combined"
    annotation_context, literature_context, manifest_record = load_manifest_context(
        manifest, clip_id, include_literature
    )
    condition_dir = output_dir / condition
    raw_dir = condition_dir / "raw_responses"
    pred_dir = condition_dir / "predictions"
    trace_dir = condition_dir / "retrieval_traces"
    for path in (raw_dir, pred_dir, trace_dir):
        path.mkdir(parents=True, exist_ok=True)
    clip_duration = read_clip_duration(eval_dir, clip_id)
    proposal_rows = format_proposals_for_prompt(
        load_proposal_payload(proposal_dir_for_clip(clip_id), clip_id)
    )
    system, user = build_prompt(
        clip_id=clip_id,
        clip_duration_seconds=clip_duration,
        proposal_rows=proposal_rows,
        annotation_context=annotation_context,
        literature_context=literature_context,
    )
    raw = call_ollama(
        image_path=image_dir / f"{clip_id}_spectrogram.png",
        system=system,
        user=user,
        timeout=timeout,
        num_predict=num_predict,
    )
    raw_response_path = raw_dir / f"{clip_id}_raw_response.txt"
    raw_response_path.write_text(raw, encoding="utf-8")
    trace = {
        "clip_id": clip_id,
        "condition": condition,
        "retrieval_type": "oracle",
        "annotation_memory_enabled": True,
        "literature_evidence_enabled": include_literature,
        "annotation_matches": [
            {
                "retrieved_id": case_id,
                "retrieval_score": 999.0,
                "match_reasons": ["manual oracle selection"],
            }
            for case_id in manifest_record["selected_annotation_case_ids"]
        ],
        "literature_matches": [
            {
                "retrieved_id": evidence_id,
                "retrieval_score": 999.0,
                "match_reasons": ["manual oracle selection"],
            }
            for evidence_id in (
                manifest_record["selected_literature_evidence_ids"]
                if include_literature
                else []
            )
        ],
        "annotation_store_version": manifest["annotation_store_version"],
        "literature_store_version": manifest["literature_store_version"],
        "target_case_excluded": True,
        "prompt_hash_sha256": prompt_hash(system, user),
    }
    (trace_dir / f"{clip_id}_retrieval_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    return parse_and_write_response(
        raw=raw,
        clip_id=clip_id,
        condition=condition,
        output_dir=pred_dir,
        eval_dir=eval_dir,
        image_dir=image_dir,
        annotation_context=annotation_context,
        literature_context=literature_context,
    )


def evaluate_oracle_conditions(output_dir: Path, eval_dir: Path) -> dict[str, list[dict[str, Any]]]:
    results_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in ORACLE_CONDITIONS:
        condition_dir = output_dir / condition
        _, results = run_evaluation(
            pred_dir=condition_dir / "predictions",
            eval_dir=eval_dir,
            clip_ids=list(ORACLE_CLIPS),
            output_dir=condition_dir / "evaluation",
        )
        evaluation_rows = load_evaluation_csvs(condition_dir / "evaluation")
        for clip_id in ORACLE_CLIPS:
            plot_diagnostic_clip(
                clip_id=clip_id,
                pred_dir=condition_dir / "predictions",
                eval_dir=eval_dir,
                evaluation_rows=evaluation_rows,
                output_dir=condition_dir / "diagnostic_figures",
            )
        results_by_condition[condition] = results
    return results_by_condition


def parse_geometry_actions(payload: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for event in payload.get("events", []):
        action = str(event.get("geometry_action") or "unknown")
        counts[action] = counts.get(action, 0) + 1
    rejected = len(payload.get("rejected_proposals", []))
    if rejected:
        counts["reject_proposal"] = rejected
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def human_review_value(payload: dict[str, Any]) -> str:
    count = sum(bool(event.get("human_review_needed")) for event in payload.get("events", []))
    return str(count)


def p7c_case_rows_path(p7c_dir: Path) -> Path:
    with_retry = p7c_dir / "p7c_case_level_results_with_retry.csv"
    return with_retry if with_retry.is_file() else p7c_dir / "p7c_case_level_results.csv"


def load_existing_case_row(
    *, p7c_dir: Path, condition: str, clip_id: str
) -> dict[str, Any]:
    rows = load_csv(p7c_case_rows_path(p7c_dir))
    for row in rows:
        if row["condition"] == condition and row["clip_id"] == clip_id:
            retry_prediction = (
                p7c_dir
                / "technical_retries"
                / "baseline"
                / TARGET_RETRY_CLIP
                / f"{TARGET_RETRY_CLIP}_predictions.json"
            )
            prediction_path = (
                retry_prediction
                if condition == "baseline"
                and clip_id == TARGET_RETRY_CLIP
                and p7c_case_rows_path(p7c_dir).name.endswith("_with_retry.csv")
                and retry_prediction.is_file()
                else p7c_dir
                / condition
                / "predictions"
                / f"{clip_id}_predictions.json"
            )
            payload = json.loads(
                prediction_path.read_text(encoding="utf-8")
            )
            return {
                "clip_id": clip_id,
                "condition": condition,
                "retrieval_type": "automatic" if condition != "baseline" else "baseline",
                "parse_success": row["parse_status"] == "success",
                "selected_annotation_case_ids": row.get("retrieved_annotation_case_ids", ""),
                "selected_literature_evidence_ids": row.get("retrieved_literature_evidence_ids", ""),
                "TP": int(row["tp"]),
                "FP": int(row["fp"]),
                "FN": int(row["fn"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "F1": float(row["f1"]),
                "mean_time_iou": float(row["mean_time_iou"]),
                "mean_frequency_iou": float(row["mean_frequency_iou"]),
                "mean_box_iou": float(row["mean_box_iou"]),
                "geometry_actions": parse_geometry_actions(payload),
                "human_review_needed": human_review_value(payload),
            }
    raise ValueError(f"Missing P7C row for {condition} {clip_id}")


def oracle_case_rows(
    *, output_dir: Path, p7c_dir: Path, manifest: dict[str, Any],
    oracle_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip_id in ORACLE_CLIPS:
        baseline_condition = "baseline"
        rows.append(
            load_existing_case_row(
                p7c_dir=p7c_dir, condition=baseline_condition, clip_id=clip_id
            )
        )
        for condition in ("annotation_memory_only", "literature_only", "combined"):
            rows.append(
                load_existing_case_row(
                    p7c_dir=p7c_dir, condition=condition, clip_id=clip_id
                )
            )
        for condition in ORACLE_CONDITIONS:
            result = next(
                item
                for item in oracle_results[condition]
                if item["metrics"]["clip_id"] == clip_id
            )
            metric = result["metrics"]
            payload = json.loads(
                (
                    output_dir
                    / condition
                    / "predictions"
                    / f"{clip_id}_predictions.json"
                ).read_text(encoding="utf-8")
            )
            manifest_record = next(
                row for row in manifest["records"] if row["target_clip"] == clip_id
            )
            rows.append(
                {
                    "clip_id": clip_id,
                    "condition": condition,
                    "retrieval_type": "oracle",
                    "parse_success": payload.get("parse_status") == "success",
                    "selected_annotation_case_ids": "|".join(
                        manifest_record["selected_annotation_case_ids"]
                    ),
                    "selected_literature_evidence_ids": "|".join(
                        manifest_record["selected_literature_evidence_ids"]
                        if condition == "oracle_combined"
                        else []
                    ),
                    "TP": metric["tp"],
                    "FP": metric["fp"],
                    "FN": metric["fn"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "F1": metric["f1"],
                    "mean_time_iou": metric["mean_time_iou"],
                    "mean_frequency_iou": metric["mean_frequency_iou"],
                    "mean_box_iou": metric["mean_box_iou"],
                    "geometry_actions": parse_geometry_actions(payload),
                    "human_review_needed": human_review_value(payload),
                }
            )
    return rows


def improvement_label(delta_f1: float, delta_box: float) -> str:
    if delta_f1 > 1e-12 or (abs(delta_f1) <= 1e-12 and delta_box > 1e-12):
        return "improvement"
    if delta_f1 < -1e-12 or (abs(delta_f1) <= 1e-12 and delta_box < -1e-12):
        return "degradation"
    return "neutral"


def delta_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["clip_id"], row["condition"]): row for row in case_rows}
    rows: list[dict[str, Any]] = []
    for clip_id in ORACLE_CLIPS:
        baseline = lookup[(clip_id, "baseline")]
        pairs = (
            ("oracle_annotation_memory", "annotation_memory_only"),
            ("oracle_combined", "combined"),
        )
        for oracle_condition, automatic_condition in pairs:
            oracle = lookup[(clip_id, oracle_condition)]
            automatic = lookup[(clip_id, automatic_condition)]
            for reference_name, reference in (
                ("corresponding_automatic", automatic),
                ("baseline", baseline),
            ):
                delta_f1 = float(oracle["F1"]) - float(reference["F1"])
                delta_box = float(oracle["mean_box_iou"]) - float(reference["mean_box_iou"])
                rows.append(
                    {
                        "clip_id": clip_id,
                        "oracle_condition": oracle_condition,
                        "reference": reference_name,
                        "reference_condition": reference["condition"],
                        "delta_TP": int(oracle["TP"]) - int(reference["TP"]),
                        "delta_FP": int(oracle["FP"]) - int(reference["FP"]),
                        "delta_FN": int(oracle["FN"]) - int(reference["FN"]),
                        "delta_F1": delta_f1,
                        "delta_mean_time_iou": float(oracle["mean_time_iou"]) - float(reference["mean_time_iou"]),
                        "delta_mean_frequency_iou": float(oracle["mean_frequency_iou"]) - float(reference["mean_frequency_iou"]),
                        "delta_mean_box_iou": delta_box,
                        "geometry_decision_changed": oracle["geometry_actions"] != reference["geometry_actions"],
                        "effect_label": improvement_label(delta_f1, delta_box),
                    }
                )
    return rows


def write_oracle_report(
    *, output_dir: Path, case_rows: list[dict[str, Any]],
    deltas: list[dict[str, Any]], manifest: dict[str, Any],
    retry_status: str,
) -> None:
    lines = [
        "# P7C Oracle-Retrieval Sensitivity Check",
        "",
        "## Scope",
        "",
        "This is a diagnostic oracle-retrieval upper-bound check, not a new unbiased benchmark condition. It reuses the frozen P7C prompt structure, parser, and evaluator, and changes only the manually selected non-target retrieval context for `OP_032` and `OP_045`.",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Endpoint: `{OLLAMA_ENDPOINT}`",
        f"- Technical retry status for baseline/OP_032: `{retry_status}`",
        "",
        "## Oracle Manifest",
        "",
    ]
    for record in manifest["records"]:
        lines.append(
            f"- `{record['target_clip']}`: annotation cases `{', '.join(record['selected_annotation_case_ids'])}`; literature `{', '.join(record['selected_literature_evidence_ids'])}`. Rationale: {record['human_selection_rationale']}"
        )
    lines.extend(
        [
            "",
            "## Case Metrics",
            "",
            "| Clip | Condition | Retrieval | Parsed | TP | FP | FN | F1 | Time IoU | Freq IoU | Box IoU | Actions | Review |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in case_rows:
        lines.append(
            f"| {row['clip_id']} | {row['condition']} | {row['retrieval_type']} | {row['parse_success']} | {row['TP']} | {row['FP']} | {row['FN']} | {float(row['F1']):.3f} | {float(row['mean_time_iou']):.3f} | {float(row['mean_frequency_iou']):.3f} | {float(row['mean_box_iou']):.3f} | {row['geometry_actions']} | {row['human_review_needed']} |"
        )
    lines.extend(["", "## Delta Summary", ""])
    for row in deltas:
        lines.append(
            f"- `{row['clip_id']}` `{row['oracle_condition']}` vs `{row['reference_condition']}` ({row['reference']}): ΔTP={row['delta_TP']:+d}, ΔFP={row['delta_FP']:+d}, ΔFN={row['delta_FN']:+d}, ΔF1={float(row['delta_F1']):+.3f}, Δbox IoU={float(row['delta_mean_box_iou']):+.3f}; {row['effect_label']}."
        )
    improved = [row for row in deltas if row["effect_label"] == "improvement"]
    degraded = [row for row in deltas if row["effect_label"] == "degradation"]
    changed = [row for row in deltas if row["geometry_decision_changed"]]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The formal P7C result remains that automatic annotation-memory and literature retrieval produced no measurable localisation benefit on the eight evaluated clips. This follow-up is intentionally narrower: it asks whether highly relevant non-target context changes localisation on two known hard cases.",
            "",
            f"Oracle retrieval produced {len(improved)} improving delta rows, {len(degraded)} degrading delta rows, and changed geometry-action summaries in {len(changed)} comparisons. Because `n=2`, this is only diagnostic evidence.",
        ]
    )
    if improved and degraded:
        interpretation = "mixed evidence: oracle context can change outputs, but the direction is not consistently beneficial."
    elif improved:
        interpretation = "retrieval ranking or coverage may be a limitation because privileged context improved at least one target."
    elif changed:
        interpretation = "model utilisation limitation: relevant context changed decisions without reliably improving localisation."
    else:
        interpretation = "insufficiently actionable knowledge or model utilisation limitation: oracle context did not create meaningful localisation change."
    lines.extend(["", f"Final interpretation: **{interpretation}**"])
    (output_dir / "p7c_oracle_retrieval_sensitivity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_oracle(
    *,
    output_dir: Path,
    p7c_dir: Path,
    eval_dir: Path,
    image_dir: Path,
    timeout: float,
    num_predict: int,
    retry_status: str,
) -> dict[str, Any]:
    manifest = build_oracle_manifest(output_dir)
    outcomes: dict[str, dict[str, ParseOutcome]] = {condition: {} for condition in ORACLE_CONDITIONS}
    for condition in ORACLE_CONDITIONS:
        for clip_id in ORACLE_CLIPS:
            outcomes[condition][clip_id] = run_oracle_clip(
                condition=condition,
                clip_id=clip_id,
                output_dir=output_dir,
                eval_dir=eval_dir,
                image_dir=image_dir,
                manifest=manifest,
                timeout=timeout,
                num_predict=num_predict,
            )
    oracle_results = evaluate_oracle_conditions(output_dir, eval_dir)
    case_rows = oracle_case_rows(
        output_dir=output_dir,
        p7c_dir=p7c_dir,
        manifest=manifest,
        oracle_results=oracle_results,
    )
    deltas = delta_rows(case_rows)
    write_csv(output_dir / "p7c_oracle_case_results.csv", case_rows)
    write_csv(output_dir / "p7c_oracle_delta_summary.csv", deltas)
    write_oracle_report(
        output_dir=output_dir,
        case_rows=case_rows,
        deltas=deltas,
        manifest=manifest,
        retry_status=retry_status,
    )
    return {
        "manifest": manifest,
        "outcomes": outcomes,
        "case_rows": case_rows,
        "deltas": deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p7c-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-dir", type=Path, default=ORACLE_OUTPUT_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument(
        "--skip-technical-retry",
        action="store_true",
        help="Only run oracle checks; useful if the retry was already performed.",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Only run the technical retry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_list = require_qwen_available()
    print("Model availability check passed:")
    print(model_list)
    retry_status = "skipped"
    if not args.skip_technical_retry:
        retry_metadata = technical_retry(
            output_dir=args.p7c_dir,
            eval_dir=args.eval_dir,
            image_dir=args.image_dir,
            timeout=args.timeout,
            num_predict=args.num_predict,
        )
        retry_status = str(retry_metadata["parse_status"])
        print(f"Technical retry parse status: {retry_status}")
    if not args.skip_oracle:
        result = run_oracle(
            output_dir=args.oracle_dir,
            p7c_dir=args.p7c_dir,
            eval_dir=args.eval_dir,
            image_dir=args.image_dir,
            timeout=args.timeout,
            num_predict=args.num_predict,
            retry_status=retry_status,
        )
        parse_count = sum(
            outcome.parse_status == "success"
            for condition_outcomes in result["outcomes"].values()
            for outcome in condition_outcomes.values()
        )
        print(f"Oracle parse success: {parse_count}/4")
        print(f"Oracle manifest: {args.oracle_dir / 'oracle_retrieval_manifest.json'}")


if __name__ == "__main__":
    main()
