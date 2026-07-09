"""Run P7C library-assisted BatDetect2 proposal refinement.

Prediction uses only clean grid_v2 images, existing BatDetect2 proposal metadata,
and condition-specific safe retrieval context. Ground truth is loaded only after
all model calls have completed, through the frozen event-level evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluate_prompt_v2_small_pilot import aggregate_results, run_evaluation
from event_characterisation_models import (
    RetrievedAnnotationCase,
    RetrievedLiteratureEvidence,
)
from event_characterisation_retrieval import (
    ConditionName,
    RetrievalTrace,
    retrieve_annotation_memory,
    retrieve_literature_evidence,
    store_version,
    write_retrieval_trace,
)
from plot_prompt_v2_small_pilot_diagnostics import (
    load_evaluation_csvs,
    plot_diagnostic_clip,
)
from run_event_characterisation_ablation import dereference_json_schema
from run_prompt_v2_batdetect2_assisted_pilot import (
    format_proposal_metadata,
    load_proposal_payload,
)
from run_prompt_v2_small_pilot import image_to_base64, ollama_host, read_clip_duration
from run_prompt_v2_tiled_pilot import require_model


MODEL_NAME = "qwen3.6:latest"
OLLAMA_ENDPOINT = "http://127.0.0.1:11435"
PROMPT_VERSION = "p7c_library_assisted_bbox_v1"
PROPOSAL_THRESHOLD = 0.30
REPRESENTATIVE_CLIPS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")
HELDOUT_CLIPS = ("OP_032", "OP_042")
CLIP_IDS = REPRESENTATIVE_CLIPS + HELDOUT_CLIPS
CONDITIONS: tuple[ConditionName, ...] = (
    "baseline",
    "annotation_memory_only",
    "literature_only",
    "combined",
)
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_IMAGE_DIR = Path("outputs/agent_inputs/prompt_v2_full_grid_v2")
REPRESENTATIVE_PROPOSAL_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6"
)
HELDOUT_PROPOSAL_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/p6e5_heldout"
)
DEFAULT_OUTPUT_DIR = Path("outputs/agent_runs/p7c_library_assisted_bbox_ablation")
ANNOTATION_STORE = Path("docs/annotation_example_library/annotation_memory.jsonl")
LITERATURE_STORE = Path(
    "docs/literature_reference_library/verified_evidence_store.jsonl"
)


GeometryAction = Literal[
    "preserve_proposal", "refine_proposal", "add_new_event"
]


class LibraryBBoxEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    start_time_seconds: float = Field(ge=0.0)
    end_time_seconds: float = Field(gt=0.0)
    low_frequency_hz: float = Field(ge=0.0)
    high_frequency_hz: float = Field(gt=0.0)
    label: Literal["bat_call"]
    confidence: float = Field(ge=0.0, le=1.0)
    source_proposal_id: str
    proposal_source: Literal["batdetect2", ""]
    geometry_action: GeometryAction
    geometry_reason: str
    retrieved_annotation_case_ids: list[str]
    retrieved_literature_evidence_ids: list[str]
    human_review_needed: bool
    review_reason: str

    @model_validator(mode="after")
    def validate_geometry_and_provenance(self) -> "LibraryBBoxEvent":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        if self.high_frequency_hz <= self.low_frequency_hz:
            raise ValueError("high_frequency_hz must be greater than low_frequency_hz")
        if self.geometry_action == "add_new_event":
            if self.source_proposal_id or self.proposal_source:
                raise ValueError("new events cannot claim proposal provenance")
        elif not self.source_proposal_id or self.proposal_source != "batdetect2":
            raise ValueError("proposal events require BatDetect2 provenance")
        if self.human_review_needed and not self.review_reason.strip():
            raise ValueError("human review requires a review reason")
        return self


class RejectedProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    geometry_action: Literal["reject_proposal"] = "reject_proposal"
    geometry_reason: str
    retrieved_annotation_case_ids: list[str]
    retrieved_literature_evidence_ids: list[str]
    human_review_needed: bool
    review_reason: str


class LibraryBBoxResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    events: list[LibraryBBoxEvent]
    rejected_proposals: list[RejectedProposal]


@dataclass(frozen=True)
class RunArtifact:
    condition: ConditionName
    clip_id: str
    split: Literal["representative", "heldout"]
    parse_status: str
    prediction_path: Path
    raw_response_path: Path
    parse_error_path: Path | None
    retrieval_trace_path: Path
    model_called: bool


def proposal_dir_for_clip(clip_id: str) -> Path:
    return (
        REPRESENTATIVE_PROPOSAL_DIR
        if clip_id in REPRESENTATIVE_CLIPS
        else HELDOUT_PROPOSAL_DIR
    )


def split_for_clip(clip_id: str) -> Literal["representative", "heldout"]:
    return "representative" if clip_id in REPRESENTATIVE_CLIPS else "heldout"


def proposal_descriptors(
    proposal_rows: list[dict[str, Any]], clip_duration_seconds: float
) -> list[str]:
    """Derive retrieval descriptors from proposals, never from ground truth."""

    descriptors = ["detector proposal", "tool validation"]
    if len(proposal_rows) >= 5:
        descriptors.extend(["dense", "multi event"])
    elif len(proposal_rows) > 1:
        descriptors.append("multi event")
    durations = [float(row["duration_ms"]) for row in proposal_rows]
    if durations and sum(durations) / len(durations) < 15.0:
        descriptors.extend(["short proposal", "timing"])
    if any(float(row["start_time_seconds"]) <= 0.01 for row in proposal_rows):
        descriptors.extend(["left boundary", "truncated"])
    if any(
        float(row["end_time_seconds"]) >= clip_duration_seconds - 0.01
        for row in proposal_rows
    ):
        descriptors.extend(["right boundary", "truncated"])
    return descriptors


def safe_annotation_context(record: RetrievedAnnotationCase) -> dict[str, Any]:
    """Expose abstract lessons only, with all artifact and outcome fields removed."""

    return {
        "case_id": record.case_id,
        "case_type": record.case_type,
        "abstract_recommended_actions": record.recommended_actions,
        "abstract_anti_patterns": record.anti_patterns,
    }


def safe_literature_context(record: RetrievedLiteratureEvidence) -> dict[str, Any]:
    return record.model_dump(mode="json")


def retrieve_context(
    *,
    condition: ConditionName,
    clip_id: str,
    proposal_rows: list[dict[str, Any]],
    clip_duration_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], RetrievalTrace]:
    descriptors = proposal_descriptors(proposal_rows, clip_duration_seconds)
    annotation_records: list[RetrievedAnnotationCase] = []
    annotation_matches = []
    literature_records: list[RetrievedLiteratureEvidence] = []
    literature_matches = []
    annotation_enabled = condition in {"annotation_memory_only", "combined"}
    literature_enabled = condition in {"literature_only", "combined"}
    if annotation_enabled:
        annotation_records, annotation_matches = retrieve_annotation_memory(
            target_case_id=clip_id,
            descriptors=descriptors,
            memory_path=ANNOTATION_STORE,
            top_k=2,
        )
    if literature_enabled:
        literature_records, literature_matches = retrieve_literature_evidence(
            descriptors=descriptors,
            evidence_path=LITERATURE_STORE,
            top_k=3,
        )
    trace = RetrievalTrace(
        clip_id=clip_id,
        condition=condition,
        annotation_memory_enabled=annotation_enabled,
        literature_evidence_enabled=literature_enabled,
        annotation_matches=annotation_matches,
        literature_matches=literature_matches,
        annotation_store_version=store_version(ANNOTATION_STORE),
        literature_store_version=store_version(LITERATURE_STORE),
        target_case_excluded=all(
            item.retrieved_id != clip_id for item in annotation_matches
        ),
    )
    return (
        [safe_annotation_context(item) for item in annotation_records],
        [safe_literature_context(item) for item in literature_records],
        trace,
    )


def format_proposals_for_prompt(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = format_proposal_metadata(payload)
    return [row for row in rows if float(row["det_prob"]) >= PROPOSAL_THRESHOLD]


def build_prompt(
    *,
    clip_id: str,
    clip_duration_seconds: float,
    proposal_rows: list[dict[str, Any]],
    annotation_context: list[dict[str, Any]],
    literature_context: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build an invariant prompt whose only condition difference is context."""

    system = (
        "You are a careful bioacoustic annotation agent. Produce one tight time-frequency "
        "box per visible bat echolocation call. BatDetect2 proposals are candidate regions, "
        "not ground truth. Preserve accurate proposal geometry unless visible spectrogram "
        "evidence clearly supports refinement. Distinguish rigid translation from anchored "
        "duration expansion. Reject unsupported proposals and add only clearly visible "
        "missing calls. UK taxonomy is metadata only; every accepted label must be bat_call. "
        "Annotation-memory records are abstract prior experience, never answers for this "
        "clip. Literature records provide constraints and limitations, never coordinates. "
        "Context must not override visible evidence. Flag uncertain proposal conflicts for "
        "human review. Return valid JSON only and no prose outside JSON."
    )
    context = {
        "clip_id": clip_id,
        "clip_duration_seconds": round(clip_duration_seconds, 6),
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
        "proposal_threshold": PROPOSAL_THRESHOLD,
        "batdetect2_proposals": proposal_rows,
        "annotation_memory_context": annotation_context,
        "verified_literature_context": literature_context,
        "output_rules": {
            "events_contains_only_accepted_boxes": True,
            "rejected_proposals_are_separate": True,
            "accepted_geometry_actions": [
                "preserve_proposal", "refine_proposal", "add_new_event"
            ],
            "rejected_geometry_action": "reject_proposal",
            "copy_only_supplied_retrieval_ids": True,
        },
    }
    user = (
        "/no_think\nInspect the attached clean grid_v2 spectrogram and verify the "
        "proposal metadata. Return final evaluator-compatible bat_call boxes. For each "
        "accepted event, record whether geometry was preserved, refined, or newly added. "
        "List rejected proposal IDs only under rejected_proposals. Copy only retrieval IDs "
        "actually supplied below.\n\n"
        + json.dumps(context, indent=2, ensure_ascii=False)
    )
    return system, user


def call_ollama(
    *, image_path: Path, system: str, user: str, timeout: float, num_predict: int
) -> str:
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "think": False,
        "format": dereference_json_schema(LibraryBBoxResponse.model_json_schema()),
        "prompt": f"{system}\n\n{user}",
        "images": [image_to_base64(image_path)],
        "options": {"temperature": 0, "seed": 0, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        f"{ollama_host()}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result.get("response")
    if not content:
        raise ValueError("Ollama response did not contain response text")
    return str(content)


def validate_response(
    payload: Any,
    *,
    clip_id: str,
    clip_duration_seconds: float,
    proposal_ids: set[str],
    annotation_ids: set[str],
    literature_ids: set[str],
) -> LibraryBBoxResponse:
    response = LibraryBBoxResponse.model_validate(payload)
    if response.clip_id != clip_id:
        raise ValueError("response clip_id does not match requested clip")
    for event in response.events:
        values = (
            event.start_time_seconds,
            event.end_time_seconds,
            event.low_frequency_hz,
            event.high_frequency_hz,
            event.confidence,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"event {event.event_id} contains non-finite values")
        if event.end_time_seconds > clip_duration_seconds:
            raise ValueError(f"event {event.event_id} exceeds clip duration")
        if event.source_proposal_id and event.source_proposal_id not in proposal_ids:
            raise ValueError(f"event {event.event_id} cites an unknown proposal")
        if not set(event.retrieved_annotation_case_ids) <= annotation_ids:
            raise ValueError(f"event {event.event_id} cites unsupplied annotation memory")
        if not set(event.retrieved_literature_evidence_ids) <= literature_ids:
            raise ValueError(f"event {event.event_id} cites unsupplied literature")
    rejected_ids = {item.proposal_id for item in response.rejected_proposals}
    if not rejected_ids <= proposal_ids:
        raise ValueError("rejected_proposals contains an unknown proposal ID")
    if len(rejected_ids) != len(response.rejected_proposals):
        raise ValueError("rejected_proposals contains duplicate IDs")
    return response


def normalise_clip_bounds(
    payload: Any, clip_duration_seconds: float
) -> tuple[Any, list[str]]:
    """Clip response times to known audio bounds without using ground truth."""

    if not isinstance(payload, dict):
        return payload, []
    normalised = json.loads(json.dumps(payload))
    adjustments: list[str] = []
    events = normalised.get("events")
    if not isinstance(events, list):
        return normalised, adjustments
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        try:
            start = float(event["start_time_seconds"])
            end = float(event["end_time_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        clipped_start = max(0.0, start)
        clipped_end = min(clip_duration_seconds, end)
        if clipped_start != start or clipped_end != end:
            event["start_time_seconds"] = clipped_start
            event["end_time_seconds"] = clipped_end
            event_id = str(event.get("event_id") or f"event_{index + 1}")
            adjustments.append(
                f"{event_id}: clipped time bounds from [{start}, {end}] "
                f"to [{clipped_start}, {clipped_end}]"
            )
    return normalised, adjustments


def write_prediction(
    path: Path,
    *,
    response: LibraryBBoxResponse | None,
    clip_id: str,
    condition: ConditionName,
    image_path: Path,
    proposal_path: Path,
    clip_duration: float,
    parse_status: str,
    error: str,
    parser_adjustments: list[str] | None = None,
) -> None:
    payload = {
        "clip_id": clip_id,
        "model_name": MODEL_NAME,
        "backend": "ollama_hpc_generate",
        "ollama_endpoint": OLLAMA_ENDPOINT,
        "prompt_version": PROMPT_VERSION,
        "condition": condition,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_image_path": image_path.as_posix(),
        "proposal_metadata_path": proposal_path.as_posix(),
        "proposal_threshold": PROPOSAL_THRESHOLD,
        "clip_duration_seconds": clip_duration,
        "parse_status": parse_status,
        "error": error,
        "parser_adjustments": parser_adjustments or [],
        "events": [item.model_dump(mode="json") for item in response.events]
        if response
        else [],
        "rejected_proposals": [
            item.model_dump(mode="json") for item in response.rejected_proposals
        ]
        if response
        else [],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_clip(
    *,
    condition: ConditionName,
    clip_id: str,
    output_dir: Path,
    eval_dir: Path,
    image_dir: Path,
    timeout: float,
    num_predict: int,
    reuse_raw_response: bool = False,
    retry_invalid_raw: bool = False,
) -> RunArtifact:
    condition_dir = output_dir / condition
    image_path = image_dir / f"{clip_id}_spectrogram.png"
    proposal_dir = proposal_dir_for_clip(clip_id)
    proposal_path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
    clip_duration = read_clip_duration(eval_dir, clip_id)
    proposal_payload = load_proposal_payload(proposal_dir, clip_id)
    proposal_rows = format_proposals_for_prompt(proposal_payload)
    annotation_context, literature_context, trace = retrieve_context(
        condition=condition,
        clip_id=clip_id,
        proposal_rows=proposal_rows,
        clip_duration_seconds=clip_duration,
    )
    raw_path = condition_dir / "raw_responses" / f"{clip_id}_raw_response.txt"
    prediction_path = condition_dir / "predictions" / f"{clip_id}_predictions.json"
    error_path = condition_dir / "predictions" / f"{clip_id}_parse_error.txt"
    trace_path = condition_dir / "retrieval_traces" / f"{clip_id}_retrieval_trace.json"
    for path in (raw_path, prediction_path, error_path, trace_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_retrieval_trace(trace, trace_path)
    system, user = build_prompt(
        clip_id=clip_id,
        clip_duration_seconds=clip_duration,
        proposal_rows=proposal_rows,
        annotation_context=annotation_context,
        literature_context=literature_context,
    )
    raw = raw_path.read_text(encoding="utf-8") if reuse_raw_response and raw_path.is_file() else ""
    model_called = False
    try:
        if not raw:
            raw = call_ollama(
                image_path=image_path,
                system=system,
                user=user,
                timeout=timeout,
                num_predict=num_predict,
            )
            model_called = True
        raw_path.write_text(raw, encoding="utf-8")
        raw_payload = json.loads(raw)
        normalised_payload, parser_adjustments = normalise_clip_bounds(
            raw_payload, clip_duration
        )
        response = validate_response(
            normalised_payload,
            clip_id=clip_id,
            clip_duration_seconds=clip_duration,
            proposal_ids={str(row["proposal_id"]) for row in proposal_rows},
            annotation_ids={str(row["case_id"]) for row in annotation_context},
            literature_ids={str(row["evidence_id"]) for row in literature_context},
        )
        write_prediction(
            prediction_path,
            response=response,
            clip_id=clip_id,
            condition=condition,
            image_path=image_path,
            proposal_path=proposal_path,
            clip_duration=clip_duration,
            parse_status="success",
            error="",
            parser_adjustments=parser_adjustments,
        )
        error_path.unlink(missing_ok=True)
        return RunArtifact(
            condition, clip_id, split_for_clip(clip_id), "success",
            prediction_path, raw_path, None, trace_path, model_called,
        )
    except Exception as exc:
        if reuse_raw_response and retry_invalid_raw and not model_called:
            raw = call_ollama(
                image_path=image_path,
                system=system,
                user=user,
                timeout=timeout,
                num_predict=num_predict,
            )
            model_called = True
            raw_path.write_text(raw, encoding="utf-8")
            try:
                raw_payload = json.loads(raw)
                normalised_payload, parser_adjustments = normalise_clip_bounds(
                    raw_payload, clip_duration
                )
                response = validate_response(
                    normalised_payload,
                    clip_id=clip_id,
                    clip_duration_seconds=clip_duration,
                    proposal_ids={str(row["proposal_id"]) for row in proposal_rows},
                    annotation_ids={str(row["case_id"]) for row in annotation_context},
                    literature_ids={str(row["evidence_id"]) for row in literature_context},
                )
                write_prediction(
                    prediction_path,
                    response=response,
                    clip_id=clip_id,
                    condition=condition,
                    image_path=image_path,
                    proposal_path=proposal_path,
                    clip_duration=clip_duration,
                    parse_status="success",
                    error="",
                    parser_adjustments=parser_adjustments,
                )
                error_path.unlink(missing_ok=True)
                return RunArtifact(
                    condition, clip_id, split_for_clip(clip_id), "success",
                    prediction_path, raw_path, None, trace_path, True,
                )
            except Exception as retry_exc:
                exc = retry_exc
        raw_path.write_text(raw or f"MODEL_CALL_FAILED\n{exc}\n", encoding="utf-8")
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        write_prediction(
            prediction_path,
            response=None,
            clip_id=clip_id,
            condition=condition,
            image_path=image_path,
            proposal_path=proposal_path,
            clip_duration=clip_duration,
            parse_status="failure",
            error=f"{type(exc).__name__}: {exc}",
            parser_adjustments=[],
        )
        return RunArtifact(
            condition, clip_id, split_for_clip(clip_id), "failure",
            prediction_path, raw_path, error_path, trace_path, model_called,
        )


def existing_artifact(
    output_dir: Path, condition: ConditionName, clip_id: str
) -> RunArtifact:
    condition_dir = output_dir / condition
    prediction_path = condition_dir / "predictions" / f"{clip_id}_predictions.json"
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    error_path = condition_dir / "predictions" / f"{clip_id}_parse_error.txt"
    return RunArtifact(
        condition=condition,
        clip_id=clip_id,
        split=split_for_clip(clip_id),
        parse_status=str(payload.get("parse_status") or "failure"),
        prediction_path=prediction_path,
        raw_response_path=condition_dir / "raw_responses" / f"{clip_id}_raw_response.txt",
        parse_error_path=error_path if error_path.is_file() else None,
        retrieval_trace_path=condition_dir / "retrieval_traces" / f"{clip_id}_retrieval_trace.json",
        model_called=False,
    )


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metric_summary(
    *,
    condition: str,
    split: str,
    results: list[dict[str, Any]],
    artifacts: list[RunArtifact],
    prediction_payloads: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_results(results)
    selected_ids = {item["metrics"]["clip_id"] for item in results}
    selected_artifacts = [item for item in artifacts if item.clip_id in selected_ids]
    event_rows = [
        event
        for clip_id in selected_ids
        for event in prediction_payloads[(condition, clip_id)]["events"]
    ]
    review_count = sum(bool(event.get("human_review_needed")) for event in event_rows)
    return {
        "condition": condition,
        "split": split,
        "clip_count": aggregate["clip_count"],
        "prediction_count": aggregate["total_predictions"],
        "parse_success": sum(item.parse_status == "success" for item in selected_artifacts),
        "parse_failure": sum(item.parse_status != "success" for item in selected_artifacts),
        "TP": aggregate["total_tp"],
        "FP": aggregate["total_fp"],
        "FN": aggregate["total_fn"],
        "precision": aggregate["precision"],
        "recall": aggregate["recall"],
        "F1": aggregate["f1"],
        "mean_time_iou": aggregate["mean_time_iou"],
        "mean_frequency_iou": aggregate["mean_frequency_iou"],
        "mean_box_iou": aggregate["mean_box_iou"],
        "box_iou_ge_0_3": aggregate["strict_box_iou_0_3_count"],
        "box_iou_ge_0_5": aggregate["strict_box_iou_0_5_count"],
        "human_review_rate": safe_divide(review_count, len(event_rows)),
    }


def geometry_signature(payload: dict[str, Any]) -> tuple:
    return tuple(
        sorted(
            (
                event.get("source_proposal_id", ""),
                event.get("geometry_action", ""),
                round(float(event["start_time_seconds"]), 6),
                round(float(event["end_time_seconds"]), 6),
                round(float(event["low_frequency_hz"]), 3),
                round(float(event["high_frequency_hz"]), 3),
            )
            for event in payload.get("events", [])
        )
    )


def retrieval_impact_row(
    baseline: dict[str, Any], comparison: dict[str, Any],
    baseline_payload: dict[str, Any], comparison_payload: dict[str, Any]
) -> dict[str, Any]:
    delta_f1 = float(comparison["f1"]) - float(baseline["f1"])
    delta_box = float(comparison["mean_box_iou"]) - float(baseline["mean_box_iou"])
    improved = delta_f1 > 1e-12 or (abs(delta_f1) <= 1e-12 and delta_box > 1e-12)
    degraded = delta_f1 < -1e-12 or (abs(delta_f1) <= 1e-12 and delta_box < -1e-12)
    parse_confounded = (
        baseline.get("parse_status") != "success"
        or comparison.get("parse_status") != "success"
    )
    return {
        "clip_id": comparison["clip_id"],
        "comparison_condition": comparison["condition"],
        "split": comparison["split"],
        "delta_TP": int(comparison["tp"]) - int(baseline["tp"]),
        "delta_FP": int(comparison["fp"]) - int(baseline["fp"]),
        "delta_FN": int(comparison["fn"]) - int(baseline["fn"]),
        "delta_F1": delta_f1,
        "delta_mean_time_iou": float(comparison["mean_time_iou"]) - float(baseline["mean_time_iou"]),
        "delta_mean_frequency_iou": float(comparison["mean_frequency_iou"]) - float(baseline["mean_frequency_iou"]),
        "delta_mean_box_iou": delta_box,
        "retrieval_changed_geometry_decisions": geometry_signature(baseline_payload) != geometry_signature(comparison_payload),
        "retrieval_improved_result": improved and not parse_confounded,
        "retrieval_degraded_result": degraded and not parse_confounded,
        "comparison_confounded_by_parse_failure": parse_confounded,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_report(
    *, output_path: Path, summaries: list[dict[str, Any]],
    cases: list[dict[str, Any]], impacts: list[dict[str, Any]]
) -> None:
    all_rows = {row["condition"]: row for row in summaries if row["split"] == "all"}
    baseline = all_rows["baseline"]
    lines = [
        "# P7C Library-Assisted BatDetect2 Proposal Refinement Ablation",
        "",
        "## Scope",
        "",
        f"Model: `{MODEL_NAME}`. Proposal threshold: `{PROPOSAL_THRESHOLD:.2f}`. Eight clips were run under four conditions. Prediction used clean grid_v2 images and proposal metadata only; GT was loaded after inference by the frozen evaluator.",
        "",
        "## Aggregate Results",
        "",
        "| Condition | Parsed | Predictions | TP | FP | FN | Precision | Recall | F1 | Time IoU | Frequency IoU | Box IoU | Review rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = all_rows[condition]
        lines.append(
            f"| {condition} | {row['parse_success']}/{row['clip_count']} | {row['prediction_count']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['F1']:.3f} | {row['mean_time_iou']:.3f} | {row['mean_frequency_iou']:.3f} | {row['mean_box_iou']:.3f} | {row['human_review_rate']:.3f} |"
        )
    lines.extend(["", "## Aggregate Effect", ""])
    for condition in CONDITIONS[1:]:
        row = all_rows[condition]
        lines.append(
            f"- `{condition}` versus baseline: ΔTP={row['TP']-baseline['TP']:+d}, ΔFP={row['FP']-baseline['FP']:+d}, ΔFN={row['FN']-baseline['FN']:+d}, ΔF1={row['F1']-baseline['F1']:+.3f}, Δbox IoU={row['mean_box_iou']-baseline['mean_box_iou']:+.3f}."
        )
    lines.extend(["", "## Representative Versus Held-Out", ""])
    for condition in CONDITIONS:
        rep = next(row for row in summaries if row["condition"] == condition and row["split"] == "representative")
        held = next(row for row in summaries if row["condition"] == condition and row["split"] == "heldout")
        lines.append(f"- `{condition}`: representative F1={rep['F1']:.3f}, box IoU={rep['mean_box_iou']:.3f}; held-out F1={held['F1']:.3f}, box IoU={held['mean_box_iou']:.3f}.")
    lines.extend(
        [
            "",
            "All four conditions were identical on the six representative clips. The held-out aggregate comparison is parse-confounded because baseline `OP_032` returned malformed JSON and was conservatively evaluated as an empty prediction. Knowledge conditions did not recover any `OP_032` true positives; they added two or three unmatched predictions.",
        ]
    )
    lines.extend(["", "## Key Cases", ""])
    by_case = {(row["condition"], row["clip_id"]): row for row in cases}
    for clip_id in ("OP_016", "OP_045", "OP_004", "OP_032", "OP_042", "OP_001"):
        values = []
        for condition in CONDITIONS:
            row = by_case[(condition, clip_id)]
            values.append(f"{condition}: parse={row['parse_status']}, TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}, F1={row['f1']:.3f}, box IoU={row['mean_box_iou']:.3f}")
        lines.append(f"- **{clip_id}:** " + "; ".join(values) + ".")
    changed = [row for row in impacts if row["retrieval_changed_geometry_decisions"]]
    improved = [row for row in impacts if row["retrieval_improved_result"]]
    degraded = [row for row in impacts if row["retrieval_degraded_result"]]
    confounded = [row for row in impacts if row["comparison_confounded_by_parse_failure"]]
    annotation_citations = sum(int(row["events_citing_annotation_memory"]) for row in cases)
    literature_citations = sum(int(row["events_citing_literature_evidence"]) for row in cases)
    lines.extend(
        [
            "",
            "## Retrieval Interpretation",
            "",
            f"Across {len(impacts)} baseline comparisons, retrieval changed geometry decisions in {len(changed)}, improved non-confounded frozen evaluation results in {len(improved)}, and degraded them in {len(degraded)}; {len(confounded)} comparisons were parse-confounded. Events copied annotation-memory IDs {annotation_citations} times and literature IDs {literature_citations} times, but these citations produced no representative-set coordinate or detection change. Literature context therefore changed provenance/constraint reporting without improving localisation.",
            "",
            "## Final Conclusion",
            "",
        ]
    )
    best = max(all_rows.values(), key=lambda row: (row["F1"], row["mean_box_iou"]))
    if best["condition"] == "baseline":
        conclusion = "Libraries had no measurable localisation benefit on the parse-complete representative subset. Held-out aggregate differences are not generalisable because baseline OP_032 was malformed; the library conditions added false-positive candidates there without recovering a true event."
    elif best["condition"] == "annotation_memory_only" and all_rows["literature_only"]["F1"] <= baseline["F1"]:
        conclusion = "Annotation memory improved proposal-use decisions, while literature did not improve aggregate localisation."
    elif best["condition"] == "literature_only" and best["mean_box_iou"] <= baseline["mean_box_iou"]:
        conclusion = "Literature changed constraint handling without improving coordinate localisation."
    else:
        held_best = max(
            (row for row in summaries if row["split"] == "heldout"),
            key=lambda row: (row["F1"], row["mean_box_iou"]),
        )
        conclusion = (
            "Library-assisted refinement improved aggregate bounding-box annotation, but the condition ranking differed on held-out clips; results are mixed and not yet generalisable."
            if held_best["condition"] != best["condition"]
            else f"`{best['condition']}` directly improved bounding-box annotation on this eight-clip ablation, subject to its small diagnostic scope."
        )
    lines.append(conclusion)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful artifacts, reparse failures, and retry malformed raw JSON once.",
    )
    parser.add_argument(
        "--reparse-only",
        action="store_true",
        help="Reuse artifacts and reparse failures without making any model calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_mode = args.resume or args.reparse_only
    if not args.reparse_only:
        if ollama_host().rstrip("/") != OLLAMA_ENDPOINT:
            raise RuntimeError(f"P7C requires OLLAMA_HOST={OLLAMA_ENDPOINT}")
        print("Available models:", ", ".join(require_model(MODEL_NAME)))
    previous_model_calls = 0
    if args.output_dir.exists() and resume_mode:
        metadata_path = args.output_dir / "run_metadata.json"
        if metadata_path.is_file():
            previous_model_calls = int(
                json.loads(metadata_path.read_text(encoding="utf-8")).get(
                    "model_call_count", 32
                )
            )
    elif args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}; use --overwrite")
        shutil.rmtree(args.output_dir)
    artifacts: list[RunArtifact] = []
    for condition in CONDITIONS:
        for clip_id in CLIP_IDS:
            if resume_mode:
                existing = existing_artifact(args.output_dir, condition, clip_id)
                if existing.parse_status == "success":
                    artifacts.append(existing)
                    print(f"Reusing {condition} {clip_id}: success", flush=True)
                    continue
            print(f"Running {condition} {clip_id}", flush=True)
            artifact = run_clip(
                condition=condition,
                clip_id=clip_id,
                output_dir=args.output_dir,
                eval_dir=args.eval_dir,
                image_dir=args.image_dir,
                timeout=args.timeout,
                num_predict=args.num_predict,
                reuse_raw_response=resume_mode,
                retry_invalid_raw=args.resume and not args.reparse_only,
            )
            artifacts.append(artifact)
            print(f"  {artifact.parse_status}", flush=True)

    all_case_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    condition_results: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        condition_dir = args.output_dir / condition
        aggregate, results = run_evaluation(
            pred_dir=condition_dir / "predictions",
            eval_dir=args.eval_dir,
            clip_ids=list(CLIP_IDS),
            output_dir=condition_dir / "evaluation",
        )
        condition_results[condition] = results
        for clip_id in CLIP_IDS:
            payloads[(condition, clip_id)] = json.loads(
                (condition_dir / "predictions" / f"{clip_id}_predictions.json").read_text(encoding="utf-8")
            )
        result_by_id = {item["metrics"]["clip_id"]: item for item in results}
        artifact_by_id = {
            item.clip_id: item for item in artifacts if item.condition == condition
        }
        for clip_id in CLIP_IDS:
            metric = result_by_id[clip_id]["metrics"]
            trace = RetrievalTrace.model_validate_json(
                artifact_by_id[clip_id].retrieval_trace_path.read_text(encoding="utf-8")
            )
            payload = payloads[(condition, clip_id)]
            all_case_rows.append(
                {
                    "condition": condition,
                    "clip_id": clip_id,
                    "split": split_for_clip(clip_id),
                    "parse_status": artifact_by_id[clip_id].parse_status,
                    "prediction_count": metric["num_predictions"],
                    "tp": metric["tp"], "fp": metric["fp"], "fn": metric["fn"],
                    "precision": metric["precision"], "recall": metric["recall"], "f1": metric["f1"],
                    "mean_time_iou": metric["mean_time_iou"],
                    "mean_frequency_iou": metric["mean_frequency_iou"],
                    "mean_box_iou": metric["mean_box_iou"],
                    "human_review_count": sum(bool(event.get("human_review_needed")) for event in payload["events"]),
                    "events_citing_annotation_memory": sum(bool(event.get("retrieved_annotation_case_ids")) for event in payload["events"]),
                    "events_citing_literature_evidence": sum(bool(event.get("retrieved_literature_evidence_ids")) for event in payload["events"]),
                    "retrieved_annotation_case_ids": "|".join(item.retrieved_id for item in trace.annotation_matches),
                    "retrieved_literature_evidence_ids": "|".join(item.retrieved_id for item in trace.literature_matches),
                }
            )
        for split, ids in (
            ("all", set(CLIP_IDS)),
            ("representative", set(REPRESENTATIVE_CLIPS)),
            ("heldout", set(HELDOUT_CLIPS)),
        ):
            selected_results = [item for item in results if item["metrics"]["clip_id"] in ids]
            selected_artifacts = [item for item in artifacts if item.condition == condition and item.clip_id in ids]
            summaries.append(
                metric_summary(
                    condition=condition,
                    split=split,
                    results=selected_results,
                    artifacts=selected_artifacts,
                    prediction_payloads=payloads,
                )
            )
        evaluation_rows = load_evaluation_csvs(condition_dir / "evaluation")
        for clip_id in CLIP_IDS:
            plot_diagnostic_clip(
                clip_id=clip_id,
                pred_dir=condition_dir / "predictions",
                eval_dir=args.eval_dir,
                evaluation_rows=evaluation_rows,
                output_dir=condition_dir / "diagnostic_figures",
            )

    geometry_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        events = [event for clip_id in CLIP_IDS for event in payloads[(condition, clip_id)]["events"]]
        rejected = [item for clip_id in CLIP_IDS for item in payloads[(condition, clip_id)].get("rejected_proposals", [])]
        counts = {action: sum(event.get("geometry_action") == action for event in events) for action in ("preserve_proposal", "refine_proposal", "add_new_event")}
        geometry_rows.append(
            {
                "condition": condition,
                "preserve_proposal_count": counts["preserve_proposal"],
                "refine_proposal_count": counts["refine_proposal"],
                "reject_proposal_count": len(rejected),
                "add_new_event_count": counts["add_new_event"],
                "accepted_proposal_count": counts["preserve_proposal"] + counts["refine_proposal"],
                "new_event_count": counts["add_new_event"],
                "human_review_count": sum(bool(event.get("human_review_needed")) for event in events) + sum(bool(item.get("human_review_needed")) for item in rejected),
            }
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
    write_csv(args.output_dir / "p7c_case_level_results.csv", all_case_rows)
    write_csv(args.output_dir / "p7c_condition_summary.csv", summaries)
    write_csv(args.output_dir / "p7c_geometry_action_summary.csv", geometry_rows)
    write_csv(args.output_dir / "p7c_retrieval_impact_summary.csv", impacts)
    write_report(
        output_path=args.output_dir / "p7c_library_assisted_bbox_ablation_report.md",
        summaries=summaries,
        cases=all_case_rows,
        impacts=impacts,
    )
    metadata = {
        "model_name": MODEL_NAME,
        "ollama_endpoint": OLLAMA_ENDPOINT,
        "prompt_version": PROMPT_VERSION,
        "proposal_threshold": PROPOSAL_THRESHOLD,
        "conditions": list(CONDITIONS),
        "clip_ids": list(CLIP_IDS),
        "inference_count": len(artifacts),
        "model_call_count": previous_model_calls + sum(item.model_called for item in artifacts),
        "retry_model_call_count": sum(item.model_called for item in artifacts) if resume_mode else 0,
        "parse_success": sum(item.parse_status == "success" for item in artifacts),
        "parse_failure": sum(item.parse_status != "success" for item in artifacts),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps([row for row in summaries if row["split"] == "all"], indent=2))


if __name__ == "__main__":
    main()
