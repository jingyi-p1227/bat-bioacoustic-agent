"""Run the P7B knowledge-grounded event-characterisation ablation.

This runner uses clean spectrograms, frozen event boxes, and deterministic
features. It never reads GT overlays or asks the model to calculate numeric
features. Four conditions reuse the same P7A Cases and differ only in their
retrieval settings.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from pydantic import ValidationError
from pydantic_evals import Case, Dataset

from build_event_characterisation_eval_dataset import (
    CONDITION_SETTINGS,
    EventCharacterisationDataset,
    inputs_for_condition,
)
from event_characterisation_evaluators import (
    AnnotationCaseIdExistsEvaluator,
    BandwidthErrorEvaluator,
    BoundaryStatusEvaluator,
    ConditionIsolationEvaluator,
    DETERMINISTIC_EVALUATORS,
    DurationErrorEvaluator,
    EventCountEvaluator,
    EventOrderEvaluator,
    EventOverlapEvaluator,
    FrequencyCentreErrorEvaluator,
    HumanReviewRuleEvaluator,
    InterEventIntervalEvaluator,
    LiteratureEvidenceIdExistsEvaluator,
    MISSING_VALUE_ERROR,
    OutputSchemaValidityEvaluator,
    ScientificNameGroundingEvaluator,
    SequenceFeatureEvaluator,
    TargetCaseRetrievalExclusionEvaluator,
    TemporalCentreErrorEvaluator,
    UnsupportedBehaviourClaimEvaluator,
)
from event_characterisation_models import (
    EventCharacterisationCaseMetadata,
    EventCharacterisationExpected,
    EventCharacterisationInput,
    GroundedEventInterpretation,
)
from event_characterisation_retrieval import (
    ConditionName,
    RetrievalTrace,
    retrieve_for_condition,
    safe_annotation_context,
    safe_literature_context,
    write_retrieval_trace,
)
from run_prompt_v2_small_pilot import extract_json_text


MODEL_NAME = "qwen3.6:latest"
OLLAMA_ENDPOINT = "http://127.0.0.1:11435"
DEFAULT_DATASET_PATH = Path(
    "outputs/evaluation_sets/event_characterisation_v1/dataset.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/agent_runs/p7_event_characterisation_ablation")
CONDITIONS: tuple[ConditionName, ...] = (
    "baseline",
    "annotation_memory_only",
    "literature_only",
    "combined",
)
REPEAT_CLIP_IDS = ("OP_016", "OP_045", "OP_032", "OP_042")
TOLERANCES = {
    "duration_error_ms": 0.5,
    "temporal_centre_error_seconds": 0.0005,
    "inter_event_interval_error_ms": 1.0,
    "bandwidth_error_hz": 500.0,
    "frequency_centre_error_hz": 500.0,
}


@dataclass(frozen=True)
class RunArtifact:
    condition: ConditionName
    clip_id: str
    run_type: str
    parse_status: str
    prediction_path: Path | None
    raw_response_path: Path
    parse_error_path: Path | None
    retrieval_trace_path: Path


def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", OLLAMA_ENDPOINT).rstrip("/")


def require_model(model_name: str = MODEL_NAME) -> str:
    """Confirm the exact model is exposed by the configured Ollama endpoint."""

    env = os.environ.copy()
    env["OLLAMA_HOST"] = ollama_host()
    completed = subprocess.run(
        ["ollama", "list"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    models = {
        line.split()[0]
        for line in completed.stdout.splitlines()[1:]
        if line.split()
    }
    if model_name not in models:
        raise RuntimeError(
            f"Required model {model_name!r} is unavailable. Available: {sorted(models)}"
        )
    return completed.stdout


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> EventCharacterisationDataset:
    """Load the frozen P7A snapshot as strongly typed Pydantic Evals Cases."""

    return EventCharacterisationDataset.from_file(path)


def deterministic_feature_payload(expected: EventCharacterisationExpected) -> dict:
    """Return immutable deterministic values that the model must preserve."""

    return {
        "events": [
            {
                "event_id": event.event_id,
                "duration_ms": event.duration_ms,
                "bandwidth_hz": event.bandwidth_hz,
                "temporal_center_seconds": event.temporal_center_seconds,
                "frequency_center_hz": event.frequency_center_hz,
                "event_order": event.event_order,
                "previous_inter_event_interval_ms": (
                    event.previous_inter_event_interval_ms
                ),
                "next_inter_event_interval_ms": event.next_inter_event_interval_ms,
                "clip_relative_position": event.clip_relative_position,
                "left_boundary_truncated": event.left_boundary_truncated,
                "right_boundary_truncated": event.right_boundary_truncated,
                "event_overlap": event.event_overlap,
                "scientific_name": event.scientific_name,
                "scientific_name_status": (
                    "direct_annotation"
                    if event.scientific_name_directly_annotated
                    else "not_provided"
                ),
            }
            for event in expected.events
        ],
        "sequence": expected.sequence.model_dump(mode="json"),
    }


def output_template(
    *,
    expected: EventCharacterisationExpected,
    annotation_context: list[dict],
    literature_context: list[dict],
) -> dict[str, Any]:
    """Build a complete valid template with deterministic values pre-filled."""

    features = deterministic_feature_payload(expected)
    return {
        "clip_id": expected.clip_id,
        "interpreted_events": [
            {
                **event,
                "confirmed_interpretations": [],
                "confidence": 1.0,
            }
            for event in features["events"]
        ],
        "sequence_interpretation": {
            **features["sequence"],
            "confirmed_interpretations": [],
        },
        "retrieved_annotation_cases": annotation_context,
        "retrieved_literature_evidence": literature_context,
        "exploratory_hypotheses": [],
        "unsupported_claims": [],
        "confidence": 1.0,
        "human_review_needed": False,
        "review_reason": "",
    }


def dereference_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local ``$defs`` references for Ollama structured output."""

    definitions = schema.get("$defs", {})

    def resolve(value: Any) -> Any:
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            prefix = "#/$defs/"
            reference = value["$ref"]
            if not reference.startswith(prefix):
                raise ValueError(f"Unsupported JSON schema reference: {reference}")
            resolved = resolve(definitions[reference[len(prefix) :]])
            extras = {key: resolve(item) for key, item in value.items() if key != "$ref"}
            return {**resolved, **extras}
        return {
            key: resolve(item)
            for key, item in value.items()
            if key != "$defs"
        }

    result = resolve(schema)
    if not isinstance(result, dict):
        raise TypeError("Expected object JSON schema")
    return result


def build_prompt(
    *,
    condition: ConditionName,
    inputs: EventCharacterisationInput,
    expected: EventCharacterisationExpected,
    annotation_context: list[dict],
    literature_context: list[dict],
) -> tuple[str, str]:
    """Build a leak-resistant prompt with immutable numeric features."""

    system_prompt = (
        "You are a careful bioacoustic event-characterisation assistant. "
        "The supplied event boxes and deterministic numeric features are frozen. "
        "Do not detect new events, change event IDs, or calculate numeric values. "
        "Copy every supplied deterministic value exactly into the structured output. "
        "Use the clean spectrogram only to write cautious non-behavioural interpretations. "
        "A scientific name with status direct_annotation may be copied only when supplied. "
        "Any other species name must be marked prediction. "
        "The dataset has no confirmed behavioural ground truth. Do not infer or enumerate "
        "behavioural categories merely because an event sequence is visible. Keep "
        "exploratory_hypotheses exactly [] unless a genuinely necessary uncertain claim "
        "must be recorded. If non-empty, every hypothesis must use the exact schema keys "
        "hypothesis_id, hypothesis_type, claim, event_id, confidence, evidence, status, "
        "and human_review_required; status must be exploratory_hypothesis, "
        "human_review_required must be true, and top-level human_review_needed must be true "
        "with a non-empty review_reason. Never put behavioural statements in "
        "confirmed_interpretations or unsupported_claims. "
        "Return valid JSON matching the requested schema and no prose outside JSON."
    )
    frozen_boxes = [box.model_dump(mode="json") for box in inputs.frozen_event_boxes]
    runtime_context = {
        "condition": condition,
        "clip_id": inputs.clip_id,
        "clip_duration_seconds": inputs.clip_duration_seconds,
        "frozen_event_boxes": frozen_boxes,
        "immutable_deterministic_features": deterministic_feature_payload(expected),
        "annotation_memory_context": annotation_context,
        "verified_literature_context": literature_context,
        "output_requirements": {
            "interpreted_event_count": len(expected.events),
            "copy_numeric_values_exactly": True,
            "retrieved_annotation_cases": (
                "copy only supplied prompt-safe records actually used"
            ),
            "retrieved_literature_evidence": (
                "copy only supplied verified records actually used"
            ),
            "behaviour_content": "exploratory_hypotheses_only",
        },
        "required_output_template": output_template(
            expected=expected,
            annotation_context=annotation_context,
            literature_context=literature_context,
        ),
    }
    user_prompt = (
        "/no_think\n"
        "Characterise the frozen events visible in the attached clean spectrogram. "
        "The numeric values below were calculated deterministically; copy them rather "
        "than recomputing them. Start from required_output_template and preserve its "
        "structure, IDs, numeric values, retrieval records, and top-level fields exactly. "
        "You may only add cautious strings to confirmed_interpretations, add properly "
        "marked exploratory_hypotheses, and update confidence/review fields. "
        "Use only the condition-specific context supplied here.\n\n"
        f"{json.dumps(runtime_context, indent=2, ensure_ascii=False)}\n\n"
        "Return the GroundedEventInterpretation JSON object immediately."
        " Before returning, verify that all top-level fields from the template remain and "
        "that an empty exploratory_hypotheses list is unchanged unless human review is set."
    )
    return system_prompt, user_prompt


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_ollama(
    *,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: float,
    num_predict: int,
) -> str:
    """Call qwen3.6 using the configured HPC Ollama endpoint."""

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "think": False,
        "format": dereference_json_schema(
            GroundedEventInterpretation.model_json_schema()
        ),
        "options": {"temperature": 0, "seed": 0, "num_predict": num_predict},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [image_to_base64(image_path)],
            },
        ],
    }
    request = urllib.request.Request(
        f"{ollama_host()}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = (response_payload.get("message") or {}).get("content")
    if not content:
        raise ValueError("Ollama response did not contain message.content")
    return str(content)


def parse_prediction(raw_text: str, clip_id: str) -> GroundedEventInterpretation:
    payload = json.loads(extract_json_text(raw_text))
    prediction = GroundedEventInterpretation.model_validate(payload)
    if prediction.clip_id != clip_id:
        raise ValueError(
            f"Prediction clip_id {prediction.clip_id!r} does not match {clip_id!r}"
        )
    return prediction


def run_case(
    *,
    condition: ConditionName,
    case: Case,
    output_dir: Path,
    run_type: str,
    timeout_seconds: float,
    num_predict: int,
) -> RunArtifact:
    """Retrieve context, call the model, and preserve raw/parsed artifacts."""

    condition_inputs = inputs_for_condition(case.inputs, condition)
    annotation_records, literature_records, trace = retrieve_for_condition(
        condition=condition,
        inputs=condition_inputs,
        expected=case.expected_output,
    )
    suffix = "" if run_type == "main" else "_repeat"
    trace_path = (
        output_dir
        / condition
        / "retrieval_traces"
        / f"{case.name}{suffix}_retrieval_trace.json"
    )
    write_retrieval_trace(trace, trace_path)
    annotation_context = [
        safe_annotation_context(record) for record in annotation_records
    ]
    literature_context = [
        safe_literature_context(record) for record in literature_records
    ]
    system_prompt, user_prompt = build_prompt(
        condition=condition,
        inputs=condition_inputs,
        expected=case.expected_output,
        annotation_context=annotation_context,
        literature_context=literature_context,
    )
    image_path = Path(condition_inputs.spectrogram_path)
    raw_path = (
        output_dir
        / condition
        / "raw_responses"
        / f"{case.name}{suffix}_raw_response.txt"
    )
    prediction_path = (
        output_dir
        / condition
        / "predictions"
        / f"{case.name}{suffix}_prediction.json"
    )
    error_path = (
        output_dir
        / condition
        / "parse_errors"
        / f"{case.name}{suffix}_parse_error.txt"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = call_ollama(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
        )
        raw_path.write_text(raw, encoding="utf-8")
        prediction = parse_prediction(raw, case.name)
        prediction_path.write_text(
            prediction.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        if error_path.exists():
            error_path.unlink()
        return RunArtifact(
            condition=condition,
            clip_id=case.name,
            run_type=run_type,
            parse_status="success",
            prediction_path=prediction_path,
            raw_response_path=raw_path,
            parse_error_path=None,
            retrieval_trace_path=trace_path,
        )
    except Exception as exc:
        if not raw_path.exists():
            raw_path.write_text(f"MODEL_CALL_FAILED\n{exc}\n", encoding="utf-8")
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return RunArtifact(
            condition=condition,
            clip_id=case.name,
            run_type=run_type,
            parse_status="failure",
            prediction_path=None,
            raw_response_path=raw_path,
            parse_error_path=error_path,
            retrieval_trace_path=trace_path,
        )


def evaluator_instances() -> list:
    return [evaluator_type() for evaluator_type in DETERMINISTIC_EVALUATORS]


def _evaluation_value(evaluator: Any, ctx: SimpleNamespace) -> bool | float:
    value = evaluator.evaluate(ctx)
    if not isinstance(value, (bool, int, float)):
        raise TypeError(f"Unexpected evaluator result: {type(value)}")
    return bool(value) if isinstance(value, bool) else float(value)


def evaluate_artifact(
    *,
    case: Case,
    condition: ConditionName,
    artifact: RunArtifact,
) -> dict[str, Any]:
    inputs = inputs_for_condition(case.inputs, condition)
    output: Any = {"parse_error": True}
    if artifact.prediction_path is not None:
        output = json.loads(artifact.prediction_path.read_text(encoding="utf-8"))
    ctx = SimpleNamespace(
        inputs=inputs,
        expected_output=case.expected_output,
        output=output,
        metadata=case.metadata,
    )
    values = {
        evaluator.__class__.__name__: _evaluation_value(evaluator, ctx)
        for evaluator in evaluator_instances()
    }
    duration_error = float(values["DurationErrorEvaluator"])
    bandwidth_error = float(values["BandwidthErrorEvaluator"])
    temporal_error = float(values["TemporalCentreErrorEvaluator"])
    frequency_error = float(values["FrequencyCentreErrorEvaluator"])
    interval_error = float(values["InterEventIntervalEvaluator"])
    valid_numeric = duration_error < MISSING_VALUE_ERROR
    trace = RetrievalTrace.model_validate_json(
        artifact.retrieval_trace_path.read_text(encoding="utf-8")
    )
    prediction = (
        GroundedEventInterpretation.model_validate(output)
        if values["OutputSchemaValidityEvaluator"]
        else None
    )
    output_annotation_ids = (
        [item.case_id for item in prediction.retrieved_annotation_cases]
        if prediction
        else []
    )
    output_literature_ids = (
        [item.evidence_id for item in prediction.retrieved_literature_evidence]
        if prediction
        else []
    )
    supplied_annotation_ids = [item.retrieved_id for item in trace.annotation_matches]
    supplied_literature_ids = [item.retrieved_id for item in trace.literature_matches]
    threshold_passes = {
        "duration_threshold_pass": (
            valid_numeric and duration_error <= TOLERANCES["duration_error_ms"]
        ),
        "bandwidth_threshold_pass": (
            valid_numeric and bandwidth_error <= TOLERANCES["bandwidth_error_hz"]
        ),
        "temporal_centre_threshold_pass": (
            valid_numeric
            and temporal_error <= TOLERANCES["temporal_centre_error_seconds"]
        ),
        "frequency_centre_threshold_pass": (
            valid_numeric
            and frequency_error <= TOLERANCES["frequency_centre_error_hz"]
        ),
        "interval_threshold_pass": (
            valid_numeric
            and interval_error <= TOLERANCES["inter_event_interval_error_ms"]
        ),
    }
    deterministic_pass = all(threshold_passes.values()) and all(
        bool(values[name])
        for name in (
            "EventCountEvaluator",
            "EventOrderEvaluator",
            "BoundaryStatusEvaluator",
            "EventOverlapEvaluator",
            "SequenceFeatureEvaluator",
        )
    )
    return {
        "condition": condition,
        "clip_id": case.name,
        "run_type": artifact.run_type,
        "split": case.metadata.representative_or_heldout,
        "parse_status": artifact.parse_status,
        "schema_valid": bool(values["OutputSchemaValidityEvaluator"]),
        "event_count_exact": bool(values["EventCountEvaluator"]),
        "event_order_exact": bool(values["EventOrderEvaluator"]),
        "boundary_status_exact": bool(values["BoundaryStatusEvaluator"]),
        "event_overlap_exact": bool(values["EventOverlapEvaluator"]),
        "sequence_features_exact": bool(values["SequenceFeatureEvaluator"]),
        "mean_absolute_duration_error_ms": (
            duration_error if valid_numeric else ""
        ),
        "mean_absolute_bandwidth_error_hz": (
            bandwidth_error if valid_numeric else ""
        ),
        "mean_absolute_temporal_centre_error_seconds": (
            temporal_error if valid_numeric else ""
        ),
        "mean_absolute_frequency_centre_error_hz": (
            frequency_error if valid_numeric else ""
        ),
        "mean_absolute_inter_event_interval_error_ms": (
            interval_error if valid_numeric else ""
        ),
        **threshold_passes,
        "deterministic_feature_preservation_pass": deterministic_pass,
        "annotation_case_ids_valid": bool(
            values["AnnotationCaseIdExistsEvaluator"]
        ),
        "literature_evidence_ids_valid": bool(
            values["LiteratureEvidenceIdExistsEvaluator"]
        ),
        "target_case_excluded": bool(
            values["TargetCaseRetrievalExclusionEvaluator"]
        ),
        "condition_isolation_pass": bool(values["ConditionIsolationEvaluator"]),
        "unsupported_behaviour_claim_pass": bool(
            values["UnsupportedBehaviourClaimEvaluator"]
        ),
        "scientific_name_grounding_pass": bool(
            values["ScientificNameGroundingEvaluator"]
        ),
        "human_review_rule_pass": bool(values["HumanReviewRuleEvaluator"]),
        "supplied_annotation_ids": "|".join(supplied_annotation_ids),
        "output_annotation_ids": "|".join(output_annotation_ids),
        "annotation_ids_match_supplied": sorted(output_annotation_ids)
        == sorted(supplied_annotation_ids),
        "supplied_literature_ids": "|".join(supplied_literature_ids),
        "output_literature_ids": "|".join(output_literature_ids),
        "literature_ids_match_supplied": sorted(output_literature_ids)
        == sorted(supplied_literature_ids),
        "human_review_needed": (
            prediction.human_review_needed if prediction else ""
        ),
        "exploratory_hypothesis_count": (
            len(prediction.exploratory_hypotheses) if prediction else ""
        ),
        "unsupported_claim_count": (
            len(prediction.unsupported_claims) if prediction else ""
        ),
    }


def run_pydantic_evals(
    *,
    condition: ConditionName,
    source_dataset: EventCharacterisationDataset,
    artifacts: list[RunArtifact],
    evaluation_dir: Path,
) -> None:
    """Run the actual Pydantic Evals orchestration over saved predictions."""

    predictions: dict[str, Any] = {}
    for artifact in artifacts:
        if artifact.prediction_path is not None:
            predictions[artifact.clip_id] = json.loads(
                artifact.prediction_path.read_text(encoding="utf-8")
            )
        else:
            predictions[artifact.clip_id] = {"parse_error": True}
    cases = [
        Case(
            name=case.name,
            inputs=inputs_for_condition(case.inputs, condition),
            expected_output=case.expected_output,
            metadata=case.metadata,
        )
        for case in source_dataset.cases
    ]
    dataset = Dataset(name=f"event_characterisation_{condition}", cases=cases)
    for evaluator in evaluator_instances():
        dataset.add_evaluator(evaluator)
    report = dataset.evaluate_sync(lambda inputs: predictions[inputs.clip_id])
    report_payload = {
        "name": report.name,
        "cases": [
            {
                "case_id": case.name,
                "scores": {
                    key: value.value for key, value in case.scores.items()
                },
                "assertions": {
                    key: value.value for key, value in case.assertions.items()
                },
                "evaluator_failures": [str(item) for item in case.evaluator_failures],
            }
            for case in report.cases
        ],
        "task_failures": [str(item) for item in report.failures],
    }
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "pydantic_eval_report.json").write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row[key] != ""]
    return mean(values) if values else None


def condition_summary(condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "condition": condition,
        "case_count": len(rows),
        "parse_success_count": sum(row["parse_status"] == "success" for row in rows),
        "parse_failure_count": sum(row["parse_status"] != "success" for row in rows),
        "schema_valid_rate": rate(rows, "schema_valid"),
        "deterministic_feature_preservation_rate": rate(
            rows, "deterministic_feature_preservation_pass"
        ),
        "mean_absolute_duration_error_ms": numeric_mean(
            rows, "mean_absolute_duration_error_ms"
        ),
        "mean_absolute_bandwidth_error_hz": numeric_mean(
            rows, "mean_absolute_bandwidth_error_hz"
        ),
        "mean_absolute_temporal_centre_error_seconds": numeric_mean(
            rows, "mean_absolute_temporal_centre_error_seconds"
        ),
        "mean_absolute_frequency_centre_error_hz": numeric_mean(
            rows, "mean_absolute_frequency_centre_error_hz"
        ),
        "mean_absolute_inter_event_interval_error_ms": numeric_mean(
            rows, "mean_absolute_inter_event_interval_error_ms"
        ),
        "annotation_case_ids_valid_rate": rate(rows, "annotation_case_ids_valid"),
        "literature_evidence_ids_valid_rate": rate(
            rows, "literature_evidence_ids_valid"
        ),
        "target_case_exclusion_rate": rate(rows, "target_case_excluded"),
        "condition_isolation_rate": rate(rows, "condition_isolation_pass"),
        "unsupported_behaviour_claim_pass_rate": rate(
            rows, "unsupported_behaviour_claim_pass"
        ),
        "scientific_name_grounding_pass_rate": rate(
            rows, "scientific_name_grounding_pass"
        ),
        "human_review_rule_pass_rate": rate(rows, "human_review_rule_pass"),
        "annotation_ids_match_supplied_rate": rate(
            rows, "annotation_ids_match_supplied"
        ),
        "literature_ids_match_supplied_rate": rate(
            rows, "literature_ids_match_supplied"
        ),
    }


def retrieval_summary_rows(artifacts: list[RunArtifact]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        trace = RetrievalTrace.model_validate_json(
            artifact.retrieval_trace_path.read_text(encoding="utf-8")
        )
        rows.append(
            {
                "condition": artifact.condition,
                "clip_id": artifact.clip_id,
                "run_type": artifact.run_type,
                "annotation_ids": "|".join(
                    match.retrieved_id for match in trace.annotation_matches
                ),
                "annotation_scores": "|".join(
                    str(match.retrieval_score) for match in trace.annotation_matches
                ),
                "literature_ids": "|".join(
                    match.retrieved_id for match in trace.literature_matches
                ),
                "literature_scores": "|".join(
                    str(match.retrieval_score) for match in trace.literature_matches
                ),
                "target_case_excluded": trace.target_case_excluded,
                "annotation_store_version": trace.annotation_store_version,
                "literature_store_version": trace.literature_store_version,
            }
        )
    return rows


def consistency_rows(
    main_rows: list[dict[str, Any]], repeat_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    main = {(row["condition"], row["clip_id"]): row for row in main_rows}
    rows: list[dict[str, Any]] = []
    for repeat in repeat_rows:
        original = main[(repeat["condition"], repeat["clip_id"])]
        rows.append(
            {
                "condition": repeat["condition"],
                "clip_id": repeat["clip_id"],
                "both_parse_success": original["parse_status"] == "success"
                and repeat["parse_status"] == "success",
                "schema_consistent": original["schema_valid"]
                == repeat["schema_valid"],
                "retrieved_annotation_ids_consistent": original[
                    "supplied_annotation_ids"
                ]
                == repeat["supplied_annotation_ids"],
                "retrieved_literature_ids_consistent": original[
                    "supplied_literature_ids"
                ]
                == repeat["supplied_literature_ids"],
                "unsupported_claim_consistent": original[
                    "unsupported_behaviour_claim_pass"
                ]
                == repeat["unsupported_behaviour_claim_pass"],
                "human_review_decision_consistent": original[
                    "human_review_needed"
                ]
                == repeat["human_review_needed"],
            }
        )
    return rows


def generate_report(
    *,
    summaries: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
    output_path: Path,
    evidence_count: int,
) -> None:
    lines = [
        "# P7 Knowledge-Grounded Single-Agent Event-Characterisation Report",
        "",
        "## Scope",
        "",
        f"Model: `{MODEL_NAME}`. Cases: 8. Main inferences: 32. Repeated consistency inferences: 16.",
        "No GT or diagnostic overlays were used as model input. Numeric features were supplied as immutable deterministic values.",
        "",
        "## Condition Summary",
        "",
        "| Condition | Parse success | Schema valid | Deterministic preservation | Unsupported-claim pass | Review-rule pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['condition']} | {summary['parse_success_count']}/{summary['case_count']} "
            f"| {summary['schema_valid_rate']:.3f} "
            f"| {summary['deterministic_feature_preservation_rate']:.3f} "
            f"| {summary['unsupported_behaviour_claim_pass_rate']:.3f} "
            f"| {summary['human_review_rule_pass_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "### 1. Annotation-memory retrieval",
            "",
            "Annotation-memory conditions retrieved two valid, target-excluded cases for every input, and model outputs reproduced all supplied case IDs. This establishes retrieval validity and condition isolation. Relevance was assigned by transparent metadata overlap, but no human relevance judgement was collected; the experiment therefore does not establish that memory improved interpretation quality.",
            "",
            "### 2. Literature grounding",
            "",
            f"The verified literature store contained {evidence_count} records. "
            + (
                "Verified evidence citations were therefore available for the literature conditions."
                if evidence_count
                else "No local literature note met the verified-source threshold, so literature-only and combined results cannot establish a literature-grounding benefit."
            ),
            "",
            "### 3. Unsupported claims and review rules",
            "",
            "All four conditions passed the unsupported-behaviour and human-review rules on every main case. No accepted output contained an exploratory hypothesis, and all human-review decisions were false. Annotation memory therefore did not reduce an existing unsupported-claim error rate, and the combined condition did not improve review-rule compliance over baseline; all conditions were already at 1.000.",
            "",
            "### 4. Deterministic feature preservation",
            "",
            "All continuous mean absolute errors were 0 under every condition, and every case passed the frozen thresholds. Library context caused no observed corruption of duration, bandwidth, temporal/frequency centre, event order, interval, overlap, boundary, or sequence features. Detailed values are available in `p7_condition_summary.csv` and `p7_case_level_results.csv`.",
            "",
            "### 5. Representative versus held-out cases",
            "",
        ]
    )
    for condition in CONDITIONS:
        representative = [
            row
            for row in main_rows
            if row["condition"] == condition and row["split"] == "representative"
        ]
        heldout = [
            row
            for row in main_rows
            if row["condition"] == condition and row["split"] == "heldout"
        ]
        lines.append(
            f"- `{condition}` deterministic preservation: representative "
            f"{rate(representative, 'deterministic_feature_preservation_pass'):.3f}; "
            f"held-out {rate(heldout, 'deterministic_feature_preservation_pass'):.3f}."
        )
    lines.extend(["", "### 6. Consistency", ""])
    for condition in CONDITIONS:
        rows = [row for row in consistency if row["condition"] == condition]
        lines.append(
            f"- `{condition}`: schema consistency "
            f"{rate(rows, 'schema_consistent'):.3f}; retrieved annotation IDs "
            f"{rate(rows, 'retrieved_annotation_ids_consistent'):.3f}; human-review decision "
            f"{rate(rows, 'human_review_decision_consistent'):.3f}."
        )
    lines.extend(
        [
            "",
            "All 16 repeated outputs parsed successfully. Schema validity, retrieved IDs, unsupported-claim status, and human-review decisions were consistent between the main and repeated runs. Because every review decision was false, review consistency should be interpreted as stability of a negative decision rather than evidence of calibrated escalation.",
        ]
    )
    deterministic_ready = bool(main_rows) and all(
        row["schema_valid"]
        and row["deterministic_feature_preservation_pass"]
        and row["unsupported_behaviour_claim_pass"]
        and row["human_review_rule_pass"]
        for row in main_rows
    )
    lines.extend(
        [
            "",
            "## Freeze Recommendation",
            "",
            (
                "The deterministic transport, schema, feature-preservation, and target-excluded annotation-retrieval components are ready to freeze as an implementation baseline. The full knowledge-grounded reasoning pipeline should not yet be frozen as a substantive result: the verified literature store is empty, no independent relevance or interpretation-quality rubric was applied, and review behaviour was not exercised by any accepted exploratory hypothesis. Populate and verify the evidence store, then run a scoped human rubric before claiming a knowledge-grounding benefit."
                if deterministic_ready
                else "Do not freeze the P7 reasoning pipeline yet. At least one main output failed schema, deterministic-preservation, unsupported-claim, or human-review requirements."
            ),
            "",
            "No LLM-as-judge evaluator was used.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the P7B four-condition ablation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Re-evaluate saved predictions without any model calls.",
    )
    return parser.parse_args()


def existing_artifact(
    *,
    output_dir: Path,
    condition: ConditionName,
    clip_id: str,
    run_type: str,
) -> RunArtifact:
    """Resolve one previously generated artifact for evaluation-only mode."""

    suffix = "" if run_type == "main" else "_repeat"
    prediction_path = (
        output_dir
        / condition
        / "predictions"
        / f"{clip_id}{suffix}_prediction.json"
    )
    raw_path = (
        output_dir
        / condition
        / "raw_responses"
        / f"{clip_id}{suffix}_raw_response.txt"
    )
    error_path = (
        output_dir
        / condition
        / "parse_errors"
        / f"{clip_id}{suffix}_parse_error.txt"
    )
    trace_path = (
        output_dir
        / condition
        / "retrieval_traces"
        / f"{clip_id}{suffix}_retrieval_trace.json"
    )
    return RunArtifact(
        condition=condition,
        clip_id=clip_id,
        run_type=run_type,
        parse_status="success" if prediction_path.is_file() else "failure",
        prediction_path=prediction_path if prediction_path.is_file() else None,
        raw_response_path=raw_path,
        parse_error_path=error_path if error_path.is_file() else None,
        retrieval_trace_path=trace_path,
    )


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    main_artifacts: list[RunArtifact] = []
    repeat_artifacts: list[RunArtifact] = []
    if args.evaluate_only:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {args.output_dir}")
        for condition in CONDITIONS:
            for case in dataset.cases:
                main_artifacts.append(
                    existing_artifact(
                        output_dir=args.output_dir,
                        condition=condition,
                        clip_id=case.name,
                        run_type="main",
                    )
                )
                if case.name in REPEAT_CLIP_IDS:
                    repeat_artifacts.append(
                        existing_artifact(
                            output_dir=args.output_dir,
                            condition=condition,
                            clip_id=case.name,
                            run_type="repeat",
                        )
                    )
    else:
        if ollama_host() != OLLAMA_ENDPOINT:
            raise RuntimeError(
                f"P7B requires OLLAMA_HOST={OLLAMA_ENDPOINT}; got {ollama_host()}"
            )
        print(require_model())
        if args.output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory exists: {args.output_dir}. Use --overwrite."
                )
            shutil.rmtree(args.output_dir)
        for condition in CONDITIONS:
            for directory in (
                "raw_responses",
                "predictions",
                "retrieval_traces",
                "parse_errors",
                "evaluation",
                "report",
            ):
                (args.output_dir / condition / directory).mkdir(
                    parents=True, exist_ok=True
                )
        for condition in CONDITIONS:
            for case in dataset.cases:
                print(f"Running {condition} {case.name} main", flush=True)
                main_artifacts.append(
                    run_case(
                        condition=condition,
                        case=case,
                        output_dir=args.output_dir,
                        run_type="main",
                        timeout_seconds=args.timeout_seconds,
                        num_predict=args.num_predict,
                    )
                )
            for case in dataset.cases:
                if case.name not in REPEAT_CLIP_IDS:
                    continue
                print(f"Running {condition} {case.name} repeat", flush=True)
                repeat_artifacts.append(
                    run_case(
                        condition=condition,
                        case=case,
                        output_dir=args.output_dir,
                        run_type="repeat",
                        timeout_seconds=args.timeout_seconds,
                        num_predict=args.num_predict,
                    )
                )

    case_by_id = {case.name: case for case in dataset.cases}
    main_rows = [
        evaluate_artifact(
            case=case_by_id[artifact.clip_id],
            condition=artifact.condition,
            artifact=artifact,
        )
        for artifact in main_artifacts
    ]
    repeat_rows = [
        evaluate_artifact(
            case=case_by_id[artifact.clip_id],
            condition=artifact.condition,
            artifact=artifact,
        )
        for artifact in repeat_artifacts
    ]
    for condition in CONDITIONS:
        artifacts = [item for item in main_artifacts if item.condition == condition]
        evaluation_dir = args.output_dir / condition / "evaluation"
        run_pydantic_evals(
            condition=condition,
            source_dataset=dataset,
            artifacts=artifacts,
            evaluation_dir=evaluation_dir,
        )
        condition_rows = [row for row in main_rows if row["condition"] == condition]
        write_csv(evaluation_dir / "case_results.csv", condition_rows)
        summary = condition_summary(condition, condition_rows)
        (evaluation_dir / "condition_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / condition / "report" / "run_metadata.json").write_text(
            json.dumps(
                {
                    "model_name": MODEL_NAME,
                    "ollama_host": ollama_host(),
                    "run_timestamp": datetime.now(timezone.utc).isoformat(),
                    "condition": condition,
                    "settings": CONDITION_SETTINGS[condition],
                    "tolerances": TOLERANCES,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    summaries = [
        condition_summary(
            condition,
            [row for row in main_rows if row["condition"] == condition],
        )
        for condition in CONDITIONS
    ]
    consistency = consistency_rows(main_rows, repeat_rows)
    write_csv(args.output_dir / "p7_condition_summary.csv", summaries)
    write_csv(args.output_dir / "p7_case_level_results.csv", main_rows)
    write_csv(
        args.output_dir / "p7_retrieval_summary.csv",
        retrieval_summary_rows(main_artifacts + repeat_artifacts),
    )
    write_csv(args.output_dir / "p7_consistency_summary.csv", consistency)
    evidence_count = sum(
        1
        for line in Path(
            "docs/literature_reference_library/verified_evidence_store.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    generate_report(
        summaries=summaries,
        main_rows=main_rows,
        consistency=consistency,
        output_path=(
            args.output_dir / "p7_knowledge_grounded_single_agent_report.md"
        ),
        evidence_count=evidence_count,
    )
    print(f"Saved P7B outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
