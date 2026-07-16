"""Utilities for P9-light Walters-style acoustic-parameter guidance.

P9-light intentionally uses Walters et al. only as a generic checklist of
acoustic dimensions. It must never construct or imply an Ozimops petersi
species prior, and it must not transfer European species-specific numeric
ranges.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from toy_audio_agent.evaluation.event_matching import (
    ClipEvaluation,
    EventBox,
    MatchingProtocol,
    aggregate_clip_evaluations,
    evaluate_clip,
)


REPRESENTATIVE_SIX: tuple[str, ...] = (
    "OP_001",
    "OP_003",
    "OP_004",
    "OP_010",
    "OP_016",
    "OP_045",
)
HELDOUT_TEN: tuple[str, ...] = (
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
TARGET_CLIPS: tuple[str, ...] = REPRESENTATIVE_SIX + HELDOUT_TEN
REQUIRED_AGENT_CONDITIONS: tuple[str, ...] = (
    "agent_proposals_only",
    "agent_methodological_literature",
    "agent_walters_guidance",
)
OPTIONAL_AGENT_CONDITION = "agent_annotation_memory_walters"
PROPOSAL_ONLY_CONDITION = "proposal_only"
ALL_CONDITIONS: tuple[str, ...] = (
    PROPOSAL_ONLY_CONDITION,
    *REQUIRED_AGENT_CONDITIONS,
    OPTIONAL_AGENT_CONDITION,
)
PROTOCOLS: tuple[MatchingProtocol, ...] = (
    MatchingProtocol.TEMPORAL_IOU_0_1,
    MatchingProtocol.TEMPORAL_IOU_0_3,
    MatchingProtocol.START_TIME_PROXIMITY_10MS,
)
PROPOSAL_THRESHOLD = 0.30
DISALLOWED_EVENT_FIELDS = {
    "species",
    "species_label",
    "predicted_species",
    "scientific_name",
    "label",
    "confidence",
    "behaviour",
    "behavior",
    "risk_flags",
    "human_review_needed",
    "review_reason",
    "evidence",
    "evidence_scope",
    "citation",
    "citations",
    "limitations",
}


class P9Condition(StrEnum):
    """P9-light condition names."""

    PROPOSAL_ONLY = PROPOSAL_ONLY_CONDITION
    AGENT_PROPOSALS_ONLY = "agent_proposals_only"
    AGENT_METHODOLOGICAL_LITERATURE = "agent_methodological_literature"
    AGENT_WALTERS_GUIDANCE = "agent_walters_guidance"
    AGENT_ANNOTATION_MEMORY_WALTERS = OPTIONAL_AGENT_CONDITION


class P9LightEvent(BaseModel):
    """Simple P9-light event schema for bounding-box geometry only."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    low_frequency: float = Field(ge=0.0)
    high_frequency: float = Field(gt=0.0)
    linked_proposal_id: str | None = None
    brief_reason: str | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> "P9LightEvent":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        if self.high_frequency <= self.low_frequency:
            raise ValueError("high_frequency must be greater than low_frequency")
        return self


class P9LightPrediction(BaseModel):
    """Clip-level P9-light prediction schema."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    events: list[P9LightEvent]


@dataclass(frozen=True)
class P9Paths:
    """Resolved paths for P9-light."""

    repo_root: Path
    eval_dir: Path
    image_dir: Path
    run_dir: Path
    analysis_dir: Path
    representative_proposal_dir: Path
    heldout_proposal_dir: Path
    annotation_memory_path: Path
    literature_store_path: Path
    walters_card_path: Path
    previous_p9_blocker_path: Path


def repo_root_from(path: Path) -> Path:
    """Return repository root from any script path under scripts/*."""

    return path.resolve().parents[2]


def default_paths(repo_root: Path) -> P9Paths:
    """Return current repository paths for P9-light."""

    return P9Paths(
        repo_root=repo_root,
        eval_dir=repo_root / "outputs/evaluation_sets/ozimops_petersi_v1",
        image_dir=repo_root / "outputs/agent_inputs/prompt_v2_full_grid_v2",
        run_dir=repo_root / "outputs/agent_runs/p9_light_walters_acoustic_parameter_guidance",
        analysis_dir=repo_root
        / "outputs/analysis_reports/p9_light_walters_acoustic_parameter_guidance",
        representative_proposal_dir=repo_root
        / "outputs/tool_outputs/batdetect2_proposals/representative6",
        heldout_proposal_dir=repo_root
        / "outputs/tool_outputs/batdetect2_proposals/p6e5_heldout",
        annotation_memory_path=repo_root
        / "docs/annotation_example_library/annotation_memory.jsonl",
        literature_store_path=repo_root
        / "docs/literature_reference_library/verified_evidence_store.jsonl",
        walters_card_path=repo_root
        / "docs/acoustic_reference_library/walters_2012_generic_acoustic_parameter_guidance.json",
        previous_p9_blocker_path=repo_root
        / "docs/acoustic_reference_library/ozimops_petersi_acoustic_reference.json",
    )


def sha256_file(path: Path) -> str:
    """Return SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def ollama_host(default: str = "http://127.0.0.1:11436") -> str:
    return os.getenv("OLLAMA_HOST", default).rstrip("/")


def proposal_dir_for_clip(paths: P9Paths, clip_id: str) -> Path:
    return paths.representative_proposal_dir if clip_id in REPRESENTATIVE_SIX else paths.heldout_proposal_dir


def proposal_path_for_clip(paths: P9Paths, clip_id: str) -> Path:
    return proposal_dir_for_clip(paths, clip_id) / f"{clip_id}_batdetect2_proposals.json"


def image_path_for_clip(paths: P9Paths, clip_id: str) -> Path:
    return paths.image_dir / f"{clip_id}_spectrogram.png"


def gt_path_for_clip(paths: P9Paths, clip_id: str) -> Path:
    return paths.eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"


def read_clip_duration(paths: P9Paths, clip_id: str) -> float:
    return float(sf.info(paths.eval_dir / "audio" / f"{clip_id}.wav").duration)


def validate_walters_card(card: dict[str, Any]) -> None:
    """Validate P9-light safety markers on the Walters guidance card."""

    required_true = (
        "not_species_specific",
        "not_op_prior",
        "no_numeric_species_ranges",
        "no_european_numeric_transfer",
    )
    if card.get("status") != "usable_for_generic_guidance":
        raise ValueError("Walters guidance card must be generic guidance")
    for field in required_true:
        if card.get(field) is not True:
            raise ValueError(f"Walters guidance card must set {field}=true")
    import re

    guidance_text = json.dumps(
        {
            "generic_dimensions": card.get("generic_dimensions", []),
            "warnings": card.get("warnings", []),
        },
        ensure_ascii=False,
    ).lower()
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:ms|khz|hz)\b", guidance_text):
        raise ValueError("Walters card contains numeric acoustic ranges")


def walters_prompt_insert(card: dict[str, Any]) -> str:
    """Return the required short Walters-style generic checklist prompt insert."""

    validate_walters_card(card)
    return (
        "Generic acoustic-parameter guidance:\n\n"
        "Bat acoustic identification studies commonly use time-frequency parameters "
        "such as duration, minimum and maximum frequency, bandwidth, centre or peak "
        "frequency, and slope or call-shape features.\n\n"
        "Use this only as an annotation checklist. For each candidate event, inspect "
        "the coherent time-frequency ridge and consider:\n\n"
        "- visible onset and offset;\n"
        "- lower and upper frequency extent;\n"
        "- main bandwidth and energy region;\n"
        "- centre or peak energy region if visible;\n"
        "- slope or call-shape trend;\n"
        "- signal quality.\n\n"
        "Avoid extending boxes into background noise, echo, isolated artefacts or "
        "empty spectrogram regions.\n\n"
        "This is not an Ozimops petersi prior. It provides no expected numeric range. "
        "Do not apply European species-specific values to Ozimops petersi. The "
        "spectrogram evidence and BatDetect2 proposals remain primary."
    )


def format_proposal_rows(payload: dict[str, Any], threshold: float = PROPOSAL_THRESHOLD) -> list[dict[str, Any]]:
    """Return compact proposal rows sorted by time."""

    rows: list[dict[str, Any]] = []
    for index, event in enumerate(payload.get("events", []), start=1):
        start = float(event["start_time_seconds"])
        end = float(event["end_time_seconds"])
        low = float(event["low_frequency_hz"])
        high = float(event["high_frequency_hz"])
        det_prob = float(event.get("det_prob", 0.0))
        if det_prob < threshold:
            continue
        rows.append(
            {
                "proposal_id": str(event.get("proposal_id") or f"bd2_{index:03d}"),
                "start_time": round(start, 6),
                "end_time": round(end, 6),
                "duration_ms": round((end - start) * 1000.0, 3),
                "low_frequency": round(low, 3),
                "high_frequency": round(high, 3),
                "det_prob": round(det_prob, 6),
                "class_prob": round(float(event.get("class_prob", 0.0)), 6),
                "original_label": str(event.get("label") or ""),
            }
        )
    return sorted(rows, key=lambda row: (row["start_time"], row["proposal_id"]))


def proposal_prediction_events(payload: dict[str, Any], threshold: float = PROPOSAL_THRESHOLD) -> list[dict[str, Any]]:
    """Convert BatDetect2 proposals to P9-light evaluator-compatible events."""

    events = []
    for row in format_proposal_rows(payload, threshold=threshold):
        events.append(
            {
                "event_id": row["proposal_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "low_frequency": row["low_frequency"],
                "high_frequency": row["high_frequency"],
                "linked_proposal_id": row["proposal_id"],
                "brief_reason": "BatDetect2 proposal-only baseline.",
            }
        )
    return events


def clip_descriptors(proposal_rows: list[dict[str, Any]], clip_duration: float) -> list[str]:
    """Derive retrieval descriptors from proposals only."""

    descriptors = ["bat call", "detector proposal", "bounding box"]
    if len(proposal_rows) >= 5:
        descriptors.extend(["dense", "multi event"])
    elif len(proposal_rows) > 1:
        descriptors.append("multi event")
    durations = [float(row["duration_ms"]) for row in proposal_rows]
    if durations and sum(durations) / len(durations) < 15.0:
        descriptors.extend(["short call", "timing"])
    if any(float(row["start_time"]) <= 0.01 for row in proposal_rows):
        descriptors.extend(["left boundary", "truncated"])
    if any(float(row["end_time"]) >= clip_duration - 0.01 for row in proposal_rows):
        descriptors.extend(["right boundary", "truncated"])
    return descriptors


def _tokens(values: Iterable[str]) -> set[str]:
    import re

    stop = {"call", "calls", "case", "clip", "event", "events"}
    text = " ".join(values).lower()
    return {token for token in re.findall(r"[a-z0-9]+", text.replace("_", " ")) if len(token) > 2 and token not in stop}


def retrieve_annotation_memory(memory_path: Path, *, target_clip: str, descriptors: list[str], top_k: int = 2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return safe annotation memory records, excluding the target clip."""

    records = read_jsonl(memory_path)
    query = _tokens(descriptors)
    ranked = []
    for record in records:
        if record.get("case_id") == target_clip:
            continue
        values = list(record.get("case_type", [])) + list(record.get("observable_features", [])) + list(record.get("known_failure_modes", []))
        overlap = sorted(query & _tokens(values))
        score = float(len(overlap))
        ranked.append((score, str(record.get("case_id")), record, overlap))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[:top_k]
    safe_records = [
        {
            "case_id": item[2]["case_id"],
            "case_type": item[2].get("case_type", []),
            "recommended_actions": item[2].get("recommended_actions", []),
            "anti_patterns": item[2].get("anti_patterns", []),
        }
        for item in selected
    ]
    trace = [
        {"retrieved_id": item[1], "score": item[0], "match_tokens": item[3]}
        for item in selected
    ]
    return safe_records, trace


def retrieve_literature(literature_path: Path, *, descriptors: list[str], top_k: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return verified literature evidence records."""

    records = read_jsonl(literature_path)
    query = _tokens(descriptors + ["tool validation", "event localisation", "reference library"])
    ranked = []
    for record in records:
        values = [record.get("claim", ""), record.get("scope", ""), record.get("limitations", "")]
        overlap = sorted(query & _tokens(values))
        score = float(len(overlap))
        ranked.append((score, str(record.get("evidence_id")), record, overlap))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[:top_k]
    safe_records = [
        {
            "evidence_id": item[2]["evidence_id"],
            "claim": item[2]["claim"],
            "scope": item[2]["scope"],
            "limitations": item[2]["limitations"],
            "source_citation": item[2]["source_citation"],
        }
        for item in selected
    ]
    trace = [
        {"retrieved_id": item[1], "score": item[0], "match_tokens": item[3]}
        for item in selected
    ]
    return safe_records, trace


P9_LIGHT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clip_id": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "low_frequency": {"type": "number"},
                    "high_frequency": {"type": "number"},
                    "linked_proposal_id": {"type": ["string", "null"]},
                    "brief_reason": {"type": ["string", "null"]},
                },
                "required": [
                    "event_id",
                    "start_time",
                    "end_time",
                    "low_frequency",
                    "high_frequency",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["clip_id", "events"],
    "additionalProperties": False,
}


def build_prompt(
    *,
    condition: str,
    clip_id: str,
    clip_duration: float,
    proposal_rows: list[dict[str, Any]],
    literature_records: list[dict[str, Any]],
    annotation_records: list[dict[str, Any]],
    walters_card: dict[str, Any] | None,
) -> tuple[str, str, dict[str, Any]]:
    """Build the P9-light prompt and return trace context."""

    system = (
        "You are a careful bioacoustic bounding-box annotation agent. "
        "The task is geometry only: return one tight time-frequency box for each "
        "visible bat-call candidate that should be kept. The spectrogram is the "
        "primary evidence. BatDetect2 proposals are candidate regions, not ground truth. "
        "Preserve proposal geometry when it already fits the visible signal; adjust "
        "boxes only when the visible coherent ridge supports the adjustment. Reject "
        "unsupported candidates by omitting them. Add missing visible calls only when "
        "there is clear spectrogram evidence. Return valid JSON only."
    )
    condition_context: dict[str, Any] = {
        "condition": condition,
        "methodological_literature": literature_records,
        "annotation_memory": annotation_records,
        "walters_guidance": walters_prompt_insert(walters_card) if walters_card else "",
    }
    context = {
        "clip_id": clip_id,
        "clip_duration_seconds": round(clip_duration, 6),
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
        "batdetect2_proposals": proposal_rows,
        "condition_context": condition_context,
        "output_schema_rules": {
            "required_event_fields": [
                "event_id",
                "start_time",
                "end_time",
                "low_frequency",
                "high_frequency",
                "linked_proposal_id",
                "brief_reason",
            ],
            "forbidden_fields": sorted(DISALLOWED_EVENT_FIELDS),
            "frequency_values_are_hz": True,
        },
    }
    user = (
        "/no_think\n"
        "Inspect the attached clean grid_v2 spectrogram and the BatDetect2 proposal metadata. "
        "Return final bounding-box geometry only. Do not provide species, behaviour, risk, "
        "human-review, citation, evidence-scope, or limitation fields.\n\n"
        f"{json.dumps(context, indent=2, ensure_ascii=False)}"
    )
    return system, user, condition_context


def check_output_schema_fields(payload: dict[str, Any]) -> None:
    """Reject forbidden P9-light output fields."""

    for event in payload.get("events", []):
        forbidden = DISALLOWED_EVENT_FIELDS & set(event)
        if forbidden:
            raise ValueError(f"P9-light output contains forbidden fields: {sorted(forbidden)}")


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start != -1 and end >= start else text


def parse_prediction(raw_text: str, *, clip_id: str, clip_duration: float) -> P9LightPrediction:
    payload = json.loads(extract_json_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("P9-light response must be a JSON object")
    check_output_schema_fields(payload)
    prediction = P9LightPrediction.model_validate(payload)
    if prediction.clip_id != clip_id:
        raise ValueError("clip_id mismatch")
    for event in prediction.events:
        if event.end_time > clip_duration:
            raise ValueError(f"event {event.event_id} exceeds clip duration")
        if not all(math.isfinite(v) for v in (event.start_time, event.end_time, event.low_frequency, event.high_frequency)):
            raise ValueError(f"event {event.event_id} contains non-finite geometry")
    return prediction


def prediction_to_payload(
    prediction: P9LightPrediction | None,
    *,
    clip_id: str,
    condition: str,
    model_name: str,
    endpoint: str,
    image_path: Path,
    proposal_path: Path,
    clip_duration: float,
    parse_status: str,
    error: str = "",
    latency_seconds: float | None = None,
    retry_status: str = "not_retried",
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "condition": condition,
        "prompt_version": "p9_light_walters_acoustic_parameter_guidance",
        "model_name": model_name,
        "backend": "ollama_generate",
        "endpoint": endpoint,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_image_path": image_path.as_posix(),
        "proposal_metadata_path": proposal_path.as_posix(),
        "clip_duration_seconds": clip_duration,
        "parse_status": parse_status,
        "error": error,
        "latency_seconds": latency_seconds,
        "retry_status": retry_status,
        "events": [event.model_dump(mode="json") for event in prediction.events] if prediction else [],
    }


def call_ollama_generate(
    *,
    endpoint: str,
    model_name: str,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
    num_predict: int,
) -> tuple[str, dict[str, Any], float]:
    """Call Ollama generate and return response text, full payload, and latency."""

    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": P9_LIGHT_JSON_SCHEMA,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "images": [image_to_base64(image_path)],
        "options": {"temperature": 0, "seed": 0, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        f"{endpoint}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    latency = time.monotonic() - started
    content = response_payload.get("response")
    if content:
        return str(content), response_payload, latency
    return json.dumps(response_payload, ensure_ascii=False), response_payload, latency


def require_ollama_model(endpoint: str, model_name: str, timeout: float = 30.0) -> list[str]:
    """Confirm a model exists at an Ollama endpoint."""

    request = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    available = sorted(str(item.get("name", "")) for item in payload.get("models", []))
    if model_name not in available:
        raise RuntimeError(f"Required model {model_name!r} not available at {endpoint}. Available: {available}")
    return available


def preflight_check(paths: P9Paths, *, include_optional: bool = False) -> dict[str, Any]:
    """Validate required P9-light inputs without modifying frozen artifacts."""

    missing: list[str] = []
    for path in (
        paths.previous_p9_blocker_path,
        paths.walters_card_path,
        paths.literature_store_path,
        paths.annotation_memory_path,
        paths.eval_dir / "manifest.csv",
        paths.analysis_dir.parent / "p8a_multi_protocol_detection/matching_protocol_audit.md",
    ):
        if not path.exists():
            missing.append(path.as_posix())
    for clip_id in TARGET_CLIPS:
        for path in (image_path_for_clip(paths, clip_id), proposal_path_for_clip(paths, clip_id), gt_path_for_clip(paths, clip_id)):
            if not path.exists():
                missing.append(path.as_posix())
    if missing:
        raise FileNotFoundError("Missing P9-light inputs:\n" + "\n".join(missing))
    card = load_json(paths.walters_card_path)
    validate_walters_card(card)
    conditions = list(REQUIRED_AGENT_CONDITIONS)
    if include_optional:
        conditions.append(OPTIONAL_AGENT_CONDITION)
    return {
        "target_clips": list(TARGET_CLIPS),
        "agent_conditions": conditions,
        "proposal_only_condition": PROPOSAL_ONLY_CONDITION,
        "p8_evaluator_available": True,
        "p8_matching_protocol_audit_available": True,
        "previous_p9_blocker_exists": paths.previous_p9_blocker_path.exists(),
    }


def write_proposal_only_predictions(paths: P9Paths) -> list[dict[str, Any]]:
    """Write P9-light proposal-only baseline predictions for all target clips."""

    rows: list[dict[str, Any]] = []
    pred_dir = paths.run_dir / PROPOSAL_ONLY_CONDITION / "predictions"
    for clip_id in TARGET_CLIPS:
        proposal_path = proposal_path_for_clip(paths, clip_id)
        image_path = image_path_for_clip(paths, clip_id)
        clip_duration = read_clip_duration(paths, clip_id)
        payload = load_json(proposal_path)
        events = proposal_prediction_events(payload)
        output = prediction_to_payload(
            P9LightPrediction(clip_id=clip_id, events=[P9LightEvent.model_validate(event) for event in events]),
            clip_id=clip_id,
            condition=PROPOSAL_ONLY_CONDITION,
            model_name="batdetect2",
            endpoint="not_applicable",
            image_path=image_path,
            proposal_path=proposal_path,
            clip_duration=clip_duration,
            parse_status="success",
        )
        output["proposal_threshold"] = PROPOSAL_THRESHOLD
        out_path = pred_dir / f"{clip_id}_predictions.json"
        write_json(out_path, output)
        rows.append({"clip_id": clip_id, "condition": PROPOSAL_ONLY_CONDITION, "prediction_path": out_path.as_posix(), "event_count": len(events), "parse_status": "success"})
    return rows


def load_ground_truth(paths: P9Paths, clip_id: str) -> list[EventBox]:
    payload = load_json(gt_path_for_clip(paths, clip_id))
    events: list[EventBox] = []
    for index, event in enumerate(payload.get("events", [])):
        events.append(
            EventBox(
                event_id=str(event.get("event_id") or f"gt_{index + 1:03d}"),
                start_time=float(event["start_time"]),
                end_time=float(event["end_time"]),
                low_frequency=float(event["low_frequency"]),
                high_frequency=float(event["high_frequency"]),
                source_index=index,
                metadata=event,
            )
        )
    return events


def prediction_file(condition_dir: Path, clip_id: str) -> Path:
    return condition_dir / "predictions" / f"{clip_id}_predictions.json"


def load_prediction_events(path: Path) -> tuple[list[EventBox], str, str]:
    if not path.exists():
        return [], "missing_prediction_file", f"Missing {path}"
    try:
        payload = load_json(path)
    except json.JSONDecodeError as exc:
        return [], "json_parse_failure", str(exc)
    parse_status = str(payload.get("parse_status") or "success")
    if parse_status != "success":
        return [], parse_status, str(payload.get("error", ""))
    events: list[EventBox] = []
    for index, event in enumerate(payload.get("events", [])):
        try:
            events.append(
                EventBox(
                    event_id=str(event.get("event_id") or f"pred_{index + 1:03d}"),
                    start_time=float(event["start_time"]),
                    end_time=float(event["end_time"]),
                    low_frequency=float(event["low_frequency"]),
                    high_frequency=float(event["high_frequency"]),
                    confidence=float(event.get("det_prob", 1.0 if event.get("linked_proposal_id") else 0.5)),
                    source_index=index,
                    metadata=event,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return events, "success", ""


def evaluate_condition(paths: P9Paths, condition: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one P9-light condition under all P8 protocols."""

    condition_dir = paths.run_dir / condition
    condition_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        clip_results: list[ClipEvaluation] = []
        for clip_id in TARGET_CLIPS:
            pred_path = prediction_file(condition_dir, clip_id)
            predictions, parse_status, parse_error = load_prediction_events(pred_path)
            result = evaluate_clip(
                clip_id,
                predictions,
                load_ground_truth(paths, clip_id),
                protocol,
                parse_status=parse_status,
                parse_error=parse_error,
            )
            clip_results.append(result)
            case_rows.append(
                {
                    "condition": condition,
                    "clip_id": clip_id,
                    "protocol": protocol.value,
                    "prediction_path": pred_path.relative_to(paths.repo_root).as_posix(),
                    "parse_status": parse_status,
                    "predicted_count": result.predicted_count,
                    "ground_truth_count": result.ground_truth_count,
                    "TP": result.tp,
                    "FP": result.fp,
                    "FN": result.fn,
                    "precision": result.precision,
                    "recall": result.recall,
                    "F1": result.f1,
                }
            )
            for pair in result.matched:
                pair_rows.append(
                    {
                        "condition": condition,
                        "clip_id": clip_id,
                        "protocol": protocol.value,
                        "prediction_event_id": pair.prediction.event_id,
                        "ground_truth_event_id": pair.ground_truth.event_id,
                        "match_score": pair.match_score,
                        "onset_error_ms": pair.start_time_error_ms,
                        "offset_error_ms": pair.end_time_error_ms,
                        "center_time_error_ms": pair.center_time_error_ms,
                        "duration_error_ms": pair.duration_error_ms,
                        "temporal_iou": pair.temporal_iou,
                        "frequency_iou": pair.frequency_iou,
                        "box_iou": pair.box_iou,
                    }
                )
        aggregate = aggregate_clip_evaluations(clip_results)
        condition_rows.append({"condition": condition, "protocol": protocol.value, **aggregate})
    return condition_rows, case_rows, pair_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise_parse(paths: P9Paths, conditions: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for condition in conditions:
        successes = failures = retries = calls = 0
        for clip_id in TARGET_CLIPS:
            payload_path = prediction_file(paths.run_dir / condition, clip_id)
            if not payload_path.exists():
                failures += 1
                continue
            payload = load_json(payload_path)
            if payload.get("parse_status") == "success":
                successes += 1
            else:
                failures += 1
            if payload.get("retry_status") not in (None, "", "not_retried"):
                retries += 1
            if condition != PROPOSAL_ONLY_CONDITION:
                calls += 1
        rows.append(
            {
                "condition": condition,
                "clip_count": len(TARGET_CLIPS),
                "model_calls": calls,
                "parse_success_count": successes,
                "parse_failure_count": failures,
                "retry_count": retries,
            }
        )
    return rows
