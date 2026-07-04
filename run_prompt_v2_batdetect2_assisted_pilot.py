"""Run Prompt V2 with clean spectrograms and BatDetect2 proposal metadata.

BatDetect2 proposals are untrusted hints. This runner does not read ground truth,
proposal overlays, diagnostics, or prior VLM predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_prompt_v2_small_pilot import (
    PROMPT_VERSION,
    extract_json_text,
    image_to_base64,
    load_prompt,
    ollama_host,
    read_clip_duration,
    resolve_input_image,
    validate_prediction,
)
from run_prompt_v2_tiled_pilot import require_model


DEFAULT_MODEL_NAME = "qwen3.6:latest"
DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_IMAGE_DIR = Path("outputs/agent_inputs/prompt_v2_full_grid_v2")
DEFAULT_PROPOSAL_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6"
)

PROVENANCE_FIELDS = {
    "used_proposal_id": str,
    "proposal_source": str,
    "refinement_note": str,
}

ASSISTED_PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "clip_id": {"type": "string"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "start_time_seconds": {"type": "number"},
                    "end_time_seconds": {"type": "number"},
                    "low_frequency_hz": {"type": "number"},
                    "high_frequency_hz": {"type": "number"},
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                    "human_review_needed": {"type": "boolean"},
                    "review_reason": {"type": "string"},
                    "used_proposal_id": {"type": "string"},
                    "proposal_source": {"type": "string"},
                    "refinement_note": {"type": "string"},
                },
                "required": [
                    "event_id",
                    "start_time_seconds",
                    "end_time_seconds",
                    "low_frequency_hz",
                    "high_frequency_hz",
                    "label",
                    "confidence",
                    "evidence",
                    "human_review_needed",
                    "review_reason",
                    *PROVENANCE_FIELDS,
                ],
            },
        },
    },
    "required": ["clip_id", "events"],
}


@dataclass(frozen=True)
class AssistedRunResult:
    clip_id: str
    parse_status: str
    predicted_event_count: int | None
    prediction_path: Path
    raw_response_path: Path
    parse_error_path: Path | None


def parse_clip_ids(value: str) -> list[str]:
    """Parse a stable, de-duplicated comma-separated clip list."""
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def load_proposal_payload(proposal_dir: Path, clip_id: str) -> dict[str, Any]:
    """Load and minimally validate one BatDetect2 proposal artifact."""
    path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
    if not path.is_file():
        raise FileNotFoundError(f"BatDetect2 proposal file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("clip_id") != clip_id:
        raise ValueError(
            f"Proposal clip_id {payload.get('clip_id')!r} does not match {clip_id!r}"
        )
    if not isinstance(payload.get("events"), list):
        raise ValueError("Proposal payload events must be a list")
    return payload


def format_proposal_metadata(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact, sorted proposal metadata suitable for the user message."""
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(payload.get("events", [])):
        try:
            start = float(event["start_time_seconds"])
            end = float(event["end_time_seconds"])
            low = float(event["low_frequency_hz"])
            high = float(event["high_frequency_hz"])
            det_prob = float(event["det_prob"])
            class_prob = float(event["class_prob"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid proposal event at index {index}: {exc}") from exc
        if not all(math.isfinite(value) for value in (start, end, low, high, det_prob, class_prob)):
            raise ValueError(f"Proposal event {index} contains a non-finite value")
        if start >= end or low >= high:
            raise ValueError(f"Proposal event {index} has invalid geometry")
        rows.append(
            {
                "proposal_id": str(event.get("proposal_id") or f"bd2_{index + 1:03d}"),
                "start_time_seconds": round(start, 6),
                "end_time_seconds": round(end, 6),
                "duration_ms": round((end - start) * 1000.0, 3),
                "low_frequency_hz": round(low, 3),
                "high_frequency_hz": round(high, 3),
                "det_prob": round(det_prob, 6),
                "class_prob": round(class_prob, 6),
                "original_label": str(event.get("label") or ""),
            }
        )
    return sorted(rows, key=lambda row: (row["start_time_seconds"], row["proposal_id"]))


def build_assisted_user_message(
    *,
    clip_id: str,
    clip_duration_seconds: float,
    proposal_rows: list[dict[str, Any]],
) -> str:
    """Build a proposal-aware request without introducing GT information."""
    context = {
        "clip_id": clip_id,
        "clip_duration_seconds": round(clip_duration_seconds, 6),
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
        "task_label": "bat_call",
    }
    return (
        "/no_think\n"
        "Annotate the attached clean grid_v2 spectrogram according to the system prompt.\n"
        "BatDetect2 supplied the candidate metadata below as an external detector tool. "
        "These proposals are hints, not labels or ground truth. Its UK taxonomy labels are "
        "unreliable for this Australian clip and must not be used for species identification.\n"
        "Verify every proposal against visible spectrogram evidence. Remove false positives; "
        "correct start/end and frequency bounds; split proposals when necessary; and add any "
        "visible calls missing from the proposal list. Do not blindly copy proposal geometry.\n"
        "Return generic label `bat_call` for every final event. For a proposal-derived event, "
        "set used_proposal_id and proposal_source=`batdetect2`; explain changes briefly in "
        "refinement_note. For a newly added event, use empty strings for those provenance fields.\n\n"
        "Runtime context:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "BatDetect2 proposal metadata:\n"
        f"{json.dumps(proposal_rows, indent=2)}\n\n"
        "Return the final corrected annotation as valid JSON only. Do not explain reasoning "
        "outside the JSON object."
    )


def validate_assisted_prediction(
    payload: Any,
    *,
    expected_clip_id: str,
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Validate Prompt V2 structure, geometry, confidence, and provenance fields."""
    prediction = validate_prediction(payload, expected_clip_id=expected_clip_id)
    for index, event in enumerate(prediction["events"]):
        for field, expected_type in PROVENANCE_FIELDS.items():
            if field not in event or not isinstance(event[field], expected_type):
                raise ValueError(f"events[{index}].{field} must be a string")
        start = float(event["start_time_seconds"])
        end = float(event["end_time_seconds"])
        low = float(event["low_frequency_hz"])
        high = float(event["high_frequency_hz"])
        confidence = float(event["confidence"])
        if not 0 <= start < end <= clip_duration_seconds:
            raise ValueError(f"events[{index}] has invalid time geometry")
        if not 0 <= low < high:
            raise ValueError(f"events[{index}] has invalid frequency geometry")
        if not 0 <= confidence <= 1:
            raise ValueError(f"events[{index}].confidence must be between 0 and 1")
        if event["label"] != "bat_call":
            raise ValueError(f"events[{index}].label must be 'bat_call'")
    return prediction


def parse_assisted_prediction(
    raw_text: str,
    *,
    expected_clip_id: str,
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Extract and validate one assisted response."""
    payload = json.loads(extract_json_text(raw_text))
    return validate_assisted_prediction(
        payload,
        expected_clip_id=expected_clip_id,
        clip_duration_seconds=clip_duration_seconds,
    )


def call_assisted_ollama(
    *,
    image_path: Path,
    system_prompt: str,
    user_message: str,
    model_name: str,
    timeout: float,
    num_predict: int,
) -> str:
    """Call Ollama generate with image input and the assisted output schema."""
    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": ASSISTED_PREDICTION_SCHEMA,
        "prompt": f"{system_prompt}\n\n{user_message}",
        "images": [image_to_base64(image_path)],
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        f"{ollama_host()}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = response_payload.get("response")
    if content:
        return str(content)
    thinking = response_payload.get("thinking")
    if thinking:
        return str(thinking)
    return json.dumps(response_payload, indent=2, ensure_ascii=False)


def _write_prediction(
    *,
    path: Path,
    clip_id: str,
    events: list[dict[str, Any]],
    model_name: str,
    image_path: Path,
    proposal_path: Path,
    clip_duration_seconds: float,
    proposal_threshold: Any,
    parse_status: str,
    error: str = "",
) -> None:
    payload = {
        "clip_id": clip_id,
        "prompt_version": PROMPT_VERSION,
        "model_name": model_name,
        "backend": "ollama_generate_batdetect2_metadata_assisted",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_image_path": image_path.as_posix(),
        "proposal_metadata_path": proposal_path.as_posix(),
        "proposal_source": "batdetect2",
        "proposal_threshold": proposal_threshold,
        "clip_duration_seconds": clip_duration_seconds,
        "parse_status": parse_status,
        "error": error,
        "events": events,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_clip(
    *,
    clip_id: str,
    prompt_text: str,
    eval_dir: Path,
    image_dir: Path,
    proposal_dir: Path,
    prediction_dir: Path,
    raw_response_dir: Path,
    model_name: str,
    timeout: float,
    num_predict: int,
    overwrite: bool,
) -> AssistedRunResult:
    """Run one assisted clip while preserving invalid responses and continuing."""
    prediction_dir.mkdir(parents=True, exist_ok=True)
    raw_response_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{clip_id}_predictions.json"
    raw_path = raw_response_dir / f"{clip_id}_raw_response.txt"
    error_path = prediction_dir / f"{clip_id}_parse_error.txt"
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(f"Prediction exists: {prediction_path}. Use --overwrite.")
    if overwrite:
        prediction_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)

    image_path = resolve_input_image(image_dir, clip_id)
    proposal_path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
    clip_duration = read_clip_duration(eval_dir, clip_id)
    proposal_payload = load_proposal_payload(proposal_dir, clip_id)
    proposal_rows = format_proposal_metadata(proposal_payload)
    try:
        raw_text = call_assisted_ollama(
            image_path=image_path,
            system_prompt=prompt_text,
            user_message=build_assisted_user_message(
                clip_id=clip_id,
                clip_duration_seconds=clip_duration,
                proposal_rows=proposal_rows,
            ),
            model_name=model_name,
            timeout=timeout,
            num_predict=num_predict,
        )
        raw_path.write_text(raw_text, encoding="utf-8")
        prediction = parse_assisted_prediction(
            raw_text,
            expected_clip_id=clip_id,
            clip_duration_seconds=clip_duration,
        )
        _write_prediction(
            path=prediction_path,
            clip_id=clip_id,
            events=prediction["events"],
            model_name=model_name,
            image_path=image_path,
            proposal_path=proposal_path,
            clip_duration_seconds=clip_duration,
            proposal_threshold=proposal_payload.get("proposal_threshold"),
            parse_status="success",
        )
        return AssistedRunResult(
            clip_id, "success", len(prediction["events"]), prediction_path, raw_path, None
        )
    except Exception as exc:
        if not raw_path.exists():
            raw_path.write_text("", encoding="utf-8")
        message = f"{type(exc).__name__}: {exc}"
        error_path.write_text(message + "\n", encoding="utf-8")
        _write_prediction(
            path=prediction_path,
            clip_id=clip_id,
            events=[],
            model_name=model_name,
            image_path=image_path,
            proposal_path=proposal_path,
            clip_duration_seconds=clip_duration,
            proposal_threshold=proposal_payload.get("proposal_threshold"),
            parse_status="failed",
            error=message,
        )
        return AssistedRunResult(
            clip_id, "failed", None, prediction_path, raw_path, error_path
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/prompt_v2_bat_strong_label.md"))
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-list", default=",".join(DEFAULT_CLIP_IDS))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=8000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"OLLAMA_HOST={ollama_host()}")
    require_model(args.model_name)
    print(f"Confirmed model: {args.model_name}")
    prompt_text = load_prompt(args.prompt)
    clip_ids = parse_clip_ids(args.clip_list)
    prediction_dir = args.output_dir / "predictions"
    raw_dir = args.output_dir / "raw_responses"
    results: list[AssistedRunResult] = []
    for index, clip_id in enumerate(clip_ids, start=1):
        print(f"[{index}/{len(clip_ids)}] Running assisted inference for {clip_id}...", flush=True)
        results.append(
            run_clip(
                clip_id=clip_id,
                prompt_text=prompt_text,
                eval_dir=args.eval_dir,
                image_dir=args.image_dir,
                proposal_dir=args.proposal_dir,
                prediction_dir=prediction_dir,
                raw_response_dir=raw_dir,
                model_name=args.model_name,
                timeout=args.timeout,
                num_predict=args.num_predict,
                overwrite=args.overwrite,
            )
        )
    print("clip_id | parse_status | predicted_events")
    print("--------+--------------+-----------------")
    for result in results:
        count = "" if result.predicted_event_count is None else result.predicted_event_count
        print(f"{result.clip_id} | {result.parse_status} | {count}")
    successes = sum(result.parse_status == "success" for result in results)
    print(f"Clips: {len(results)} | parse successes: {successes} | parse failures: {len(results) - successes}")


if __name__ == "__main__":
    main()
