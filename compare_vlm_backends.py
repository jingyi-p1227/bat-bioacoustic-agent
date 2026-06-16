"""Compare VLM backends on one grid-overlay spectrogram image.

The script always saves the raw model response and appends one row to
outputs/model_tests/backend_comparison_log.csv. Valid EventResult JSON from
annotation mode is also saved under annotations/.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from main import ANNOTATION_DIR, OLLAMA_HOST, OUTPUT_DIR, EventResult


DEFAULT_IMAGE_PATH = Path("outputs/grid/pseudo_petersi_001_zoom_short_event_grid.png")
DEFAULT_GROUND_TRUTH_PATH = Path("ground_truth/pseudo_petersi_001_ground_truth.json")
MODEL_TEST_DIR = OUTPUT_DIR / "model_tests"
LOG_PATH = MODEL_TEST_DIR / "backend_comparison_log.csv"

FailureMode = Literal[
    "empty_response",
    "invalid_json",
    "invalid_eventresult_schema",
    "reasoning_text_instead_of_json",
    "api_error",
    "image_not_supported",
    "invalid_model_for_backend",
    "model_not_found",
    "timeout_or_memory_issue",
    "ollama_api_format_issue",
    "server_error",
    "success",
    "success_text_response",
]

PROMPTS = {
    "debug-simple": "Describe this spectrogram in one sentence. Answer directly.",
    "debug-json": 'Return JSON only: {"is_spectrogram": true, "description": "..."}',
    "annotate": (
        "Find up to 10 clear bat echolocation pulses in this grid-overlay spectrogram. "
        "Use tight time-frequency boxes. The 20-100 kHz range is only the search region, "
        "not the box boundary. Return EventResult JSON only."
    ),
    "annotate-coordinates": (
        "Find up to 10 clearest bat echolocation pulses in this grid-overlay spectrogram. "
        "Do not return box_2d. Do not return image pixel coordinates. "
        "Read the x-axis as time in seconds. Read the y-axis as frequency in Hz. "
        "Use the cyan grid lines to estimate coordinates. "
        "The 20-100 kHz range is only the search region, not the box boundary. "
        "Use tight boxes only. Return EventResult JSON only using exactly this schema: "
        '{"audio_path":"pseudo_petersi_001.wav","events":[{"event_id":"event_001",'
        '"start_time_seconds":0.0,"end_time_seconds":0.0,"low_frequency_hz":0.0,'
        '"high_frequency_hz":0.0,"label":"bat echolocation pulse","confidence":0.0,'
        '"evidence":"...","tools_used":["grid_overlay_spectrogram"],'
        '"human_review_needed":true,"review_reason":"..."}],"notes":"..."}'
    ),
    "annotate-coordinates-high-recall": (
        "You are annotating a grid-overlay spectrogram. "
        "Important: Read the x-axis as time in seconds. "
        "Read the y-axis as frequency in Hz. "
        "Use the cyan grid lines to estimate coordinates. "
        "Do not return box_2d. "
        "Do not return image pixel coordinates. "
        "Return EventResult JSON only. "
        "Task: Scan the spectrogram from left to right and annotate every visible "
        "bat echolocation pulse you can identify. "
        "The pulses may appear as repeated short vertical or slightly curved traces "
        "in a similar frequency band. "
        "Do not only select representative examples. "
        "Do not skip repeated pulses. "
        "Each visible pulse should be a separate event. "
        "Return up to 20 candidate events. "
        "Use tight time-frequency boxes. "
        "The 20-100 kHz range is only the search region, not the box boundary. "
        "It is better to include uncertain but plausible pulses with "
        "human_review_needed=true than to miss obvious repeated pulses. "
        "Use the exact EventResult schema: "
        '{"audio_path":"pseudo_petersi_001.wav","events":[{"event_id":"event_001",'
        '"start_time_seconds":0.0,"end_time_seconds":0.0,"low_frequency_hz":0.0,'
        '"high_frequency_hz":0.0,"label":"bat echolocation pulse","confidence":0.0,'
        '"evidence":"...","tools_used":["grid_overlay_spectrogram"],'
        '"human_review_needed":true,"review_reason":"..."}],"notes":"..."}'
    ),
}

EVENTRESULT_MODES = {"annotate", "annotate-coordinates", "annotate-coordinates-high-recall"}

DEBUG_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "is_spectrogram": {"type": "boolean"},
        "description": {"type": "string"},
    },
    "required": ["is_spectrogram", "description"],
}

EVENT_RESULT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "audio_path": {"type": "string"},
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
                    "tools_used": {"type": "array", "items": {"type": "string"}},
                    "human_review_needed": {"type": "boolean"},
                    "review_reason": {"type": "string"},
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
                    "tools_used",
                    "human_review_needed",
                    "review_reason",
                ],
            },
        },
    },
    "required": ["audio_path", "events"],
}

LOG_COLUMNS = [
    "timestamp",
    "backend",
    "model",
    "image_path",
    "mode",
    "valid_json",
    "valid_eventresult",
    "predicted_count",
    "output_path",
    "raw_output_path",
    "failure_mode",
]


class InvalidModelForBackendError(ValueError):
    """Raised when a model name clearly belongs to another backend."""


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def looks_like_ollama_model_name(model: str) -> bool:
    """Return true for common Ollama-style local model names, e.g. qwen3-vl:latest."""
    return ":" in model or model.lower().startswith(("qwen", "llava", "gemma", "mistral"))


def selected_mode(args: argparse.Namespace) -> str:
    if args.debug_simple:
        return "debug-simple"
    if args.debug_json:
        return "debug-json"
    if args.annotate_coordinates:
        return "annotate-coordinates"
    if args.annotate_coordinates_high_recall:
        return "annotate-coordinates-high-recall"
    return "annotate"


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def format_for_mode(mode: str) -> dict[str, Any] | None:
    if mode == "debug-json":
        return DEBUG_JSON_SCHEMA
    if mode in EVENTRESULT_MODES:
        return EVENT_RESULT_JSON_SCHEMA
    return None


def num_predict_for_mode(mode: str) -> int:
    if mode == "annotate-coordinates-high-recall":
        return 3500
    if mode in EVENTRESULT_MODES:
        return 2500
    return 800


def call_ollama(image_path: Path, model: str, mode: str, timeout: float) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict_for_mode(mode),
        },
        "messages": [
            {
                "role": "user",
                "content": f"/no_think\n{PROMPTS[mode]}",
                "images": [image_to_base64(image_path)],
            }
        ],
    }
    response_format = format_for_mode(mode)
    if response_format is not None:
        payload["format"] = response_format

    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    message = response_payload.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(str(part) for part in content if part)
    if content:
        return str(content)

    thinking = message.get("thinking") or message.get("reasoning")
    if thinking:
        return str(thinking)
    return json.dumps(response_payload, indent=2, ensure_ascii=False)


def call_backend(backend: str, image_path: Path, model: str, mode: str, timeout: float) -> str:
    if backend == "ollama":
        return call_ollama(image_path=image_path, model=model, mode=mode, timeout=timeout)
    if backend == "openai" and looks_like_ollama_model_name(model):
        raise InvalidModelForBackendError(
            f"Model '{model}' looks like an Ollama/local model name, but backend=openai was selected. "
            "Use --backend ollama for this model, or pass an OpenAI model name for --backend openai."
        )
    if backend in {"openai", "claude"}:
        raise NotImplementedError(f"{backend} image backend is a placeholder in this script.")
    raise ValueError(f"Unsupported backend: {backend}")


def classify_api_exception(exc: BaseException) -> tuple[str, FailureMode]:
    """Turn transport/API exceptions into a more useful failure mode."""
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except OSError:
            body = ""
        raw_text = f"HTTPError {exc.code}: {exc.reason}"
        if body:
            raw_text = f"{raw_text}\n{body}"

        body_lower = body.lower()
        raw_lower = raw_text.lower()
        if exc.code == 404:
            return raw_text, "model_not_found"
        if "image" in body_lower and ("not support" in body_lower or "unsupported" in body_lower):
            return raw_text, "image_not_supported"
        if exc.code == 400:
            return raw_text, "ollama_api_format_issue"
        if exc.code >= 500:
            if "memory" in raw_lower or "timed out" in raw_lower or "timeout" in raw_lower:
                return raw_text, "timeout_or_memory_issue"
            return raw_text, "server_error"
        return raw_text, "api_error"

    raw_text = repr(exc)
    raw_lower = raw_text.lower()
    if "timed out" in raw_lower or "timeout" in raw_lower or "memory" in raw_lower:
        return raw_text, "timeout_or_memory_issue"
    return raw_text, "api_error"


def strip_reasoning(raw_text: str) -> tuple[str, bool]:
    text = raw_text.strip()
    had_reasoning = False
    if text.startswith("<think>"):
        had_reasoning = True
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        else:
            text = ""
    return text, had_reasoning


def extract_json_text(raw_text: str) -> tuple[str, bool]:
    text, had_reasoning = strip_reasoning(raw_text)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    object_start = text.find("{")
    array_start = text.find("[")
    starts = [index for index in (object_start, array_start) if index != -1]
    if not starts:
        return text, had_reasoning

    start = min(starts)
    end_char = "}" if text[start] == "{" else "]"
    end = text.rfind(end_char)
    if end == -1:
        return text, had_reasoning
    return text[start : end + 1], had_reasoning


def normalize_event_result(parsed: Any) -> EventResult:
    if isinstance(parsed, list):
        payload = {"audio_path": "pseudo_petersi_001.wav", "events": parsed}
    elif isinstance(parsed, dict):
        if "events" in parsed:
            payload = {"audio_path": "pseudo_petersi_001.wav", **parsed}
        else:
            payload = {"audio_path": "pseudo_petersi_001.wav", "events": [parsed]}
    else:
        raise ValueError("Expected EventResult JSON object or event list.")
    return EventResult.model_validate(payload)


def validate_response(raw_text: str, mode: str) -> tuple[bool, bool, int, EventResult | None, FailureMode]:
    if not raw_text.strip():
        return False, False, 0, None, "empty_response"

    if mode == "debug-simple":
        return False, False, 0, None, "success_text_response"

    json_text, had_reasoning = extract_json_text(raw_text)
    if not json_text:
        failure: FailureMode = "reasoning_text_instead_of_json" if had_reasoning else "invalid_json"
        return False, False, 0, None, failure

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        failure = "reasoning_text_instead_of_json" if had_reasoning else "invalid_json"
        return False, False, 0, None, failure

    if mode not in EVENTRESULT_MODES:
        return True, False, 0, None, "success"

    try:
        event_result = normalize_event_result(parsed)
    except (ValidationError, ValueError):
        return True, False, 0, None, "invalid_eventresult_schema"

    return True, True, len(event_result.events), event_result, "success"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_event_result(path: Path, event_result: EventResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(event_result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_log(row: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = selected_mode(args)
    timestamp = datetime.now().isoformat(timespec="seconds")
    image_path = args.image_path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.ground_truth and not args.ground_truth.exists():
        raise FileNotFoundError(f"Ground truth not found: {args.ground_truth}")

    backend_slug = slug(args.backend)
    model_slug = slug(args.model)
    mode_slug = slug(mode)
    timestamp_slug = timestamp.replace(":", "").replace("-", "").replace("T", "_")
    raw_output_path = (
        MODEL_TEST_DIR / f"{timestamp_slug}_{backend_slug}_{model_slug}_{mode_slug}_raw.txt"
    )

    output_path: Path | None = None
    valid_json = False
    valid_eventresult = False
    predicted_count = 0
    failure_mode: FailureMode = "success"

    try:
        raw_text = call_backend(
            backend=args.backend,
            image_path=image_path,
            model=args.model,
            mode=mode,
            timeout=args.timeout,
        )
    except InvalidModelForBackendError as exc:
        raw_text = str(exc)
        failure_mode = "invalid_model_for_backend"
    except NotImplementedError as exc:
        raw_text = str(exc)
        failure_mode = "image_not_supported"
    except (urllib.error.URLError, TimeoutError, OSError, MemoryError, ValueError) as exc:
        raw_text, failure_mode = classify_api_exception(exc)

    save_text(raw_output_path, raw_text)

    if failure_mode == "success":
        valid_json, valid_eventresult, predicted_count, event_result, failure_mode = validate_response(
            raw_text, mode
        )
        if valid_eventresult and event_result is not None:
            output_path = (
                ANNOTATION_DIR
                / f"{image_path.stem}_{backend_slug}_{model_slug}_{mode_slug}_events.json"
            )
            save_event_result(output_path, event_result)

    row = {
        "timestamp": timestamp,
        "backend": args.backend,
        "model": args.model,
        "image_path": str(image_path),
        "mode": mode,
        "valid_json": valid_json,
        "valid_eventresult": valid_eventresult,
        "predicted_count": predicted_count,
        "output_path": str(output_path) if output_path is not None else "",
        "raw_output_path": str(raw_output_path),
        "failure_mode": failure_mode,
    }
    append_log(row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare VLM backends on the same grid-overlay spectrogram image."
    )
    parser.add_argument("--backend", choices=["ollama", "openai", "claude"], default="ollama")
    parser.add_argument("--model", default="qwen3-vl:latest")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--timeout", type=float, default=180.0)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--debug-simple", action="store_true")
    mode_group.add_argument("--debug-json", action="store_true")
    mode_group.add_argument("--annotate", action="store_true")
    mode_group.add_argument("--annotate-coordinates", action="store_true")
    mode_group.add_argument("--annotate-coordinates-high-recall", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    row = run(args)
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
