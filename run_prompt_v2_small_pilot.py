"""Run Prompt V2 on the six clean spectrogram inputs without ground-truth access."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf

from main import OLLAMA_HOST


DEFAULT_PROMPT_PATH = Path("prompts/prompt_v2_bat_strong_label.md")
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_IMAGE_DIR = Path("outputs/agent_inputs/prompt_v2_small_pilot")
DEFAULT_OUTPUT_DIR = Path("outputs/agent_runs/prompt_v2_small_pilot")
DEFAULT_CLIP_IDS = ["OP_001", "OP_010", "OP_045", "OP_003", "OP_004", "OP_016"]
DEFAULT_MODEL_NAME = "qwen3-vl:latest"
PROMPT_VERSION = "prompt_v2_bat_strong_label"
REQUIRED_EVENT_FIELDS = {
    "event_id": str,
    "start_time_seconds": (int, float),
    "end_time_seconds": (int, float),
    "low_frequency_hz": (int, float),
    "high_frequency_hz": (int, float),
    "label": str,
    "confidence": (int, float),
    "evidence": str,
    "human_review_needed": bool,
    "review_reason": str,
}

PREDICTION_JSON_SCHEMA = {
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
                },
                "required": list(REQUIRED_EVENT_FIELDS),
            },
        },
    },
    "required": ["clip_id", "events"],
}


@dataclass(frozen=True)
class PilotResult:
    clip_id: str
    parse_status: str
    predicted_event_count: int | None
    output_json_path: Path | None
    raw_response_path: Path
    parse_error_path: Path | None


def load_prompt(prompt_path: str | Path = DEFAULT_PROMPT_PATH) -> str:
    """Load the versioned prompt text from disk."""
    path = Path(prompt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


def read_clip_duration(eval_dir: str | Path, clip_id: str) -> float:
    """Read clip duration from the evaluation-set WAV metadata."""
    audio_path = Path(eval_dir) / "audio" / f"{clip_id}.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio for clip id {clip_id!r} not found: {audio_path}")
    return float(sf.info(audio_path).duration)


def resolve_input_image(image_dir: str | Path, clip_id: str) -> Path:
    """Resolve one clean spectrogram image."""
    image_path = Path(image_dir) / f"{clip_id}_spectrogram.png"
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Clean spectrogram for clip id {clip_id!r} not found: {image_path}"
        )
    return image_path


def build_user_message(
    *,
    clip_id: str,
    clip_duration_seconds: float,
) -> str:
    """Build the clip-specific user message sent with the clean image."""
    context = {
        "clip_id": clip_id,
        "clip_duration_seconds": clip_duration_seconds,
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
    }
    return (
        "/no_think\n"
        "Annotate the attached clean spectrogram according to the system prompt.\n\n"
        "Runtime context:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Do not explain your reasoning. Return the valid JSON object immediately."
    )


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def call_ollama(
    *,
    image_path: Path,
    system_prompt: str,
    user_message: str,
    model_name: str,
    timeout: float,
    num_predict: int,
) -> str:
    """Call the existing local Ollama vision workflow."""
    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": PREDICTION_JSON_SCHEMA,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
                "images": [image_to_base64(image_path)],
            }
        ],
    }
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

    reasoning = message.get("thinking") or message.get("reasoning")
    if reasoning:
        return str(reasoning)
    return json.dumps(response_payload, indent=2, ensure_ascii=False)


def call_ollama_generate(
    *,
    image_path: Path,
    system_prompt: str,
    user_message: str,
    model_name: str,
    timeout: float,
    num_predict: int,
) -> str:
    """Call Ollama's generate endpoint and return its clean constrained output."""
    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": PREDICTION_JSON_SCHEMA,
        "prompt": f"{system_prompt}\n\n{user_message}",
        "images": [image_to_base64(image_path)],
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
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


def call_ollama_cli(
    *,
    image_path: Path,
    system_prompt: str,
    user_message: str,
    model_name: str,
    timeout: float,
) -> str:
    """Call Ollama CLI, whose no-thinking mode works for the local vision model."""
    combined_prompt = (
        f"{system_prompt}\n\n"
        f"{user_message}\n\n"
        f"Input spectrogram image: {image_path.resolve()}"
    )
    env = os.environ.copy()
    env["OLLAMA_NOHISTORY"] = "1"
    completed = subprocess.run(
        [
            "ollama",
            "run",
            model_name,
            "--think=false",
            "--format",
            "json",
            combined_prompt,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"ollama CLI exited with code {completed.returncode}: {error_text}"
        )
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", completed.stdout).strip()


def extract_json_text(raw_text: str) -> str:
    """Extract the most likely JSON object from a model response."""
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
    if start == -1 or end < start:
        return text
    return text[start : end + 1]


def validate_prediction(payload: Any, *, expected_clip_id: str) -> dict[str, Any]:
    """Validate the basic Prompt V2 prediction structure."""
    if not isinstance(payload, dict):
        raise ValueError("Top-level model output must be a JSON object")
    if payload.get("clip_id") != expected_clip_id:
        raise ValueError(
            f"clip_id must be {expected_clip_id!r}, got {payload.get('clip_id')!r}"
        )

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("Top-level 'events' must be a list")

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"events[{index}] must be a JSON object")
        for field, expected_type in REQUIRED_EVENT_FIELDS.items():
            if field not in event:
                raise ValueError(f"events[{index}] is missing required field {field!r}")
            value = event[field]
            if isinstance(value, bool) and expected_type == (int, float):
                raise ValueError(f"events[{index}].{field} must be numeric")
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"events[{index}].{field} has invalid type "
                    f"{type(value).__name__}"
                )
    return payload


def parse_prediction(raw_text: str, *, expected_clip_id: str) -> dict[str, Any]:
    """Extract, decode, and validate a raw model response."""
    parsed = json.loads(extract_json_text(raw_text))
    return validate_prediction(parsed, expected_clip_id=expected_clip_id)


def write_prediction_output(
    *,
    output_path: Path,
    prediction: dict[str, Any],
    model_name: str,
    backend: str,
    run_timestamp: str,
    input_image_path: Path,
    clip_duration_seconds: float,
) -> None:
    """Write validated events with reproducibility metadata."""
    payload = {
        "clip_id": prediction["clip_id"],
        "prompt_version": PROMPT_VERSION,
        "model_name": model_name,
        "backend": backend,
        "run_timestamp": run_timestamp,
        "input_image_path": input_image_path.as_posix(),
        "clip_duration_seconds": clip_duration_seconds,
        "events": prediction["events"],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_clip(
    *,
    clip_id: str,
    prompt_text: str,
    eval_dir: Path,
    image_dir: Path,
    output_dir: Path,
    model_name: str,
    backend: str,
    timeout: float,
    num_predict: int,
) -> PilotResult:
    """Run one clip, preserve raw output, and continue cleanly on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_response_path = output_dir / f"{clip_id}_raw_response.txt"
    prediction_path = output_dir / f"{clip_id}_predictions.json"
    parse_error_path = output_dir / f"{clip_id}_parse_error.txt"
    prediction_path.unlink(missing_ok=True)
    parse_error_path.unlink(missing_ok=True)

    try:
        clip_duration = read_clip_duration(eval_dir, clip_id)
        image_path = resolve_input_image(image_dir, clip_id)
        user_message = build_user_message(
            clip_id=clip_id,
            clip_duration_seconds=clip_duration,
        )
        if backend == "ollama_generate":
            raw_text = call_ollama_generate(
                image_path=image_path,
                system_prompt=prompt_text,
                user_message=user_message,
                model_name=model_name,
                timeout=timeout,
                num_predict=num_predict,
            )
        elif backend == "ollama_cli":
            raw_text = call_ollama_cli(
                image_path=image_path,
                system_prompt=prompt_text,
                user_message=user_message,
                model_name=model_name,
                timeout=timeout,
            )
        elif backend == "ollama_api":
            raw_text = call_ollama(
                image_path=image_path,
                system_prompt=prompt_text,
                user_message=user_message,
                model_name=model_name,
                timeout=timeout,
                num_predict=num_predict,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        raw_response_path.write_text(raw_text, encoding="utf-8")

        prediction = parse_prediction(raw_text, expected_clip_id=clip_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        write_prediction_output(
            output_path=prediction_path,
            prediction=prediction,
            model_name=model_name,
            backend=backend,
            run_timestamp=timestamp,
            input_image_path=image_path,
            clip_duration_seconds=clip_duration,
        )
        return PilotResult(
            clip_id=clip_id,
            parse_status="success",
            predicted_event_count=len(prediction["events"]),
            output_json_path=prediction_path,
            raw_response_path=raw_response_path,
            parse_error_path=None,
        )
    except Exception as exc:
        if not raw_response_path.exists():
            raw_response_path.write_text("", encoding="utf-8")
        parse_error_path.write_text(
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        return PilotResult(
            clip_id=clip_id,
            parse_status="failed",
            predicted_event_count=None,
            output_json_path=None,
            raw_response_path=raw_response_path,
            parse_error_path=parse_error_path,
        )


def print_summary(results: list[PilotResult]) -> None:
    """Print a concise, dependency-free summary table."""
    headers = [
        "clip_id",
        "parse_status",
        "predicted_event_count",
        "output_json_path",
        "raw_response_path",
        "parse_error_path",
    ]
    rows = [
        [
            result.clip_id,
            result.parse_status,
            "" if result.predicted_event_count is None else str(result.predicted_event_count),
            "" if result.output_json_path is None else str(result.output_json_path),
            str(result.raw_response_path),
            "" if result.parse_error_path is None else str(result.parse_error_path),
        ]
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prompt V2 on the six clean representative spectrograms."
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--clip-list",
        default=",".join(DEFAULT_CLIP_IDS),
        help="Comma-separated clip ids. Defaults to the six representative clips.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--backend",
        choices=["ollama_generate", "ollama_cli", "ollama_api"],
        default="ollama_generate",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--num-predict",
        type=int,
        default=8000,
        help="Maximum generated tokens per clip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_text = load_prompt(args.prompt)
    results = []
    for clip_id in parse_clip_ids(args.clip_list):
        print(f"Running {clip_id} with {args.backend}/{args.model}...")
        results.append(
            run_clip(
                clip_id=clip_id,
                prompt_text=prompt_text,
                eval_dir=args.eval_dir,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                model_name=args.model,
                backend=args.backend,
                timeout=args.timeout,
                num_predict=args.num_predict,
            )
        )
    print_summary(results)


if __name__ == "__main__":
    main()
