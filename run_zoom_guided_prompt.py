"""Run a fresh zoom-guided visual annotation prompt outside the web chat.

This script intentionally bypasses the Pydantic-AI web chat history. It sends
one zoomed spectrogram image plus one short prompt directly to the local Ollama
native chat endpoint.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from typing import Any

from pydantic import ValidationError

from main import (
    ANNOTATION_DIR,
    OLLAMA_HOST,
    OUTPUT_DIR,
    EventResult,
    figure_to_image,
    make_spectrogram,
    plot_spectrogram_with_grid,
    read_mono_audio,
)


AUDIO_PATH = "pseudo_petersi_001.wav"
START_TIME = 0.0
END_TIME = 4.0
LOW_FREQUENCY = 20_000
HIGH_FREQUENCY = 100_000
PRESET = "short_event"

MODEL_NAME = "qwen3-vl:latest"
OUTPUT_JSON_PATH = ANNOTATION_DIR / "pseudo_petersi_001_zoom_guided_events.json"
RAW_RESPONSE_PATH = OUTPUT_DIR / "raw_zoom_guided_response.txt"

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
                    "tools_used": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
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

PROMPT = (
    "/no_think\n"
    "Find up to 10 clear bat echolocation pulses. Use tight time-frequency boxes. "
    "The 20-100 kHz range is only the search region, not the box boundary. "
    "Return JSON events only."
)


def build_zoom_image_bytes() -> bytes:
    """Generate only the requested zoomed spectrogram image."""
    _, audio, sr = read_mono_audio(AUDIO_PATH)
    spec, local_stft = make_spectrogram(audio, sr)
    fig = plot_spectrogram_with_grid(
        spec,
        audio,
        local_stft,
        sr,
        start_time=START_TIME,
        end_time=END_TIME,
        low_frequency=LOW_FREQUENCY,
        high_frequency=HIGH_FREQUENCY,
        preset=PRESET,
    )
    return figure_to_image(fig).read()


def call_qwen_with_zoom_image(image_bytes: bytes) -> str:
    """Send a fresh one-message image prompt to Ollama/qwen3-vl."""
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "think": False,
        "format": EVENT_RESULT_JSON_SCHEMA,
        "options": {
            "temperature": 0,
            "num_predict": 2500,
        },
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64],
            }
        ],
    }

    print(f"Sending one zoomed spectrogram image to {MODEL_NAME}...")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    message = response_payload.get("message", {})
    content = message.get("content")
    if not content:
        reasoning = message.get("thinking") or message.get("reasoning")
        if reasoning:
            return str(reasoning)
        return json.dumps(response_payload, indent=2, ensure_ascii=False)
    if isinstance(content, list):
        return "\n".join(str(part) for part in content if part)
    return str(content)


def extract_json_text(raw_text: str) -> str:
    """Return the most likely JSON substring from the model response."""
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

    object_start = text.find("{")
    array_start = text.find("[")
    starts = [index for index in (object_start, array_start) if index != -1]
    if not starts:
        return text

    start = min(starts)
    end_char = "}" if text[start] == "{" else "]"
    end = text.rfind(end_char)
    if end == -1:
        return text
    return text[start : end + 1]


def normalise_event_result(parsed: Any) -> EventResult:
    """Validate model JSON against the local EventResult schema."""
    if isinstance(parsed, list):
        payload = {"audio_path": AUDIO_PATH, "events": parsed}
    elif isinstance(parsed, dict):
        if "events" in parsed:
            payload = {"audio_path": AUDIO_PATH, **parsed}
        else:
            payload = {"audio_path": AUDIO_PATH, "events": [parsed]}
    else:
        raise ValueError("Expected a JSON object or array.")

    return EventResult.model_validate(payload)


def save_raw_response(raw_text: str) -> None:
    RAW_RESPONSE_PATH.parent.mkdir(exist_ok=True)
    RAW_RESPONSE_PATH.write_text(raw_text, encoding="utf-8")


def main() -> None:
    ANNOTATION_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Generating zoomed spectrogram image...")
    image_bytes = build_zoom_image_bytes()
    raw_text = call_qwen_with_zoom_image(image_bytes)

    try:
        parsed = json.loads(extract_json_text(raw_text))
        event_result = normalise_event_result(parsed)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        save_raw_response(raw_text)
        if raw_text.lstrip().startswith("<think>"):
            raise SystemExit(
                f"Model returned Qwen thinking text instead of JSON. "
                f"Raw response saved to {RAW_RESPONSE_PATH}."
            ) from exc
        raise SystemExit(
            f"Failed to parse/validate JSON. Raw response saved to {RAW_RESPONSE_PATH}. "
            f"Error: {exc}"
        ) from exc

    OUTPUT_JSON_PATH.write_text(
        json.dumps(event_result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved zoom-guided events to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
