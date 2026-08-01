"""Repair simple EventResult schema issues in model raw outputs.

This is intentionally conservative:
- removes Markdown JSON fences
- accepts either {"audio_path": ..., "events": [...]} or a top-level event list
- fills missing/null review_reason with an empty string
- validates against the local EventResult schema

It does not convert image-space boxes such as box_2d into time/frequency
coordinates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import EventResult


def strip_markdown_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    return text


def load_raw_json(path: Path) -> Any:
    text = strip_markdown_json_fence(path.read_text(encoding="utf-8"))
    return json.loads(text)


def repair_payload(payload: Any, audio_path: str) -> EventResult:
    if isinstance(payload, list):
        payload = {"audio_path": audio_path, "events": payload}
    elif isinstance(payload, dict):
        payload.setdefault("audio_path", audio_path)
    else:
        raise ValueError("Expected a JSON object or list of events.")

    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("Expected an EventResult object with an events list.")

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each event must be a JSON object.")
        if "box_2d" in event:
            raise ValueError("Refusing to repair image-space box_2d coordinates.")
        if event.get("review_reason") is None:
            event["review_reason"] = ""

    return EventResult.model_validate(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair simple schema issues in raw EventResult JSON."
    )
    parser.add_argument("input", type=Path, help="Raw model response text/JSON file.")
    parser.add_argument("output", type=Path, help="Repaired EventResult JSON output path.")
    parser.add_argument("--audio-path", default="pseudo_petersi_001.wav")
    args = parser.parse_args()

    payload = load_raw_json(args.input)
    repaired = repair_payload(payload, args.audio_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(repaired.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved repaired EventResult to {args.output}")
    print(f"events: {len(repaired.events)}")


if __name__ == "__main__":
    main()
