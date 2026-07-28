"""Run Stage 2 pilot80 joint localisation and species classification.

This runner uses the label-safe clean no-box Stage 1 images. It does not read
GT-box-marker images, human-review overlays, image exemplars, raw PDFs, or full
text. The pilot subset is deterministic: the first 10 manifest rows per species
in manifest order.

BatDetect2 proposal metadata is included only when a matching sample-level
proposal file exists. The current multi-species Stage 1 dataset does not ship
sample-level BatDetect2 proposals, so the runner records proposal availability
per sample rather than fabricating proposals from ground truth.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_prompt_v2_small_pilot import extract_json_text  # noqa: E402
from scripts.inference.run_stage1a_multispecies_classification import (  # noqa: E402
    ALLOWED_LABELS,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_NAME,
    image_to_base64,
    load_manifest,
    resolve_repo_path,
)


CONDITION_NAME = "qwen3_6_stage2_joint_proposal_constrained_pilot80"
STAGE2B_CONDITION_NAME = "qwen3_6_stage2b_true_proposal_constrained_pilot80"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/agent_runs/multispecies_classification/"
    / CONDITION_NAME
)
DEFAULT_PROPOSAL_DIR = (
    REPO_ROOT
    / "outputs/tool_outputs/batdetect2_proposals/multispecies_stage2_pilot80"
)
DEFAULT_AUDIO_WINDOW_MANIFEST = (
    REPO_ROOT
    / "outputs/tool_outputs/batdetect2_multispecies_stage1_windows/"
    "audio_window_manifest.csv"
)
DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["retain", "refine"]},
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "low_freq": {"type": "number"},
                    "high_freq": {"type": "number"},
                    "predicted_species": {"type": "string", "enum": list(ALLOWED_LABELS)},
                    "confidence": {"type": "number"},
                    "reasoning_brief": {"type": "string"},
                },
                "required": [
                    "start_time",
                    "end_time",
                    "low_freq",
                    "high_freq",
                    "predicted_species",
                    "confidence",
                ],
            },
        },
        "rejected_proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["proposal_id", "reason"],
            },
        },
    },
    "required": ["detections"],
}


def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def select_balanced_subset(rows: list[dict[str, str]], per_species: int = 10) -> list[dict[str, str]]:
    counts: Counter[str] = Counter()
    selected: list[dict[str, str]] = []
    for row in rows:
        species = row["species"]
        if species not in ALLOWED_LABELS:
            continue
        if counts[species] >= per_species:
            continue
        counts[species] += 1
        selected.append(row)
    missing = {label: per_species - counts[label] for label in ALLOWED_LABELS if counts[label] < per_species}
    if missing:
        raise ValueError(f"Insufficient samples for balanced subset: {missing}")
    return selected


def load_sample_proposals(proposal_dir: Path, anonymous_sample_id: str) -> tuple[list[dict[str, Any]], str]:
    path = proposal_dir / f"{anonymous_sample_id}_batdetect2_proposals.json"
    if not path.is_file():
        return [], "missing_sample_level_proposal_file"
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events")
    if not isinstance(events, list):
        return [], "invalid_proposal_payload"
    return events, "available"


def compact_proposal(event: dict[str, Any]) -> dict[str, Any]:
    """Keep only detector localisation metadata, not detector species labels."""

    start = float(event.get("start_time_seconds", event.get("start_time")))
    end = float(event.get("end_time_seconds", event.get("end_time")))
    return {
        "proposal_id": event.get("proposal_id"),
        "start_time": round(start, 6),
        "end_time": round(end, 6),
        "duration_ms": round((end - start) * 1000, 3),
        "low_freq": float(event.get("low_frequency_hz", event.get("low_freq"))),
        "high_freq": float(event.get("high_frequency_hz", event.get("high_freq"))),
        "det_prob": float(event.get("det_prob", 0.0)),
    }


def load_audio_window_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["anonymous_sample_id"]: row for row in csv.DictReader(handle)}


def build_system_prompt() -> str:
    labels = "\n".join(f"- {label}" for label in ALLOWED_LABELS)
    return (
        "You are a bioacoustic annotation model for event-level bat calls. "
        "Your task is joint localisation and species classification from one "
        "clean spectrogram image.\n\n"
        "Use the supplied BatDetect2 proposals as candidate events, not ground "
        "truth. Identify which proposals correspond to real target bat calls, "
        "reject obvious false positives/noise/echo/background artefacts, and "
        "refine coordinates only when the spectrogram gives clear evidence. "
        "Prefer conservative retain/reject/refine over free-form redrawing. "
        "Preserve proposal timing unless there is clear visual evidence to "
        "adjust it. Classify only retained detections.\n\n"
        "Output all times in local 0.000-0.300 second window coordinates, even "
        "if the image axis tick labels display the original source-time range. "
        "Use Hz for frequency. Do not use any species label unless it is chosen "
        "from the allowed list.\n\n"
        "Allowed species labels:\n"
        f"{labels}\n\n"
        "BatDetect2 proposal IDs are provenance handles, not class labels. "
        "Do not blindly keep all proposals. Do not use GT overlays, marker boxes, "
        "image exemplars, raw PDFs, full text, or hidden labels.\n\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "detections": [\n'
        "    {\n"
        '      "proposal_id": "bd2_001",\n'
        '      "decision": "retain",\n'
        '      "start_time": 0.0,\n'
        '      "end_time": 0.0,\n'
        '      "low_freq": 0.0,\n'
        '      "high_freq": 0.0,\n'
        '      "predicted_species": "one allowed label",\n'
        '      "confidence": 0.0,\n'
        '      "reasoning_brief": "short visual reason"\n'
        "    }\n"
        "  ],\n"
        '  "rejected_proposals": [\n'
        '    {"proposal_id": "bd2_002", "reason": "noise or not visible"}\n'
        "  ]\n"
        "}"
    )


def build_user_message(row: dict[str, str], proposals: list[dict[str, Any]], proposal_status: str) -> str:
    payload = {
        "anonymous_sample_id": row["anonymous_sample_id"],
        "image_variant": "centred_crop_no_box",
        "target_note": (
            "The image is a clean label-safe spectrogram crop without GT marker. "
            "No species label or target box is provided in this prompt."
        ),
        "coordinate_frame": "local_window_seconds_0.000_to_0.300",
        "proposal_status": proposal_status,
        "batdetect2_proposals": [compact_proposal(item) for item in proposals],
        "instructions": [
            "Use local 0.000-0.300 s window coordinates for start_time and end_time.",
            "Treat the proposals as candidates, not ground truth.",
            "Reject proposals that look like noise, echo, partial artefact, or background.",
            "Preserve proposal timing unless clear visual evidence supports a small refinement.",
            "Classify each detection using exactly one allowed species label.",
            "Ignore padding artefacts, gridlines, axes, and background texture.",
            "Return valid JSON only.",
        ],
    }
    return "/no_think\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def call_ollama_generate(
    *,
    image_path: Path,
    system_prompt: str,
    user_message: str,
    model_name: str,
    timeout: float,
    num_predict: int,
) -> str:
    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": DETECTION_SCHEMA,
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
    return json.dumps(response_payload, indent=2, ensure_ascii=False)


def parse_joint_payload(raw_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(extract_json_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON must be an object")
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise ValueError("detections must be a list")
    parsed: list[dict[str, Any]] = []
    for index, detection in enumerate(detections, start=1):
        if not isinstance(detection, dict):
            raise ValueError(f"detection {index} must be an object")
        species = detection.get("predicted_species")
        if species not in ALLOWED_LABELS:
            raise ValueError(f"invalid predicted_species in detection {index}: {species!r}")
        confidence = detection.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"confidence in detection {index} must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ValueError(f"confidence in detection {index} must be between 0 and 1")
        parsed.append(
            {
                "proposal_id": str(detection.get("proposal_id") or ""),
                "decision": str(detection.get("decision") or ""),
                "start_time": detection.get("start_time"),
                "end_time": detection.get("end_time"),
                "low_freq": detection.get("low_freq"),
                "high_freq": detection.get("high_freq"),
                "predicted_species": species,
                "confidence": float(confidence),
                "reasoning_brief": str(detection.get("reasoning_brief") or ""),
            }
        )
    rejected = payload.get("rejected_proposals") or []
    if not isinstance(rejected, list):
        raise ValueError("rejected_proposals must be a list when provided")
    rejected_rows = [
        {
            "proposal_id": str(item.get("proposal_id") or ""),
            "reason": str(item.get("reason") or ""),
        }
        for item in rejected
        if isinstance(item, dict)
    ]
    return parsed, rejected_rows


def parse_joint_response(raw_text: str) -> list[dict[str, Any]]:
    detections, _rejected = parse_joint_payload(raw_text)
    return detections


def write_subset_manifest(
    path: Path,
    rows: list[dict[str, str]],
    proposal_status: dict[str, str],
    proposal_counts: dict[str, int],
    audio_window_metadata: dict[str, dict[str, str]],
) -> None:
    fields = [
        "sample_id",
        "anonymous_sample_id",
        "species",
        "source_dataset",
        "source_recording",
        "source_recording_id",
        "event_index",
        "event_start_time",
        "event_end_time",
        "event_low_freq",
        "event_high_freq",
        "local_gt_start_time",
        "local_gt_end_time",
        "centred_crop_image_path",
        "proposal_status",
        "proposal_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            anon_id = row["anonymous_sample_id"]
            window_row = audio_window_metadata.get(anon_id, {})
            out = {field: row.get(field, "") for field in fields}
            out["proposal_status"] = proposal_status.get(anon_id, "")
            out["proposal_count"] = proposal_counts.get(anon_id, 0)
            out["local_gt_start_time"] = window_row.get("local_gt_start_time", "")
            out["local_gt_end_time"] = window_row.get("local_gt_end_time", "")
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-field", default="centred_crop_image_path")
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--audio-window-manifest", type=Path, default=DEFAULT_AUDIO_WINDOW_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-name", default=CONDITION_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=1600)
    parser.add_argument("--per-species", type=int, default=10)
    args = parser.parse_args()

    if args.image_field != "centred_crop_image_path":
        raise ValueError("Stage 2 pilot must use clean centred_crop_image_path only")
    if args.output_dir.exists() and (args.output_dir / "parsed_predictions.csv").exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    raw_dir = args.output_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = select_balanced_subset(load_manifest(args.manifest), args.per_species)
    system_prompt = build_system_prompt()
    prediction_rows: list[dict[str, Any]] = []
    parse_failure_rows: list[dict[str, str]] = []
    proposal_status: dict[str, str] = {}
    proposal_counts: dict[str, int] = {}
    audio_window_metadata = load_audio_window_metadata(args.audio_window_manifest)

    for index, row in enumerate(rows, start=1):
        anon_id = row["anonymous_sample_id"]
        print(f"[{index}/{len(rows)}] {anon_id}", flush=True)
        raw_path = raw_dir / f"{anon_id}_raw_response.txt"
        image_path = resolve_repo_path(row["centred_crop_image_path"])
        proposals, status = load_sample_proposals(args.proposal_dir, anon_id)
        proposal_status[anon_id] = status
        proposal_counts[anon_id] = len(proposals)
        try:
            raw_text = call_ollama_generate(
                image_path=image_path,
                system_prompt=system_prompt,
                user_message=build_user_message(row, proposals, status),
                model_name=args.model_name,
                timeout=args.timeout,
                num_predict=args.num_predict,
            )
            raw_path.write_text(raw_text, encoding="utf-8")
            detections, rejected = parse_joint_payload(raw_text)
            parse_status = "success"
            parse_error = ""
        except Exception as exc:
            if not raw_path.exists():
                raw_path.write_text("", encoding="utf-8")
            detections = []
            rejected = []
            parse_status = "failed"
            parse_error = f"{type(exc).__name__}: {exc}"
            parse_failure_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "anonymous_sample_id": anon_id,
                    "parse_error": parse_error,
                    "raw_response_path": raw_path.as_posix(),
                }
            )
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "anonymous_sample_id": anon_id,
                "true_species": row["species"],
                "parse_status": parse_status,
                "proposal_status": status,
                "proposal_count": len(proposals),
                "prediction_count": len(detections),
                "detections_json": json.dumps(detections, ensure_ascii=False),
                "rejected_proposals_json": json.dumps(rejected, ensure_ascii=False),
                "raw_response_path": raw_path.as_posix(),
                "parse_error": parse_error,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_subset_manifest(
        args.output_dir / "pilot80_subset_manifest.csv",
        rows,
        proposal_status,
        proposal_counts,
        audio_window_metadata,
    )
    pred_fields = list(prediction_rows[0])
    with (args.output_dir / "parsed_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    failure_fields = ["sample_id", "anonymous_sample_id", "parse_error", "raw_response_path"]
    with (args.output_dir / "parse_failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(parse_failure_rows)

    successes = sum(row["parse_status"] == "success" for row in prediction_rows)
    print(
        f"Condition={args.condition_name} samples={len(prediction_rows)} "
        f"parse_success={successes} parse_failure={len(prediction_rows) - successes}"
    )


if __name__ == "__main__":
    main()
