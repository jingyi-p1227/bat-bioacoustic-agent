"""Run Stage 1A zero-shot multi-species GT-event classification.

The runner uses only label-safe centred crops from `centred_crop_image_path`.
It does not read GT diagnostic overlays, marker images, species cards,
Walters guidance, or annotation exemplars. Species labels are used only to
write the prediction table for later evaluation.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_prompt_v2_small_pilot import extract_json_text  # noqa: E402


CONDITION_NAME = "qwen3_6_stage1a_gt_centred_no_box_zero_shot"
DEFAULT_MODEL_NAME = "qwen3.6:latest"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_stage1_gt_event_classification_dataset/"
    "stage1_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/agent_runs/multispecies_classification/"
    / CONDITION_NAME
)
ALLOWED_LABELS = (
    "Rhinolophus hipposideros",
    "Rhinolophus ferrumequinum",
    "Myotis daubentonii",
    "Myotis nattereri",
    "Myotis mystacinus",
    "Plecotus auritus",
    "Pipistrellus pipistrellus",
    "Ozimops petersi",
)
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_species": {"type": "string", "enum": list(ALLOWED_LABELS)},
        "confidence": {"type": "number"},
        "reasoning_brief": {"type": "string"},
        "visual_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "predicted_species",
        "confidence",
        "reasoning_brief",
        "visual_evidence",
    ],
}


@dataclass(frozen=True)
class RunResult:
    sample_id: str
    anonymous_sample_id: str
    true_species: str
    parse_status: str
    predicted_species: str
    confidence: str
    raw_response_path: Path
    parse_error: str


def ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_system_prompt() -> str:
    labels = "\n".join(f"- {label}" for label in ALLOWED_LABELS)
    return (
        "You are a bioacoustic image classification model. Your task is species "
        "classification for one target bat echolocation call.\n\n"
        "The target call is horizontally centred in the image because the "
        "ground-truth event location was used only to construct the crop. "
        "Classify only the centred target call. Ignore background calls, silence "
        "padding artefacts, noise, axes, gridlines, or any non-target signal.\n\n"
        "Choose exactly one species label from this allowed list:\n"
        f"{labels}\n\n"
        "Do not use species descriptions, examples, external acoustic libraries, "
        "or prior species cards. Use only the visual evidence in the image.\n\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "predicted_species": "one allowed label",\n'
        '  "confidence": 0.0,\n'
        '  "reasoning_brief": "short visual justification",\n'
        '  "visual_evidence": ["brief visible cue"]\n'
        "}"
    )


def build_user_message(row: dict[str, str]) -> str:
    return (
        "/no_think\n"
        "Classify the species of the horizontally centred target bat call in "
        "the attached spectrogram crop. Choose exactly one allowed label. "
        "Ignore background calls, padding artefacts, noise, axes, and gridlines. "
        "Return valid JSON only.\n\n"
        f"anonymous_sample_id: {row['anonymous_sample_id']}\n"
        f"image_variant: centred_crop_no_box\n"
        f"target_center_x_fraction: {row['target_center_x_fraction']}\n"
    )


def call_ollama_generate(
    *,
    image_path: Path,
    system_prompt: str,
    user_message: str,
    model_name: str,
    timeout: float,
    num_predict: int,
    response_schema: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": response_schema or CLASSIFICATION_SCHEMA,
        "prompt": f"{system_prompt}\n\n{user_message}",
        "images": [image_to_base64(image_path)],
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
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


def parse_classification(raw_text: str) -> dict[str, Any]:
    payload = json.loads(extract_json_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON must be an object")
    predicted = payload.get("predicted_species")
    if predicted not in ALLOWED_LABELS:
        raise ValueError(f"invalid predicted_species: {predicted!r}")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(payload.get("reasoning_brief"), str):
        raise ValueError("reasoning_brief must be a string")
    evidence = payload.get("visual_evidence")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise ValueError("visual_evidence must be a list of strings")
    return payload


def run_sample(
    *,
    row: dict[str, str],
    output_dir: Path,
    model_name: str,
    timeout: float,
    num_predict: int,
    system_prompt: str,
) -> RunResult:
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{row['anonymous_sample_id']}_raw_response.txt"
    image_path = resolve_repo_path(row["centred_crop_image_path"])
    try:
        raw_text = call_ollama_generate(
            image_path=image_path,
            system_prompt=system_prompt,
            user_message=build_user_message(row),
            model_name=model_name,
            timeout=timeout,
            num_predict=num_predict,
        )
        raw_path.write_text(raw_text, encoding="utf-8")
        parsed = parse_classification(raw_text)
        return RunResult(
            sample_id=row["sample_id"],
            anonymous_sample_id=row["anonymous_sample_id"],
            true_species=row["species"],
            parse_status="success",
            predicted_species=str(parsed["predicted_species"]),
            confidence=str(float(parsed["confidence"])),
            raw_response_path=raw_path,
            parse_error="",
        )
    except Exception as exc:
        if not raw_path.exists():
            raw_path.write_text("", encoding="utf-8")
        return RunResult(
            sample_id=row["sample_id"],
            anonymous_sample_id=row["anonymous_sample_id"],
            true_species=row["species"],
            parse_status="failed",
            predicted_species="",
            confidence="",
            raw_response_path=raw_path,
            parse_error=f"{type(exc).__name__}: {exc}",
        )


def write_results(output_dir: Path, results: list[RunResult]) -> None:
    rows = [
        {
            "sample_id": result.sample_id,
            "anonymous_sample_id": result.anonymous_sample_id,
            "true_species": result.true_species,
            "parse_status": result.parse_status,
            "predicted_species": result.predicted_species,
            "confidence": result.confidence,
            "raw_response_path": result.raw_response_path.as_posix(),
            "parse_error": result.parse_error,
        }
        for result in results
    ]
    fieldnames = list(rows[0]) if rows else []
    with (output_dir / "parsed_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["parse_status"] != "success"]
    with (output_dir / "parse_failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-field", default="centred_crop_image_path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_field != "centred_crop_image_path":
        raise ValueError("Stage 1A must use centred_crop_image_path only")
    if args.output_dir.exists() and (args.output_dir / "parsed_predictions.csv").exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    system_prompt = build_system_prompt()
    results: list[RunResult] = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['anonymous_sample_id']}", flush=True)
        results.append(
            run_sample(
                row=row,
                output_dir=args.output_dir,
                model_name=args.model_name,
                timeout=args.timeout,
                num_predict=args.num_predict,
                system_prompt=system_prompt,
            )
        )
    write_results(args.output_dir, results)
    successes = sum(result.parse_status == "success" for result in results)
    print(
        f"Condition={CONDITION_NAME} samples={len(results)} "
        f"parse_success={successes} parse_failure={len(results) - successes}"
    )


if __name__ == "__main__":
    main()
