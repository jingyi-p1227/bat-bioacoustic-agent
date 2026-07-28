"""Run Stage 1C GT-box-marker classification with compact acoustic guidance.

The runner uses only label-safe marker images from `gt_box_marker_image_path`.
It provides compact generic acoustic guidance and provisional genus/species
diagnostic notes, but does not use image exemplars, GT overlays, centred
no-box images, or sample-level ground-truth labels in the prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_prompt_v2_small_pilot import extract_json_text  # noqa: E402
from scripts.inference.run_stage1a_multispecies_classification import (  # noqa: E402
    ALLOWED_LABELS,
    CLASSIFICATION_SCHEMA,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_NAME,
    RunResult,
    call_ollama_generate,
    load_manifest,
    resolve_repo_path,
    write_results,
)


CONDITION_NAME = "qwen3_6_stage1c_gt_box_marker_species_guidance"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/agent_runs/multispecies_classification/"
    / CONDITION_NAME
)

GUIDANCE_SCHEMA = {
    **CLASSIFICATION_SCHEMA,
    "properties": {
        **CLASSIFICATION_SCHEMA["properties"],
        "guidance_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        *CLASSIFICATION_SCHEMA["required"],
        "guidance_used",
    ],
}

VERIFIED_EVIDENCE_IDS = (
    "walters_2012_chunk_002_call_detection_measurement_quality",
    "walters_2012_chunk_003_parameter_dimensions",
    "walters_2012_chunk_004_thresholds_and_uncertainty",
)


def compact_guidance_block() -> str:
    return (
        "Compact acoustic guidance for this baseline:\n"
        "\n"
        "Verified generic acoustic checklist from Walters et al. 2012 evidence "
        "chunks `walters_2012_chunk_002_call_detection_measurement_quality`, "
        "`walters_2012_chunk_003_parameter_dimensions`, and "
        "`walters_2012_chunk_004_thresholds_and_uncertainty`:\n"
        "- Compare visible call shape, frequency position, bandwidth, duration, "
        "ridge slope/modulation, and signal quality.\n"
        "- Treat onset/offset, lower/upper frequency, bandwidth, peak/centre "
        "frequency, and ridge shape as visual cues.\n"
        "- Use weak, noisy, truncated, or low-contrast evidence cautiously; do "
        "not force high confidence from brightness alone.\n"
        "- Use this as a checklist, not as a rigid decision rule.\n"
        "\n"
        "Provisional diagnostic guidance, not verified species-specific "
        "literature rules:\n"
        "- Rhinolophus calls are often expected to look more quasi-constant-"
        "frequency or narrowband than strongly frequency-modulated sweep calls; "
        "distinguishing the two Rhinolophus labels may still be hard.\n"
        "- Myotis labels are expected to be difficult to separate from image "
        "appearance alone; do not guess a Myotis species solely from noise, "
        "brightness, or a weak partial ridge.\n"
        "- Pipistrellus and Plecotus may be tempting default labels in zero-shot "
        "settings; avoid choosing them unless the boxed call itself supports it.\n"
        "- Ozimops petersi is an Australian benchmark anchor and lacks verified "
        "species guidance here; do not transfer European species priors to it.\n"
        "- If cues conflict, choose the single best allowed label but lower "
        "confidence and say which guidance cue was uncertain.\n"
    )


def build_system_prompt() -> str:
    labels = "\n".join(f"- {label}" for label in ALLOWED_LABELS)
    return (
        "You are a bioacoustic image classification model. Your task is species "
        "classification for one boxed target bat echolocation call.\n\n"
        "A neutral box in the image marks the ground-truth target event location. "
        "Classify only the boxed target call. Ignore other calls, background calls, "
        "silence padding artefacts, noise, axes, gridlines, or any non-target "
        "signal.\n\n"
        "Choose exactly one species label from this allowed list:\n"
        f"{labels}\n\n"
        f"{compact_guidance_block()}\n"
        "Do not use image exemplars, sample-level labels, or any information "
        "outside the provided image and compact guidance. Do not treat the "
        "provisional guidance as a rigid rule.\n\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "predicted_species": "one allowed label",\n'
        '  "confidence": 0.0,\n'
        '  "reasoning_brief": "short visual justification",\n'
        '  "visual_evidence": ["brief visible cue"],\n'
        '  "guidance_used": ["brief checklist cue used"]\n'
        "}"
    )


def build_user_message(row: dict[str, str]) -> str:
    return (
        "/no_think\n"
        "Classify the species of the boxed target bat call in the attached "
        "spectrogram crop. The neutral box marks the target event location. "
        "Use the compact acoustic guidance as a checklist, not as a rigid rule. "
        "Myotis species may be difficult to separate, so do not guess from noise "
        "or brightness alone. Choose exactly one allowed label. Return valid "
        "JSON only.\n\n"
        f"anonymous_sample_id: {row['anonymous_sample_id']}\n"
        f"image_variant: gt_box_marker\n"
    )


def parse_guided_classification(raw_text: str) -> dict[str, Any]:
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
    guidance = payload.get("guidance_used")
    if not isinstance(guidance, list) or not all(isinstance(item, str) for item in guidance):
        raise ValueError("guidance_used must be a list of strings")
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
    image_path = resolve_repo_path(row["gt_box_marker_image_path"])
    try:
        raw_text = call_ollama_generate(
            image_path=image_path,
            system_prompt=system_prompt,
            user_message=build_user_message(row),
            model_name=model_name,
            timeout=timeout,
            num_predict=num_predict,
            response_schema=GUIDANCE_SCHEMA,
        )
        raw_path.write_text(raw_text, encoding="utf-8")
        parsed = parse_guided_classification(raw_text)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-field", default="gt_box_marker_image_path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=1400)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_field != "gt_box_marker_image_path":
        raise ValueError("Stage 1C must use gt_box_marker_image_path only")
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
