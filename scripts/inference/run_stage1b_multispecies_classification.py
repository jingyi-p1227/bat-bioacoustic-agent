"""Run Stage 1B zero-shot multi-species GT-box-marker classification.

The runner uses only label-safe marker images from `gt_box_marker_image_path`.
It does not read human-review diagnostic overlays, centred crops, species cards,
Walters guidance, or annotation exemplars. Species labels are used only to
write the prediction table for later evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.run_stage1a_multispecies_classification import (
    ALLOWED_LABELS,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_NAME,
    RunResult,
    call_ollama_generate,
    load_manifest,
    parse_classification,
    resolve_repo_path,
    write_results,
)


CONDITION_NAME = "qwen3_6_stage1b_gt_box_marker_zero_shot"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/agent_runs/multispecies_classification/"
    / CONDITION_NAME
)


def build_system_prompt() -> str:
    labels = "\n".join(f"- {label}" for label in ALLOWED_LABELS)
    return (
        "You are a bioacoustic image classification model. Your task is species "
        "classification for one target bat echolocation call.\n\n"
        "A neutral box in the image marks the ground-truth target event location. "
        "Classify only the boxed target call. Ignore other calls, background calls, "
        "silence padding artefacts, noise, axes, gridlines, or any non-target signal.\n\n"
        "Choose exactly one species label from this allowed list:\n"
        f"{labels}\n\n"
        "Do not use species descriptions, examples, Walters acoustic guidance, "
        "external acoustic libraries, or prior species cards. Use only the visual "
        "evidence in the image.\n\n"
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
        "Classify the species of the boxed target bat call in the attached "
        "spectrogram crop. The neutral box marks the target event location. "
        "Choose exactly one allowed label. Ignore other calls, background calls, "
        "padding artefacts, noise, axes, and gridlines. Return valid JSON only.\n\n"
        f"anonymous_sample_id: {row['anonymous_sample_id']}\n"
        f"image_variant: gt_box_marker\n"
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-field", default="gt_box_marker_image_path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_field != "gt_box_marker_image_path":
        raise ValueError("Stage 1B must use gt_box_marker_image_path only")
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
