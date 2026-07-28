"""Run Stage 2C selected-proposal species classification.

This experiment preserves the deterministic nearest-centre BatDetect2 proposal
coordinates and asks the VLM only to classify the selected proposal. The model
is not asked to redraw, refine, add, or remove boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.run_stage1a_multispecies_classification import (  # noqa: E402
    ALLOWED_LABELS,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_NAME,
    call_ollama_generate,
    load_manifest,
    parse_classification,
    resolve_repo_path,
)
from scripts.inference.run_stage2_joint_proposal_constrained_pilot80 import (  # noqa: E402
    select_balanced_subset,
)


CONDITION_NAME = "qwen3_6_stage2c_nearest_centre_proposal_classification_pilot80"
DEFAULT_SELECTED_PROPOSALS = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_classification/"
    "stage2_central_proposal_selection_baseline/selected_proposals.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/agent_runs/multispecies_classification" / CONDITION_NAME
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_proposal_index(path: Path, clip_scope: str = "pilot80") -> dict[str, dict[str, str]]:
    rows = [
        row
        for row in read_csv(path)
        if row["clip_scope"] == clip_scope and row["rule"] == "nearest_to_centre"
    ]
    return {row["anonymous_sample_id"]: row for row in rows}


def build_system_prompt() -> str:
    labels = "\n".join(f"- {label}" for label in ALLOWED_LABELS)
    return (
        "You are a bioacoustic species-classification model. A detector proposal "
        "has already been selected as the target candidate. Your task is only to "
        "classify the species of that selected proposal.\n\n"
        "Do not change bbox coordinates. Do not refine, redraw, add, or remove "
        "detections. Do not output additional detections. Classify only the "
        "selected proposal. Ignore other calls, noise, axes, gridlines, and "
        "padding artefacts.\n\n"
        "Choose exactly one species label from this allowed list:\n"
        f"{labels}\n\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "selected_proposal_id": "bd2_001",\n'
        '  "predicted_species": "one allowed label",\n'
        '  "confidence": 0.0,\n'
        '  "reasoning_brief": "short visual justification",\n'
        '  "visual_evidence": ["brief visible cue"]\n'
        "}"
    )


def build_user_message(row: dict[str, str], proposal: dict[str, str]) -> str:
    proposal_payload = {
        "proposal_id": proposal["proposal_id"],
        "coordinate_frame": "local_window_seconds_0.000_to_0.300",
        "start_time": float(proposal["start_time"]),
        "end_time": float(proposal["end_time"]),
        "low_freq": float(proposal["low_freq"]),
        "high_freq": float(proposal["high_freq"]),
        "det_prob": float(proposal["det_prob"]),
    }
    payload = {
        "anonymous_sample_id": row["anonymous_sample_id"],
        "image_variant": "centred_crop_no_box",
        "selected_detector_proposal": proposal_payload,
        "instructions": [
            "Classify only this selected detector proposal.",
            "Do not alter start_time, end_time, low_freq, or high_freq.",
            "Do not output additional detections.",
            "Choose exactly one allowed species label.",
            "Return valid JSON only.",
        ],
    }
    return "/no_think\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def parse_selected_classification(raw_text: str) -> dict[str, Any]:
    parsed = parse_classification(raw_text)
    proposal_id = parsed.get("selected_proposal_id")
    if proposal_id is not None and not isinstance(proposal_id, str):
        raise ValueError("selected_proposal_id must be a string when provided")
    return parsed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected-proposals", type=Path, default=DEFAULT_SELECTED_PROPOSALS)
    parser.add_argument("--clip-scope", default="pilot80", choices=("pilot80", "full240"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=800)
    parser.add_argument("--per-species", type=int, default=10)
    args = parser.parse_args()

    if args.output_dir.exists() and (args.output_dir / "parsed_predictions.csv").exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    subset_rows = select_balanced_subset(load_manifest(args.manifest), args.per_species)
    selected_by_anon = selected_proposal_index(args.selected_proposals, args.clip_scope)
    raw_dir = args.output_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = build_system_prompt()
    prediction_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for index, row in enumerate(subset_rows, start=1):
        anon_id = row["anonymous_sample_id"]
        print(f"[{index}/{len(subset_rows)}] {anon_id}", flush=True)
        proposal = selected_by_anon.get(anon_id)
        raw_path = raw_dir / f"{anon_id}_raw_response.txt"
        base_row: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "anonymous_sample_id": anon_id,
            "true_species": row["species"],
            "selection_rule": "nearest_to_centre",
            "clip_scope": args.clip_scope,
            "selected_proposal_available": str(proposal is not None).lower(),
            "raw_response_path": raw_path.as_posix(),
        }
        if proposal is None:
            output = {
                **base_row,
                "parse_status": "no_selected_proposal",
                "selected_proposal_id": "",
                "predicted_species": "",
                "confidence": "",
                "reasoning_brief": "",
                "visual_evidence_json": "[]",
                "parse_error": "nearest_to_centre selected no proposal for this sample",
            }
            prediction_rows.append(output)
            failure_rows.append(output)
            raw_path.write_text("", encoding="utf-8")
            continue
        try:
            raw_text = call_ollama_generate(
                image_path=resolve_repo_path(row["centred_crop_image_path"]),
                system_prompt=system_prompt,
                user_message=build_user_message(row, proposal),
                model_name=args.model_name,
                timeout=args.timeout,
                num_predict=args.num_predict,
            )
            raw_path.write_text(raw_text, encoding="utf-8")
            parsed = parse_selected_classification(raw_text)
            parse_status = "success"
            parse_error = ""
        except Exception as exc:
            if not raw_path.exists():
                raw_path.write_text("", encoding="utf-8")
            parsed = {}
            parse_status = "failed"
            parse_error = f"{type(exc).__name__}: {exc}"
        output = {
            **base_row,
            "parse_status": parse_status,
            "selected_proposal_id": proposal["proposal_id"],
            "selected_start_time": proposal["start_time"],
            "selected_end_time": proposal["end_time"],
            "selected_low_freq": proposal["low_freq"],
            "selected_high_freq": proposal["high_freq"],
            "selected_det_prob": proposal["det_prob"],
            "predicted_species": parsed.get("predicted_species", ""),
            "confidence": parsed.get("confidence", ""),
            "reasoning_brief": parsed.get("reasoning_brief", ""),
            "visual_evidence_json": json.dumps(parsed.get("visual_evidence", []), ensure_ascii=False),
            "parse_error": parse_error,
        }
        prediction_rows.append(output)
        if parse_status != "success":
            failure_rows.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "parsed_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "parse_failures.csv", failure_rows)
    write_csv(args.output_dir / "pilot80_subset_manifest.csv", subset_rows)
    successes = sum(row["parse_status"] == "success" for row in prediction_rows)
    print(
        f"Condition={args.output_dir.name} samples={len(prediction_rows)} "
        f"parse_success={successes} parse_failure={len(prediction_rows) - successes}"
    )


if __name__ == "__main__":
    main()
