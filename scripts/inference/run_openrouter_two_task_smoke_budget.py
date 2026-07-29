"""Run two one-sample OpenRouter smoke tests for budget estimation.

The script reads OPENROUTER_API_KEY from the environment or local .env file.
It never prints, logs, or writes the API key. Raw saved responses are only the
OpenRouter API response payloads.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_prompt_v2_small_pilot import (  # noqa: E402
    extract_json_text,
    load_prompt,
    parse_prediction,
    read_clip_duration,
)
from scripts.inference.run_full45_localisation_condition import (  # noqa: E402
    build_condition_user_message,
    load_proposal_context,
)
from scripts.inference.run_stage1a_multispecies_classification import (  # noqa: E402
    ALLOWED_LABELS,
    load_manifest,
    resolve_repo_path,
)
from scripts.inference.run_stage2c_selected_proposal_classification import (  # noqa: E402
    build_system_prompt as build_stage2c_system_prompt,
    build_user_message as build_stage2c_user_message,
    parse_selected_classification,
    selected_proposal_index,
)


MODEL_NAME = "openai/gpt-5.6-sol"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
INPUT_PRICE_PER_1M = 5.0
OUTPUT_PRICE_PER_1M = 30.0
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/openrouter_smoke_tests/gpt_5_6_sol_two_task_budget"
)
LOCALISATION_CLIP_ID = "OP_016"
CLASSIFICATION_ANON_ID = "sample_000001"


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment or .env")
    return api_key


def extract_message_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response choice has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    raise ValueError("OpenRouter response message content is not text")


def call_openrouter(
    *,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_message: str,
    image_path: Path,
    timeout_seconds: float,
    max_tokens: int,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            },
        ],
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.toy-audio-agent",
            "X-Title": "toy-audio-agent OpenRouter smoke budget",
            "X-OpenRouter-Metadata": "enabled",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    latency = time.perf_counter() - started
    return response_payload, latency


def cost_from_usage(usage: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    input_cost = prompt_tokens / 1_000_000 * INPUT_PRICE_PER_1M
    output_cost = completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M
    estimated_cost = input_cost + output_cost
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": estimated_cost,
        "pricing_assumption": {
            "input_usd_per_1m_tokens": INPUT_PRICE_PER_1M,
            "output_usd_per_1m_tokens": OUTPUT_PRICE_PER_1M,
        },
    }


def build_localisation_task() -> dict[str, Any]:
    clip_id = LOCALISATION_CLIP_ID
    image_path = REPO_ROOT / "outputs/agent_inputs/prompt_v2_full_grid_v2" / f"{clip_id}_spectrogram.png"
    proposal_path = (
        REPO_ROOT
        / "outputs/tool_outputs/batdetect2_proposals/full45"
        / f"{clip_id}_batdetect2_proposals.json"
    )
    prompt_path = REPO_ROOT / "prompts/prompt_v2_bat_strong_label.md"
    eval_dir = REPO_ROOT / "outputs/evaluation_sets/ozimops_petersi_v1"
    proposals = load_proposal_context(
        REPO_ROOT / "outputs/tool_outputs/batdetect2_proposals/full45", clip_id
    )
    duration = read_clip_duration(eval_dir, clip_id)
    return {
        "task_name": "localisation",
        "sample_id": clip_id,
        "image_path": image_path,
        "proposal_metadata_path": proposal_path,
        "prompt_template": prompt_path,
        "system_prompt": load_prompt(prompt_path),
        "user_message": build_condition_user_message(
            condition="proposal_constrained",
            clip_id=clip_id,
            clip_duration_seconds=duration,
            condition_context=proposals,
        ),
        "clip_duration_seconds": duration,
        "max_tokens": 3500,
    }


def first_wrong_stage2c_sample() -> tuple[dict[str, str], dict[str, str]]:
    manifest_rows = load_manifest(
        REPO_ROOT
        / "outputs/analysis_reports/multispecies_stage1_gt_event_classification_dataset/stage1_manifest.csv"
    )
    manifest_by_anon = {row["anonymous_sample_id"]: row for row in manifest_rows}
    selected_by_anon = selected_proposal_index(
        REPO_ROOT
        / "outputs/analysis_reports/multispecies_classification/"
        "stage2_central_proposal_selection_baseline/selected_proposals.csv",
        "full240",
    )
    previous_path = (
        REPO_ROOT
        / "outputs/agent_runs/multispecies_classification/"
        "qwen3_6_stage2c_nearest_centre_proposal_classification_full240/parsed_predictions.csv"
    )
    with previous_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["parse_status"] != "success":
            continue
        if row["selected_proposal_available"] != "true":
            continue
        if row["predicted_species"] != row["true_species"]:
            anon_id = row["anonymous_sample_id"]
            return manifest_by_anon[anon_id], selected_by_anon[anon_id]
    raise RuntimeError("No deterministic wrong Stage 2C sample found")


def build_classification_task() -> dict[str, Any]:
    row, proposal = first_wrong_stage2c_sample()
    image_path = resolve_repo_path(row["centred_crop_image_path"])
    return {
        "task_name": "classification",
        "sample_id": row["anonymous_sample_id"],
        "source_sample_id": row["sample_id"],
        "true_species": row["species"],
        "image_path": image_path,
        "selected_proposal_id": proposal["proposal_id"],
        "system_prompt": build_stage2c_system_prompt(),
        "user_message": build_stage2c_user_message(row, proposal),
        "max_tokens": 900,
    }


def validate_localisation(response_text: str, expected_clip_id: str) -> dict[str, Any]:
    prediction = parse_prediction(response_text, expected_clip_id=expected_clip_id)
    valid_events = []
    for event in prediction["events"]:
        valid = (
            float(event["start_time_seconds"]) < float(event["end_time_seconds"])
            and float(event["low_frequency_hz"]) < float(event["high_frequency_hz"])
        )
        valid_events.append(valid)
    return {
        "parse_status": "success",
        "parsed_output": prediction,
        "structurally_valid": all(valid_events),
        "event_count": len(prediction["events"]),
    }


def validate_classification(response_text: str, true_species: str) -> dict[str, Any]:
    parsed = parse_selected_classification(response_text)
    predicted = parsed["predicted_species"]
    return {
        "parse_status": "success",
        "parsed_output": parsed,
        "structurally_valid": predicted in ALLOWED_LABELS,
        "predicted_species": predicted,
        "true_species": true_species,
        "species_correct": predicted == true_species,
    }


def run_task(
    *,
    api_key: str,
    task: dict[str, Any],
    output_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    response_payload, latency = call_openrouter(
        api_key=api_key,
        model_name=MODEL_NAME,
        system_prompt=task["system_prompt"],
        user_message=task["user_message"],
        image_path=task["image_path"],
        timeout_seconds=timeout_seconds,
        max_tokens=int(task["max_tokens"]),
    )
    task_name = task["task_name"]
    write_json(output_dir / f"{task_name}_raw_response.json", response_payload)
    response_text = extract_message_text(response_payload)
    if task_name == "localisation":
        parsed = validate_localisation(response_text, task["sample_id"])
    elif task_name == "classification":
        parsed = validate_classification(response_text, task["true_species"])
    else:
        raise ValueError(f"Unknown task: {task_name}")
    write_json(output_dir / f"{task_name}_parsed_output.json", parsed)
    usage = response_payload.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    usage_cost = cost_from_usage(usage)
    usage_cost.update(
        {
            "task_name": task_name,
            "model": MODEL_NAME,
            "sample_id": task["sample_id"],
            "image_path": task["image_path"].as_posix(),
            "latency_seconds": latency,
            "response_parse_status": parsed["parse_status"],
            "structurally_valid": parsed["structurally_valid"],
            "raw_usage": usage,
        }
    )
    if task_name == "localisation":
        usage_cost.update(
            {
                "proposal_metadata_path": task["proposal_metadata_path"].as_posix(),
                "prompt_template": task["prompt_template"].as_posix(),
                "parsed_detection_count": parsed["event_count"],
                "projected_45_samples_usd": usage_cost["estimated_total_cost_usd"] * 45,
                "projected_45_samples_1p5x_usd": usage_cost["estimated_total_cost_usd"] * 45 * 1.5,
            }
        )
    else:
        usage_cost.update(
            {
                "source_sample_id": task["source_sample_id"],
                "selected_proposal_id": task["selected_proposal_id"],
                "true_species": task["true_species"],
                "predicted_species": parsed["predicted_species"],
                "species_correct": parsed["species_correct"],
                "projected_240_samples_usd": usage_cost["estimated_total_cost_usd"] * 240,
                "projected_240_samples_1p5x_usd": usage_cost["estimated_total_cost_usd"] * 240 * 1.5,
            }
        )
    write_json(output_dir / f"{task_name}_usage_and_cost.json", usage_cost)
    return usage_cost


def write_report(output_dir: Path, loc: dict[str, Any], cls: dict[str, Any]) -> None:
    combined = loc["projected_45_samples_usd"] + cls["projected_240_samples_usd"]
    combined_buffer = (
        loc["projected_45_samples_1p5x_usd"] + cls["projected_240_samples_1p5x_usd"]
    )
    rows = [
        {
            "task": "single_agent_localisation",
            "sample_count": 45,
            "one_sample_cost_usd": loc["estimated_total_cost_usd"],
            "projected_cost_usd": loc["projected_45_samples_usd"],
            "projected_cost_1p5x_usd": loc["projected_45_samples_1p5x_usd"],
            "prompt_tokens": loc["prompt_tokens"],
            "completion_tokens": loc["completion_tokens"],
            "total_tokens": loc["total_tokens"],
        },
        {
            "task": "multispecies_classification",
            "sample_count": 240,
            "one_sample_cost_usd": cls["estimated_total_cost_usd"],
            "projected_cost_usd": cls["projected_240_samples_usd"],
            "projected_cost_1p5x_usd": cls["projected_240_samples_1p5x_usd"],
            "prompt_tokens": cls["prompt_tokens"],
            "completion_tokens": cls["completion_tokens"],
            "total_tokens": cls["total_tokens"],
        },
        {
            "task": "combined_full_comparison",
            "sample_count": 285,
            "one_sample_cost_usd": "",
            "projected_cost_usd": combined,
            "projected_cost_1p5x_usd": combined_buffer,
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
        },
    ]
    write_csv(output_dir / "projected_budget_summary.csv", rows)
    report = f"""# OpenRouter GPT-5.6 Sol Two-Task Smoke Budget

## Status

- OpenRouter worked: yes
- Model requested: `{MODEL_NAME}`
- Localisation response parsed: `{loc["response_parse_status"]}`
- Classification response parsed: `{cls["response_parse_status"]}`
- Pricing assumption: input `${INPUT_PRICE_PER_1M}` per 1M tokens, output `${OUTPUT_PRICE_PER_1M}` per 1M tokens

## Smoke Test A: Single-Agent Localisation

- Sample ID: `{loc["sample_id"]}`
- Image path: `{loc["image_path"]}`
- Proposal metadata path: `{loc["proposal_metadata_path"]}`
- Prompt/template used: `{loc["prompt_template"]}` with `proposal_constrained` runtime message
- Structurally valid localisation output: `{loc["structurally_valid"]}`
- Parsed detections: `{loc["parsed_detection_count"]}`
- Prompt tokens: `{loc["prompt_tokens"]}`
- Completion tokens: `{loc["completion_tokens"]}`
- Total tokens: `{loc["total_tokens"]}`
- Latency seconds: `{loc["latency_seconds"]:.3f}`
- Estimated one-sample cost: `${loc["estimated_total_cost_usd"]:.6f}`
- Projected full45 cost: `${loc["projected_45_samples_usd"]:.4f}`
- Projected full45 cost with 1.5x buffer: `${loc["projected_45_samples_1p5x_usd"]:.4f}`

## Smoke Test B: Multi-Species Selected-Proposal Classification

- Sample ID: `{cls["sample_id"]}`
- Source sample ID: `{cls["source_sample_id"]}`
- Image path: `{cls["image_path"]}`
- Selected proposal ID: `{cls["selected_proposal_id"]}`
- True species: `{cls["true_species"]}`
- Predicted species: `{cls["predicted_species"]}`
- Species correct: `{cls["species_correct"]}`
- Structurally valid classification output: `{cls["structurally_valid"]}`
- Prompt tokens: `{cls["prompt_tokens"]}`
- Completion tokens: `{cls["completion_tokens"]}`
- Total tokens: `{cls["total_tokens"]}`
- Latency seconds: `{cls["latency_seconds"]:.3f}`
- Estimated one-sample cost: `${cls["estimated_total_cost_usd"]:.6f}`
- Projected full240 cost: `${cls["projected_240_samples_usd"]:.4f}`
- Projected full240 cost with 1.5x buffer: `${cls["projected_240_samples_1p5x_usd"]:.4f}`

## Combined Projection

- Combined full comparison cost: `${combined:.4f}`
- Combined full comparison cost with 1.5x buffer: `${combined_buffer:.4f}`

## Recommendation for Santiago

Based on these two one-sample smoke tests, the projected direct API cost is low enough to justify a small frontier-model pilot before committing to full runs. A 5-sample pilot for each task is recommended before full45/full240, because token usage can vary by image and proposal count, and the one-sample estimate may understate worst-case costs for dense clips.
"""
    (output_dir / "smoke_test_budget_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key()
    localisation = run_task(
        api_key=api_key,
        task=build_localisation_task(),
        output_dir=args.output_dir,
        timeout_seconds=args.timeout,
    )
    classification = run_task(
        api_key=api_key,
        task=build_classification_task(),
        output_dir=args.output_dir,
        timeout_seconds=args.timeout,
    )
    write_report(args.output_dir, localisation, classification)
    print(
        "OpenRouter smoke budget complete: "
        f"localisation_sample={localisation['sample_id']} "
        f"classification_sample={classification['sample_id']} "
        f"localisation_tokens={localisation['total_tokens']} "
        f"classification_tokens={classification['total_tokens']}"
    )


if __name__ == "__main__":
    main()
