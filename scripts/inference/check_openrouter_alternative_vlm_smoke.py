"""Check alternative OpenRouter frontier VLM availability and run two smoke tests.

The script reads OPENROUTER_API_KEY from the environment or local .env file.
It never prints, logs, or writes the API key. Candidate models are checked with
a tiny text-only request first. Only the first available model receives the two
one-sample image smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.run_openrouter_two_task_smoke_budget import (  # noqa: E402
    INPUT_PRICE_PER_1M,
    MODEL_NAME as PREVIOUS_MODEL_NAME,
    OPENROUTER_URL,
    OUTPUT_PRICE_PER_1M,
    build_classification_task,
    build_localisation_task,
    call_openrouter,
    cost_from_usage,
    extract_message_text,
    load_api_key,
    validate_classification,
    validate_localisation,
    write_json,
)


CANDIDATE_MODELS = ("openai/gpt-5.6-terra", "openai/gpt-5.6-luna")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/openrouter_smoke_tests/alternative_model_availability"
)


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


def call_openrouter_text_availability(
    *,
    api_key: str,
    model_name: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 20,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": 'Return exactly this JSON object: {"available": true}',
            }
        ],
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.toy-audio-agent",
            "X-Title": "toy-audio-agent OpenRouter alternative model check",
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
    return response_payload, time.perf_counter() - started


def safe_error_payload(model_name: str, error: Exception) -> dict[str, Any]:
    return {
        "model": model_name,
        "availability_check_passed": False,
        "error": str(error),
    }


def run_availability_checks(
    *,
    api_key: str,
    output_dir: Path,
    timeout: float,
) -> tuple[str | None, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for model_name in CANDIDATE_MODELS:
        try:
            response_payload, latency = call_openrouter_text_availability(
                api_key=api_key,
                model_name=model_name,
                timeout_seconds=timeout,
            )
            write_json(raw_dir / f"{model_name.replace('/', '__')}_availability_raw.json", response_payload)
            usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
            cost = cost_from_usage(usage)
            row = {
                "task": "availability_check",
                "model_name": model_name,
                "availability_check_passed": True,
                "parse_status": "not_applicable",
                "latency_seconds": latency,
                "prompt_tokens": cost["prompt_tokens"],
                "completion_tokens": cost["completion_tokens"],
                "total_tokens": cost["total_tokens"],
                "estimated_cost_usd": cost["estimated_total_cost_usd"],
                "classification_correct": "",
                "error": "",
            }
            rows.append(row)
            return model_name, rows
        except Exception as exc:
            write_json(
                raw_dir / f"{model_name.replace('/', '__')}_availability_error.json",
                safe_error_payload(model_name, exc),
            )
            rows.append(
                {
                    "task": "availability_check",
                    "model_name": model_name,
                    "availability_check_passed": False,
                    "parse_status": "failed",
                    "latency_seconds": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "classification_correct": "",
                    "error": str(exc),
                }
            )
    return None, rows


def run_smoke_task(
    *,
    api_key: str,
    model_name: str,
    task: dict[str, Any],
    output_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    task_name = task["task_name"]
    response_payload, latency = call_openrouter(
        api_key=api_key,
        model_name=model_name,
        system_prompt=task["system_prompt"],
        user_message=task["user_message"],
        image_path=task["image_path"],
        timeout_seconds=timeout,
        max_tokens=int(task["max_tokens"]),
    )
    write_json(raw_dir / f"{task_name}_raw_response.json", response_payload)
    response_text = extract_message_text(response_payload)
    if task_name == "localisation":
        parsed = validate_localisation(response_text, task["sample_id"])
        classification_correct: str | bool = ""
        parsed_short = {
            "parse_status": parsed["parse_status"],
            "structurally_valid": parsed["structurally_valid"],
            "event_count": parsed["event_count"],
            "parsed_output": parsed["parsed_output"],
        }
    elif task_name == "classification":
        parsed = validate_classification(response_text, task["true_species"])
        classification_correct = parsed["species_correct"]
        parsed_short = parsed
    else:
        raise ValueError(task_name)
    write_json(output_dir / f"{task_name}_parsed_output.json", parsed_short)
    usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
    cost = cost_from_usage(usage)
    return {
        "task": task_name,
        "model_name": model_name,
        "availability_check_passed": True,
        "sample_id": task["sample_id"],
        "parse_status": parsed["parse_status"],
        "structurally_valid": parsed["structurally_valid"],
        "latency_seconds": latency,
        "prompt_tokens": cost["prompt_tokens"],
        "completion_tokens": cost["completion_tokens"],
        "total_tokens": cost["total_tokens"],
        "estimated_cost_usd": cost["estimated_total_cost_usd"],
        "classification_correct": classification_correct,
        "error": "",
    }


def write_report(output_dir: Path, available_model: str | None, rows: list[dict[str, Any]]) -> None:
    availability_lines = []
    for row in rows:
        if row["task"] == "availability_check":
            status = "passed" if row["availability_check_passed"] else "failed"
            availability_lines.append(f"- `{row['model_name']}`: {status}")
    smoke_rows = [row for row in rows if row["task"] in {"localisation", "classification"}]
    smoke_lines = []
    for row in smoke_rows:
        smoke_lines.append(
            f"- `{row['task']}` sample `{row.get('sample_id', '')}`: parse `{row['parse_status']}`, "
            f"tokens `{row['total_tokens']}`, cost `${float(row['estimated_cost_usd']):.6f}`"
        )
    if not smoke_lines:
        smoke_lines.append("- No image smoke tests were run because no candidate model passed availability.")
    report = f"""# Alternative OpenRouter Model Availability

## Candidate Models

{chr(10).join(availability_lines)}

## Selected Available Model

`{available_model or 'none'}`

## Smoke Tests

{chr(10).join(smoke_lines)}

## Notes

- Previous unavailable model: `{PREVIOUS_MODEL_NAME}` returned HTTP 403 region unavailable in the full-run attempt.
- This check used at most one text-only availability request per candidate.
- Only the first available candidate received the two one-sample image smoke tests.
- API keys were loaded from the environment or `.env` and were not written to outputs.
"""
    (output_dir / "alternative_model_availability_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and (args.output_dir / "usage_and_cost_summary.csv").exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key()
    available_model, rows = run_availability_checks(
        api_key=api_key,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )
    if available_model:
        rows.append(
            run_smoke_task(
                api_key=api_key,
                model_name=available_model,
                task=build_localisation_task(),
                output_dir=args.output_dir,
                timeout=args.timeout,
            )
        )
        rows.append(
            run_smoke_task(
                api_key=api_key,
                model_name=available_model,
                task=build_classification_task(),
                output_dir=args.output_dir,
                timeout=args.timeout,
            )
        )
    write_csv(args.output_dir / "usage_and_cost_summary.csv", rows)
    write_report(args.output_dir, available_model, rows)
    print(f"Alternative OpenRouter availability complete: selected_model={available_model or 'none'}")


if __name__ == "__main__":
    main()
