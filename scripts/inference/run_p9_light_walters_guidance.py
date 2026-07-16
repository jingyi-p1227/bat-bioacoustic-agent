"""Run P9-light Walters-style generic acoustic-parameter guidance.

This runner uses clean grid_v2 spectrograms and BatDetect2 proposal metadata.
It does not read ground truth, diagnostic overlays, or prior model predictions
during inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_audio_agent.experiments.p9_light import (  # noqa: E402
    OPTIONAL_AGENT_CONDITION,
    REQUIRED_AGENT_CONDITIONS,
    TARGET_CLIPS,
    build_prompt,
    call_ollama_generate,
    check_output_schema_fields,
    clip_descriptors,
    default_paths,
    format_proposal_rows,
    image_path_for_clip,
    load_json,
    ollama_host,
    parse_prediction,
    preflight_check,
    prediction_to_payload,
    proposal_path_for_clip,
    read_clip_duration,
    require_ollama_model,
    retrieve_annotation_memory,
    retrieve_literature,
    sha256_file,
    walters_prompt_insert,
    write_json,
    write_proposal_only_predictions,
)


def condition_context(
    *,
    condition: str,
    descriptors: list[str],
    paths,
    clip_id: str,
    walters_card: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return prompt-safe context records and a retrieval trace."""

    literature_records: list[dict[str, Any]] = []
    literature_trace: list[dict[str, Any]] = []
    annotation_records: list[dict[str, Any]] = []
    annotation_trace: list[dict[str, Any]] = []
    if condition == "agent_methodological_literature":
        literature_records, literature_trace = retrieve_literature(
            paths.literature_store_path,
            descriptors=descriptors,
        )
    if condition == OPTIONAL_AGENT_CONDITION:
        annotation_records, annotation_trace = retrieve_annotation_memory(
            paths.annotation_memory_path,
            target_clip=clip_id,
            descriptors=descriptors,
        )
    trace = {
        "condition": condition,
        "clip_id": clip_id,
        "annotation_memory_enabled": condition == OPTIONAL_AGENT_CONDITION,
        "literature_enabled": condition == "agent_methodological_literature",
        "walters_guidance_enabled": condition in {"agent_walters_guidance", OPTIONAL_AGENT_CONDITION},
        "annotation_matches": annotation_trace,
        "literature_matches": literature_trace,
        "target_clip_excluded_from_annotation_memory": all(
            item.get("retrieved_id") != clip_id for item in annotation_trace
        ),
        "walters_guidance_card_status": walters_card.get("status"),
    }
    return literature_records, annotation_records, trace


def run_one_call(
    *,
    condition: str,
    clip_id: str,
    paths,
    endpoint: str,
    model_name: str,
    timeout: float,
    num_predict: int,
    retry: bool,
) -> dict[str, Any]:
    """Run one model call and persist all trace artifacts."""

    condition_dir = paths.run_dir / condition
    raw_dir = condition_dir / "raw_responses"
    prompt_dir = condition_dir / "prompts"
    pred_dir = condition_dir / "predictions"
    trace_dir = condition_dir / "traces"
    for directory in (raw_dir, prompt_dir, pred_dir, trace_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_path = image_path_for_clip(paths, clip_id)
    proposal_path = proposal_path_for_clip(paths, clip_id)
    proposal_payload = load_json(proposal_path)
    proposal_rows = format_proposal_rows(proposal_payload)
    clip_duration = read_clip_duration(paths, clip_id)
    descriptors = clip_descriptors(proposal_rows, clip_duration)
    walters_card = load_json(paths.walters_card_path)
    literature_records, annotation_records, retrieval_trace = condition_context(
        condition=condition,
        descriptors=descriptors,
        paths=paths,
        clip_id=clip_id,
        walters_card=walters_card,
    )
    prompt_walters_card = walters_card if condition in {"agent_walters_guidance", OPTIONAL_AGENT_CONDITION} else None
    system, user, condition_prompt_context = build_prompt(
        condition=condition,
        clip_id=clip_id,
        clip_duration=clip_duration,
        proposal_rows=proposal_rows,
        literature_records=literature_records,
        annotation_records=annotation_records,
        walters_card=prompt_walters_card,
    )

    prompt_payload = {
        "condition": condition,
        "clip_id": clip_id,
        "system_prompt": system,
        "user_prompt": user,
        "condition_prompt_context": condition_prompt_context,
    }
    prompt_path = prompt_dir / f"{clip_id}_prompt.json"
    raw_path = raw_dir / f"{clip_id}_raw_response.txt"
    pred_path = pred_dir / f"{clip_id}_predictions.json"
    error_path = pred_dir / f"{clip_id}_parse_error.txt"
    trace_path = trace_dir / f"{clip_id}_trace.json"
    write_json(prompt_path, prompt_payload)

    retry_status = "not_retried"
    response_payload: dict[str, Any] = {}
    latency: float | None = None
    raw_text = ""
    error = ""
    prediction = None

    for attempt in (1, 2):
        try:
            raw_text, response_payload, latency = call_ollama_generate(
                endpoint=endpoint,
                model_name=model_name,
                image_path=image_path,
                system_prompt=system,
                user_prompt=user,
                timeout=timeout,
                num_predict=num_predict,
            )
            raw_path.write_text(raw_text, encoding="utf-8")
            prediction = parse_prediction(raw_text, clip_id=clip_id, clip_duration=clip_duration)
            check_output_schema_fields({"events": [event.model_dump(mode="json") for event in prediction.events]})
            error_path.unlink(missing_ok=True)
            break
        except Exception as exc:  # noqa: BLE001 - preserve raw failures and continue
            error = f"{type(exc).__name__}: {exc}"
            if raw_text:
                raw_path.write_text(raw_text, encoding="utf-8")
            else:
                raw_path.write_text("", encoding="utf-8")
            error_path.write_text(error + "\n", encoding="utf-8")
            if attempt == 1 and retry:
                retry_status = "technical_retry_after_parse_failure"
                continue
            prediction = None
            break

    output = prediction_to_payload(
        prediction,
        clip_id=clip_id,
        condition=condition,
        model_name=model_name,
        endpoint=endpoint,
        image_path=image_path,
        proposal_path=proposal_path,
        clip_duration=clip_duration,
        parse_status="success" if prediction else "failed",
        error="" if prediction else error,
        latency_seconds=latency,
        retry_status=retry_status,
    )
    output["image_sha256"] = sha256_file(image_path)
    output["proposal_sha256"] = sha256_file(proposal_path)
    output["guidance_card_content"] = walters_prompt_insert(walters_card) if prompt_walters_card else ""
    output["retrieved_methodological_literature_records"] = literature_records
    output["retrieved_annotation_memory_records"] = annotation_records
    write_json(pred_path, output)

    trace = {
        "condition": condition,
        "clip_id": clip_id,
        "model_id": model_name,
        "endpoint": endpoint,
        "image_path": image_path.as_posix(),
        "image_sha256": output["image_sha256"],
        "proposal_path": proposal_path.as_posix(),
        "proposal_sha256": output["proposal_sha256"],
        "prompt_path": prompt_path.as_posix(),
        "raw_response_path": raw_path.as_posix(),
        "parsed_output_path": pred_path.as_posix(),
        "parse_status": output["parse_status"],
        "latency_seconds": latency,
        "retry_status": retry_status,
        "error": output["error"],
        "retrieval_trace": retrieval_trace,
        "ollama_response_metadata": {
            key: response_payload.get(key)
            for key in (
                "model",
                "created_at",
                "done",
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if key in response_payload
        },
    }
    write_json(trace_path, trace)
    return {
        "condition": condition,
        "clip_id": clip_id,
        "parse_status": output["parse_status"],
        "event_count": len(output["events"]),
        "retry_status": retry_status,
        "latency_seconds": latency,
        "prediction_path": pred_path.as_posix(),
        "raw_response_path": raw_path.as_posix(),
        "trace_path": trace_path.as_posix(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="qwen3.6:latest")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=5000)
    parser.add_argument("--include-optional-condition", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--no-retry", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_paths(REPO_ROOT)
    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    preflight = preflight_check(paths, include_optional=args.include_optional_condition)
    write_json(paths.analysis_dir / "p9_light_preflight.json", preflight)

    endpoint = (args.ollama_host or ollama_host()).rstrip("/")
    if not args.skip_model_check:
        available = require_ollama_model(endpoint, args.model_name)
        print(f"Confirmed {args.model_name} at {endpoint}. Available models: {', '.join(available)}")

    proposal_rows = write_proposal_only_predictions(paths)
    conditions = list(REQUIRED_AGENT_CONDITIONS)
    if args.include_optional_condition:
        conditions.append(OPTIONAL_AGENT_CONDITION)

    results: list[dict[str, Any]] = []
    for condition in conditions:
        for index, clip_id in enumerate(TARGET_CLIPS, start=1):
            print(f"[{condition} {index}/{len(TARGET_CLIPS)}] {clip_id}", flush=True)
            results.append(
                run_one_call(
                    condition=condition,
                    clip_id=clip_id,
                    paths=paths,
                    endpoint=endpoint,
                    model_name=args.model_name,
                    timeout=args.timeout,
                    num_predict=args.num_predict,
                    retry=not args.no_retry,
                )
            )
    write_json(
        paths.run_dir / "run_summary.json",
        {
            "model_name": args.model_name,
            "endpoint": endpoint,
            "target_clips": list(TARGET_CLIPS),
            "proposal_only_predictions": proposal_rows,
            "agent_results": results,
            "optional_condition_run": args.include_optional_condition,
        },
    )
    print("condition,clip_id,parse_status,event_count,retry_status")
    for row in results:
        print(f"{row['condition']},{row['clip_id']},{row['parse_status']},{row['event_count']},{row['retry_status']}")


if __name__ == "__main__":
    main()
