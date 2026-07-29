"""Run full OpenRouter GPT-5.6 Sol localisation and classification comparison.

This script reads OPENROUTER_API_KEY from the environment or local .env file.
It never prints, logs, or writes the API key. It does not read GT overlays,
human-review overlays, or modify frozen qwen outputs.
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
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_prompt_v2_small_pilot import (  # noqa: E402
    PROMPT_VERSION,
    load_prompt,
    parse_prediction,
    read_clip_duration,
    resolve_all_clip_ids,
)
from scripts.evaluation.evaluate_multi_protocol_detection import (  # noqa: E402
    FrozenRun,
    evaluate_runs as evaluate_detection_runs,
    write_csv as write_detection_csv,
)
from scripts.evaluation.evaluate_stage2c_selected_proposal_classification import (  # noqa: E402
    evaluate as evaluate_stage2c,
)
from scripts.inference.run_full45_localisation_condition import (  # noqa: E402
    build_condition_user_message,
    load_proposal_context,
)
from scripts.inference.run_openrouter_two_task_smoke_budget import (  # noqa: E402
    MODEL_NAME,
    OPENROUTER_URL,
    call_openrouter,
    cost_from_usage,
    extract_message_text,
    load_api_key,
    write_json,
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
    write_csv,
)


DEFAULT_EVAL_DIR = REPO_ROOT / "outputs/evaluation_sets/ozimops_petersi_v1"
DEFAULT_IMAGE_DIR = REPO_ROOT / "outputs/agent_inputs/prompt_v2_full_grid_v2"
DEFAULT_PROMPT = REPO_ROOT / "prompts/prompt_v2_bat_strong_label.md"
DEFAULT_PROPOSAL_DIR = REPO_ROOT / "outputs/tool_outputs/batdetect2_proposals/full45"
DEFAULT_STAGE1_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_stage1_gt_event_classification_dataset/stage1_manifest.csv"
)
DEFAULT_SELECTED_PROPOSALS = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_classification/"
    "stage2_central_proposal_selection_baseline/selected_proposals.csv"
)
DEFAULT_LOCALISATION_RUN_DIR = (
    REPO_ROOT
    / "outputs/agent_runs/openrouter_model_comparison/gpt_5_6_sol_uk_node_localisation_full45"
)
DEFAULT_LOCALISATION_ANALYSIS_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/openrouter_model_comparison/gpt_5_6_sol_uk_node_localisation_full45"
)
DEFAULT_CLASSIFICATION_RUN_DIR = (
    REPO_ROOT
    / "outputs/agent_runs/openrouter_model_comparison/"
    "gpt_5_6_sol_uk_node_stage2c_classification_full240"
)
DEFAULT_CLASSIFICATION_ANALYSIS_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/openrouter_model_comparison/"
    "gpt_5_6_sol_uk_node_stage2c_classification_full240"
)
DEFAULT_FINAL_REPORT_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/openrouter_model_comparison/gpt_5_6_sol_uk_node_final_comparison"
)
QWEN_STAGE2C_ANALYSIS = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_classification/"
    "qwen3_6_stage2c_nearest_centre_proposal_classification_full240"
)
QWEN_STAGE1C_SUMMARY = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_classification/"
    "qwen3_6_stage1c_gt_box_marker_species_guidance/metrics_summary.md"
)
SINGLE_AGENT_METRICS = (
    REPO_ROOT / "outputs/analysis_reports/single_agent_full45_summary/single_agent_full45_metrics.csv"
)
STAGE2C_V2_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_event_level_dataset_v2_centred/"
    "multispecies_event_dataset_manifest.csv"
)
MAX_PROJECTED_COST_USD = 15.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_jsonl_like_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def estimated_cost_from_response(response_payload: dict[str, Any]) -> dict[str, Any]:
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return cost_from_usage(usage) | {"raw_usage": usage}


def response_provider(response_payload: dict[str, Any]) -> str:
    provider = response_payload.get("provider")
    if isinstance(provider, str):
        return provider
    provider_info = response_payload.get("provider_info")
    if isinstance(provider_info, dict):
        name = provider_info.get("name") or provider_info.get("provider")
        if isinstance(name, str):
            return name
    return ""


def assert_expected_model(response_payload: dict[str, Any]) -> None:
    returned_model = response_payload.get("model")
    if returned_model != MODEL_NAME:
        raise RuntimeError(
            f"OpenRouter returned model {returned_model!r}; expected {MODEL_NAME!r}. "
            "Stopping to avoid silent model substitution."
        )


def call_openrouter_text_availability(
    *,
    api_key: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": MODEL_NAME,
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
            "X-Title": "toy-audio-agent OpenRouter full comparison availability",
            "X-OpenRouter-Metadata": "enabled",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    return payload, time.perf_counter() - started


def preflight_availability_check(api_key: str, output_dir: Path, timeout: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    response_payload, latency = call_openrouter_text_availability(
        api_key=api_key,
        timeout_seconds=timeout,
    )
    assert_expected_model(response_payload)
    write_json(output_dir / "availability_check_response.json", response_payload)
    usage = estimated_cost_from_response(response_payload)
    report = {
        "model_requested": MODEL_NAME,
        "returned_model": response_payload.get("model", ""),
        "provider": response_provider(response_payload),
        "availability_status": "passed",
        "latency_seconds": latency,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "estimated_cost_usd": usage["estimated_total_cost_usd"],
    }
    write_json(output_dir / "availability_check_summary.json", report)
    return report


def assert_fresh_or_resume(path: Path, expected_file: str) -> None:
    if path.exists() and (path / expected_file).exists():
        raise FileExistsError(f"Existing completed output found: {path / expected_file}")


def projected_combined_cost(
    localisation_costs: list[float],
    classification_costs: list[float],
    *,
    localisation_total: int,
    classification_total: int,
) -> float:
    loc_avg = mean(localisation_costs) if localisation_costs else 0.0
    cls_avg = mean(classification_costs) if classification_costs else 0.0
    return loc_avg * localisation_total + cls_avg * classification_total


def check_cost_guard(projected_cost: float) -> None:
    if projected_cost > MAX_PROJECTED_COST_USD:
        raise RuntimeError(
            f"Projected combined cost ${projected_cost:.4f} exceeds guardrail "
            f"${MAX_PROJECTED_COST_USD:.2f}; stopping before further calls."
        )


def is_fatal_openrouter_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        isinstance(exc, RuntimeError)
        and ("OpenRouter HTTP" in message or "silent model substitution" in message)
    )


def localisation_prediction_payload(
    *,
    prediction: dict[str, Any],
    image_path: Path,
    clip_duration: float,
) -> dict[str, Any]:
    return {
        "clip_id": prediction["clip_id"],
        "prompt_version": PROMPT_VERSION,
        "condition": "openrouter_gpt_5_6_sol_proposal_constrained",
        "model_name": MODEL_NAME,
        "backend": "openrouter_chat_completions",
        "input_image_path": image_path.as_posix(),
        "clip_duration_seconds": clip_duration,
        "events": prediction["events"],
    }


def failed_localisation_payload(
    *,
    clip_id: str,
    image_path: Path,
    clip_duration: float,
    error: str,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "prompt_version": PROMPT_VERSION,
        "condition": "openrouter_gpt_5_6_sol_proposal_constrained",
        "model_name": MODEL_NAME,
        "backend": "openrouter_chat_completions",
        "input_image_path": image_path.as_posix(),
        "clip_duration_seconds": clip_duration,
        "parse_status": "failed",
        "parse_error": error,
        "events": [],
    }


def run_localisation_full45(
    *,
    api_key: str,
    run_dir: Path,
    eval_dir: Path,
    image_dir: Path,
    prompt_path: Path,
    proposal_dir: Path,
    timeout: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    assert_fresh_or_resume(run_dir, "usage_and_cost.csv")
    predictions_dir = run_dir / "predictions"
    raw_dir = run_dir / "raw_responses"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = load_prompt(prompt_path)
    clip_ids = list(resolve_all_clip_ids(eval_dir))
    usage_rows: list[dict[str, Any]] = []
    cost_values: list[float] = []
    for index, clip_id in enumerate(clip_ids, start=1):
        print(f"[localisation {index}/{len(clip_ids)}] {clip_id}", flush=True)
        image_path = image_dir / f"{clip_id}_spectrogram.png"
        clip_duration = read_clip_duration(eval_dir, clip_id)
        proposals = load_proposal_context(proposal_dir, clip_id)
        try:
            response_payload, latency = call_openrouter(
                api_key=api_key,
                model_name=MODEL_NAME,
                system_prompt=prompt_text,
                user_message=build_condition_user_message(
                    condition="proposal_constrained",
                    clip_id=clip_id,
                    clip_duration_seconds=clip_duration,
                    condition_context=proposals,
                ),
                image_path=image_path,
                timeout_seconds=timeout,
                max_tokens=max_tokens,
            )
            write_json(raw_dir / f"{clip_id}_raw_response.json", response_payload)
            assert_expected_model(response_payload)
            returned_model = str(response_payload.get("model", ""))
            provider = response_provider(response_payload)
            cost = estimated_cost_from_response(response_payload)
            response_text = extract_message_text(response_payload)
            prediction = parse_prediction(response_text, expected_clip_id=clip_id)
            parsed_payload = localisation_prediction_payload(
                prediction=prediction,
                image_path=image_path,
                clip_duration=clip_duration,
            )
            parse_status = "success"
            parse_error = ""
            event_count = len(prediction["events"])
        except Exception as exc:
            if is_fatal_openrouter_error(exc):
                raise
            cost = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_total_cost_usd": 0.0,
                "raw_usage": {},
            }
            latency = 0.0
            returned_model = ""
            provider = ""
            parse_status = "failed"
            parse_error = f"{type(exc).__name__}: {exc}"
            event_count = 0
            parsed_payload = failed_localisation_payload(
                clip_id=clip_id,
                image_path=image_path,
                clip_duration=clip_duration,
                error=parse_error,
            )
            (predictions_dir / f"{clip_id}_parse_error.txt").write_text(
                parse_error + "\n", encoding="utf-8"
            )
        write_json(predictions_dir / f"{clip_id}_predictions.json", parsed_payload)
        cost_values.append(float(cost["estimated_total_cost_usd"]))
        projected = projected_combined_cost(
            cost_values,
            [],
            localisation_total=len(clip_ids),
            classification_total=0,
        )
        check_cost_guard(projected)
        usage_rows.append(
            {
                "task": "localisation",
                "sample_id": clip_id,
                "image_path": image_path.as_posix(),
                "proposal_metadata_path": (proposal_dir / f"{clip_id}_batdetect2_proposals.json").as_posix(),
                "returned_model": returned_model,
                "provider": provider,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "event_count": event_count,
                "latency_seconds": latency,
                "prompt_tokens": cost["prompt_tokens"],
                "completion_tokens": cost["completion_tokens"],
                "total_tokens": cost["total_tokens"],
                "estimated_cost_usd": cost["estimated_total_cost_usd"],
                "raw_usage_json": json.dumps(cost.get("raw_usage", {}), ensure_ascii=False),
            }
        )
    write_jsonl_like_csv(run_dir / "usage_and_cost.csv", usage_rows)
    return usage_rows


def run_classification_full240(
    *,
    api_key: str,
    run_dir: Path,
    manifest_path: Path,
    selected_proposals_path: Path,
    timeout: float,
    max_tokens: int,
    existing_localisation_costs: list[float],
) -> list[dict[str, Any]]:
    assert_fresh_or_resume(run_dir, "usage_and_cost.csv")
    raw_dir = run_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest(manifest_path)
    selected_by_anon = selected_proposal_index(selected_proposals_path, "full240")
    system_prompt = build_stage2c_system_prompt()
    prediction_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    classification_costs: list[float] = []
    for index, row in enumerate(manifest_rows, start=1):
        anon_id = row["anonymous_sample_id"]
        print(f"[classification {index}/{len(manifest_rows)}] {anon_id}", flush=True)
        proposal = selected_by_anon.get(anon_id)
        image_path = resolve_repo_path(row["centred_crop_image_path"])
        raw_path = raw_dir / f"{anon_id}_raw_response.json"
        base_row: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "anonymous_sample_id": anon_id,
            "true_species": row["species"],
            "selection_rule": "nearest_to_centre",
            "clip_scope": "full240",
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
            usage_rows.append(
                {
                    "task": "classification",
                    "sample_id": anon_id,
                    "source_sample_id": row["sample_id"],
                    "image_path": image_path.as_posix(),
                    "selected_proposal_id": "",
                    "parse_status": "no_selected_proposal",
                    "parse_error": output["parse_error"],
                    "returned_model": "",
                    "provider": "",
                    "latency_seconds": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "raw_usage_json": "{}",
                }
            )
            continue
        try:
            response_payload, latency = call_openrouter(
                api_key=api_key,
                model_name=MODEL_NAME,
                system_prompt=system_prompt,
                user_message=build_stage2c_user_message(row, proposal),
                image_path=image_path,
                timeout_seconds=timeout,
                max_tokens=max_tokens,
            )
            write_json(raw_path, response_payload)
            assert_expected_model(response_payload)
            returned_model = str(response_payload.get("model", ""))
            provider = response_provider(response_payload)
            cost = estimated_cost_from_response(response_payload)
            response_text = extract_message_text(response_payload)
            parsed = parse_selected_classification(response_text)
            parse_status = "success"
            parse_error = ""
        except Exception as exc:
            if is_fatal_openrouter_error(exc):
                raise
            cost = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_total_cost_usd": 0.0,
                "raw_usage": {},
            }
            latency = 0.0
            returned_model = ""
            provider = ""
            parsed = {}
            parse_status = "failed"
            parse_error = f"{type(exc).__name__}: {exc}"
        classification_costs.append(float(cost["estimated_total_cost_usd"]))
        projected = projected_combined_cost(
            existing_localisation_costs,
            classification_costs,
            localisation_total=45,
            classification_total=240,
        )
        check_cost_guard(projected)
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
        usage_rows.append(
            {
                "task": "classification",
                "sample_id": anon_id,
                "source_sample_id": row["sample_id"],
                "image_path": image_path.as_posix(),
                "selected_proposal_id": proposal["proposal_id"],
                "true_species": row["species"],
                "predicted_species": parsed.get("predicted_species", ""),
                "species_correct": str(parsed.get("predicted_species", "") == row["species"]).lower(),
                "returned_model": returned_model,
                "provider": provider,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "latency_seconds": latency,
                "prompt_tokens": cost["prompt_tokens"],
                "completion_tokens": cost["completion_tokens"],
                "total_tokens": cost["total_tokens"],
                "estimated_cost_usd": cost["estimated_total_cost_usd"],
                "raw_usage_json": json.dumps(cost.get("raw_usage", {}), ensure_ascii=False),
            }
        )
    write_csv(run_dir / "parsed_predictions.csv", prediction_rows)
    write_csv(run_dir / "parse_failures.csv", failure_rows)
    write_csv(run_dir / "pilot80_subset_manifest.csv", manifest_rows)
    write_jsonl_like_csv(run_dir / "usage_and_cost.csv", usage_rows)
    return usage_rows


def evaluate_localisation(run_dir: Path, analysis_dir: Path, eval_dir: Path) -> dict[str, Any]:
    clip_ids = tuple(resolve_all_clip_ids(eval_dir))
    frozen_run = FrozenRun(
        experiment_id="openrouter_gpt_5_6_sol_localisation_full45",
        experiment_group="full_45",
        clip_scope="full_45",
        method="proposal_constrained_vlm",
        model=MODEL_NAME,
        prediction_dir=run_dir / "predictions",
        clip_ids=clip_ids,
        notes="OpenRouter GPT-5.6 Sol proposal-constrained full45 localisation.",
    )
    experiment_rows, case_rows, pair_rows = evaluate_detection_runs([frozen_run], eval_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_detection_csv(analysis_dir / "protocol_metrics.csv", experiment_rows)
    write_detection_csv(analysis_dir / "case_level_results.csv", case_rows)
    write_detection_csv(analysis_dir / "matched_pair_errors.csv", pair_rows)
    primary = next(row for row in experiment_rows if row["protocol"] == "temporal_iou_0.3")
    summary = {
        "condition": frozen_run.experiment_id,
        "model": MODEL_NAME,
        "primary_protocol": "temporal_iou_0.3",
        "primary": primary,
        "all_protocols": experiment_rows,
    }
    write_json(analysis_dir / "aggregate_summary.json", summary)
    return summary


def evaluate_classification(run_dir: Path, analysis_dir: Path) -> dict[str, Any]:
    return evaluate_stage2c(
        run_dir=run_dir,
        output_dir=analysis_dir,
        stage1_manifest=DEFAULT_STAGE1_MANIFEST,
        v2_manifest=STAGE2C_V2_MANIFEST,
    )


def sum_usage(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    return {
        "task": task,
        "sample_rows": len(rows),
        "api_call_count": sum(1 for row in rows if int(row.get("total_tokens") or 0) > 0),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "estimated_cost_usd": sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows),
        "mean_latency_seconds": mean(
            [float(row.get("latency_seconds") or 0.0) for row in rows if float(row.get("latency_seconds") or 0.0) > 0]
        )
        if any(float(row.get("latency_seconds") or 0.0) > 0 for row in rows)
        else 0.0,
    }


def protocol_value(rows: list[dict[str, str]], condition: str, protocol: str, field: str) -> str:
    for row in rows:
        if row.get("experiment_id") == condition and row.get("protocol") == protocol:
            return row.get(field, "")
    return ""


def write_final_report(
    *,
    report_dir: Path,
    localisation_summary: dict[str, Any],
    classification_summary: dict[str, Any],
    localisation_usage: list[dict[str, Any]],
    classification_usage: list[dict[str, Any]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    loc_usage = sum_usage(localisation_usage, "full45_localisation")
    cls_usage = sum_usage(classification_usage, "full240_classification")
    total_usage = {
        "task": "combined",
        "sample_rows": loc_usage["sample_rows"] + cls_usage["sample_rows"],
        "api_call_count": loc_usage["api_call_count"] + cls_usage["api_call_count"],
        "prompt_tokens": loc_usage["prompt_tokens"] + cls_usage["prompt_tokens"],
        "completion_tokens": loc_usage["completion_tokens"] + cls_usage["completion_tokens"],
        "total_tokens": loc_usage["total_tokens"] + cls_usage["total_tokens"],
        "estimated_cost_usd": loc_usage["estimated_cost_usd"] + cls_usage["estimated_cost_usd"],
        "mean_latency_seconds": "",
    }
    write_csv(report_dir / "token_cost_summary.csv", [loc_usage, cls_usage, total_usage])

    single_rows = read_csv(SINGLE_AGENT_METRICS)
    loc_primary = localisation_summary["primary"]
    loc_protocols = {row["protocol"]: row for row in localisation_summary["all_protocols"]}
    localisation_rows = [
        {
            "condition": "qwen3.6 grid_v2 baseline",
            "protocol": "temporal_iou_0.3",
            "TP": protocol_value(single_rows, "single_agent_full45_grid_v2_baseline", "temporal_iou_0.3", "TP"),
            "FP": protocol_value(single_rows, "single_agent_full45_grid_v2_baseline", "temporal_iou_0.3", "FP"),
            "FN": protocol_value(single_rows, "single_agent_full45_grid_v2_baseline", "temporal_iou_0.3", "FN"),
            "precision": protocol_value(single_rows, "single_agent_full45_grid_v2_baseline", "temporal_iou_0.3", "precision"),
            "recall": protocol_value(single_rows, "single_agent_full45_grid_v2_baseline", "temporal_iou_0.3", "recall"),
            "F1": protocol_value(single_rows, "single_agent_full45_grid_v2_baseline", "temporal_iou_0.3", "F1"),
            "F1_iou_0p1": protocol_value(single_rows, "single_agent_full45_grid_v2_baseline", "temporal_iou_0.1", "F1"),
            "F1_10ms": protocol_value(single_rows, "single_agent_full45_grid_v2_baseline", "start_time_proximity_10ms", "F1"),
        },
        {
            "condition": "BatDetect2 proposal-only",
            "protocol": "temporal_iou_0.3",
            "TP": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_only", "temporal_iou_0.3", "TP"),
            "FP": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_only", "temporal_iou_0.3", "FP"),
            "FN": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_only", "temporal_iou_0.3", "FN"),
            "precision": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_only", "temporal_iou_0.3", "precision"),
            "recall": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_only", "temporal_iou_0.3", "recall"),
            "F1": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_only", "temporal_iou_0.3", "F1"),
            "F1_iou_0p1": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_only", "temporal_iou_0.1", "F1"),
            "F1_10ms": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_only", "start_time_proximity_10ms", "F1"),
        },
        {
            "condition": "qwen3.6 proposal-constrained VLM",
            "protocol": "temporal_iou_0.3",
            "TP": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_constrained_vlm", "temporal_iou_0.3", "TP"),
            "FP": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_constrained_vlm", "temporal_iou_0.3", "FP"),
            "FN": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_constrained_vlm", "temporal_iou_0.3", "FN"),
            "precision": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_constrained_vlm", "temporal_iou_0.3", "precision"),
            "recall": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_constrained_vlm", "temporal_iou_0.3", "recall"),
            "F1": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_constrained_vlm", "temporal_iou_0.3", "F1"),
            "F1_iou_0p1": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_constrained_vlm", "temporal_iou_0.1", "F1"),
            "F1_10ms": protocol_value(single_rows, "single_agent_full45_batdetect2_proposal_constrained_vlm", "start_time_proximity_10ms", "F1"),
        },
        {
            "condition": "P14 conservative variant",
            "protocol": "temporal_iou_0.3",
            "TP": protocol_value(single_rows, "single_agent_full45_best_stack_qwen3_6", "temporal_iou_0.3", "TP"),
            "FP": protocol_value(single_rows, "single_agent_full45_best_stack_qwen3_6", "temporal_iou_0.3", "FP"),
            "FN": protocol_value(single_rows, "single_agent_full45_best_stack_qwen3_6", "temporal_iou_0.3", "FN"),
            "precision": protocol_value(single_rows, "single_agent_full45_best_stack_qwen3_6", "temporal_iou_0.3", "precision"),
            "recall": protocol_value(single_rows, "single_agent_full45_best_stack_qwen3_6", "temporal_iou_0.3", "recall"),
            "F1": protocol_value(single_rows, "single_agent_full45_best_stack_qwen3_6", "temporal_iou_0.3", "F1"),
            "F1_iou_0p1": protocol_value(single_rows, "single_agent_full45_best_stack_qwen3_6", "temporal_iou_0.1", "F1"),
            "F1_10ms": protocol_value(single_rows, "single_agent_full45_best_stack_qwen3_6", "start_time_proximity_10ms", "F1"),
        },
        {
            "condition": "OpenRouter gpt-5.6-sol proposal-constrained VLM",
            "protocol": "temporal_iou_0.3",
            "TP": loc_primary["TP"],
            "FP": loc_primary["FP"],
            "FN": loc_primary["FN"],
            "precision": loc_primary["precision"],
            "recall": loc_primary["recall"],
            "F1": loc_primary["F1"],
            "F1_iou_0p1": loc_protocols["temporal_iou_0.1"]["F1"],
            "F1_10ms": loc_protocols["start_time_proximity_10ms"]["F1"],
        },
    ]
    write_csv(report_dir / "localisation_comparison_table.csv", localisation_rows)

    qwen_stage2c = json.loads((QWEN_STAGE2C_ANALYSIS / "aggregate_summary.json").read_text(encoding="utf-8"))
    classification_rows = [
        {
            "condition": "qwen3.6 Stage 1C GT-box species-guided",
            "scope": "full240",
            "accuracy": "0.192",
            "macro_F1": "0.105",
            "balanced_accuracy": "0.192",
            "matched_species_accuracy": "",
        },
        {
            "condition": "qwen3.6 Stage 2C selected-proposal classification",
            "scope": "full240",
            "accuracy": "",
            "macro_F1": qwen_stage2c["classification_on_matched"]["macro_F1"],
            "balanced_accuracy": qwen_stage2c["classification_on_matched"]["balanced_accuracy"],
            "matched_species_accuracy": qwen_stage2c["classification_on_matched"]["species_accuracy"],
        },
        {
            "condition": "OpenRouter gpt-5.6-sol Stage 2C selected-proposal classification",
            "scope": "full240",
            "accuracy": "",
            "macro_F1": classification_summary["classification_on_matched"]["macro_F1"],
            "balanced_accuracy": classification_summary["classification_on_matched"]["balanced_accuracy"],
            "matched_species_accuracy": classification_summary["classification_on_matched"]["species_accuracy"],
        },
    ]
    write_csv(report_dir / "classification_comparison_table.csv", classification_rows)
    joint_rows = [
        {
            "condition": "qwen3.6 Stage 2C selected-proposal classification",
            "scope": "full240",
            "joint_correct": qwen_stage2c["joint"]["joint_correct"],
            "joint_precision": qwen_stage2c["joint"]["precision"],
            "joint_recall": qwen_stage2c["joint"]["recall"],
            "joint_F1": qwen_stage2c["joint"]["F1"],
        },
        {
            "condition": "OpenRouter gpt-5.6-sol Stage 2C selected-proposal classification",
            "scope": "full240",
            "joint_correct": classification_summary["joint"]["joint_correct"],
            "joint_precision": classification_summary["joint"]["precision"],
            "joint_recall": classification_summary["joint"]["recall"],
            "joint_F1": classification_summary["joint"]["F1"],
        },
    ]
    write_csv(report_dir / "joint_task_comparison_table.csv", joint_rows)
    report = f"""# GPT-5.6 Sol OpenRouter Model Comparison

## Scope

This comparison ran `openai/gpt-5.6-sol` through OpenRouter on two frozen workflows:

1. Full45 `Ozimops petersi` proposal-constrained single-agent localisation.
2. Full240 Stage 2C selected-proposal multi-species classification.

No GT files, images, prompts, qwen predictions, GT overlays, or human-review overlays were modified or used as model input.

## Localisation Result

OpenRouter GPT-5.6 Sol achieved temporal IoU>=0.3 F1 `{float(loc_primary['F1']):.3f}` on full45, with TP/FP/FN `{loc_primary['TP']}/{loc_primary['FP']}/{loc_primary['FN']}`. The corresponding qwen3.6 proposal-constrained VLM result was F1 `0.785`, and BatDetect2 proposal-only was F1 `0.716`.

Under the onset-sensitive 10 ms protocol, GPT-5.6 Sol achieved F1 `{float(loc_protocols['start_time_proximity_10ms']['F1']):.3f}`. The P14 conservative qwen3.6 variant remains useful as secondary onset-preserving context.

## Multi-Species Classification Result

OpenRouter GPT-5.6 Sol Stage 2C full240 matched-proposal species accuracy was `{classification_summary['classification_on_matched']['species_accuracy']:.3f}`, macro-F1 was `{classification_summary['classification_on_matched']['macro_F1']:.3f}`, and balanced accuracy was `{classification_summary['classification_on_matched']['balanced_accuracy']:.3f}`.

Joint F1 was `{classification_summary['joint']['F1']:.3f}`, compared with qwen3.6 Stage 2C full240 joint F1 `{qwen_stage2c['joint']['F1']:.3f}`.

## Token and Cost Summary

- Localisation API calls: `{loc_usage['api_call_count']}`
- Classification API calls: `{cls_usage['api_call_count']}`
- Total prompt tokens: `{total_usage['prompt_tokens']}`
- Total completion tokens: `{total_usage['completion_tokens']}`
- Total tokens: `{total_usage['total_tokens']}`
- Estimated total cost: `${total_usage['estimated_cost_usd']:.4f}`

## Answers

1. **Does GPT-5.6 Sol improve localisation over qwen3.6?** See `localisation_comparison_table.csv`; the primary comparison is against qwen3.6 proposal-constrained VLM.
2. **Does it improve species classification over qwen3.6?** See `classification_comparison_table.csv` and `joint_task_comparison_table.csv`.
3. **Does it reduce label collapse into Pipistrellus/Plecotus?** Inspect the generated classification confusion matrix in the classification analysis directory.
4. **Does it improve Myotis or Ozimops?** Inspect per-species metrics and joint recall in the classification analysis directory.
5. **Is it worth reporting?** Yes: this is a clean stronger-model comparison using the same frozen workflows and gives a dissertation-facing check on whether earlier failures were model-capability limited.
6. **Total token usage and cost:** `{total_usage['total_tokens']}` tokens and `${total_usage['estimated_cost_usd']:.4f}` estimated cost under the requested pricing assumption.
"""
    (report_dir / "gpt_5_6_sol_uk_node_final_comparison_report.md").write_text(
        report, encoding="utf-8"
    )
    paragraphs = f"""# Dissertation Paragraphs

To test whether the observed limitations were specific to qwen3.6, the final workflows were re-run with `openai/gpt-5.6-sol` via OpenRouter. The stronger model used the same model-facing inputs and structured output schemas as the frozen qwen3.6 experiments: BatDetect2 proposal-constrained full45 localisation for `Ozimops petersi`, and Stage 2C nearest-centre selected-proposal classification for the full 240-sample multi-species dataset.

For localisation, GPT-5.6 Sol achieved temporal IoU>=0.3 F1 `{float(loc_primary['F1']):.3f}` on the full45 benchmark. This provides a direct frontier-model comparison against the qwen3.6 proposal-constrained result and the BatDetect2 proposal-only baseline.

For multi-species classification, GPT-5.6 Sol achieved matched-proposal species accuracy `{classification_summary['classification_on_matched']['species_accuracy']:.3f}` and joint F1 `{classification_summary['joint']['F1']:.3f}` on Stage 2C full240. These results show whether species-level acoustic classification improves when the same deterministic localisation support is paired with a stronger commercial model.
"""
    (report_dir / "dissertation_paragraphs.md").write_text(paragraphs, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localisation-run-dir", type=Path, default=DEFAULT_LOCALISATION_RUN_DIR)
    parser.add_argument("--localisation-analysis-dir", type=Path, default=DEFAULT_LOCALISATION_ANALYSIS_DIR)
    parser.add_argument("--classification-run-dir", type=Path, default=DEFAULT_CLASSIFICATION_RUN_DIR)
    parser.add_argument("--classification-analysis-dir", type=Path, default=DEFAULT_CLASSIFICATION_ANALYSIS_DIR)
    parser.add_argument("--final-report-dir", type=Path, default=DEFAULT_FINAL_REPORT_DIR)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--localisation-max-tokens", type=int, default=3500)
    parser.add_argument("--classification-max-tokens", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key()
    availability = preflight_availability_check(api_key, args.final_report_dir, args.timeout)
    print(
        "OpenRouter availability check passed: "
        f"model={availability['returned_model']} provider={availability['provider']} "
        f"latency_seconds={float(availability['latency_seconds']):.3f}",
        flush=True,
    )
    localisation_usage = run_localisation_full45(
        api_key=api_key,
        run_dir=args.localisation_run_dir,
        eval_dir=DEFAULT_EVAL_DIR,
        image_dir=DEFAULT_IMAGE_DIR,
        prompt_path=DEFAULT_PROMPT,
        proposal_dir=DEFAULT_PROPOSAL_DIR,
        timeout=args.timeout,
        max_tokens=args.localisation_max_tokens,
    )
    localisation_summary = evaluate_localisation(
        args.localisation_run_dir, args.localisation_analysis_dir, DEFAULT_EVAL_DIR
    )
    localisation_costs = [float(row.get("estimated_cost_usd") or 0.0) for row in localisation_usage]
    classification_usage = run_classification_full240(
        api_key=api_key,
        run_dir=args.classification_run_dir,
        manifest_path=DEFAULT_STAGE1_MANIFEST,
        selected_proposals_path=DEFAULT_SELECTED_PROPOSALS,
        timeout=args.timeout,
        max_tokens=args.classification_max_tokens,
        existing_localisation_costs=localisation_costs,
    )
    classification_summary = evaluate_classification(
        args.classification_run_dir, args.classification_analysis_dir
    )
    write_final_report(
        report_dir=args.final_report_dir,
        localisation_summary=localisation_summary,
        classification_summary=classification_summary,
        localisation_usage=localisation_usage,
        classification_usage=classification_usage,
    )
    total = sum_usage(localisation_usage, "loc")["estimated_cost_usd"] + sum_usage(
        classification_usage, "cls"
    )["estimated_cost_usd"]
    print(
        "OpenRouter full comparison complete: "
        f"localisation_calls={sum_usage(localisation_usage, 'loc')['api_call_count']} "
        f"classification_calls={sum_usage(classification_usage, 'cls')['api_call_count']} "
        f"estimated_cost_usd={total:.4f}"
    )


if __name__ == "__main__":
    main()
