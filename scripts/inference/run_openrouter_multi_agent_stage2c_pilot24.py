"""Run GPT-5.6 Sol multi-agent Stage 2C classification pilot on fixed samples."""

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

from scripts.inference.run_openrouter_two_task_smoke_budget import (  # noqa: E402
    MODEL_NAME,
    OPENROUTER_URL,
    call_openrouter,
    cost_from_usage,
    extract_message_text,
    load_api_key,
    write_json,
)
from scripts.inference.run_qwen_stage2c_multi_agent_pilot24 import (  # noqa: E402
    AGENT1_SCHEMA,
    AGENT2_SCHEMA,
    AGENT3_SCHEMA,
    ALLOWED_LABELS,
    DEFAULT_SELECTED_PROPOSALS,
    GPT_STAGE2C_FULL240,
    QWEN_STAGE2C_FULL240,
    aggregate_metrics,
    build_agent1_system_prompt,
    build_agent2_system_prompt,
    build_agent3_prompt,
    build_image_user_message,
    comparison_prediction_index,
    comparison_rows,
    confusion_matrix,
    parse_agent1,
    parse_agent2,
    parse_agent3,
    per_species_metrics,
    read_csv,
    selected_proposal_index,
    subset_accuracy,
    write_csv,
    write_jsonl,
)
from scripts.inference.run_stage1a_multispecies_classification import resolve_repo_path  # noqa: E402


CONDITION_NAME = "gpt_5_6_sol_multi_agent_stage2c_pilot24_same_samples"
DEFAULT_QWEN_SAMPLE_LIST = (
    REPO_ROOT
    / "outputs/analysis_reports/multi_agent/qwen3_6_stage2c_pilot24/selected_pilot_samples.csv"
)
QWEN_MULTI_AGENT_PREDICTIONS = (
    REPO_ROOT
    / "outputs/analysis_reports/multi_agent/qwen3_6_stage2c_pilot24/multi_agent_pilot_predictions.csv"
)
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/agent_runs/openrouter_model_comparison" / CONDITION_NAME
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs/analysis_reports/openrouter_model_comparison" / CONDITION_NAME
MAX_PILOT_COST_USD = 5.0


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
            "Stopping to avoid silent substitution."
        )


def usage_row_from_response(
    *,
    task: str,
    sample_id: str,
    anonymous_sample_id: str,
    agent: str,
    response_payload: dict[str, Any],
    latency: float,
    parse_status: str,
    parse_error: str,
) -> dict[str, Any]:
    usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
    cost = cost_from_usage(usage)
    return {
        "task": task,
        "sample_id": sample_id,
        "anonymous_sample_id": anonymous_sample_id,
        "agent": agent,
        "returned_model": response_payload.get("model", ""),
        "provider": response_provider(response_payload),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "latency_seconds": latency,
        "prompt_tokens": cost["prompt_tokens"],
        "completion_tokens": cost["completion_tokens"],
        "total_tokens": cost["total_tokens"],
        "estimated_cost_usd": cost["estimated_total_cost_usd"],
        "raw_usage_json": json.dumps(usage, ensure_ascii=False),
    }


def call_openrouter_text(
    *,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_message: str,
    timeout_seconds: float,
    max_tokens: int,
    response_schema: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    # OpenRouter supports JSON-object mode; schema is included in the prompt to
    # keep parity with image calls where response_format is also json_object.
    payload["messages"][0]["content"] += "\n\nRequired output schema:\n" + json.dumps(
        response_schema, ensure_ascii=False
    )
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.toy-audio-agent",
            "X-Title": "toy-audio-agent OpenRouter multi-agent pilot",
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
    return response_payload, time.perf_counter() - started


def availability_check(api_key: str, output_dir: Path, timeout: float) -> dict[str, Any]:
    response_payload, latency = call_openrouter_text(
        api_key=api_key,
        model_name=MODEL_NAME,
        system_prompt="You are a JSON health-check endpoint.",
        user_message='Return exactly this JSON object: {"available": true}',
        timeout_seconds=timeout,
        max_tokens=20,
        response_schema={
            "type": "object",
            "properties": {"available": {"type": "boolean"}},
            "required": ["available"],
        },
    )
    assert_expected_model(response_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "availability_check_response.json", response_payload)
    usage = usage_row_from_response(
        task="availability_check",
        sample_id="availability_check",
        anonymous_sample_id="availability_check",
        agent="availability_check",
        response_payload=response_payload,
        latency=latency,
        parse_status="success",
        parse_error="",
    )
    summary = {
        "model_requested": MODEL_NAME,
        "returned_model": response_payload.get("model", ""),
        "provider": response_provider(response_payload),
        "availability_status": "passed",
        "latency_seconds": latency,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "estimated_cost_usd": usage["estimated_cost_usd"],
    }
    write_json(output_dir / "availability_check_summary.json", summary)
    return summary


def run_agent_image_call(
    *,
    api_key: str,
    row: dict[str, str],
    proposal: dict[str, str],
    agent: str,
    system_prompt: str,
    user_message: str,
    image_path: Path,
    schema: dict[str, Any],
    parser,
    raw_path: Path,
    timeout: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        response_payload, latency = call_openrouter(
            api_key=api_key,
            model_name=MODEL_NAME,
            system_prompt=system_prompt + "\n\nRequired output schema:\n" + json.dumps(schema),
            user_message=user_message,
            image_path=image_path,
            timeout_seconds=timeout,
            max_tokens=max_tokens,
        )
        write_json(raw_path, response_payload)
        assert_expected_model(response_payload)
        raw_text = extract_message_text(response_payload)
        parsed = parser(raw_text)
        status, error = "success", ""
    except Exception as exc:
        if isinstance(exc, RuntimeError) and (
            "OpenRouter HTTP" in str(exc) or "silent substitution" in str(exc)
        ):
            raise
        if not raw_path.exists():
            write_json(raw_path, {"error": f"{type(exc).__name__}: {exc}"})
        response_payload = {}
        latency = 0.0
        parsed = {}
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    record = {
        "agent": agent,
        "sample_id": row["sample_id"],
        "anonymous_sample_id": row["anonymous_sample_id"],
        "selected_proposal_id": proposal["proposal_id"],
        "parse_status": status,
        "parsed": parsed,
        "raw_response_path": raw_path.as_posix(),
        "parse_error": error,
    }
    usage = usage_row_from_response(
        task="multi_agent_stage2c",
        sample_id=row["sample_id"],
        anonymous_sample_id=row["anonymous_sample_id"],
        agent=agent,
        response_payload=response_payload,
        latency=latency,
        parse_status=status,
        parse_error=error,
    )
    return parsed, record, usage


def run_agent_text_call(
    *,
    api_key: str,
    row: dict[str, str],
    proposal: dict[str, str],
    system_prompt: str,
    user_message: str,
    raw_path: Path,
    timeout: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        response_payload, latency = call_openrouter_text(
            api_key=api_key,
            model_name=MODEL_NAME,
            system_prompt=system_prompt,
            user_message=user_message,
            timeout_seconds=timeout,
            max_tokens=max_tokens,
            response_schema=AGENT3_SCHEMA,
        )
        write_json(raw_path, response_payload)
        assert_expected_model(response_payload)
        raw_text = extract_message_text(response_payload)
        parsed = parse_agent3(raw_text)
        status, error = "success", ""
    except Exception as exc:
        if isinstance(exc, RuntimeError) and (
            "OpenRouter HTTP" in str(exc) or "silent substitution" in str(exc)
        ):
            raise
        if not raw_path.exists():
            write_json(raw_path, {"error": f"{type(exc).__name__}: {exc}"})
        response_payload = {}
        latency = 0.0
        parsed = {}
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    record = {
        "agent": "agent3_adjudicator",
        "sample_id": row["sample_id"],
        "anonymous_sample_id": row["anonymous_sample_id"],
        "selected_proposal_id": proposal["proposal_id"],
        "parse_status": status,
        "parsed": parsed,
        "raw_response_path": raw_path.as_posix(),
        "parse_error": error,
    }
    usage = usage_row_from_response(
        task="multi_agent_stage2c",
        sample_id=row["sample_id"],
        anonymous_sample_id=row["anonymous_sample_id"],
        agent="agent3_adjudicator",
        response_payload=response_payload,
        latency=latency,
        parse_status=status,
        parse_error=error,
    )
    return parsed, record, usage


def check_cost_guard(usage_rows: list[dict[str, Any]]) -> None:
    total = sum(float(row.get("estimated_cost_usd") or 0.0) for row in usage_rows)
    if total > MAX_PILOT_COST_USD:
        raise RuntimeError(
            f"Pilot cost ${total:.4f} exceeds ${MAX_PILOT_COST_USD:.2f}; stopping."
        )


def final_prediction_row(
    row: dict[str, str],
    proposal: dict[str, str],
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    agent3: dict[str, Any],
    statuses: tuple[str, str, str],
    errors: tuple[str, str, str],
) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "anonymous_sample_id": row["anonymous_sample_id"],
        "true_species": row["species"],
        "selected_proposal_id": proposal["proposal_id"],
        "selected_start_time": proposal["start_time"],
        "selected_end_time": proposal["end_time"],
        "selected_low_freq": proposal["low_freq"],
        "selected_high_freq": proposal["high_freq"],
        "agent1_parse_status": statuses[0],
        "agent1_predicted_species": agent1.get("predicted_species", ""),
        "agent1_confidence": agent1.get("confidence", ""),
        "agent2_parse_status": statuses[1],
        "agent2_review_decision": agent2.get("review_decision", ""),
        "agent2_revised_species": agent2.get("revised_species", ""),
        "agent2_human_review_recommended": str(agent2.get("human_review_recommended", "")).lower(),
        "agent3_parse_status": statuses[2],
        "final_species": agent3.get("final_species", ""),
        "final_confidence": agent3.get("confidence", ""),
        "review_status": agent3.get("review_status", ""),
        "human_review_recommended": str(agent3.get("human_review_recommended", "")).lower(),
        "final_correct": str(agent3.get("final_species", "") == row["species"]).lower(),
        "parse_error": " | ".join(error for error in errors if error),
    }


def write_token_cost_summary(path: Path, usage_rows: list[dict[str, Any]]) -> None:
    aggregate = {
        "task": "aggregate",
        "sample_id": "",
        "anonymous_sample_id": "",
        "agent": "all",
        "returned_model": "",
        "provider": "",
        "parse_status": "",
        "parse_error": "",
        "latency_seconds": mean(
            [float(row.get("latency_seconds") or 0.0) for row in usage_rows if float(row.get("latency_seconds") or 0.0) > 0]
        )
        if any(float(row.get("latency_seconds") or 0.0) > 0 for row in usage_rows)
        else 0.0,
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in usage_rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in usage_rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in usage_rows),
        "estimated_cost_usd": sum(float(row.get("estimated_cost_usd") or 0.0) for row in usage_rows),
        "raw_usage_json": "",
    }
    write_csv(path, usage_rows + [aggregate])


def same_sample_metric(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    return aggregate_metrics(rows, prediction_field)


def comparison_metric_rows(
    selected_rows: list[dict[str, str]],
    qwen_single: dict[str, dict[str, str]],
    qwen_multi: dict[str, dict[str, str]],
    gpt_single: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = comparison_rows(selected_rows, qwen_single, gpt_single)
    qwen_multi_by_anon = qwen_multi
    for row in rows:
        multi = qwen_multi_by_anon.get(row["anonymous_sample_id"], {})
        row["qwen_multi_agent_species"] = multi.get("final_species", "")
        row["qwen_multi_agent_correct"] = str(multi.get("final_species", "") == row["true_species"]).lower()
    return rows


def write_report(
    path: Path,
    *,
    predictions: list[dict[str, Any]],
    final_metrics: dict[str, Any],
    qwen_single_metrics: dict[str, Any],
    qwen_multi_metrics: dict[str, Any],
    gpt_single_metrics: dict[str, Any],
    usage_rows: list[dict[str, Any]],
) -> None:
    revised = sum(row.get("agent2_review_decision") == "revise" for row in predictions)
    uncertain = sum(
        row.get("review_status") == "uncertain" or row.get("human_review_recommended") == "true"
        for row in predictions
    )
    accepted = subset_accuracy(predictions, "review_status", "accepted")
    review = subset_accuracy(predictions, "human_review_recommended", "true")
    no_review = subset_accuracy(predictions, "human_review_recommended", "false")
    total_cost = sum(float(row.get("estimated_cost_usd") or 0.0) for row in usage_rows)
    projected_full240 = total_cost / 24 * 240 if predictions else 0.0
    lines = [
        "# GPT-5.6 Sol Multi-Agent Stage 2C Pilot24",
        "",
        "## Scope",
        "",
        "This pilot used the exact same 24 selected proposal samples as the qwen3.6 multi-agent pilot. It used label-safe centred crops and deterministic nearest-centre BatDetect2 proposal coordinates. The agents were instructed not to redraw, refine, add, or remove detections. Ground-truth species labels were used only after prediction for evaluation.",
        "",
        "## Parse Status",
        "",
        f"- Agent 1 parse success: {sum(row['agent1_parse_status'] == 'success' for row in predictions)}/{len(predictions)}",
        f"- Agent 2 parse success: {sum(row['agent2_parse_status'] == 'success' for row in predictions)}/{len(predictions)}",
        f"- Agent 3 parse success: {sum(row['agent3_parse_status'] == 'success' for row in predictions)}/{len(predictions)}",
        "",
        "## Final Classification Metrics",
        "",
        f"- Accuracy: {final_metrics['accuracy']:.3f}",
        f"- Macro-F1: {final_metrics['macro_F1']:.3f}",
        f"- Balanced accuracy: {final_metrics['balanced_accuracy']:.3f}",
        "",
        "## Review Metrics",
        "",
        f"- Revised cases: {revised}",
        f"- Uncertain or human-review cases: {uncertain}",
        f"- Accepted cases: {accepted['count']}, accuracy {accepted['accuracy']:.3f}",
        f"- Human-review cases: {review['count']}, accuracy {review['accuracy']:.3f}",
        f"- Non-review cases: {no_review['count']}, accuracy {no_review['accuracy']:.3f}",
        "",
        "## Same-Sample Comparisons",
        "",
        f"- qwen3.6 single-agent accuracy: {qwen_single_metrics['accuracy']:.3f}",
        f"- qwen3.6 multi-agent accuracy: {qwen_multi_metrics['accuracy']:.3f}",
        f"- GPT-5.6 Sol single-agent accuracy: {gpt_single_metrics['accuracy']:.3f}",
        f"- GPT-5.6 Sol multi-agent accuracy: {final_metrics['accuracy']:.3f}",
        "",
        "## Token and Cost",
        "",
        f"- Estimated pilot cost: ${total_cost:.4f}",
        f"- Projected full240 multi-agent cost: ${projected_full240:.4f}",
        "",
        "## Interpretation",
        "",
        "This pilot tests whether a classifier-critic-adjudicator structure improves GPT-5.6 Sol species decisions beyond the single-agent selected-proposal baseline on the same samples. Review flags should be interpreted as useful only if they are enriched among incorrect or difficult samples.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-list", type=Path, default=DEFAULT_QWEN_SAMPLE_LIST)
    parser.add_argument("--selected-proposals", type=Path, default=DEFAULT_SELECTED_PROPOSALS)
    parser.add_argument("--qwen-single", type=Path, default=QWEN_STAGE2C_FULL240)
    parser.add_argument("--qwen-multi", type=Path, default=QWEN_MULTI_AGENT_PREDICTIONS)
    parser.add_argument("--gpt-single", type=Path, default=GPT_STAGE2C_FULL240)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--agent-max-tokens", type=int, default=900)
    parser.add_argument("--adjudicator-max-tokens", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir.exists() and (args.run_dir / "multi_agent_pilot_predictions.csv").exists():
        raise FileExistsError(f"Output already exists: {args.run_dir}")
    api_key = load_api_key()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    availability = availability_check(api_key, args.analysis_dir, args.timeout)
    print(
        "OpenRouter availability passed: "
        f"model={availability['returned_model']} provider={availability['provider']}",
        flush=True,
    )

    selected_rows = read_csv(args.sample_list)
    if len(selected_rows) != 24:
        raise RuntimeError(f"Expected 24 fixed pilot samples, got {len(selected_rows)}")
    selected_by_anon = selected_proposal_index(args.selected_proposals, "full240")
    qwen_single = comparison_prediction_index(args.qwen_single)
    qwen_multi = comparison_prediction_index(args.qwen_multi)
    gpt_single = comparison_prediction_index(args.gpt_single)
    write_csv(args.run_dir / "selected_pilot_samples.csv", selected_rows)
    write_csv(args.analysis_dir / "selected_pilot_samples.csv", selected_rows)

    agent1_records: list[dict[str, Any]] = []
    agent2_records: list[dict[str, Any]] = []
    agent3_records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []

    for index, row in enumerate(selected_rows, start=1):
        anon_id = row["anonymous_sample_id"]
        print(f"[{index}/{len(selected_rows)}] {anon_id}", flush=True)
        proposal = selected_by_anon[anon_id]
        image_path = resolve_repo_path(row["centred_crop_image_path"])
        raw_dir = args.run_dir / "raw_responses" / anon_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        parsed1, rec1, usage1 = run_agent_image_call(
            api_key=api_key,
            row=row,
            proposal=proposal,
            agent="agent1_classifier",
            system_prompt=build_agent1_system_prompt(),
            user_message=build_image_user_message(row, proposal),
            image_path=image_path,
            schema=AGENT1_SCHEMA,
            parser=parse_agent1,
            raw_path=raw_dir / "agent1_classifier_raw_response.json",
            timeout=args.timeout,
            max_tokens=args.agent_max_tokens,
        )
        agent1_records.append(rec1)
        usage_rows.append(usage1)
        check_cost_guard(usage_rows)

        parsed2, rec2, usage2 = run_agent_image_call(
            api_key=api_key,
            row=row,
            proposal=proposal,
            agent="agent2_reviewer",
            system_prompt=build_agent2_system_prompt(),
            user_message=build_image_user_message(row, proposal, {"agent1_classifier_output": parsed1}),
            image_path=image_path,
            schema=AGENT2_SCHEMA,
            parser=parse_agent2,
            raw_path=raw_dir / "agent2_reviewer_raw_response.json",
            timeout=args.timeout,
            max_tokens=args.agent_max_tokens,
        )
        agent2_records.append(rec2)
        usage_rows.append(usage2)
        check_cost_guard(usage_rows)

        parsed3, rec3, usage3 = run_agent_text_call(
            api_key=api_key,
            row=row,
            proposal=proposal,
            system_prompt=(
                "You are Agent 3, the final adjudicator. Use only Agent 1 and "
                "Agent 2 outputs. Choose one final forced-choice species label. "
                "Recommend human review when evidence is weak, ambiguous, or "
                "the reviewer disagrees with the classifier."
            ),
            user_message=build_agent3_prompt(parsed1, parsed2),
            raw_path=raw_dir / "agent3_adjudicator_raw_response.json",
            timeout=args.timeout,
            max_tokens=args.adjudicator_max_tokens,
        )
        agent3_records.append(rec3)
        usage_rows.append(usage3)
        check_cost_guard(usage_rows)

        predictions.append(
            final_prediction_row(
                row,
                proposal,
                parsed1,
                parsed2,
                parsed3,
                (rec1["parse_status"], rec2["parse_status"], rec3["parse_status"]),
                (rec1["parse_error"], rec2["parse_error"], rec3["parse_error"]),
            )
        )

    write_jsonl(args.run_dir / "agent1_classifier_outputs.jsonl", agent1_records)
    write_jsonl(args.run_dir / "agent2_reviewer_outputs.jsonl", agent2_records)
    write_jsonl(args.run_dir / "agent3_adjudicator_outputs.jsonl", agent3_records)
    write_csv(args.run_dir / "multi_agent_pilot_predictions.csv", predictions)
    write_csv(args.analysis_dir / "multi_agent_pilot_predictions.csv", predictions)
    write_token_cost_summary(args.run_dir / "token_cost_summary.csv", usage_rows)
    write_token_cost_summary(args.analysis_dir / "token_cost_summary.csv", usage_rows)

    final_metrics = aggregate_metrics(predictions, "final_species")
    species_rows = per_species_metrics(predictions, "final_species")
    confusion_rows = confusion_matrix(predictions, "final_species")
    write_csv(args.analysis_dir / "per_species_metrics.csv", species_rows)
    write_csv(args.analysis_dir / "confusion_matrix.csv", confusion_rows)

    same_sample_rows = comparison_metric_rows(selected_rows, qwen_single, qwen_multi, gpt_single)
    for row in same_sample_rows:
        pred = next(item for item in predictions if item["anonymous_sample_id"] == row["anonymous_sample_id"])
        row["gpt_multi_agent_species"] = pred["final_species"]
        row["gpt_multi_agent_correct"] = pred["final_correct"]
    write_csv(args.analysis_dir / "same_sample_comparison.csv", same_sample_rows)

    qwen_single_metrics = aggregate_metrics(
        [{"true_species": row["true_species"], "predicted": row["qwen_single_agent_species"]} for row in same_sample_rows],
        "predicted",
    )
    qwen_multi_metrics = aggregate_metrics(
        [{"true_species": row["true_species"], "predicted": row["qwen_multi_agent_species"]} for row in same_sample_rows],
        "predicted",
    )
    gpt_single_metrics = aggregate_metrics(
        [{"true_species": row["true_species"], "predicted": row["gpt_single_agent_species"]} for row in same_sample_rows],
        "predicted",
    )
    summary = {
        "condition": CONDITION_NAME,
        "model_name": MODEL_NAME,
        "sample_count": len(predictions),
        "agent1_parse_success": sum(row["agent1_parse_status"] == "success" for row in predictions),
        "agent2_parse_success": sum(row["agent2_parse_status"] == "success" for row in predictions),
        "agent3_parse_success": sum(row["agent3_parse_status"] == "success" for row in predictions),
        "final_metrics": final_metrics,
        "qwen_single_agent_same_sample_metrics": qwen_single_metrics,
        "qwen_multi_agent_same_sample_metrics": qwen_multi_metrics,
        "gpt_single_agent_same_sample_metrics": gpt_single_metrics,
        "revised_cases": sum(row.get("agent2_review_decision") == "revise" for row in predictions),
        "uncertain_or_human_review_cases": sum(
            row.get("review_status") == "uncertain" or row.get("human_review_recommended") == "true"
            for row in predictions
        ),
        "estimated_pilot_cost_usd": sum(float(row.get("estimated_cost_usd") or 0.0) for row in usage_rows),
        "projected_full240_multi_agent_cost_usd": (
            sum(float(row.get("estimated_cost_usd") or 0.0) for row in usage_rows) / 24 * 240
        ),
    }
    write_json(args.analysis_dir / "aggregate_summary.json", summary)
    write_report(
        args.analysis_dir / "multi_agent_pilot_report.md",
        predictions=predictions,
        final_metrics=final_metrics,
        qwen_single_metrics=qwen_single_metrics,
        qwen_multi_metrics=qwen_multi_metrics,
        gpt_single_metrics=gpt_single_metrics,
        usage_rows=usage_rows,
    )
    write_report(
        args.run_dir / "multi_agent_pilot_report.md",
        predictions=predictions,
        final_metrics=final_metrics,
        qwen_single_metrics=qwen_single_metrics,
        qwen_multi_metrics=qwen_multi_metrics,
        gpt_single_metrics=gpt_single_metrics,
        usage_rows=usage_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
