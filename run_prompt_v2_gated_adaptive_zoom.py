"""Run P5D gated adaptive zoom on representative Prompt V2 clips."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_agent_spectrogram_inputs import DEFAULT_MAX_FREQ_HZ, DEFAULT_MIN_DB
from run_prompt_v2_adaptive_zoom import (
    DEFAULT_CLIP_IDS,
    DEFAULT_EVAL_DIR,
    DEFAULT_FINAL_PROMPT,
    DEFAULT_MODEL_NAME,
    DEFAULT_OVERVIEW_DIR,
    ZoomWindow,
    call_ollama_generate_images,
    copy_overview,
    failed_view_plan,
    final_user_message,
    make_composite_image,
    save_zoom_spectrogram,
    write_view_plan,
)
from run_prompt_v2_small_pilot import (
    PREDICTION_JSON_SCHEMA,
    extract_json_text,
    load_prompt,
    parse_clip_ids,
    parse_prediction,
    read_clip_duration,
    resolve_all_clip_ids,
    write_failed_prediction_output,
    write_prediction_output,
)


DEFAULT_GATED_PLAN_PROMPT = Path("prompts/prompt_v2_gated_adaptive_view_planning.md")
DEFAULT_GATED_INPUT_DIR = Path("outputs/agent_inputs/prompt_v2_gated_adaptive_zoom_qwen3_6")
DEFAULT_GATED_OUTPUT_DIR = Path(
    "outputs/agent_runs/prompt_v2_gated_adaptive_zoom_qwen3_6_representative6"
)
ALLOWED_GATING_REASONS = {
    "dense_adjacent_calls",
    "boundary_truncated_calls",
    "weak_or_faint_calls",
    "unclear_frequency_bounds",
    "ambiguous_region",
}


GATED_VIEW_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "clip_id": {"type": "string"},
        "overview_sufficient": {"type": "boolean"},
        "zoom_needed": {"type": "boolean"},
        "gating_reasons": {
            "type": "array",
            "items": {"type": "string"},
        },
        "zoom_requests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "zoom_id": {"type": "string"},
                    "start_time_seconds": {"type": "number"},
                    "end_time_seconds": {"type": "number"},
                    "low_frequency_hz": {"type": "number"},
                    "high_frequency_hz": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "zoom_id",
                    "start_time_seconds",
                    "end_time_seconds",
                    "low_frequency_hz",
                    "high_frequency_hz",
                    "reason",
                ],
            },
        },
        "reason": {"type": "string"},
    },
    "required": [
        "clip_id",
        "overview_sufficient",
        "zoom_needed",
        "gating_reasons",
        "zoom_requests",
        "reason",
    ],
}


@dataclass(frozen=True)
class GatedClipResult:
    clip_id: str
    plan_parse_success: bool
    final_parse_status: str
    predicted_event_count: int | None
    overview_sufficient: bool
    zoom_needed: bool
    gating_reasons: list[str]
    requested_zoom_count: int
    accepted_zoom_count: int
    rejected_zoom_count: int


def parse_gated_view_plan(raw_text: str, *, expected_clip_id: str) -> dict[str, Any]:
    """Parse and validate the gated view-plan JSON response."""
    payload = json.loads(extract_json_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("View plan must be a JSON object")
    if payload.get("clip_id") != expected_clip_id:
        raise ValueError(
            f"clip_id must be {expected_clip_id!r}, got {payload.get('clip_id')!r}"
        )
    if not isinstance(payload.get("overview_sufficient"), bool):
        raise ValueError("overview_sufficient must be boolean")
    if not isinstance(payload.get("zoom_needed"), bool):
        raise ValueError("zoom_needed must be boolean")
    if not isinstance(payload.get("gating_reasons"), list):
        raise ValueError("gating_reasons must be a list")
    if not isinstance(payload.get("zoom_requests"), list):
        raise ValueError("zoom_requests must be a list")

    valid_reasons = [
        str(reason)
        for reason in payload["gating_reasons"]
        if str(reason) in ALLOWED_GATING_REASONS
    ]
    payload["gating_reasons"] = list(dict.fromkeys(valid_reasons))

    if payload["overview_sufficient"]:
        payload["zoom_needed"] = False
        payload["zoom_requests"] = []
        payload["gating_reasons"] = []
    elif payload["zoom_needed"] and not payload["gating_reasons"]:
        raise ValueError("zoom_needed requires at least one valid gating reason")
    return payload


def gated_view_plan_user_message(clip_id: str, clip_duration_seconds: float) -> str:
    context = {
        "clip_id": clip_id,
        "clip_duration_seconds": clip_duration_seconds,
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
        "default_max_zoom_requests": 2,
        "max_zoom_requests_for_dense_or_boundary": 3,
        "allowed_gating_reasons": sorted(ALLOWED_GATING_REASONS),
    }
    return (
        "Plan gated adaptive clean zoom views for this spectrogram.\n\n"
        f"Runtime context:\n{json.dumps(context, indent=2)}\n\n"
        "If the overview is sufficient, do not request zoom. Return valid JSON only."
    )


def validate_zoom_request(
    request: dict[str, Any],
    *,
    index: int,
    clip_duration_seconds: float,
    max_frequency_hz: float,
) -> ZoomWindow | None:
    """Clip one zoom request to valid bounds or discard if degenerate."""
    try:
        start = max(0.0, float(request["start_time_seconds"]))
        end = min(clip_duration_seconds, float(request["end_time_seconds"]))
        low = max(0.0, float(request["low_frequency_hz"]))
        high = min(max_frequency_hz, float(request["high_frequency_hz"]))
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start or high <= low:
        return None
    zoom_id = str(request.get("zoom_id") or f"zoom_{index + 1:03d}")
    if not zoom_id.startswith("zoom_"):
        zoom_id = f"zoom_{index + 1:03d}"
    return ZoomWindow(
        zoom_id=zoom_id,
        start_time_seconds=start,
        end_time_seconds=end,
        low_frequency_hz=low,
        high_frequency_hz=high,
        reason=str(request.get("reason") or ""),
    )


def accepted_gated_zoom_windows(
    plan: dict[str, Any],
    *,
    clip_duration_seconds: float,
    max_frequency_hz: float,
) -> tuple[list[ZoomWindow], int]:
    """Apply gating policy and return accepted zooms plus rejection count."""
    requested = plan.get("zoom_requests", [])
    if plan.get("overview_sufficient") or not plan.get("zoom_needed"):
        return [], len(requested)
    reasons = set(plan.get("gating_reasons", [])) & ALLOWED_GATING_REASONS
    if not reasons:
        return [], len(requested)
    max_zoom = (
        3
        if reasons & {"dense_adjacent_calls", "boundary_truncated_calls"}
        else 2
    )
    accepted: list[ZoomWindow] = []
    rejected = max(0, len(requested) - max_zoom)
    for index, request in enumerate(requested[:max_zoom]):
        window = validate_zoom_request(
            request,
            index=index,
            clip_duration_seconds=clip_duration_seconds,
            max_frequency_hz=max_frequency_hz,
        )
        if window is None:
            rejected += 1
        else:
            accepted.append(window)
    return accepted, rejected


def fallback_gated_plan(clip_id: str, error: Exception) -> dict[str, Any]:
    plan = failed_view_plan(clip_id, error)
    plan["overview_sufficient"] = True
    plan["gating_reasons"] = []
    return plan


def run_gated_clip(
    *,
    clip_id: str,
    planning_prompt: str,
    final_prompt: str,
    eval_dir: Path,
    overview_dir: Path,
    adaptive_input_dir: Path,
    output_dir: Path,
    model_name: str,
    timeout: float,
    num_predict: int,
    min_db: float,
    max_frequency_hz: float,
) -> GatedClipResult:
    """Run gated planning, optional zoom generation, and final annotation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    adaptive_input_dir.mkdir(parents=True, exist_ok=True)
    clip_duration = read_clip_duration(eval_dir, clip_id)
    overview_path = copy_overview(overview_dir, adaptive_input_dir, clip_id)

    plan_raw_path = output_dir / f"{clip_id}_view_plan_raw_response.txt"
    plan_json_path = output_dir / f"{clip_id}_view_plan.json"
    plan_parse_success = True
    try:
        raw_plan = call_ollama_generate_images(
            image_paths=[overview_path],
            prompt=f"{planning_prompt}\n\n{gated_view_plan_user_message(clip_id, clip_duration)}",
            model_name=model_name,
            response_schema=GATED_VIEW_PLAN_SCHEMA,
            timeout=timeout,
            num_predict=2000,
        )
        plan_raw_path.write_text(raw_plan, encoding="utf-8")
        plan = parse_gated_view_plan(raw_plan, expected_clip_id=clip_id)
    except Exception as exc:
        plan_parse_success = False
        if not plan_raw_path.exists():
            plan_raw_path.write_text("", encoding="utf-8")
        plan = fallback_gated_plan(clip_id, exc)

    requested_zoom_count = len(plan.get("zoom_requests", []))
    zoom_windows, rejected_count = accepted_gated_zoom_windows(
        plan,
        clip_duration_seconds=clip_duration,
        max_frequency_hz=max_frequency_hz,
    )
    plan["accepted_zoom_requests"] = [window.__dict__ for window in zoom_windows]
    plan["rejected_zoom_requests"] = rejected_count
    write_view_plan(plan_json_path, plan)

    zoom_paths = [
        save_zoom_spectrogram(
            eval_dir=eval_dir,
            clip_id=clip_id,
            window=window,
            output_dir=adaptive_input_dir,
            min_db=min_db,
            max_frequency_hz=max_frequency_hz,
        )
        for window in zoom_windows
    ]
    composite_path = make_composite_image(
        clip_id=clip_id,
        overview_path=overview_path,
        zoom_paths=zoom_paths,
        output_dir=adaptive_input_dir,
    )

    raw_response_path = output_dir / f"{clip_id}_raw_response.txt"
    prediction_path = output_dir / f"{clip_id}_predictions.json"
    parse_error_path = output_dir / f"{clip_id}_parse_error.txt"
    prediction_path.unlink(missing_ok=True)
    parse_error_path.unlink(missing_ok=True)
    try:
        raw_text = call_ollama_generate_images(
            image_paths=[composite_path],
            prompt=f"{final_prompt}\n\n{final_user_message(clip_id, clip_duration, len(zoom_paths))}",
            model_name=model_name,
            response_schema=PREDICTION_JSON_SCHEMA,
            timeout=timeout,
            num_predict=num_predict,
        )
        raw_response_path.write_text(raw_text, encoding="utf-8")
        prediction = parse_prediction(raw_text, expected_clip_id=clip_id)
        write_prediction_output(
            output_path=prediction_path,
            prediction=prediction,
            model_name=model_name,
            backend="ollama_generate_gated_adaptive_zoom",
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            input_image_path=composite_path,
            clip_duration_seconds=clip_duration,
        )
        return GatedClipResult(
            clip_id=clip_id,
            plan_parse_success=plan_parse_success,
            final_parse_status="success",
            predicted_event_count=len(prediction["events"]),
            overview_sufficient=bool(plan.get("overview_sufficient")),
            zoom_needed=bool(plan.get("zoom_needed")),
            gating_reasons=list(plan.get("gating_reasons", [])),
            requested_zoom_count=requested_zoom_count,
            accepted_zoom_count=len(zoom_windows),
            rejected_zoom_count=rejected_count,
        )
    except Exception as exc:
        if not raw_response_path.exists():
            raw_response_path.write_text("", encoding="utf-8")
        error_message = f"{type(exc).__name__}: {exc}"
        parse_error_path.write_text(error_message + "\n", encoding="utf-8")
        write_failed_prediction_output(
            output_path=prediction_path,
            clip_id=clip_id,
            model_name=model_name,
            backend="ollama_generate_gated_adaptive_zoom",
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            input_image_path=composite_path,
            clip_duration_seconds=clip_duration,
            error_message=error_message,
        )
        return GatedClipResult(
            clip_id=clip_id,
            plan_parse_success=plan_parse_success,
            final_parse_status="failed",
            predicted_event_count=None,
            overview_sufficient=bool(plan.get("overview_sufficient")),
            zoom_needed=bool(plan.get("zoom_needed")),
            gating_reasons=list(plan.get("gating_reasons", [])),
            requested_zoom_count=requested_zoom_count,
            accepted_zoom_count=len(zoom_windows),
            rejected_zoom_count=rejected_count,
        )


def write_view_plan_summary(output_dir: Path, results: list[GatedClipResult]) -> Path:
    path = output_dir / "view_plan_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_id",
            "overview_sufficient",
            "zoom_needed",
            "gating_reasons",
            "requested_zoom_count",
            "accepted_zoom_count",
            "rejected_zoom_count",
            "parse_success",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "clip_id": result.clip_id,
                    "overview_sufficient": result.overview_sufficient,
                    "zoom_needed": result.zoom_needed,
                    "gating_reasons": ";".join(result.gating_reasons),
                    "requested_zoom_count": result.requested_zoom_count,
                    "accepted_zoom_count": result.accepted_zoom_count,
                    "rejected_zoom_count": result.rejected_zoom_count,
                    "parse_success": result.plan_parse_success,
                }
            )
    return path


def print_summary(results: list[GatedClipResult]) -> None:
    print("clip_id | plan | final | events | sufficient | requested | accepted | rejected | reasons")
    print("--------+------+-------+--------+------------+-----------+----------+----------+--------")
    for result in results:
        events = "" if result.predicted_event_count is None else str(result.predicted_event_count)
        print(
            f"{result.clip_id} | {result.plan_parse_success} | "
            f"{result.final_parse_status} | {events} | {result.overview_sufficient} | "
            f"{result.requested_zoom_count} | {result.accepted_zoom_count} | "
            f"{result.rejected_zoom_count} | {';'.join(result.gating_reasons)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P5D gated adaptive zoom prototype.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--overview-dir",
        "--input-dir",
        dest="overview_dir",
        type=Path,
        default=DEFAULT_OVERVIEW_DIR,
        help="Directory containing clean overview spectrogram inputs.",
    )
    parser.add_argument(
        "--adaptive-input-dir",
        "--generated-input-dir",
        dest="adaptive_input_dir",
        type=Path,
        default=DEFAULT_GATED_INPUT_DIR,
        help="Directory where copied overview, zoom, and composite inputs are saved.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GATED_OUTPUT_DIR)
    parser.add_argument("--planning-prompt", type=Path, default=DEFAULT_GATED_PLAN_PROMPT)
    parser.add_argument("--final-prompt", type=Path, default=DEFAULT_FINAL_PROMPT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--clip-list", default=",".join(DEFAULT_CLIP_IDS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--num-predict", type=int, default=8000)
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ_HZ)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planning_prompt = load_prompt(args.planning_prompt)
    final_prompt = load_prompt(args.final_prompt)
    clip_ids = resolve_all_clip_ids(args.eval_dir) if args.all else parse_clip_ids(args.clip_list)
    results = [
        run_gated_clip(
            clip_id=clip_id,
            planning_prompt=planning_prompt,
            final_prompt=final_prompt,
            eval_dir=args.eval_dir,
            overview_dir=args.overview_dir,
            adaptive_input_dir=args.adaptive_input_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            timeout=args.timeout,
            num_predict=args.num_predict,
            min_db=args.min_db,
            max_frequency_hz=args.max_freq,
        )
        for clip_id in clip_ids
    ]
    summary_path = write_view_plan_summary(args.output_dir, results)
    print_summary(results)
    print(f"Saved view-plan summary to {summary_path}")


if __name__ == "__main__":
    main()
