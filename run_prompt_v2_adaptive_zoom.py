"""Run a minimal adaptive overview-plus-zoom Prompt V2 prototype.

This script uses clean spectrogram images generated from WAV audio only. It does
not read ground-truth JSON during prediction and it does not use GT overlays or
diagnostic figures as model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageDraw, ImageOps

from main import make_spectrogram, to_decibels
from scripts.data_prep.prepare_agent_spectrogram_inputs import (
    DEFAULT_MAX_FREQ_HZ,
    DEFAULT_MIN_DB,
    apply_grid_style,
    read_mono_audio,
    resolve_audio_path,
)
from run_prompt_v2_small_pilot import (
    PREDICTION_JSON_SCHEMA,
    PROMPT_VERSION,
    PilotResult,
    build_user_message,
    default_output_dir_for_run,
    extract_json_text,
    image_to_base64,
    load_prompt,
    ollama_host,
    parse_clip_ids,
    parse_prediction,
    read_clip_duration,
    resolve_all_clip_ids,
    write_failed_prediction_output,
    write_prediction_output,
)


DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_OVERVIEW_DIR = Path("outputs/agent_inputs/prompt_v2_full_grid_v2")
DEFAULT_ADAPTIVE_INPUT_DIR = Path("outputs/agent_inputs/prompt_v2_adaptive_zoom_qwen3_6")
DEFAULT_OUTPUT_DIR = Path("outputs/agent_runs/prompt_v2_adaptive_zoom_qwen3_6_representative6")
DEFAULT_PLAN_PROMPT = Path("prompts/prompt_v2_adaptive_view_planning.md")
DEFAULT_FINAL_PROMPT = Path("prompts/prompt_v2_adaptive_zoom_final.md")
DEFAULT_MODEL_NAME = "qwen3.6:latest"
DEFAULT_CLIP_IDS = ["OP_001", "OP_010", "OP_045", "OP_003", "OP_004", "OP_016"]
MAX_ZOOM_REQUESTS = 3


VIEW_PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "clip_id": {"type": "string"},
        "preferred_grid": {"type": "string"},
        "zoom_needed": {"type": "boolean"},
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
    "required": ["clip_id", "preferred_grid", "zoom_needed", "zoom_requests", "reason"],
}


@dataclass(frozen=True)
class ZoomWindow:
    zoom_id: str
    start_time_seconds: float
    end_time_seconds: float
    low_frequency_hz: float
    high_frequency_hz: float
    reason: str


@dataclass(frozen=True)
class AdaptiveClipResult:
    clip_id: str
    view_plan_parse_success: bool
    final_parse_status: str
    predicted_event_count: int | None
    accepted_zoom_requests: int
    rejected_zoom_requests: int


def call_ollama_generate_images(
    *,
    image_paths: list[Path],
    prompt: str,
    model_name: str,
    response_schema: dict[str, Any],
    timeout: float,
    num_predict: int,
) -> str:
    """Call Ollama generate with one or more clean images."""
    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": response_schema,
        "prompt": prompt,
        "images": [image_to_base64(path) for path in image_paths],
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
    thinking = response_payload.get("thinking")
    if thinking:
        return str(thinking)
    return json.dumps(response_payload, indent=2, ensure_ascii=False)


def parse_view_plan(raw_text: str, *, expected_clip_id: str) -> dict[str, Any]:
    """Parse and minimally validate a view-plan JSON response."""
    payload = json.loads(extract_json_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("View plan must be a JSON object")
    if payload.get("clip_id") != expected_clip_id:
        raise ValueError(
            f"clip_id must be {expected_clip_id!r}, got {payload.get('clip_id')!r}"
        )
    if payload.get("preferred_grid") not in {"grid_v1", "grid_v2"}:
        raise ValueError("preferred_grid must be grid_v1 or grid_v2")
    if not isinstance(payload.get("zoom_needed"), bool):
        raise ValueError("zoom_needed must be boolean")
    if not isinstance(payload.get("zoom_requests"), list):
        raise ValueError("zoom_requests must be a list")
    return payload


def validate_zoom_request(
    request: dict[str, Any],
    *,
    index: int,
    clip_duration_seconds: float,
    max_frequency_hz: float,
) -> ZoomWindow | None:
    """Clip one zoom request to valid bounds or discard it if degenerate."""
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


def accepted_zoom_windows(
    plan: dict[str, Any],
    *,
    clip_duration_seconds: float,
    max_frequency_hz: float,
) -> tuple[list[ZoomWindow], int]:
    """Return accepted zoom windows and rejected request count."""
    accepted: list[ZoomWindow] = []
    rejected = 0
    for index, request in enumerate(plan.get("zoom_requests", [])[:MAX_ZOOM_REQUESTS]):
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
    extra = max(0, len(plan.get("zoom_requests", [])) - MAX_ZOOM_REQUESTS)
    return accepted, rejected + extra


def spectrogram_image(spec: np.ndarray, min_db: float) -> np.ndarray:
    spec_db = to_decibels(spec)
    spec_db = np.clip(spec_db, min_db, 0)
    return (spec_db - min_db) / -min_db


def save_zoom_spectrogram(
    *,
    eval_dir: Path,
    clip_id: str,
    window: ZoomWindow,
    output_dir: Path,
    min_db: float,
    max_frequency_hz: float,
) -> Path:
    """Save one clean zoom spectrogram with original clip coordinates."""
    audio_path = resolve_audio_path(eval_dir, clip_id)
    audio, sample_rate = read_mono_audio(audio_path)
    spec, stft = make_spectrogram(audio, sample_rate)
    image = spectrogram_image(spec, min_db)
    extent = list(stft.extent(len(audio)))
    extent[2] /= 1000
    extent[3] /= 1000

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )
    displayed_high_hz = min(window.high_frequency_hz, max_frequency_hz, sample_rate / 2)
    ax.set_xlim(window.start_time_seconds, window.end_time_seconds)
    ax.set_ylim(window.low_frequency_hz / 1000, displayed_high_hz / 1000)
    ax.set_xlabel("Time (s, original clip)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(
        f"{clip_id} {window.zoom_id}: "
        f"{window.start_time_seconds:.3f}-{window.end_time_seconds:.3f}s"
    )
    ax.xaxis.set_major_locator(
        MultipleLocator(max(0.01, (window.end_time_seconds - window.start_time_seconds) / 5))
    )
    ax.yaxis.set_major_locator(
        MultipleLocator(max(1.0, (displayed_high_hz - window.low_frequency_hz) / 5000))
    )
    ax.grid(which="major", color="cyan", linewidth=0.7, alpha=0.65)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clip_id}_{window.zoom_id}.png"
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def overview_output_path(output_dir: str | Path, clip_id: str) -> Path:
    return Path(output_dir) / f"{clip_id}_overview_grid_v2.png"


def composite_output_path(output_dir: str | Path, clip_id: str) -> Path:
    return Path(output_dir) / f"{clip_id}_composite.png"


def copy_overview(overview_dir: Path, output_dir: Path, clip_id: str) -> Path:
    source = overview_dir / f"{clip_id}_spectrogram.png"
    if not source.is_file():
        raise FileNotFoundError(f"Overview spectrogram not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = overview_output_path(output_dir, clip_id)
    shutil.copyfile(source, destination)
    return destination


def titled_panel(image_path: Path, title: str, width: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    ratio = width / image.width
    resized = image.resize((width, max(1, int(image.height * ratio))))
    title_height = 34
    panel = Image.new("RGB", (width, resized.height + title_height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), title, fill="black")
    panel.paste(resized, (0, title_height))
    return ImageOps.expand(panel, border=2, fill="black")


def make_composite_image(
    *,
    clip_id: str,
    overview_path: Path,
    zoom_paths: list[Path],
    output_dir: Path,
) -> Path:
    """Create a clean overview-plus-zoom montage for one model input."""
    width = 1200
    panels = [titled_panel(overview_path, f"{clip_id} overview grid_v2", width)]
    for index, path in enumerate(zoom_paths, start=1):
        panels.append(titled_panel(path, f"{clip_id} zoom {index}", width))
    total_height = sum(panel.height for panel in panels)
    composite = Image.new("RGB", (width + 4, total_height), "white")
    y = 0
    for panel in panels:
        composite.paste(panel, (0, y))
        y += panel.height
    output_path = composite_output_path(output_dir, clip_id)
    composite.save(output_path)
    return output_path


def view_plan_user_message(clip_id: str, clip_duration_seconds: float) -> str:
    context = {
        "clip_id": clip_id,
        "clip_duration_seconds": clip_duration_seconds,
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
        "max_zoom_requests": MAX_ZOOM_REQUESTS,
    }
    return (
        "Plan adaptive clean zoom views for this spectrogram.\n\n"
        f"Runtime context:\n{json.dumps(context, indent=2)}\n\n"
        "Return valid JSON only."
    )


def final_user_message(clip_id: str, clip_duration_seconds: float, zoom_count: int) -> str:
    context = {
        "clip_id": clip_id,
        "clip_duration_seconds": clip_duration_seconds,
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
        "zoom_panel_count": zoom_count,
    }
    return (
        "Annotate the attached clean composite spectrogram according to the prompt.\n\n"
        f"Runtime context:\n{json.dumps(context, indent=2)}\n\n"
        "Return valid JSON only."
    )


def write_view_plan(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def failed_view_plan(clip_id: str, error: Exception) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "preferred_grid": "grid_v2",
        "zoom_needed": False,
        "zoom_requests": [],
        "reason": f"View planning failed; using overview only. {type(error).__name__}: {error}",
        "parse_status": "failed",
    }


def run_adaptive_clip(
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
) -> AdaptiveClipResult:
    """Run view planning, zoom generation, and final annotation for one clip."""
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
            prompt=f"{planning_prompt}\n\n{view_plan_user_message(clip_id, clip_duration)}",
            model_name=model_name,
            response_schema=VIEW_PLAN_JSON_SCHEMA,
            timeout=timeout,
            num_predict=2000,
        )
        plan_raw_path.write_text(raw_plan, encoding="utf-8")
        plan = parse_view_plan(raw_plan, expected_clip_id=clip_id)
    except Exception as exc:
        plan_parse_success = False
        if not plan_raw_path.exists():
            plan_raw_path.write_text("", encoding="utf-8")
        plan = failed_view_plan(clip_id, exc)

    zoom_windows, rejected_count = accepted_zoom_windows(
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
            backend="ollama_generate_adaptive_zoom",
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            input_image_path=composite_path,
            clip_duration_seconds=clip_duration,
        )
        return AdaptiveClipResult(
            clip_id=clip_id,
            view_plan_parse_success=plan_parse_success,
            final_parse_status="success",
            predicted_event_count=len(prediction["events"]),
            accepted_zoom_requests=len(zoom_windows),
            rejected_zoom_requests=rejected_count,
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
            backend="ollama_generate_adaptive_zoom",
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            input_image_path=composite_path,
            clip_duration_seconds=clip_duration,
            error_message=error_message,
        )
        return AdaptiveClipResult(
            clip_id=clip_id,
            view_plan_parse_success=plan_parse_success,
            final_parse_status="failed",
            predicted_event_count=None,
            accepted_zoom_requests=len(zoom_windows),
            rejected_zoom_requests=rejected_count,
        )


def write_view_plan_summary(output_dir: Path, results: list[AdaptiveClipResult]) -> Path:
    path = output_dir / "view_plan_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "clip_id",
            "preferred_grid",
            "zoom_needed",
            "number_of_zoom_requests",
            "accepted_zoom_requests",
            "rejected_zoom_requests",
            "parse_success",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            plan = json.loads((output_dir / f"{result.clip_id}_view_plan.json").read_text())
            writer.writerow(
                {
                    "clip_id": result.clip_id,
                    "preferred_grid": plan.get("preferred_grid", ""),
                    "zoom_needed": plan.get("zoom_needed", False),
                    "number_of_zoom_requests": len(plan.get("zoom_requests", [])),
                    "accepted_zoom_requests": result.accepted_zoom_requests,
                    "rejected_zoom_requests": result.rejected_zoom_requests,
                    "parse_success": result.view_plan_parse_success,
                }
            )
    return path


def print_summary(results: list[AdaptiveClipResult]) -> None:
    print("clip_id | view_plan | final_parse | events | accepted_zoom | rejected_zoom")
    print("--------+-----------+-------------+--------+---------------+--------------")
    for result in results:
        events = "" if result.predicted_event_count is None else str(result.predicted_event_count)
        print(
            f"{result.clip_id} | {result.view_plan_parse_success} | "
            f"{result.final_parse_status} | {events} | "
            f"{result.accepted_zoom_requests} | {result.rejected_zoom_requests}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive Prompt V2 zoom prototype.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--overview-dir", type=Path, default=DEFAULT_OVERVIEW_DIR)
    parser.add_argument("--adaptive-input-dir", type=Path, default=DEFAULT_ADAPTIVE_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--planning-prompt", type=Path, default=DEFAULT_PLAN_PROMPT)
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
        run_adaptive_clip(
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
