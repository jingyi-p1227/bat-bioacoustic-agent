"""Run P8C: full-45 PCEN plus grid_v2 Prompt V2 experiment.

This entry point prepares clean PCEN spectrograms with the existing grid_v2
overlay from WAV audio, confirms the requested qwen3.6 Ollama endpoint, and
writes raw plus parsed outputs into a new P8C run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.data_prep.prepare_pcen_spectrogram_inputs import (  # noqa: E402
    DEFAULT_PCEN_PARAMETERS,
    GRID_STYLE,
    build_pcen_inputs,
)
from run_prompt_v2_small_pilot import (  # noqa: E402
    PROMPT_VERSION,
    build_user_message,
    call_ollama_generate,
    load_prompt,
    parse_prediction,
    read_clip_duration,
    resolve_all_clip_ids,
    resolve_input_image,
    write_failed_prediction_output,
)


CONFIG_PATH = REPO_ROOT / "configs/experiments/p8c_pcen_grid_v2_full45.yaml"
EVAL_DIR = REPO_ROOT / "outputs/evaluation_sets/ozimops_petersi_v1"
BASELINE_DIR = REPO_ROOT / "outputs/agent_inputs/prompt_v2_full_grid_v2"
INPUT_DIR = REPO_ROOT / "outputs/agent_inputs/p8c_pcen_grid_v2_full45"
RUN_DIR = REPO_ROOT / "outputs/agent_runs/p8c_pcen_grid_v2_qwen3_6_full45"
PROMPT_PATH = REPO_ROOT / "prompts/prompt_v2_bat_strong_label.md"
MODEL_NAME = "qwen3.6:latest"
REQUIRED_OLLAMA_HOST = "http://127.0.0.1:11436"
IMAGE_SUFFIX = "_pcen_grid_v2.png"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_ollama_model(model_name: str, required_host: str) -> str:
    """Confirm the requested model is exposed by the required Ollama endpoint."""

    active_host = os.getenv("OLLAMA_HOST", "").rstrip("/")
    if active_host != required_host:
        raise RuntimeError(
            f"OLLAMA_HOST must be {required_host}, got {active_host or '<unset>'}. "
            "Refusing to use local Ollama or substitute models."
        )
    completed = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    if model_name not in completed.stdout:
        raise RuntimeError(
            f"Required model {model_name!r} was not found at {required_host}.\n"
            f"Available models:\n{completed.stdout}"
        )
    return completed.stdout


def prepare_full45_inputs(*, overwrite_inputs: bool) -> list[dict[str, Any]]:
    """Generate full-45 clean PCEN+grid_v2 images and return manifest rows."""

    if GRID_STYLE != "grid_v2":
        raise RuntimeError(f"Expected grid_v2 PCEN inputs, got {GRID_STYLE!r}")
    clip_ids = resolve_all_clip_ids(EVAL_DIR)
    rows = build_pcen_inputs(
        eval_dir=EVAL_DIR,
        baseline_dir=BASELINE_DIR,
        output_dir=INPUT_DIR,
        clip_ids=clip_ids,
        parameters=DEFAULT_PCEN_PARAMETERS,
        overwrite=overwrite_inputs,
        base_dir=REPO_ROOT,
    )
    return [row.__dict__ for row in rows]


def write_prediction_with_metadata(
    *,
    output_path: Path,
    prediction: dict[str, Any],
    input_image_path: Path,
    clip_duration_seconds: float,
    run_timestamp: str,
    latency_seconds: float,
    attempt_count: int,
    prompt_sha256: str,
    image_sha256: str,
) -> None:
    payload = {
        "clip_id": prediction["clip_id"],
        "prompt_version": PROMPT_VERSION,
        "model_name": MODEL_NAME,
        "backend": "ollama_generate",
        "run_timestamp": run_timestamp,
        "input_condition": "pcen_grid_v2",
        "input_image_path": input_image_path.relative_to(REPO_ROOT).as_posix(),
        "input_image_sha256": image_sha256,
        "prompt_sha256": prompt_sha256,
        "clip_duration_seconds": clip_duration_seconds,
        "latency_seconds": round(latency_seconds, 6),
        "attempt_count": attempt_count,
        "parse_status": "success",
        "events": prediction["events"],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_one_clip(
    *,
    clip_id: str,
    prompt_text: str,
    prompt_sha256: str,
    output_dir: Path,
    raw_response_dir: Path,
    timeout: float,
    num_predict: int,
    retry_count: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_response_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{clip_id}_predictions.json"
    parse_error_path = output_dir / f"{clip_id}_parse_error.txt"
    prediction_path.unlink(missing_ok=True)
    parse_error_path.unlink(missing_ok=True)

    clip_duration = read_clip_duration(EVAL_DIR, clip_id)
    image_path = resolve_input_image(INPUT_DIR, clip_id, IMAGE_SUFFIX)
    user_message = build_user_message(clip_id=clip_id, clip_duration_seconds=clip_duration)
    image_hash = file_sha256(image_path)
    start_time = time.perf_counter()
    raw_text = ""
    last_error = ""
    attempt_count = 0

    for attempt_index in range(retry_count + 1):
        attempt_count = attempt_index + 1
        raw_path = raw_response_dir / (
            f"{clip_id}_raw_response.txt"
            if attempt_index == 0
            else f"{clip_id}_retry{attempt_index}_raw_response.txt"
        )
        try:
            raw_text = call_ollama_generate(
                image_path=image_path,
                system_prompt=prompt_text,
                user_message=user_message,
                model_name=MODEL_NAME,
                timeout=timeout,
                num_predict=num_predict,
            )
            raw_path.write_text(raw_text, encoding="utf-8")
            prediction = parse_prediction(raw_text, expected_clip_id=clip_id)
            latency = time.perf_counter() - start_time
            write_prediction_with_metadata(
                output_path=prediction_path,
                prediction=prediction,
                input_image_path=image_path,
                clip_duration_seconds=clip_duration,
                run_timestamp=datetime.now(timezone.utc).isoformat(),
                latency_seconds=latency,
                attempt_count=attempt_count,
                prompt_sha256=prompt_sha256,
                image_sha256=image_hash,
            )
            return {
                "clip_id": clip_id,
                "parse_status": "success",
                "attempt_count": attempt_count,
                "predicted_event_count": len(prediction["events"]),
                "latency_seconds": round(latency, 6),
                "prediction_path": prediction_path.relative_to(REPO_ROOT).as_posix(),
                "raw_response_path": raw_path.relative_to(REPO_ROOT).as_posix(),
                "parse_error_path": "",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if raw_text and not raw_path.exists():
                raw_path.write_text(raw_text, encoding="utf-8")

    latency = time.perf_counter() - start_time
    parse_error_path.write_text(last_error + "\n", encoding="utf-8")
    write_failed_prediction_output(
        output_path=prediction_path,
        clip_id=clip_id,
        model_name=MODEL_NAME,
        backend="ollama_generate",
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        input_image_path=image_path,
        clip_duration_seconds=clip_duration,
        error_message=last_error,
    )
    return {
        "clip_id": clip_id,
        "parse_status": "failed",
        "attempt_count": attempt_count,
        "predicted_event_count": "",
        "latency_seconds": round(latency, 6),
        "prediction_path": prediction_path.relative_to(REPO_ROOT).as_posix(),
        "raw_response_path": (raw_response_dir / f"{clip_id}_raw_response.txt").relative_to(REPO_ROOT).as_posix(),
        "parse_error_path": parse_error_path.relative_to(REPO_ROOT).as_posix(),
    }


def write_parse_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "clip_id",
        "parse_status",
        "attempt_count",
        "predicted_event_count",
        "latency_seconds",
        "prediction_path",
        "raw_response_path",
        "parse_error_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-inputs-only", action="store_true")
    parser.add_argument("--skip-input-generation", action="store_true")
    parser.add_argument("--overwrite-inputs", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--num-predict", type=int, default=8000)
    parser.add_argument("--retry-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)

    if not args.skip_input_generation:
        rows = prepare_full45_inputs(overwrite_inputs=args.overwrite_inputs)
        print(f"Prepared {len(rows)} PCEN+grid_v2 input image(s): {INPUT_DIR}")

    if args.prepare_inputs_only:
        return

    if not args.skip_model_check:
        model_listing = require_ollama_model(MODEL_NAME, REQUIRED_OLLAMA_HOST)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "ollama_model_list.txt").write_text(model_listing, encoding="utf-8")
        print(f"Confirmed {MODEL_NAME} at {REQUIRED_OLLAMA_HOST}")

    prompt_text = load_prompt(PROMPT_PATH)
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    clip_ids = resolve_all_clip_ids(EVAL_DIR)
    prediction_dir = RUN_DIR / "predictions"
    raw_response_dir = RUN_DIR / "raw_responses"
    rows = []
    for clip_id in clip_ids:
        print(f"Running P8C PCEN+grid_v2 {clip_id} with {MODEL_NAME}...")
        rows.append(
            run_one_clip(
                clip_id=clip_id,
                prompt_text=prompt_text,
                prompt_sha256=prompt_hash,
                output_dir=prediction_dir,
                raw_response_dir=raw_response_dir,
                timeout=args.timeout,
                num_predict=args.num_predict,
                retry_count=args.retry_count,
            )
        )
    write_parse_summary(rows, RUN_DIR / "p8c_parse_summary.csv")
    success_count = sum(1 for row in rows if row["parse_status"] == "success")
    print(f"P8C complete: {success_count}/{len(rows)} parsed successfully.")


if __name__ == "__main__":
    main()
