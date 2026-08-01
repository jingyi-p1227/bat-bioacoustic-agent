"""Run Prompt V2 independently on clean P6 spectrogram tiles.

The runner reads only the prompt, tile manifest, and clean tile images. It does
not read ground truth, overlays, diagnostics, or previous model predictions.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.maintenance.merge_tiled_predictions import (
    DEFAULT_MANIFEST,
    DEFAULT_TILE_SETTING,
    ManifestTile,
    group_tiles_by_clip,
    load_manifest_tiles,
)
from run_prompt_v2_small_pilot import (
    PROMPT_VERSION,
    call_ollama_generate,
    load_prompt,
    ollama_host,
    parse_prediction,
)


DEFAULT_MODEL_NAME = "qwen3.6:latest"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/agent_runs/p6_tiled_qwen3_6_tile_0p5_overlap_0p1/predictions_by_tile"
)
DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")


@dataclass(frozen=True)
class TileRunResult:
    """Outcome and artifact paths for one tile-level model request."""

    clip_id: str
    tile_id: str
    parse_status: str
    predicted_event_count: int | None
    prediction_path: Path
    raw_response_path: Path
    parse_error_path: Path | None


def parse_clip_ids(value: str) -> list[str]:
    """Parse a stable, de-duplicated comma-separated clip list."""
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def select_tiles(
    tiles: list[ManifestTile], clip_ids: list[str]
) -> list[tuple[ManifestTile, float]]:
    """Return selected tiles with their original full-clip duration."""
    groups = group_tiles_by_clip(tiles)
    missing = [clip_id for clip_id in clip_ids if clip_id not in groups]
    if missing:
        raise ValueError(f"Clip ids missing from tile manifest: {', '.join(missing)}")

    selected: list[tuple[ManifestTile, float]] = []
    for clip_id in clip_ids:
        clip_tiles = groups[clip_id]
        clip_duration = max(tile.tile_end_seconds for tile in clip_tiles)
        selected.extend((tile, clip_duration) for tile in clip_tiles)
    return selected


def tile_artifact_paths(
    output_dir: Path,
    tile: ManifestTile,
    raw_response_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Return prediction, raw response, and parse-error paths for one tile."""
    stem = tile.image_path.stem
    raw_dir = output_dir if raw_response_dir is None else raw_response_dir
    return (
        output_dir / f"{stem}.json",
        raw_dir / f"{stem}_raw_response.txt",
        output_dir / f"{stem}_parse_error.txt",
    )


def build_tile_user_message(
    *,
    tile: ManifestTile,
    clip_duration_seconds: float,
) -> str:
    """Build an original-coordinate annotation request for one clean tile."""
    context = {
        "clip_id": tile.clip_id,
        "clip_duration_seconds": clip_duration_seconds,
        "tile_id": tile.tile_id,
        "tile_start_seconds": tile.tile_start_seconds,
        "tile_end_seconds": tile.tile_end_seconds,
        "tile_setting": tile.tile_setting,
        "time_coordinate_frame": "original_clip",
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
    }
    return (
        "/no_think\n"
        "Annotate the attached clean spectrogram tile according to the system prompt.\n"
        "Annotate only bat calls visibly present inside this shown tile.\n"
        "The x-axis already uses original clip coordinates. Return those displayed "
        "times directly; do not subtract tile_start_seconds and do not restart time at 0.\n"
        "Do not infer calls outside the displayed tile window.\n\n"
        "Runtime context:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Do not explain your reasoning. Return the valid JSON object immediately."
    )


def available_ollama_models(timeout: float = 30.0) -> list[str]:
    """Return model names exposed by the configured OLLAMA_HOST."""
    request = urllib.request.Request(f"{ollama_host()}/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return sorted(
        str(row.get("name") or row.get("model"))
        for row in payload.get("models", [])
        if row.get("name") or row.get("model")
    )


def require_model(model_name: str, timeout: float = 30.0) -> list[str]:
    """Stop rather than silently substituting when the requested model is absent."""
    models = available_ollama_models(timeout=timeout)
    if model_name not in models:
        raise RuntimeError(
            f"Required Ollama model {model_name!r} is unavailable at {ollama_host()}. "
            f"Available models: {', '.join(models) or '(none)'}"
        )
    return models


def _write_tile_prediction(
    *,
    output_path: Path,
    prediction: dict[str, Any],
    tile: ManifestTile,
    clip_duration_seconds: float,
    model_name: str,
    backend: str,
    parse_status: str,
    error_message: str = "",
) -> None:
    payload = {
        "clip_id": tile.clip_id,
        "tile_id": tile.tile_id,
        "prompt_version": PROMPT_VERSION,
        "model_name": model_name,
        "backend": backend,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_image_path": tile.image_path.as_posix(),
        "clip_duration_seconds": clip_duration_seconds,
        "tile_start_seconds": tile.tile_start_seconds,
        "tile_end_seconds": tile.tile_end_seconds,
        "tile_setting": tile.tile_setting,
        "time_coordinate_frame": "original_clip",
        "parse_status": parse_status,
        "error": error_message,
        "events": prediction.get("events", []),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_tile(
    *,
    tile: ManifestTile,
    clip_duration_seconds: float,
    prompt_text: str,
    output_dir: Path,
    model_name: str,
    timeout: float,
    num_predict: int,
    overwrite: bool,
    raw_response_dir: Path | None = None,
) -> TileRunResult:
    """Run one tile and preserve both valid and invalid model responses."""
    if not tile.image_path.is_file():
        raise FileNotFoundError(f"Clean tile image not found: {tile.image_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path, raw_path, error_path = tile_artifact_paths(
        output_dir,
        tile,
        raw_response_dir,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(
            f"Tile prediction already exists: {prediction_path}. Use --overwrite."
        )
    if overwrite:
        raw_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)

    try:
        raw_text = call_ollama_generate(
            image_path=tile.image_path,
            system_prompt=prompt_text,
            user_message=build_tile_user_message(
                tile=tile,
                clip_duration_seconds=clip_duration_seconds,
            ),
            model_name=model_name,
            timeout=timeout,
            num_predict=num_predict,
        )
        raw_path.write_text(raw_text, encoding="utf-8")
        prediction = parse_prediction(raw_text, expected_clip_id=tile.clip_id)
        _write_tile_prediction(
            output_path=prediction_path,
            prediction=prediction,
            tile=tile,
            clip_duration_seconds=clip_duration_seconds,
            model_name=model_name,
            backend="ollama_generate",
            parse_status="success",
        )
        return TileRunResult(
            clip_id=tile.clip_id,
            tile_id=tile.tile_id,
            parse_status="success",
            predicted_event_count=len(prediction["events"]),
            prediction_path=prediction_path,
            raw_response_path=raw_path,
            parse_error_path=None,
        )
    except Exception as exc:
        if not raw_path.exists():
            raw_path.write_text("", encoding="utf-8")
        message = f"{type(exc).__name__}: {exc}"
        error_path.write_text(message + "\n", encoding="utf-8")
        _write_tile_prediction(
            output_path=prediction_path,
            prediction={"events": []},
            tile=tile,
            clip_duration_seconds=clip_duration_seconds,
            model_name=model_name,
            backend="ollama_generate",
            parse_status="failed",
            error_message=message,
        )
        return TileRunResult(
            clip_id=tile.clip_id,
            tile_id=tile.tile_id,
            parse_status="failed",
            predicted_event_count=None,
            prediction_path=prediction_path,
            raw_response_path=raw_path,
            parse_error_path=error_path,
        )


def print_summary(results: list[TileRunResult]) -> None:
    """Print concise per-tile parse and event counts."""
    print("clip_id | tile_id | parse_status | predicted_events")
    print("--------+---------+--------------+-----------------")
    for result in results:
        count = "" if result.predicted_event_count is None else result.predicted_event_count
        print(
            f"{result.clip_id} | {result.tile_id} | "
            f"{result.parse_status} | {count}"
        )
    successes = sum(result.parse_status == "success" for result in results)
    print(
        f"Tiles: {len(results)} | parse successes: {successes} | "
        f"parse failures: {len(results) - successes}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prompt V2 on clean P6 spectrogram tiles."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tile-setting", default=DEFAULT_TILE_SETTING)
    parser.add_argument(
        "--clip-list",
        default=",".join(DEFAULT_CLIP_IDS),
    )
    parser.add_argument("--prompt", type=Path, default=Path("prompts/prompt_v2_bat_strong_label.md"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--raw-response-dir",
        type=Path,
        default=None,
        help="Optional separate directory for raw model responses.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=8000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"OLLAMA_HOST={ollama_host()}")
    require_model(args.model_name)
    print(f"Confirmed model: {args.model_name}")
    tiles = load_manifest_tiles(
        args.manifest,
        tile_setting=args.tile_setting,
    )
    selected = select_tiles(tiles, parse_clip_ids(args.clip_list))
    prompt_text = load_prompt(args.prompt)
    results: list[TileRunResult] = []
    for index, (tile, clip_duration) in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] Running {tile.clip_id} {tile.tile_id} "
            f"with ollama_generate/{args.model_name}...",
            flush=True,
        )
        results.append(
            run_tile(
                tile=tile,
                clip_duration_seconds=clip_duration,
                prompt_text=prompt_text,
                output_dir=args.output_dir,
                model_name=args.model_name,
                timeout=args.timeout,
                num_predict=args.num_predict,
                overwrite=args.overwrite,
                raw_response_dir=args.raw_response_dir,
            )
        )
    print_summary(results)


if __name__ == "__main__":
    main()
