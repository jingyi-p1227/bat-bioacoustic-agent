"""Run isolated BatDetect2 inference for proposal-tool preparation.

BatDetect2 is intentionally not a project dependency. Run this script through
an isolated uvx environment, for example:

    uvx --from batdetect2 python run_batdetect2_proposal_inference.py ...
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6/raw_batdetect2"
)
DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def resolve_audio_paths(eval_dir: Path, clip_ids: list[str]) -> list[tuple[str, Path]]:
    """Resolve selected evaluation WAV files without reading ground truth."""
    resolved: list[tuple[str, Path]] = []
    for clip_id in clip_ids:
        path = eval_dir / "audio" / f"{clip_id}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation WAV not found: {path}")
        resolved.append((clip_id, path))
    return resolved


def run_inference(
    *,
    eval_dir: Path,
    output_dir: Path,
    clip_ids: list[str],
    device_name: str,
    detection_threshold: float,
    chunk_size: float,
    overwrite: bool,
) -> dict[str, Any]:
    """Run the official BatDetect2 API and preserve its native JSON/CSV."""
    import torch
    from batdetect2 import api
    from batdetect2.detector.parameters import DEFAULT_MODEL_PATH
    from batdetect2.utils.detector_utils import save_results_to_file

    audio_paths = resolve_audio_paths(eval_dir, clip_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"{clip_id}.json" for clip_id, _ in audio_paths]
    existing = [path for path in existing if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Raw BatDetect2 output exists: {existing[0]}. Use --overwrite."
        )

    device = torch.device(device_name)
    model_started = time.perf_counter()
    model, params = api.load_model(DEFAULT_MODEL_PATH, device=device)
    model_load_seconds = time.perf_counter() - model_started
    config = api.get_config(
        **{
            **params,
            "detection_threshold": detection_threshold,
            "time_expansion": 1,
            "spec_slices": False,
            "chunk_size": chunk_size,
            "spec_features": False,
            "cnn_features": False,
            "quiet": True,
        }
    )

    file_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for clip_id, audio_path in audio_paths:
        output_stem = output_dir / clip_id
        file_started = time.perf_counter()
        try:
            result = api.process_file(str(audio_path), model, config=config)
            save_results_to_file(result, str(output_stem))
            file_results.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(audio_path),
                    "status": "success",
                    "runtime_seconds": time.perf_counter() - file_started,
                    "prediction_count": len(result["pred_dict"].get("annotation", [])),
                    "raw_json_path": str(output_stem) + ".json",
                    "raw_csv_path": str(output_stem) + ".csv",
                }
            )
        except Exception as exc:
            file_results.append(
                {
                    "clip_id": clip_id,
                    "audio_path": str(audio_path),
                    "status": "failed",
                    "runtime_seconds": time.perf_counter() - file_started,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    successes = [row for row in file_results if row["status"] == "success"]
    failures = [row for row in file_results if row["status"] == "failed"]
    total_runtime_seconds = time.perf_counter() - started
    run_metadata: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "batdetect2_version": metadata.version("batdetect2"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "model_path": str(DEFAULT_MODEL_PATH),
        "model_name": "Net2DFast_UK_same",
        "model_load_seconds": model_load_seconds,
        "detection_threshold": detection_threshold,
        "chunk_size": chunk_size,
        "total_files": len(file_results),
        "successful_files": len(successes),
        "failed_files": len(failures),
        "total_runtime_seconds": total_runtime_seconds,
        "files": file_results,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-list", default=",".join(DEFAULT_CLIP_IDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detection-threshold", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_inference(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        clip_ids=parse_clip_ids(args.clip_list),
        device_name=args.device,
        detection_threshold=args.detection_threshold,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "batdetect2_version",
                    "device",
                    "total_files",
                    "successful_files",
                    "failed_files",
                    "total_runtime_seconds",
                )
            },
            indent=2,
        )
    )
    return 0 if result["failed_files"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

