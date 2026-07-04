"""Convert BatDetect2 proposal JSON into evaluator-compatible predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import soundfile as sf


DEFAULT_PROPOSAL_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6"
)
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/agent_runs/p6_batdetect2_proposal_only_representative6/predictions"
)
DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def prediction_output_path(output_dir: Path, clip_id: str) -> Path:
    """Return the stable proposal-only prediction filename."""
    return output_dir / f"{clip_id}_prediction.json"


def finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def convert_proposal_event(
    proposal: dict[str, Any],
    *,
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Map one proposal to Prompt V2 event fields and preserve detector metadata."""
    start = finite_float(proposal["start_time_seconds"], "start_time_seconds")
    original_end = finite_float(proposal["end_time_seconds"], "end_time_seconds")
    low = finite_float(proposal["low_frequency_hz"], "low_frequency_hz")
    high = finite_float(proposal["high_frequency_hz"], "high_frequency_hz")
    det_prob = finite_float(proposal["det_prob"], "det_prob")
    class_prob = finite_float(proposal["class_prob"], "class_prob")
    if start < 0 or start >= original_end:
        raise ValueError("Proposal must satisfy 0 <= start_time < end_time")
    if low < 0 or low >= high:
        raise ValueError("Proposal must satisfy 0 <= low_frequency < high_frequency")
    if not 0 <= det_prob <= 1 or not 0 <= class_prob <= 1:
        raise ValueError("Proposal probabilities must be between 0 and 1")
    if start >= clip_duration_seconds:
        raise ValueError("Proposal starts outside the source clip")

    end = min(original_end, clip_duration_seconds)
    clipped = end != original_end
    review_reason = (
        "BatDetect2 proposal exceeded the clip end and was clipped to the "
        "physical audio boundary."
        if clipped
        else ""
    )
    proposal_id = str(proposal.get("proposal_id") or "")
    if not proposal_id:
        raise ValueError("Proposal is missing proposal_id")
    return {
        "event_id": proposal_id,
        "start_time_seconds": start,
        "end_time_seconds": end,
        "low_frequency_hz": low,
        "high_frequency_hz": high,
        "label": "bat_call",
        "confidence": det_prob,
        "evidence": "Candidate event supplied by BatDetect2 proposal metadata.",
        "human_review_needed": clipped,
        "review_reason": review_reason,
        "proposal_id": proposal_id,
        "det_prob": det_prob,
        "class_prob": class_prob,
        "original_label": str(proposal.get("label") or ""),
        "proposal_source": str(
            proposal.get("source") or proposal.get("proposal_source") or "batdetect2"
        ),
        "original_start_time_seconds": start,
        "original_end_time_seconds": original_end,
        "clipped_to_clip_bounds": clipped,
    }


def convert_proposal_payload(
    payload: dict[str, Any],
    *,
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Convert one clip payload while retaining proposal-source metadata."""
    clip_id = str(payload.get("clip_id") or "")
    if not clip_id:
        raise ValueError("Proposal payload is missing clip_id")
    proposals = payload.get("events")
    if not isinstance(proposals, list):
        raise ValueError("Proposal payload events must be a list")
    events = [
        convert_proposal_event(
            proposal,
            clip_duration_seconds=clip_duration_seconds,
        )
        for proposal in proposals
    ]
    return {
        "clip_id": clip_id,
        "model_name": "batdetect2",
        "backend": "proposal_conversion",
        "proposal_source": str(payload.get("proposal_source") or "batdetect2"),
        "proposal_threshold": payload.get("proposal_threshold"),
        "clip_duration_seconds": clip_duration_seconds,
        "events": events,
    }


def convert_files(
    *,
    proposal_dir: Path,
    eval_dir: Path,
    output_dir: Path,
    clip_ids: list[str],
    overwrite: bool,
) -> list[Path]:
    """Convert selected proposal files using WAV duration as the physical bound."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for clip_id in clip_ids:
        proposal_path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
        audio_path = eval_dir / "audio" / f"{clip_id}.wav"
        if not proposal_path.is_file():
            raise FileNotFoundError(f"Proposal JSON not found: {proposal_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Evaluation WAV not found: {audio_path}")
        output_path = prediction_output_path(output_dir, clip_id)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Prediction output exists: {output_path}. Use --overwrite.")
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        if payload.get("clip_id") != clip_id:
            raise ValueError(
                f"Proposal clip_id {payload.get('clip_id')!r} does not match {clip_id!r}"
            )
        converted = convert_proposal_payload(
            payload,
            clip_duration_seconds=float(sf.info(audio_path).duration),
        )
        output_path.write_text(
            json.dumps(converted, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        paths.append(output_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-list", default=",".join(DEFAULT_CLIP_IDS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = convert_files(
        proposal_dir=args.proposal_dir,
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        clip_ids=parse_clip_ids(args.clip_list),
        overwrite=args.overwrite,
    )
    print(f"Created {len(paths)} BatDetect2 proposal-only prediction file(s):")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

