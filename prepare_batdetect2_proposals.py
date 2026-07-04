"""Convert native BatDetect2 JSON into stable clip-level proposal files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RAW_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6/raw_batdetect2"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6"
)
DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")
DEFAULT_MIN_DET_PROB = 0.3


@dataclass(frozen=True)
class ProposalSummaryRow:
    clip_id: str
    proposal_count: int
    mean_det_prob: float | None
    min_start_time: float | None
    max_end_time: float | None
    notes: str


SUMMARY_FIELDS = tuple(ProposalSummaryRow.__dataclass_fields__)


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def convert_raw_event(raw: dict[str, Any], proposal_id: str) -> dict[str, Any]:
    """Validate and convert one native BatDetect2 annotation row."""
    start = finite_float(raw["start_time"], "start_time")
    end = finite_float(raw["end_time"], "end_time")
    low = finite_float(raw["low_freq"], "low_freq")
    high = finite_float(raw["high_freq"], "high_freq")
    det_prob = finite_float(raw["det_prob"], "det_prob")
    class_prob = finite_float(raw["class_prob"], "class_prob")
    if start < 0 or start >= end:
        raise ValueError("BatDetect2 event must satisfy 0 <= start_time < end_time")
    if low < 0 or low >= high:
        raise ValueError("BatDetect2 event must satisfy 0 <= low_freq < high_freq")
    if not 0 <= det_prob <= 1 or not 0 <= class_prob <= 1:
        raise ValueError("BatDetect2 probabilities must be between 0 and 1")
    return {
        "proposal_id": proposal_id,
        "start_time_seconds": start,
        "end_time_seconds": end,
        "low_frequency_hz": low,
        "high_frequency_hz": high,
        "det_prob": det_prob,
        "class_prob": class_prob,
        "label": str(raw.get("class") or ""),
        "source": "batdetect2",
    }


def convert_raw_payload(
    *,
    clip_id: str,
    payload: dict[str, Any],
    min_det_prob: float,
) -> tuple[dict[str, Any], ProposalSummaryRow]:
    """Filter, validate, sort, and number proposals for one clip."""
    if not 0 <= min_det_prob <= 1:
        raise ValueError("min_det_prob must be between 0 and 1")
    raw_events = payload.get("annotation")
    if not isinstance(raw_events, list):
        raise ValueError("BatDetect2 payload annotation must be a list")

    converted: list[dict[str, Any]] = []
    invalid_count = 0
    below_threshold_count = 0
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            invalid_count += 1
            continue
        try:
            if finite_float(raw_event.get("det_prob"), "det_prob") < min_det_prob:
                below_threshold_count += 1
                continue
            converted.append(convert_raw_event(raw_event, "pending"))
        except (KeyError, ValueError):
            invalid_count += 1

    converted.sort(
        key=lambda event: (
            event["start_time_seconds"],
            event["end_time_seconds"],
            -event["det_prob"],
            event["low_frequency_hz"],
        )
    )
    for index, event in enumerate(converted, start=1):
        event["proposal_id"] = f"bd2_{index:03d}"

    notes = (
        f"det_prob >= {min_det_prob}; filtered_below_threshold="
        f"{below_threshold_count}; invalid_dropped={invalid_count}; "
        "UK model taxonomy labels are metadata, not Australian species truth."
    )
    output = {
        "clip_id": clip_id,
        "proposal_source": "batdetect2",
        "proposal_threshold": min_det_prob,
        "events": converted,
    }
    probabilities = [event["det_prob"] for event in converted]
    starts = [event["start_time_seconds"] for event in converted]
    ends = [event["end_time_seconds"] for event in converted]
    summary = ProposalSummaryRow(
        clip_id=clip_id,
        proposal_count=len(converted),
        mean_det_prob=(round(sum(probabilities) / len(probabilities), 6) if probabilities else None),
        min_start_time=(min(starts) if starts else None),
        max_end_time=(max(ends) if ends else None),
        notes=notes,
    )
    return output, summary


def write_proposal_summary(rows: list[ProposalSummaryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def prepare_proposals(
    *,
    raw_dir: Path,
    output_dir: Path,
    clip_ids: list[str],
    min_det_prob: float,
    overwrite: bool,
) -> list[ProposalSummaryRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ProposalSummaryRow] = []
    for clip_id in clip_ids:
        raw_path = raw_dir / f"{clip_id}.json"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw BatDetect2 JSON not found: {raw_path}")
        output_path = output_dir / f"{clip_id}_batdetect2_proposals.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Proposal output exists: {output_path}. Use --overwrite.")
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        output, summary = convert_raw_payload(
            clip_id=clip_id,
            payload=payload,
            min_det_prob=min_det_prob,
        )
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)
    write_proposal_summary(summaries, output_dir / "proposal_summary.csv")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-list", default=",".join(DEFAULT_CLIP_IDS))
    parser.add_argument("--min-det-prob", type=float, default=DEFAULT_MIN_DET_PROB)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_proposals(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        clip_ids=parse_clip_ids(args.clip_list),
        min_det_prob=args.min_det_prob,
        overwrite=args.overwrite,
    )
    print("clip_id | proposals | mean_det_prob | min_start | max_end")
    for row in rows:
        print(
            f"{row.clip_id} | {row.proposal_count} | {row.mean_det_prob} | "
            f"{row.min_start_time} | {row.max_end_time}"
        )


if __name__ == "__main__":
    main()

