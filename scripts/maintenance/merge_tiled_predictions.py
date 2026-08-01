"""Merge tile-level Prompt V2 predictions into clip-level prediction JSON.

This is an offline post-processing utility. It does not run a model, read
ground truth, or change the frozen evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("outputs/agent_inputs/p6_tiled_spectrograms/tile_manifest.csv")
DEFAULT_TILE_SETTING = "tile_0p5_overlap_0p1"
DEFAULT_NMS_IOU_THRESHOLD = 0.5
TOP_LEVEL_METADATA_FIELDS = (
    "prompt_version",
    "model_name",
    "backend",
    "run_timestamp",
)


@dataclass(frozen=True)
class ManifestTile:
    """One tile selected from the P6 input manifest."""

    clip_id: str
    tile_id: str
    tile_setting: str
    tile_start_seconds: float
    tile_end_seconds: float
    image_path: Path


@dataclass(frozen=True)
class MergeSummaryRow:
    """Per-clip accounting for tile prediction merging."""

    clip_id: str
    tile_setting: str
    tile_prediction_files_found: int
    raw_tile_events: int
    merged_events: int
    duplicates_removed: int
    invalid_events_dropped: int
    out_of_tile_events_dropped: int
    notes: str


@dataclass
class ClipMergeResult:
    """Merged payload plus summary and file-level metadata."""

    payload: dict[str, Any] | None
    summary: MergeSummaryRow


SUMMARY_FIELDS = tuple(MergeSummaryRow.__dataclass_fields__)


def tile_setting_from_id(tile_id: str) -> str:
    """Extract the named setting prefix from a manifest tile id."""
    marker = "_tile_"
    if marker not in tile_id:
        raise ValueError(f"Tile id does not contain {marker!r}: {tile_id}")
    return tile_id.rsplit(marker, 1)[0]


def load_manifest_tiles(
    manifest_path: Path,
    *,
    tile_setting: str,
    project_root: Path | None = None,
) -> list[ManifestTile]:
    """Load one tile setting from the portable P6 manifest."""
    project_root = Path.cwd() if project_root is None else project_root
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    tiles: list[ManifestTile] = []
    for row in rows:
        row_setting = tile_setting_from_id(row["tile_id"])
        if row_setting != tile_setting:
            continue
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = project_root / image_path
        tiles.append(
            ManifestTile(
                clip_id=row["clip_id"],
                tile_id=row["tile_id"],
                tile_setting=row_setting,
                tile_start_seconds=float(row["tile_start_seconds"]),
                tile_end_seconds=float(row["tile_end_seconds"]),
                image_path=image_path,
            )
        )
    if not tiles:
        raise ValueError(
            f"No manifest rows found for tile setting {tile_setting!r}"
        )
    return tiles


def group_tiles_by_clip(
    tiles: list[ManifestTile],
) -> dict[str, list[ManifestTile]]:
    """Group manifest tiles by clip in stable temporal order."""
    groups: dict[str, list[ManifestTile]] = {}
    for tile in tiles:
        groups.setdefault(tile.clip_id, []).append(tile)
    for group in groups.values():
        group.sort(key=lambda tile: (tile.tile_start_seconds, tile.tile_end_seconds))
    return groups


def select_clip_groups(
    groups: dict[str, list[ManifestTile]],
    clip_ids: list[str] | None,
) -> dict[str, list[ManifestTile]]:
    """Optionally restrict merge processing to an explicit clip list."""
    if clip_ids is None:
        return groups
    missing = [clip_id for clip_id in clip_ids if clip_id not in groups]
    if missing:
        raise ValueError(f"Clip ids missing from tile manifest: {', '.join(missing)}")
    return {clip_id: groups[clip_id] for clip_id in clip_ids}


def parse_clip_ids(value: str) -> list[str]:
    """Parse a stable, de-duplicated comma-separated clip list."""
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def prediction_path_for_tile(prediction_dir: Path, tile: ManifestTile) -> Path:
    """Resolve the expected JSON filename from the source tile image stem."""
    return prediction_dir / f"{tile.image_path.stem}.json"


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def normalize_tile_event(
    raw_event: dict[str, Any],
    *,
    tile: ManifestTile,
    clip_duration_seconds: float,
    event_index: int,
    drop_fully_outside_tile: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate geometry, clip to the source clip, and attach provenance."""
    try:
        start = _finite_float(raw_event["start_time_seconds"], "start_time_seconds")
        end = _finite_float(raw_event["end_time_seconds"], "end_time_seconds")
        low = _finite_float(raw_event["low_frequency_hz"], "low_frequency_hz")
        high = _finite_float(raw_event["high_frequency_hz"], "high_frequency_hz")
    except (KeyError, ValueError):
        return None, "invalid_geometry"
    if start >= end or low >= high:
        return None, "invalid_geometry"

    start = max(0.0, start)
    end = min(clip_duration_seconds, end)
    if start >= end:
        return None, "invalid_geometry"

    fully_outside_tile = (
        end <= tile.tile_start_seconds or start >= tile.tile_end_seconds
    )
    if fully_outside_tile and drop_fully_outside_tile:
        return None, "out_of_tile"

    crosses_left = start < tile.tile_start_seconds
    crosses_right = end > tile.tile_end_seconds
    try:
        confidence = _finite_float(raw_event.get("confidence", 0.0), "confidence")
    except ValueError:
        confidence = 0.0

    source_event_id = str(
        raw_event.get("event_id") or f"{tile.tile_id}_event_{event_index:03d}"
    )
    event = dict(raw_event)
    event.update(
        {
            "event_id": source_event_id,
            "source_event_id": source_event_id,
            "start_time_seconds": start,
            "end_time_seconds": end,
            "low_frequency_hz": low,
            "high_frequency_hz": high,
            "label": str(raw_event.get("label") or ""),
            "confidence": confidence,
            "evidence": str(raw_event.get("evidence") or ""),
            "human_review_needed": bool(
                raw_event.get("human_review_needed", False)
            ),
            "review_reason": str(raw_event.get("review_reason") or ""),
            "source_tile_id": tile.tile_id,
            "source_tile_start_seconds": tile.tile_start_seconds,
            "source_tile_end_seconds": tile.tile_end_seconds,
            "source_tile_setting": tile.tile_setting,
            "crosses_source_tile_boundary": crosses_left or crosses_right,
            "source_tile_boundary_sides": [
                side
                for side, crosses in (("left", crosses_left), ("right", crosses_right))
                if crosses
            ],
            "merged_duplicate_provenance": [],
        }
    )
    return event, None


def box_iou(event_a: dict[str, Any], event_b: dict[str, Any]) -> float:
    """Return two-dimensional time-frequency IoU for prediction boxes."""
    time_overlap = max(
        0.0,
        min(event_a["end_time_seconds"], event_b["end_time_seconds"])
        - max(event_a["start_time_seconds"], event_b["start_time_seconds"]),
    )
    frequency_overlap = max(
        0.0,
        min(event_a["high_frequency_hz"], event_b["high_frequency_hz"])
        - max(event_a["low_frequency_hz"], event_b["low_frequency_hz"]),
    )
    intersection = time_overlap * frequency_overlap
    area_a = (
        event_a["end_time_seconds"] - event_a["start_time_seconds"]
    ) * (event_a["high_frequency_hz"] - event_a["low_frequency_hz"])
    area_b = (
        event_b["end_time_seconds"] - event_b["start_time_seconds"]
    ) * (event_b["high_frequency_hz"] - event_b["low_frequency_hz"])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress_duplicate_events(
    events: list[dict[str, Any]],
    *,
    iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
) -> tuple[list[dict[str, Any]], int]:
    """Apply confidence-ordered box-IoU NMS and retain duplicate provenance."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    ranked = sorted(
        enumerate(events),
        key=lambda item: (-float(item[1].get("confidence", 0.0)), item[0]),
    )
    kept: list[dict[str, Any]] = []
    duplicates_removed = 0
    for _, candidate in ranked:
        duplicate_of: dict[str, Any] | None = None
        duplicate_iou = 0.0
        for selected in kept:
            score = box_iou(candidate, selected)
            if score >= iou_threshold:
                duplicate_of = selected
                duplicate_iou = score
                break
        if duplicate_of is None:
            kept.append(candidate)
            continue

        duplicates_removed += 1
        duplicate_of["merged_duplicate_provenance"].append(
            {
                "source_event_id": candidate["source_event_id"],
                "source_tile_id": candidate["source_tile_id"],
                "source_tile_start_seconds": candidate[
                    "source_tile_start_seconds"
                ],
                "source_tile_end_seconds": candidate["source_tile_end_seconds"],
                "confidence": candidate["confidence"],
                "box_iou_with_kept_event": duplicate_iou,
            }
        )

    kept.sort(
        key=lambda event: (
            event["start_time_seconds"],
            event["end_time_seconds"],
            -event["confidence"],
        )
    )
    return kept, duplicates_removed


def _load_prediction_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("top-level prediction JSON must be an object")
    if not isinstance(payload.get("events"), list):
        raise ValueError("prediction JSON events must be a list")
    return payload


def merge_clip_predictions(
    *,
    clip_id: str,
    tiles: list[ManifestTile],
    prediction_dir: Path,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    drop_fully_outside_tile: bool = True,
) -> ClipMergeResult:
    """Merge all available tile files for one original clip."""
    clip_duration_seconds = max(tile.tile_end_seconds for tile in tiles)
    found_files = 0
    valid_payloads = 0
    raw_event_count = 0
    invalid_events_dropped = 0
    out_of_tile_events_dropped = 0
    normalized_events: list[dict[str, Any]] = []
    source_files: list[str] = []
    metadata: dict[str, Any] = {}
    notes: list[str] = []

    for tile in tiles:
        prediction_path = prediction_path_for_tile(prediction_dir, tile)
        if not prediction_path.is_file():
            continue
        found_files += 1
        source_files.append(prediction_path.name)
        try:
            payload = _load_prediction_payload(prediction_path)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            notes.append(f"{prediction_path.name}: {exc}")
            continue
        if payload.get("clip_id") != clip_id:
            notes.append(
                f"{prediction_path.name}: clip_id {payload.get('clip_id')!r} "
                f"does not match {clip_id!r}"
            )
            continue
        valid_payloads += 1
        for field in TOP_LEVEL_METADATA_FIELDS:
            if field not in metadata and payload.get(field) is not None:
                metadata[field] = payload[field]

        raw_events = payload["events"]
        raw_event_count += len(raw_events)
        for event_index, raw_event in enumerate(raw_events, start=1):
            if not isinstance(raw_event, dict):
                invalid_events_dropped += 1
                continue
            event, rejection_reason = normalize_tile_event(
                raw_event,
                tile=tile,
                clip_duration_seconds=clip_duration_seconds,
                event_index=event_index,
                drop_fully_outside_tile=drop_fully_outside_tile,
            )
            if rejection_reason == "invalid_geometry":
                invalid_events_dropped += 1
            elif rejection_reason == "out_of_tile":
                out_of_tile_events_dropped += 1
            elif event is not None:
                normalized_events.append(event)

    missing_count = len(tiles) - found_files
    if missing_count:
        notes.append(f"{missing_count} expected tile prediction file(s) missing")

    merged_events, duplicates_removed = suppress_duplicate_events(
        normalized_events,
        iou_threshold=nms_iou_threshold,
    )
    for merged_index, event in enumerate(merged_events, start=1):
        event["event_id"] = f"{clip_id}_merged_{merged_index:03d}"

    payload: dict[str, Any] | None = None
    if valid_payloads:
        payload = {
            "clip_id": clip_id,
            **metadata,
            "clip_duration_seconds": clip_duration_seconds,
            "input_image_paths": source_files,
            "events": merged_events,
            "merge_metadata": {
                "tile_setting": tiles[0].tile_setting,
                "nms_type": "confidence_ordered_box_iou",
                "nms_box_iou_threshold": nms_iou_threshold,
                "drop_fully_outside_tile": drop_fully_outside_tile,
                "tile_prediction_files_expected": len(tiles),
                "tile_prediction_files_found": found_files,
                "duplicates_removed": duplicates_removed,
            },
        }
    elif found_files:
        notes.append("No valid tile prediction payloads were available")

    summary = MergeSummaryRow(
        clip_id=clip_id,
        tile_setting=tiles[0].tile_setting,
        tile_prediction_files_found=found_files,
        raw_tile_events=raw_event_count,
        merged_events=len(merged_events),
        duplicates_removed=duplicates_removed,
        invalid_events_dropped=invalid_events_dropped,
        out_of_tile_events_dropped=out_of_tile_events_dropped,
        notes="; ".join(notes),
    )
    return ClipMergeResult(payload=payload, summary=summary)


def write_merged_payload(
    payload: dict[str, Any], output_path: Path, *, overwrite: bool
) -> None:
    """Write one evaluator-compatible clip-level JSON."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Merged prediction already exists: {output_path}. Use --overwrite."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_merge_summary(rows: list[MergeSummaryRow], output_path: Path) -> None:
    """Write deterministic per-clip merge accounting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def run_merge(
    *,
    manifest_path: Path,
    prediction_dir: Path,
    output_dir: Path,
    summary_path: Path,
    tile_setting: str,
    nms_iou_threshold: float,
    drop_fully_outside_tile: bool,
    overwrite: bool,
    project_root: Path | None = None,
    clip_ids: list[str] | None = None,
) -> list[MergeSummaryRow]:
    """Merge every clip represented by one tile setting."""
    tiles = load_manifest_tiles(
        manifest_path,
        tile_setting=tile_setting,
        project_root=project_root,
    )
    groups = select_clip_groups(group_tiles_by_clip(tiles), clip_ids)
    summaries: list[MergeSummaryRow] = []
    for clip_id, clip_tiles in sorted(groups.items()):
        result = merge_clip_predictions(
            clip_id=clip_id,
            tiles=clip_tiles,
            prediction_dir=prediction_dir,
            nms_iou_threshold=nms_iou_threshold,
            drop_fully_outside_tile=drop_fully_outside_tile,
        )
        if result.payload is not None:
            write_merged_payload(
                result.payload,
                output_dir / f"{clip_id}_prediction.json",
                overwrite=overwrite,
            )
        summaries.append(result.summary)
    write_merge_summary(summaries, summary_path)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge tile-level Prompt V2 predictions into clip-level JSON."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/../merge_summary.csv.",
    )
    parser.add_argument(
        "--tile-setting",
        default=DEFAULT_TILE_SETTING,
    )
    parser.add_argument(
        "--clip-list",
        default=None,
        help="Optional comma-separated clip ids to merge.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=DEFAULT_NMS_IOU_THRESHOLD,
    )
    parser.add_argument(
        "--keep-fully-outside-tile",
        action="store_true",
        help="Keep geometrically valid events fully outside their source tile.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = (
        args.summary_file
        if args.summary_file is not None
        else args.output_dir.parent / "merge_summary.csv"
    )
    summaries = run_merge(
        manifest_path=args.manifest,
        prediction_dir=args.pred_dir,
        output_dir=args.output_dir,
        summary_path=summary_path,
        tile_setting=args.tile_setting,
        nms_iou_threshold=args.nms_iou_threshold,
        drop_fully_outside_tile=not args.keep_fully_outside_tile,
        overwrite=args.overwrite,
        clip_ids=(parse_clip_ids(args.clip_list) if args.clip_list else None),
    )
    print(f"Processed {len(summaries)} clip(s).")
    for row in summaries:
        print(
            f"{row.clip_id} | files={row.tile_prediction_files_found} | "
            f"raw={row.raw_tile_events} | merged={row.merged_events} | "
            f"duplicates={row.duplicates_removed} | "
            f"invalid={row.invalid_events_dropped} | "
            f"out_of_tile={row.out_of_tile_events_dropped}"
        )
    print(f"Merge summary: {summary_path}")


if __name__ == "__main__":
    main()
