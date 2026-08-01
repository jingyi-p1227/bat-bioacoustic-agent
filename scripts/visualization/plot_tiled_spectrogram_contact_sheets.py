"""Create clean and GT-diagnostic contact sheets for P6 tiled inputs.

Clean sheets concatenate the existing WAV-derived model inputs. Ground-truth
sheets are written to a separate diagnostic-only directory and must never be
used as model input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import make_spectrogram
from prepare_agent_spectrogram_inputs import (
    DEFAULT_MAX_FREQ_HZ,
    DEFAULT_MIN_DB,
    apply_grid_style,
    read_mono_audio,
    spectrogram_to_image,
)


DEFAULT_MANIFEST = Path("outputs/agent_inputs/p6_tiled_spectrograms/tile_manifest.csv")
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_OUTPUT_ROOT = Path("outputs/agent_inputs/p6_tiled_spectrograms")
CLEAN_SHEET_DIRNAME = "contact_sheets"
GT_SHEET_DIRNAME = "contact_sheets_gt_diagnostic_only"


@dataclass(frozen=True)
class TileManifestEntry:
    """Typed manifest values needed for contact-sheet generation."""

    clip_id: str
    tile_id: str
    tile_setting: str
    tile_start_seconds: float
    tile_end_seconds: float
    image_path: Path
    original_audio_path: Path
    overlap: float
    grid_style: str


@dataclass(frozen=True)
class CoverageSummaryRow:
    """Ground-truth coverage statistics for one clip and tile setting."""

    clip_id: str
    tile_setting: str
    total_gt_events: int
    gt_events_visible_in_at_least_one_tile: int
    gt_events_crossing_tile_boundary: int
    max_visible_fraction_per_gt_event_min: float
    notes: str


COVERAGE_FIELDS = tuple(CoverageSummaryRow.__dataclass_fields__)


def tile_setting_from_id(tile_id: str) -> str:
    """Extract a named tile setting from a manifest tile id."""
    marker = "_tile_"
    if marker not in tile_id:
        raise ValueError(f"Tile id does not contain {marker!r}: {tile_id}")
    return tile_id.rsplit(marker, 1)[0]


def load_tile_manifest(
    manifest_path: Path, *, project_root: Path | None = None
) -> list[TileManifestEntry]:
    """Load tile metadata and resolve portable paths against the project root."""
    project_root = Path.cwd() if project_root is None else project_root
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Tile manifest is empty: {manifest_path}")

    entries: list[TileManifestEntry] = []
    for row in rows:
        image_path = Path(row["image_path"])
        audio_path = Path(row["original_audio_path"])
        entries.append(
            TileManifestEntry(
                clip_id=row["clip_id"],
                tile_id=row["tile_id"],
                tile_setting=tile_setting_from_id(row["tile_id"]),
                tile_start_seconds=float(row["tile_start_seconds"]),
                tile_end_seconds=float(row["tile_end_seconds"]),
                image_path=(
                    image_path
                    if image_path.is_absolute()
                    else project_root / image_path
                ),
                original_audio_path=(
                    audio_path
                    if audio_path.is_absolute()
                    else project_root / audio_path
                ),
                overlap=float(row["overlap"]),
                grid_style=row["grid_style"],
            )
        )
    return entries


def group_manifest_entries(
    entries: list[TileManifestEntry],
) -> dict[tuple[str, str], list[TileManifestEntry]]:
    """Group entries by clip and setting in temporal order."""
    groups: dict[tuple[str, str], list[TileManifestEntry]] = {}
    for entry in entries:
        groups.setdefault((entry.clip_id, entry.tile_setting), []).append(entry)
    for group in groups.values():
        group.sort(key=lambda entry: (entry.tile_start_seconds, entry.tile_end_seconds))
    return groups


def contact_sheet_path(
    output_root: Path,
    clip_id: str,
    tile_setting: str,
    *,
    gt_diagnostic: bool,
) -> Path:
    """Return the clean or diagnostic contact-sheet destination."""
    if gt_diagnostic:
        return (
            output_root
            / GT_SHEET_DIRNAME
            / f"{clip_id}_{tile_setting}_gt_diagnostic_contact_sheet.png"
        )
    return (
        output_root
        / CLEAN_SHEET_DIRNAME
        / f"{clip_id}_{tile_setting}_clean_contact_sheet.png"
    )


def visible_event_portion(
    event: dict, tile_start_seconds: float, tile_end_seconds: float
) -> dict | None:
    """Return the event portion visible inside a tile, or None when disjoint."""
    event_start = float(event["start_time"])
    event_end = float(event["end_time"])
    if event_start >= tile_end_seconds or event_end <= tile_start_seconds:
        return None
    if event_end <= event_start:
        raise ValueError(f"Invalid event interval: {event_start} to {event_end}")

    visible_start = max(event_start, tile_start_seconds)
    visible_end = min(event_end, tile_end_seconds)
    return {
        "start_time": visible_start,
        "end_time": visible_end,
        "visible_fraction": (visible_end - visible_start) / (event_end - event_start),
        "tile_truncated_left": event_start < tile_start_seconds,
        "tile_truncated_right": event_end > tile_end_seconds,
    }


def internal_tile_boundaries(entries: list[TileManifestEntry]) -> list[float]:
    """Return unique internal starts and ends for one tile configuration."""
    clip_start = min(entry.tile_start_seconds for entry in entries)
    clip_end = max(entry.tile_end_seconds for entry in entries)
    boundaries = {
        value
        for entry in entries
        for value in (entry.tile_start_seconds, entry.tile_end_seconds)
        if clip_start < value < clip_end
    }
    return sorted(boundaries)


def summarize_gt_coverage(
    *,
    clip_id: str,
    tile_setting: str,
    entries: list[TileManifestEntry],
    events: list[dict],
) -> CoverageSummaryRow:
    """Summarize whether tile windows preserve complete event visibility."""
    max_visible_fractions: list[float] = []
    visible_event_count = 0
    for event in events:
        portions = [
            portion
            for entry in entries
            if (
                portion := visible_event_portion(
                    event, entry.tile_start_seconds, entry.tile_end_seconds
                )
            )
            is not None
        ]
        if portions:
            visible_event_count += 1
            max_visible_fractions.append(
                max(float(portion["visible_fraction"]) for portion in portions)
            )
        else:
            max_visible_fractions.append(0.0)

    boundaries = internal_tile_boundaries(entries)
    crossing_count = sum(
        any(
            float(event["start_time"]) < boundary < float(event["end_time"])
            for boundary in boundaries
        )
        for event in events
    )
    min_best_fraction = min(max_visible_fractions, default=1.0)
    if visible_event_count == len(events) and min_best_fraction >= 1.0 - 1e-9:
        notes = "All GT events have at least one complete tile view."
    elif visible_event_count == len(events):
        notes = "All GT events are covered, but at least one is partial in every tile."
    else:
        notes = f"{len(events) - visible_event_count} GT event(s) are not covered."
    if crossing_count:
        notes += f" {crossing_count} event(s) cross an internal tile boundary."

    return CoverageSummaryRow(
        clip_id=clip_id,
        tile_setting=tile_setting,
        total_gt_events=len(events),
        gt_events_visible_in_at_least_one_tile=visible_event_count,
        gt_events_crossing_tile_boundary=crossing_count,
        max_visible_fraction_per_gt_event_min=round(min_best_fraction, 6),
        notes=notes,
    )


def write_coverage_summary(
    rows: list[CoverageSummaryRow], output_path: Path
) -> None:
    """Write deterministic ground-truth tile coverage statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COVERAGE_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def save_clean_contact_sheet(
    entries: list[TileManifestEntry], output_path: Path
) -> None:
    """Stack existing clean tile PNGs without altering their content."""
    images: list[Image.Image] = []
    for entry in entries:
        if not entry.image_path.is_file():
            raise FileNotFoundError(f"Clean tile image not found: {entry.image_path}")
        with Image.open(entry.image_path) as source:
            images.append(source.convert("RGB"))

    margin = 16
    header_height = 42
    caption_height = 22
    width = max(image.width for image in images) + 2 * margin
    height = header_height + sum(image.height + caption_height + margin for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    first = entries[0]
    draw.text(
        (margin, 12),
        f"{first.clip_id} | {first.tile_setting} | clean WAV-derived tiles",
        fill="black",
    )
    y = header_height
    for entry, image in zip(entries, images, strict=True):
        draw.text(
            (margin, y),
            (
                f"{entry.tile_id}: {entry.tile_start_seconds:.6f}"
                f"-{entry.tile_end_seconds:.6f} s"
            ),
            fill="black",
        )
        y += caption_height
        sheet.paste(image, (margin, y))
        y += image.height + margin

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    for image in images:
        image.close()


def save_gt_diagnostic_contact_sheet(
    *,
    entries: list[TileManifestEntry],
    events: list[dict],
    output_path: Path,
    min_db: float,
    max_freq_hz: float,
) -> None:
    """Render diagnostic-only tiles with visible GT portions clipped per window."""
    audio, sample_rate_hz = read_mono_audio(entries[0].original_audio_path)
    fig, axes = plt.subplots(
        len(entries),
        1,
        figsize=(12, max(3.2 * len(entries), 4)),
        squeeze=False,
    )
    displayed_max_freq_hz = min(max_freq_hz, sample_rate_hz / 2)

    for ax, entry in zip(axes[:, 0], entries, strict=True):
        start_sample = round(entry.tile_start_seconds * sample_rate_hz)
        end_sample = round(entry.tile_end_seconds * sample_rate_hz)
        tile_audio = audio[start_sample:end_sample]
        spec, stft = make_spectrogram(tile_audio, sample_rate_hz)
        image = spectrogram_to_image(spec, min_db=min_db)
        extent = list(stft.extent(len(tile_audio)))
        extent[0] += entry.tile_start_seconds
        extent[1] += entry.tile_start_seconds
        extent[2] /= 1000
        extent[3] /= 1000

        ax.imshow(
            image,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="gray",
            interpolation="bilinear",
        )
        ax.set_xlim(entry.tile_start_seconds, entry.tile_end_seconds)
        ax.set_ylim(0, displayed_max_freq_hz / 1000)
        ax.set_xlabel("Time (s, original clip coordinates)")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title(
            f"{entry.tile_id}: {entry.tile_start_seconds:.3f}"
            f"-{entry.tile_end_seconds:.3f} s"
        )
        apply_grid_style(
            ax,
            grid_style=entry.grid_style,
            duration_seconds=entry.tile_end_seconds - entry.tile_start_seconds,
            displayed_max_freq_hz=displayed_max_freq_hz,
        )

        for event_index, event in enumerate(events, start=1):
            portion = visible_event_portion(
                event, entry.tile_start_seconds, entry.tile_end_seconds
            )
            if portion is None:
                continue
            low_khz = float(event["low_frequency"]) / 1000
            high_khz = float(event["high_frequency"]) / 1000
            tile_truncated = bool(
                portion["tile_truncated_left"] or portion["tile_truncated_right"]
            )
            color = "yellow" if tile_truncated else "lime"
            linestyle = "--" if tile_truncated else "-"
            rect = Rectangle(
                (portion["start_time"], low_khz),
                portion["end_time"] - portion["start_time"],
                high_khz - low_khz,
                fill=False,
                edgecolor=color,
                linestyle=linestyle,
                linewidth=1.5,
            )
            ax.add_patch(rect)
            label = f"{event_index}{'*' if tile_truncated else ''}"
            ax.text(
                portion["start_time"],
                high_khz,
                label,
                color=color,
                fontsize=7,
                va="bottom",
                ha="left",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.55,
                    "pad": 1,
                    "edgecolor": "none",
                },
            )

    first = entries[0]
    fig.suptitle(
        f"DIAGNOSTIC ONLY - {first.clip_id} | {first.tile_setting} | GT visible portions",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", bbox_inches="tight")
    plt.close(fig)


def load_gt_events(eval_dir: Path, clip_id: str) -> list[dict]:
    """Load clip-level events for diagnostic-only plotting and coverage checks."""
    path = eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError(f"Ground-truth events must be a list: {path}")
    return events


def write_sanity_report(
    *,
    rows: list[CoverageSummaryRow],
    clean_paths: list[Path],
    diagnostic_paths: list[Path],
    output_path: Path,
) -> None:
    """Write the initial P6A.2 coverage and visual-inspection report."""
    all_covered = all(
        row.total_gt_events == row.gt_events_visible_in_at_least_one_tile
        for row in rows
    )
    crossing = [row for row in rows if row.gt_events_crossing_tile_boundary > 0]
    lines = [
        "# P6A.2 Tiled Spectrogram Visual Sanity Check",
        "",
        "## Scope",
        "",
        "This diagnostic checks the P6 tiled inputs before model inference. Clean contact sheets contain only existing WAV-derived tiles. GT contact sheets are stored separately and must never be used as model inputs.",
        "",
        "## Coverage Summary",
        "",
        f"- All GT events covered by at least one tile: **{'yes' if all_covered else 'no'}**.",
        f"- Clean contact sheets generated: **{len(clean_paths)}**.",
        f"- GT diagnostic contact sheets generated: **{len(diagnostic_paths)}**.",
        "- `max_visible_fraction_per_gt_event_min` measures the worst event's best available tile view.",
        "- `gt_events_crossing_tile_boundary` uses internal tile starts/ends only; original clip boundaries are excluded.",
        "",
        "| Clip | Tile setting | GT | Covered | Crossing internal boundary | Minimum best visible fraction |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.clip_id}` | `{row.tile_setting}` | {row.total_gt_events} | "
            f"{row.gt_events_visible_in_at_least_one_tile} | "
            f"{row.gt_events_crossing_tile_boundary} | "
            f"{row.max_visible_fraction_per_gt_event_min:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Boundary-Crossing Clips",
            "",
        ]
    )
    if crossing:
        for row in crossing:
            lines.append(
                f"- `{row.clip_id}` under `{row.tile_setting}`: "
                f"{row.gt_events_crossing_tile_boundary} event(s)."
            )
    else:
        lines.append("No GT event crosses an internal tile boundary in these settings.")

    lines.extend(
        [
            "",
            "## Visual Findings",
            "",
            "### OP_016",
            "",
            "The 0.25 s tiles enlarge the time geometry of individual calls and isolate smaller activity regions more clearly than the 0.5 s tiles. This makes them useful as a targeted stress-test representation for `OP_016`. They also remove sequence-level context and create more overlapping views, so visual clarity does not by itself guarantee better merged predictions.",
            "",
            "### OP_003 and OP_004",
            "",
            "The 0.5 s tiles retain enough local context for the paired right- and left-boundary examples while making calls larger than in the full overview. They are the lower-risk first condition for testing whether tiling helps boundary localisation.",
            "",
            "### Duplicate Risk",
            "",
            "Overlap deliberately places some calls in more than one tile. Duplicate predictions are therefore expected, especially with the 0.25 s condition, which creates more views. Raw tile predictions must be retained and merged using the documented confidence-ordered NMS rule before evaluation.",
            "",
            "## Recommendation",
            "",
            "Use `tile_0p5_overlap_0p1` for the first six-clip model pilot. It covers all GT events, uses only 17 views, retains more context, and has lower duplicate and inference-cost risk than the 29-view 0.25 s condition. Use `tile_0p25_overlap_0p05` as a controlled follow-up, with particular attention to `OP_016`, if the 0.5 s pilot does not recover dense calls or improve box localisation.",
            "",
            "## Safety Note",
            "",
            "Only images under `contact_sheets/` are clean. Files under `contact_sheets_gt_diagnostic_only/` contain ground truth and are for human inspection only.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create P6 tiled-input clean and GT-diagnostic contact sheets."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ_HZ)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_tile_manifest(args.manifest)
    groups = group_manifest_entries(entries)
    coverage_rows: list[CoverageSummaryRow] = []
    clean_paths: list[Path] = []
    diagnostic_paths: list[Path] = []

    events_by_clip: dict[str, list[dict]] = {}
    for (clip_id, tile_setting), group in sorted(groups.items()):
        events = events_by_clip.setdefault(
            clip_id, load_gt_events(args.eval_dir, clip_id)
        )
        clean_path = contact_sheet_path(
            args.output_root,
            clip_id,
            tile_setting,
            gt_diagnostic=False,
        )
        diagnostic_path = contact_sheet_path(
            args.output_root,
            clip_id,
            tile_setting,
            gt_diagnostic=True,
        )
        save_clean_contact_sheet(group, clean_path)
        save_gt_diagnostic_contact_sheet(
            entries=group,
            events=events,
            output_path=diagnostic_path,
            min_db=args.min_db,
            max_freq_hz=args.max_freq,
        )
        coverage_rows.append(
            summarize_gt_coverage(
                clip_id=clip_id,
                tile_setting=tile_setting,
                entries=group,
                events=events,
            )
        )
        clean_paths.append(clean_path)
        diagnostic_paths.append(diagnostic_path)

    coverage_path = args.output_root / "tile_gt_coverage_summary.csv"
    report_path = args.output_root / "p6a2_visual_sanity_check.md"
    write_coverage_summary(coverage_rows, coverage_path)
    write_sanity_report(
        rows=coverage_rows,
        clean_paths=clean_paths,
        diagnostic_paths=diagnostic_paths,
        output_path=report_path,
    )

    print(f"Generated {len(clean_paths)} clean contact sheet(s).")
    print(f"Generated {len(diagnostic_paths)} GT diagnostic contact sheet(s).")
    print(f"Coverage summary: {coverage_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
