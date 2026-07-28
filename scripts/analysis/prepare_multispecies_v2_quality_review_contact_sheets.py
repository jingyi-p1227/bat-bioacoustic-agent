"""Create contact sheets for V2 multi-species event dataset manual review.

This script does not alter source images. It reads the V2 manifest, selects
three representative samples per species where possible, and writes clean and
human-review-only overlay contact sheets.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_event_level_dataset_v2_centred/"
    "multispecies_event_dataset_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_event_level_dataset_v2_centred/"
    "quality_review_contact_sheets"
)
THUMBNAIL_SIZE = (320, 240)
TILE_WIDTH = 360
TILE_HEIGHT = 310
SHEET_COLUMNS = 3


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def padding_category(row: dict[str, Any]) -> str:
    left = float(row.get("left_padding_seconds") or 0.0)
    right = float(row.get("right_padding_seconds") or 0.0)
    if left > 0:
        return "left_padding"
    if right > 0:
        return "right_padding"
    return "no_padding"


def select_species_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pick up to no/left/right padding examples per species."""

    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_species[row["species"]].append(row)
    selected: list[dict[str, Any]] = []
    for species in sorted(by_species):
        species_rows = by_species[species]
        chosen: list[dict[str, Any]] = []
        for category in ("no_padding", "left_padding", "right_padding"):
            match = next(
                (row for row in species_rows if padding_category(row) == category),
                None,
            )
            if match and match not in chosen:
                chosen.append(match)
        for row in species_rows:
            if len(chosen) >= 3:
                break
            if row not in chosen:
                chosen.append(row)
        selected.extend(chosen[:3])
    return selected


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def draw_contact_sheet(
    *,
    rows: list[dict[str, Any]],
    image_key: str,
    output_path: Path,
    title: str,
) -> list[str]:
    """Draw a simple labelled image contact sheet and return warnings."""

    warnings: list[str] = []
    if not rows:
        raise ValueError("No rows supplied for contact sheet")
    columns = SHEET_COLUMNS
    rows_count = (len(rows) + columns - 1) // columns
    width = columns * TILE_WIDTH
    height = rows_count * TILE_HEIGHT + 50
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((12, 12), title, fill="black", font=font)
    for index, row in enumerate(rows):
        col = index % columns
        row_index = index // columns
        x = col * TILE_WIDTH + 12
        y = row_index * TILE_HEIGHT + 48
        image_path = resolve_path(row[image_key])
        if not image_path.is_file():
            warnings.append(f"missing_image:{row['sample_id']}:{image_path}")
            continue
        with Image.open(image_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail(THUMBNAIL_SIZE)
            sheet.paste(thumb, (x, y))
        label_lines = [
            row["sample_id"],
            row["species"],
            f"L={float(row['left_padding_seconds']):.3f}s R={float(row['right_padding_seconds']):.3f}s x={row['target_center_x_fraction']}",
        ]
        for offset, line in enumerate(label_lines):
            draw.text((x, y + THUMBNAIL_SIZE[1] + 8 + offset * 14), line, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return warnings


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| sample_id | species | source_recording | left_padding_seconds | right_padding_seconds | target_center_x_fraction | image_path | overlay_path |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['species']} | {row['source_recording']} | {row['left_padding_seconds']} | {row['right_padding_seconds']} | {row['target_center_x_fraction']} | {row['image_path']} | {row['human_review_overlay_path']} |"
        )
    return "\n".join(lines)


def write_review_markdown(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    lines = [
        "# V2 Multi-Species Event Dataset Quality Review Contact Sheets",
        "",
        "This package samples three rows per species for quick manual inspection. Clean sheets contain model-facing images with no GT overlays. Overlay sheets use the separate human-review-only diagnostic images.",
        "",
        "## Review Checklist",
        "",
        "- Is the target call visibly centred?",
        "- Is the clean image free of GT overlays?",
        "- Does the overlay align with the target event?",
        "- Are any images mostly blank?",
        "- Are Myotis / Plecotus samples too visually ambiguous?",
        "- Are any samples dominated by noise or non-target artefacts?",
        "",
        "## Selected Samples",
        "",
        markdown_table(rows),
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    selected = select_species_review_rows(rows)
    warnings = []
    warnings.extend(
        draw_contact_sheet(
            rows=selected,
            image_key="image_path",
            output_path=args.output_dir / "clean_contact_sheet.png",
            title="V2 clean model-facing samples",
        )
    )
    warnings.extend(
        draw_contact_sheet(
            rows=selected,
            image_key="human_review_overlay_path",
            output_path=args.output_dir / "overlay_contact_sheet.png",
            title="V2 GT diagnostic overlays - not model input",
        )
    )
    write_review_markdown(
        args.output_dir / "species_by_species_review.md",
        selected,
        warnings,
    )
    print(f"Selected {len(selected)} review sample(s)")
    print(f"Warnings: {len(warnings)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
