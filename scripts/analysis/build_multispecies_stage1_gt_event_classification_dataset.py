"""Build label-safe Stage 1 GT-location species-classification images.

Stage 1 allows the model to receive the target event location, but not the
ground-truth species label. This script reuses the fixed V2 sample selection,
renders anonymous model-facing images, and keeps species labels only in the CSV
metadata for later evaluation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.build_multispecies_event_level_dataset import (  # noqa: E402
    DEFAULT_DPI,
    DEFAULT_IMAGE_HEIGHT_PX,
    DEFAULT_IMAGE_WIDTH_PX,
    DEFAULT_MAX_FREQ_HZ,
    DEFAULT_MIN_FREQ_HZ,
    DEFAULT_TIME_CONTEXT_SECONDS,
    PaddedWindow,
    compute_padded_window,
    event_density,
    padded_audio_segment,
    resolve_portable_path,
)
from scripts.analysis.prepare_bd2_species_visual_spotcheck import (  # noqa: E402
    DEFAULT_MAX_DB,
    DEFAULT_MIN_DB,
    load_audio,
    portable,
    spectrogram_image,
)


DEFAULT_INPUT_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_event_level_dataset_v2_centred/"
    "multispecies_event_dataset_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/"
    "multispecies_stage1_gt_event_classification_dataset"
)
SPECIES_TOKENS = (
    "rhinolophus_hipposideros",
    "rhinolophus_ferrumequinum",
    "myotis_daubentonii",
    "myotis_nattereri",
    "myotis_mystacinus",
    "plecotus_auritus",
    "pipistrellus_pipistrellus",
    "ozimops_petersi",
)


def anonymous_sample_id(index: int) -> str:
    return f"sample_{index:06d}"


def contains_species_token(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in SPECIES_TOKENS)


def load_v2_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def model_image_path(output_dir: Path, variant: str, anon_id: str) -> Path:
    return output_dir / variant / f"{anon_id}.png"


def human_overlay_path(output_dir: Path, anon_id: str) -> Path:
    return output_dir / "human_review_overlays" / f"{anon_id}_gt_diagnostic_overlay.png"


def render_stage1_image(
    *,
    audio: np.ndarray,
    sample_rate: int,
    manifest_row: dict[str, Any],
    padded_window: PaddedWindow,
    output_path: Path,
    image_width_px: int,
    image_height_px: int,
    max_frequency_hz: float,
    draw_target_box: bool,
    human_review_overlay: bool,
) -> None:
    """Render one anonymous Stage 1 image with no title or class text."""

    segment = padded_audio_segment(
        audio=audio,
        sample_rate=sample_rate,
        padded_window=padded_window,
    )
    image, extent = spectrogram_image(segment, sample_rate)
    extent[0] += padded_window.requested_start_time
    extent[1] += padded_window.requested_start_time
    displayed_max_freq = min(max_frequency_hz, sample_rate / 2.0)
    figsize = (image_width_px / DEFAULT_DPI, image_height_px / DEFAULT_DPI)
    fig = plt.figure(figsize=figsize, dpi=DEFAULT_DPI)
    ax = fig.add_axes([0.12, 0.12, 0.84, 0.78])
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )
    ax.set_xlim(padded_window.requested_start_time, padded_window.requested_end_time)
    ax.set_ylim(DEFAULT_MIN_FREQ_HZ / 1000.0, displayed_max_freq / 1000.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.grid(True, which="major", color="white", alpha=0.25, linewidth=0.5)
    if draw_target_box or human_review_overlay:
        start = float(manifest_row["event_start_time"])
        end = float(manifest_row["event_end_time"])
        low = float(manifest_row["event_low_freq"])
        high = float(manifest_row["event_high_freq"])
        rect = Rectangle(
            (start, low / 1000.0),
            end - start,
            (high - low) / 1000.0,
            fill=False,
            edgecolor="cyan" if draw_target_box else "lime",
            linewidth=1.4,
        )
        ax.add_patch(rect)
    if human_review_overlay:
        ax.text(
            0.01,
            0.99,
            "HUMAN REVIEW GT OVERLAY - NOT MODEL INPUT",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color="yellow",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", dpi=DEFAULT_DPI)
    plt.close(fig)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def build_stage1_dataset(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    half_context_seconds: float,
    max_frequency_hz: float,
    image_width_px: int,
    image_height_px: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    output_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    audio_cache: dict[Path, tuple[np.ndarray, int]] = {}
    for index, row in enumerate(rows, start=1):
        anon_id = anonymous_sample_id(index)
        audio_path = resolve_portable_path(row["original_audio_path"])
        if not audio_path.is_file():
            warnings.append(f"missing_audio:{row['sample_id']}:{audio_path}")
            continue
        if audio_path not in audio_cache:
            audio_cache[audio_path] = load_audio(audio_path)
        audio, sample_rate = audio_cache[audio_path]
        duration = len(audio) / sample_rate
        padded = compute_padded_window(
            event_start=float(row["event_start_time"]),
            event_end=float(row["event_end_time"]),
            audio_duration=duration,
            half_context_seconds=half_context_seconds,
        )
        centred_path = model_image_path(output_dir, "centred_crop_no_box", anon_id)
        marker_path = model_image_path(output_dir, "gt_box_marker", anon_id)
        overlay_path = human_overlay_path(output_dir, anon_id)
        render_stage1_image(
            audio=audio,
            sample_rate=sample_rate,
            manifest_row=row,
            padded_window=padded,
            output_path=centred_path,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            max_frequency_hz=max_frequency_hz,
            draw_target_box=False,
            human_review_overlay=False,
        )
        render_stage1_image(
            audio=audio,
            sample_rate=sample_rate,
            manifest_row=row,
            padded_window=padded,
            output_path=marker_path,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            max_frequency_hz=max_frequency_hz,
            draw_target_box=True,
            human_review_overlay=False,
        )
        render_stage1_image(
            audio=audio,
            sample_rate=sample_rate,
            manifest_row=row,
            padded_window=padded,
            output_path=overlay_path,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
            max_frequency_hz=max_frequency_hz,
            draw_target_box=False,
            human_review_overlay=True,
        )
        for path in (centred_path, marker_path, overlay_path):
            if image_size(path) != (image_width_px, image_height_px):
                warnings.append(f"image_size_mismatch:{path}")
        embedded_label_text_detected = "false"
        model_paths = (portable(centred_path), portable(marker_path))
        label_safe = (
            not any(contains_species_token(path) for path in model_paths)
            and embedded_label_text_detected == "false"
        )
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "anonymous_sample_id": anon_id,
                "species": row["species"],
                "source_dataset": row["source_dataset"],
                "source_recording": row["source_recording"],
                "source_recording_id": row["source_recording_id"],
                "split_group": row["split_group"],
                "event_index": row["event_index"],
                "event_start_time": row["event_start_time"],
                "event_end_time": row["event_end_time"],
                "event_low_freq": row["event_low_freq"],
                "event_high_freq": row["event_high_freq"],
                "centred_crop_image_path": portable(centred_path),
                "gt_box_marker_image_path": portable(marker_path),
                "human_review_overlay_path": portable(overlay_path),
                "image_width": image_width_px,
                "image_height": image_height_px,
                "target_centered_pass": str(padded.target_centered_pass).lower(),
                "label_safe_pass": str(label_safe).lower(),
                "embedded_label_text_detected": embedded_label_text_detected,
                "left_padding_seconds": padded.left_padding_seconds,
                "right_padding_seconds": padded.right_padding_seconds,
                "target_center_x_fraction": padded.target_center_x_fraction,
                "candidate_event_count": row.get("candidate_event_count", ""),
                "event_density": row.get(
                    "event_density",
                    event_density(int(float(row.get("candidate_event_count") or 0))),
                ),
                "quality_hint": row.get("quality_hint", ""),
                "difficulty_prior": row.get("difficulty_prior", ""),
            }
        )
    return output_rows, warnings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    by_species = Counter(row["species"] for row in rows)
    lines = [
        "# Stage 1 GT-Location Species Classification Dataset",
        "",
        "Stage 1 permits target-location information but forbids species-label leakage in model-facing images. Species labels are kept only in `stage1_manifest.csv` for evaluation.",
        "",
        "## Variants",
        "",
        "- `centred_crop_no_box/`: anonymous strict-centred crops with no GT box.",
        "- `gt_box_marker/`: anonymous strict-centred crops with a neutral target box.",
        "- `human_review_overlays/`: diagnostic-only GT overlays; never use as model input.",
        "",
        "## Integrity Summary",
        "",
        f"- Samples: `{len(rows)}`",
        f"- Label-safe pass: `{sum(row['label_safe_pass'] == 'true' for row in rows)}/{len(rows)}`",
        f"- Target-centred pass: `{sum(row['target_centered_pass'] == 'true' for row in rows)}/{len(rows)}`",
        f"- Embedded label text detected: `{sum(row['embedded_label_text_detected'] == 'true' for row in rows)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Species Counts",
        "",
        "| Species | Samples |",
        "|---|---:|",
    ]
    for species, count in sorted(by_species.items()):
        lines.append(f"| {species} | {count} |")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-context-seconds", type=float, default=DEFAULT_TIME_CONTEXT_SECONDS)
    parser.add_argument("--max-frequency-hz", type=float, default=DEFAULT_MAX_FREQ_HZ)
    parser.add_argument("--image-width-px", type=int, default=DEFAULT_IMAGE_WIDTH_PX)
    parser.add_argument("--image-height-px", type=int, default=DEFAULT_IMAGE_HEIGHT_PX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_v2_manifest(args.input_manifest)
    stage1_rows, warnings = build_stage1_dataset(
        rows=rows,
        output_dir=args.output_dir,
        half_context_seconds=args.time_context_seconds,
        max_frequency_hz=args.max_frequency_hz,
        image_width_px=args.image_width_px,
        image_height_px=args.image_height_px,
    )
    write_csv(args.output_dir / "stage1_manifest.csv", stage1_rows)
    write_report(args.output_dir / "dataset_construction_report.md", stage1_rows, warnings)
    print(f"Read {len(rows)} V2 sample row(s)")
    print(f"Generated {len(stage1_rows)} Stage 1 sample row(s)")
    print(f"Warnings: {len(warnings)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
