"""Build a standardised event-level spectrogram dataset for species classification.

This no-inference script reads the existing BD2 multi-species event-crop
spot-check table, re-renders every selected event from the original WAV audio,
and writes fixed-size clean spectrogram images. Clean images contain no
ground-truth overlays. Optional diagnostic overlays are written separately for
human review only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.prepare_bd2_species_visual_spotcheck import (  # noqa: E402
    DEFAULT_MAX_DB,
    DEFAULT_MIN_DB,
    load_audio,
    portable,
    slugify,
    spectrogram_image,
)


DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "outputs/analysis_reports/bd2_species_event_crop_spotcheck/event_crop_review.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/analysis_reports/multispecies_event_level_dataset"
)
DEFAULT_V1_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_event_level_dataset/multispecies_event_dataset_manifest.csv"
)
DEFAULT_V2_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_event_level_dataset_v2_centred"
)
DEFAULT_TIME_CONTEXT_SECONDS = 0.15
DEFAULT_MIN_FREQ_HZ = 0.0
DEFAULT_MAX_FREQ_HZ = 120_000.0
DEFAULT_IMAGE_WIDTH_PX = 800
DEFAULT_IMAGE_HEIGHT_PX = 600
DEFAULT_DPI = 100
TARGET_SPECIES = (
    "Rhinolophus hipposideros",
    "Rhinolophus ferrumequinum",
    "Myotis daubentonii",
    "Myotis nattereri",
    "Myotis mystacinus",
    "Plecotus auritus",
    "Pipistrellus pipistrellus",
    "Ozimops petersi",
)


@dataclass(frozen=True)
class EventSampleInput:
    """One target event selected from the existing event-crop review table."""

    species: str
    example_index: int
    source_dataset: str
    source: str
    source_recording: str
    event_index: int
    event_uuid: str
    event_start_time: float
    event_end_time: float
    event_low_freq: float
    event_high_freq: float
    audio_path: Path
    audio_duration_seconds: float
    sample_rate_hz: int
    candidate_event_count: int
    quality_hint: str
    difficulty_prior: str


@dataclass(frozen=True)
class TimeWindow:
    """Visible time window for one event-level image."""

    start_seconds: float
    end_seconds: float
    centered: bool


@dataclass(frozen=True)
class PaddedWindow:
    """Requested and realised audio window with silence padding metadata."""

    requested_start_time: float
    requested_end_time: float
    actual_audio_start_time: float
    actual_audio_end_time: float
    left_padding_seconds: float
    right_padding_seconds: float
    target_center_x_fraction: float
    target_centered_pass: bool


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def resolve_portable_path(value: str) -> Path:
    """Resolve a path recorded relative to repo root or repo parent."""

    path = Path(value)
    if path.is_absolute():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    parent_path = REPO_ROOT.parent / path
    if parent_path.exists():
        return parent_path
    return repo_path


def read_event_samples(path: Path) -> list[EventSampleInput]:
    """Read valid, written event rows from the spot-check review CSV."""

    rows: list[EventSampleInput] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("species") not in TARGET_SPECIES:
                continue
            if not parse_bool(row.get("audio_exists", "")):
                continue
            if not parse_bool(row.get("crop_written", "")):
                continue
            try:
                event_index = int(row["source_event_rank"])
                event_start = float(row["event_start_time_seconds"])
                event_end = float(row["event_end_time_seconds"])
                event_low = float(row["event_low_frequency_hz"])
                event_high = float(row["event_high_frequency_hz"])
                duration = float(row["audio_duration_seconds"])
                sample_rate = int(float(row["sample_rate_hz"]))
                candidate_event_count = int(float(row.get("candidate_event_count") or 0))
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= event_start < event_end and 0 <= event_low < event_high):
                continue
            rows.append(
                EventSampleInput(
                    species=row["species"],
                    example_index=int(row["example_index"]),
                    source_dataset=row["dataset"],
                    source=row["source"],
                    source_recording=row["recording_path"],
                    event_index=event_index,
                    event_uuid=row.get("event_uuid", ""),
                    event_start_time=event_start,
                    event_end_time=event_end,
                    event_low_freq=event_low,
                    event_high_freq=event_high,
                    audio_path=resolve_portable_path(row["audio_path"]),
                    audio_duration_seconds=duration,
                    sample_rate_hz=sample_rate,
                    candidate_event_count=candidate_event_count,
                    quality_hint=row.get("quality_hint", ""),
                    difficulty_prior=row.get("difficulty_prior", ""),
                )
            )
    return rows


def read_event_samples_from_v1_manifest(path: Path) -> list[EventSampleInput]:
    """Read the frozen V1 selected sample manifest without reselecting events."""

    rows: list[EventSampleInput] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    EventSampleInput(
                        species=row["species"],
                        example_index=int(re.search(r"_rec(\d+)_", row["sample_id"]).group(1)),  # type: ignore[union-attr]
                        source_dataset=row["source_dataset"],
                        source=row["source"],
                        source_recording=row["source_recording"],
                        event_index=int(row["event_index"]),
                        event_uuid=row.get("event_uuid", ""),
                        event_start_time=float(row["event_start_time"]),
                        event_end_time=float(row["event_end_time"]),
                        event_low_freq=float(row["event_low_freq"]),
                        event_high_freq=float(row["event_high_freq"]),
                        audio_path=resolve_portable_path(row["original_audio_path"]),
                        audio_duration_seconds=float(row["audio_duration_seconds"]),
                        sample_rate_hz=int(float(row["sample_rate_hz"])),
                        candidate_event_count=int(float(row["candidate_event_count"])),
                        quality_hint=row.get("quality_hint", ""),
                        difficulty_prior=row.get("difficulty_prior", ""),
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
    return rows


def source_recording_id(sample: EventSampleInput) -> str:
    """Return the split group; samples from this ID must not cross splits."""

    return slugify(
        f"{sample.source_dataset}_{sample.source}_{sample.source_recording.rsplit('.', 1)[0]}"
    )


def sample_id_for(sample: EventSampleInput) -> str:
    species_slug = slugify(sample.species)
    return (
        f"{species_slug}_rec{sample.example_index:02d}"
        f"_event{sample.event_index:03d}"
    )


def compute_centered_time_window(
    *,
    event_start: float,
    event_end: float,
    audio_duration: float,
    half_context_seconds: float,
) -> TimeWindow:
    """Return a fixed-width, event-centred window when boundaries allow."""

    if half_context_seconds <= 0:
        raise ValueError("half_context_seconds must be positive")
    if audio_duration <= 0:
        raise ValueError("audio_duration must be positive")
    window_width = half_context_seconds * 2.0
    if audio_duration <= window_width:
        center = (event_start + event_end) / 2.0
        return TimeWindow(0.0, audio_duration, abs(center - audio_duration / 2.0) < 1e-9)
    center = (event_start + event_end) / 2.0
    start = center - half_context_seconds
    end = center + half_context_seconds
    centered = True
    if start < 0.0:
        start = 0.0
        end = window_width
        centered = False
    elif end > audio_duration:
        end = audio_duration
        start = audio_duration - window_width
        centered = False
    return TimeWindow(round(start, 6), round(end, 6), centered)


def compute_padded_window(
    *,
    event_start: float,
    event_end: float,
    audio_duration: float,
    half_context_seconds: float,
) -> PaddedWindow:
    """Return requested event-centred window and required silence padding."""

    if half_context_seconds <= 0:
        raise ValueError("half_context_seconds must be positive")
    if audio_duration <= 0:
        raise ValueError("audio_duration must be positive")
    center = (event_start + event_end) / 2.0
    requested_start = center - half_context_seconds
    requested_end = center + half_context_seconds
    actual_start = max(0.0, requested_start)
    actual_end = min(audio_duration, requested_end)
    left_padding = max(0.0, -requested_start)
    right_padding = max(0.0, requested_end - audio_duration)
    target_fraction = (center - requested_start) / (requested_end - requested_start)
    centered_pass = abs(target_fraction - 0.5) <= 1e-9
    return PaddedWindow(
        requested_start_time=round(requested_start, 6),
        requested_end_time=round(requested_end, 6),
        actual_audio_start_time=round(actual_start, 6),
        actual_audio_end_time=round(actual_end, 6),
        left_padding_seconds=round(left_padding, 6),
        right_padding_seconds=round(right_padding, 6),
        target_center_x_fraction=round(target_fraction, 6),
        target_centered_pass=centered_pass,
    )


def clean_image_path(output_dir: Path, sample: EventSampleInput) -> Path:
    return (
        output_dir
        / "clean_images"
        / slugify(sample.species)
        / f"{sample_id_for(sample)}.png"
    )


def overlay_image_path(output_dir: Path, sample: EventSampleInput) -> Path:
    return (
        output_dir
        / "human_review_overlays"
        / slugify(sample.species)
        / f"{sample_id_for(sample)}_gt_diagnostic_overlay.png"
    )


def render_event_image(
    *,
    audio: np.ndarray,
    sample_rate: int,
    sample: EventSampleInput,
    output_path: Path,
    time_window: TimeWindow,
    max_frequency_hz: float,
    image_width_px: int,
    image_height_px: int,
    overlay_gt: bool,
) -> None:
    """Render one fixed-pixel spectrogram image."""

    image, extent = spectrogram_image(audio, sample_rate)
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
    ax.set_xlim(time_window.start_seconds, time_window.end_seconds)
    ax.set_ylim(DEFAULT_MIN_FREQ_HZ / 1000.0, displayed_max_freq / 1000.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(sample_id_for(sample), fontsize=10)
    ax.grid(True, which="major", color="white", alpha=0.25, linewidth=0.5)
    if overlay_gt:
        rect = Rectangle(
            (sample.event_start_time, sample.event_low_freq / 1000.0),
            sample.event_end_time - sample.event_start_time,
            (sample.event_high_freq - sample.event_low_freq) / 1000.0,
            fill=False,
            edgecolor="lime",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            0.01,
            0.99,
            "GT DIAGNOSTIC OVERLAY - NOT MODEL INPUT",
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


def padded_audio_segment(
    *,
    audio: np.ndarray,
    sample_rate: int,
    padded_window: PaddedWindow,
) -> np.ndarray:
    """Extract actual audio and pad missing requested context with silence."""

    start_sample = int(round(padded_window.actual_audio_start_time * sample_rate))
    end_sample = int(round(padded_window.actual_audio_end_time * sample_rate))
    left_pad_samples = int(round(padded_window.left_padding_seconds * sample_rate))
    right_pad_samples = int(round(padded_window.right_padding_seconds * sample_rate))
    segment = audio[start_sample:end_sample]
    if left_pad_samples:
        segment = np.concatenate([np.zeros(left_pad_samples, dtype=audio.dtype), segment])
    if right_pad_samples:
        segment = np.concatenate([segment, np.zeros(right_pad_samples, dtype=audio.dtype)])
    expected_samples = int(
        round(
            (padded_window.requested_end_time - padded_window.requested_start_time)
            * sample_rate
        )
    )
    if len(segment) < expected_samples:
        segment = np.concatenate(
            [segment, np.zeros(expected_samples - len(segment), dtype=audio.dtype)]
        )
    elif len(segment) > expected_samples:
        segment = segment[:expected_samples]
    return segment


def render_padded_event_image(
    *,
    audio: np.ndarray,
    sample_rate: int,
    sample: EventSampleInput,
    output_path: Path,
    padded_window: PaddedWindow,
    max_frequency_hz: float,
    image_width_px: int,
    image_height_px: int,
    overlay_gt: bool,
) -> None:
    """Render one strict-centred image from a silence-padded audio segment."""

    segment = padded_audio_segment(
        audio=audio,
        sample_rate=sample_rate,
        padded_window=padded_window,
    )
    image, extent = spectrogram_image(segment, sample_rate)
    # Remap the segment-local spectrogram extent back to requested source time.
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
    ax.set_title(sample_id_for(sample), fontsize=10)
    ax.grid(True, which="major", color="white", alpha=0.25, linewidth=0.5)
    if overlay_gt:
        rect = Rectangle(
            (sample.event_start_time, sample.event_low_freq / 1000.0),
            sample.event_end_time - sample.event_start_time,
            (sample.event_high_freq - sample.event_low_freq) / 1000.0,
            fill=False,
            edgecolor="lime",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            0.01,
            0.99,
            "GT DIAGNOSTIC OVERLAY - NOT MODEL INPUT",
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


def event_density(candidate_event_count: int) -> str:
    if candidate_event_count >= 30:
        return "high"
    if candidate_event_count >= 10:
        return "medium"
    if candidate_event_count >= 1:
        return "low"
    return "zero"


def build_dataset(
    *,
    rows: list[EventSampleInput],
    output_dir: Path,
    half_context_seconds: float,
    max_frequency_hz: float,
    image_width_px: int,
    image_height_px: int,
    write_overlays: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render images and return manifest rows plus invalid/missing rows."""

    manifest_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    audio_cache: dict[Path, tuple[np.ndarray, int]] = {}
    for sample in rows:
        sample_id = sample_id_for(sample)
        if not sample.audio_path.is_file():
            invalid_rows.append(
                {
                    "sample_id": sample_id,
                    "species": sample.species,
                    "source_recording": sample.source_recording,
                    "reason": "audio_missing",
                    "audio_path": portable(sample.audio_path),
                }
            )
            continue
        try:
            if sample.audio_path not in audio_cache:
                audio_cache[sample.audio_path] = load_audio(sample.audio_path)
            audio, sample_rate = audio_cache[sample.audio_path]
            duration = len(audio) / sample_rate
            window = compute_centered_time_window(
                event_start=sample.event_start_time,
                event_end=sample.event_end_time,
                audio_duration=duration,
                half_context_seconds=half_context_seconds,
            )
            clean_path = clean_image_path(output_dir, sample)
            overlay_path = overlay_image_path(output_dir, sample)
            render_event_image(
                audio=audio,
                sample_rate=sample_rate,
                sample=sample,
                output_path=clean_path,
                time_window=window,
                max_frequency_hz=max_frequency_hz,
                image_width_px=image_width_px,
                image_height_px=image_height_px,
                overlay_gt=False,
            )
            if write_overlays:
                render_event_image(
                    audio=audio,
                    sample_rate=sample_rate,
                    sample=sample,
                    output_path=overlay_path,
                    time_window=window,
                    max_frequency_hz=max_frequency_hz,
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                    overlay_gt=True,
                )
            actual_size = image_size(clean_path)
            if actual_size != (image_width_px, image_height_px):
                raise ValueError(f"image_size_mismatch:{actual_size}")
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "species": sample.species,
                    "source_dataset": sample.source_dataset,
                    "source": sample.source,
                    "source_recording": sample.source_recording,
                    "source_recording_id": source_recording_id(sample),
                    "split_group": source_recording_id(sample),
                    "event_index": sample.event_index,
                    "event_uuid": sample.event_uuid,
                    "event_start_time": round(sample.event_start_time, 6),
                    "event_end_time": round(sample.event_end_time, 6),
                    "event_low_freq": round(sample.event_low_freq, 6),
                    "event_high_freq": round(sample.event_high_freq, 6),
                    "event_duration_ms": round(
                        (sample.event_end_time - sample.event_start_time) * 1000.0,
                        6,
                    ),
                    "event_bandwidth_hz": round(sample.event_high_freq - sample.event_low_freq, 6),
                    "crop_time_start_seconds": window.start_seconds,
                    "crop_time_end_seconds": window.end_seconds,
                    "target_event_centered": str(window.centered).lower(),
                    "frequency_min_hz": int(DEFAULT_MIN_FREQ_HZ),
                    "frequency_max_hz": int(min(max_frequency_hz, sample_rate / 2.0)),
                    "image_width_px": image_width_px,
                    "image_height_px": image_height_px,
                    "image_path": portable(clean_path),
                    "human_review_overlay_path": portable(overlay_path) if write_overlays else "",
                    "original_audio_path": portable(sample.audio_path),
                    "sample_rate_hz": sample_rate,
                    "audio_duration_seconds": round(duration, 6),
                    "candidate_event_count": sample.candidate_event_count,
                    "event_density": event_density(sample.candidate_event_count),
                    "quality_hint": sample.quality_hint,
                    "difficulty_prior": sample.difficulty_prior,
                    "notes": (
                        "target_not_centered_due_to_audio_boundary"
                        if not window.centered
                        else ""
                    ),
                }
            )
        except Exception as exc:  # keep going so one bad file does not stop package construction
            invalid_rows.append(
                {
                    "sample_id": sample_id,
                    "species": sample.species,
                    "source_recording": sample.source_recording,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "audio_path": portable(sample.audio_path),
                }
            )
    return manifest_rows, invalid_rows


def build_strict_centred_dataset(
    *,
    rows: list[EventSampleInput],
    output_dir: Path,
    half_context_seconds: float,
    max_frequency_hz: float,
    image_width_px: int,
    image_height_px: int,
    write_overlays: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render V2 images with silence padding so every event centre is fixed."""

    manifest_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    audio_cache: dict[Path, tuple[np.ndarray, int]] = {}
    for sample in rows:
        sample_id = sample_id_for(sample)
        if not sample.audio_path.is_file():
            invalid_rows.append(
                {
                    "sample_id": sample_id,
                    "species": sample.species,
                    "source_recording": sample.source_recording,
                    "reason": "audio_missing",
                    "audio_path": portable(sample.audio_path),
                }
            )
            continue
        try:
            if sample.audio_path not in audio_cache:
                audio_cache[sample.audio_path] = load_audio(sample.audio_path)
            audio, sample_rate = audio_cache[sample.audio_path]
            duration = len(audio) / sample_rate
            padded = compute_padded_window(
                event_start=sample.event_start_time,
                event_end=sample.event_end_time,
                audio_duration=duration,
                half_context_seconds=half_context_seconds,
            )
            clean_path = clean_image_path(output_dir, sample)
            overlay_path = overlay_image_path(output_dir, sample)
            render_padded_event_image(
                audio=audio,
                sample_rate=sample_rate,
                sample=sample,
                output_path=clean_path,
                padded_window=padded,
                max_frequency_hz=max_frequency_hz,
                image_width_px=image_width_px,
                image_height_px=image_height_px,
                overlay_gt=False,
            )
            if write_overlays:
                render_padded_event_image(
                    audio=audio,
                    sample_rate=sample_rate,
                    sample=sample,
                    output_path=overlay_path,
                    padded_window=padded,
                    max_frequency_hz=max_frequency_hz,
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                    overlay_gt=True,
                )
            actual_size = image_size(clean_path)
            if actual_size != (image_width_px, image_height_px):
                raise ValueError(f"image_size_mismatch:{actual_size}")
            source_id = source_recording_id(sample)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "species": sample.species,
                    "source_dataset": sample.source_dataset,
                    "source": sample.source,
                    "source_recording": sample.source_recording,
                    "source_recording_id": source_id,
                    "split_group": source_id,
                    "event_index": sample.event_index,
                    "event_uuid": sample.event_uuid,
                    "event_start_time": round(sample.event_start_time, 6),
                    "event_end_time": round(sample.event_end_time, 6),
                    "event_low_freq": round(sample.event_low_freq, 6),
                    "event_high_freq": round(sample.event_high_freq, 6),
                    "event_duration_ms": round(
                        (sample.event_end_time - sample.event_start_time) * 1000.0,
                        6,
                    ),
                    "event_bandwidth_hz": round(sample.event_high_freq - sample.event_low_freq, 6),
                    "requested_start_time": padded.requested_start_time,
                    "requested_end_time": padded.requested_end_time,
                    "actual_audio_start_time": padded.actual_audio_start_time,
                    "actual_audio_end_time": padded.actual_audio_end_time,
                    "left_padding_seconds": padded.left_padding_seconds,
                    "right_padding_seconds": padded.right_padding_seconds,
                    "target_center_x_fraction": padded.target_center_x_fraction,
                    "target_centered_pass": str(padded.target_centered_pass).lower(),
                    "frequency_min_hz": int(DEFAULT_MIN_FREQ_HZ),
                    "frequency_max_hz": int(min(max_frequency_hz, sample_rate / 2.0)),
                    "image_width": image_width_px,
                    "image_height": image_height_px,
                    "image_width_px": image_width_px,
                    "image_height_px": image_height_px,
                    "image_path": portable(clean_path),
                    "human_review_overlay_path": portable(overlay_path) if write_overlays else "",
                    "original_audio_path": portable(sample.audio_path),
                    "sample_rate_hz": sample_rate,
                    "audio_duration_seconds": round(duration, 6),
                    "candidate_event_count": sample.candidate_event_count,
                    "event_density": event_density(sample.candidate_event_count),
                    "quality_hint": sample.quality_hint,
                    "difficulty_prior": sample.difficulty_prior,
                    "notes": (
                        "silence_padding_used"
                        if padded.left_padding_seconds or padded.right_padding_seconds
                        else ""
                    ),
                }
            )
        except Exception as exc:
            invalid_rows.append(
                {
                    "sample_id": sample_id,
                    "species": sample.species,
                    "source_recording": sample.source_recording,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "audio_path": portable(sample.audio_path),
                }
            )
    return manifest_rows, invalid_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0])
    else:
        fieldnames = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def species_counts(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_species[row["species"]].append(row)
    rows = []
    for species, species_rows in sorted(by_species.items()):
        recordings = {row["source_recording_id"] for row in species_rows}
        rows.append(
            {
                "species": species,
                "sample_count": len(species_rows),
                "recording_count": len(recordings),
                "source_datasets": ";".join(sorted({row["source_dataset"] for row in species_rows})),
                "density_counts": dict(Counter(row["event_density"] for row in species_rows)),
                "difficulty_priors": dict(Counter(row["difficulty_prior"] for row in species_rows)),
            }
        )
    return rows


def write_report(
    *,
    path: Path,
    manifest_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    image_width_px: int,
    image_height_px: int,
    half_context_seconds: float,
    max_frequency_hz: float,
) -> None:
    counts = species_counts(manifest_rows)
    all_same_size = all(
        int(row["image_width_px"]) == image_width_px
        and int(row["image_height_px"]) == image_height_px
        for row in manifest_rows
    )
    duplicate_sample_ids = [
        sample_id
        for sample_id, count in Counter(row["sample_id"] for row in manifest_rows).items()
        if count > 1
    ]
    strict_centred = bool(manifest_rows and "target_centered_pass" in manifest_rows[0])
    left_padding_values = [
        float(row.get("left_padding_seconds") or 0.0) for row in manifest_rows
    ]
    right_padding_values = [
        float(row.get("right_padding_seconds") or 0.0) for row in manifest_rows
    ]
    lines = [
        (
            "# Multi-Species Event-Level Spectrogram Dataset V2: Strict-Centred"
            if strict_centred
            else "# Multi-Species Event-Level Spectrogram Dataset V1"
        ),
        "",
        "This dataset construction is no-inference. Clean model-facing images are re-rendered from original WAV audio and contain no GT overlays. Diagnostic overlays are stored separately for human review only.",
        "",
        "## Construction settings",
        "",
        f"- Input selection file: `{(DEFAULT_V1_MANIFEST if strict_centred else DEFAULT_INPUT_CSV).relative_to(REPO_ROOT)}`",
        f"- Image size: `{image_width_px}x{image_height_px}` pixels",
        (
            f"- Time context: +/- `{half_context_seconds:.3f}` s around the target event centre with silence padding at audio boundaries"
            if strict_centred
            else f"- Time context: +/- `{half_context_seconds:.3f}` s around the target event centre where possible"
        ),
        f"- Frequency display range: `0-{int(max_frequency_hz)}` Hz, clamped to Nyquist when required",
        f"- dB scaling: `{DEFAULT_MIN_DB}` to `{DEFAULT_MAX_DB}`",
        "- Model-facing images: `clean_images/`",
        "- Human-review-only overlays: `human_review_overlays/`",
        "",
        "## Included species",
        "",
        "| Species | Samples | Source recordings | Source datasets | Difficulty prior | Density |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in counts:
        lines.append(
            f"| {row['species']} | {row['sample_count']} | {row['recording_count']} | {row['source_datasets']} | {row['difficulty_priors']} | {row['density_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            f"- Total generated samples: `{len(manifest_rows)}`",
            f"- Missing/invalid events: `{len(invalid_rows)}`",
            f"- All clean images same size: `{str(all_same_size).lower()}`",
            f"- Duplicate sample IDs: `{duplicate_sample_ids if duplicate_sample_ids else 'none'}`",
            (
                f"- Target-centred pass count: `{sum(row.get('target_centered_pass') == 'true' for row in manifest_rows)}/{len(manifest_rows)}`"
                if strict_centred
                else f"- Boundary-shifted/non-centred samples: `{sum(row['target_event_centered'] != 'true' for row in manifest_rows)}`"
            ),
            (
                f"- Samples requiring left padding: `{sum(value > 0 for value in left_padding_values)}`"
                if strict_centred
                else ""
            ),
            (
                f"- Samples requiring right padding: `{sum(value > 0 for value in right_padding_values)}`"
                if strict_centred
                else ""
            ),
            (
                f"- Max padding duration: `{max(left_padding_values + right_padding_values, default=0.0):.6f}` s"
                if strict_centred
                else ""
            ),
            "- Split safety: `split_group` is the source recording ID. Future train/test splits must assign whole source-recording groups, not individual images.",
            "",
            "## Difficulty notes",
            "",
            "- Likely easier: `Rhinolophus hipposideros` and `Rhinolophus ferrumequinum`, because these candidates tend to show high-frequency, visually distinctive calls.",
            "- Likely harder: `Myotis daubentonii`, `Myotis nattereri`, and `Myotis mystacinus`, because related Myotis calls can be visually similar and may require careful manual review.",
            "- Challenging: `Plecotus auritus`, retained as an explicit difficult candidate.",
            "- Moderate/control: `Pipistrellus pipistrellus`.",
            "- Optional benchmark anchor: `Ozimops petersi`; keep it separable from UK species when reporting cross-species results.",
            "",
            "## Manual review checklist before model inference",
            "",
            "1. Confirm every clean image contains the target call and no GT overlay.",
            (
                "2. Check whether silence-padded near-boundary crops remain interpretable."
                if strict_centred
                else "2. Check whether near-boundary target calls remain interpretable when the event cannot be perfectly centred."
            ),
            "3. Confirm that 0-120 kHz is appropriate for each species; if high-frequency Rhinolophus calls are clipped by Nyquist, note this during review.",
            "4. Inspect human-review overlays only for label-quality validation; never use them as model inputs.",
            "5. Create train/test splits by `split_group` only to avoid leakage across the same source recording.",
            "6. Mark visually noisy, ambiguous, or multi-call crops before any classifier experiment.",
        ]
    )
    if invalid_rows:
        lines.extend(["", "## Missing or invalid rows", ""])
        for row in invalid_rows[:20]:
            lines.append(f"- `{row['sample_id']}`: {row['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_checklist(path: Path) -> None:
    lines = [
        "# Multi-Species Event Dataset Quality Review Checklist",
        "",
        "Use this checklist before any classifier or VLM experiment.",
        "",
        "- [ ] Clean images contain no GT boxes, labels, or diagnostic overlay text.",
        "- [ ] All clean images are `800x600` pixels.",
        "- [ ] The target event is visually present and horizontally centred in each image.",
        "- [ ] Silence-padded boundary crops are reviewed for interpretability.",
        "- [ ] Species folders are used only for dataset organisation, not as model-visible prompt text.",
        "- [ ] Human-review overlays are never used as model input.",
        "- [ ] Train/test split is performed by `split_group`, not by individual sample.",
        "- [ ] Myotis and Plecotus examples receive extra manual ambiguity review.",
        "- [ ] Ozimops petersi is treated as an optional benchmark anchor, not mixed into UK-only conclusions without disclosure.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--strict-centred", action="store_true")
    parser.add_argument("--time-context-seconds", type=float, default=DEFAULT_TIME_CONTEXT_SECONDS)
    parser.add_argument("--max-frequency-hz", type=float, default=DEFAULT_MAX_FREQ_HZ)
    parser.add_argument("--image-width-px", type=int, default=DEFAULT_IMAGE_WIDTH_PX)
    parser.add_argument("--image-height-px", type=int, default=DEFAULT_IMAGE_HEIGHT_PX)
    parser.add_argument("--skip-overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = (
        read_event_samples_from_v1_manifest(args.input_manifest)
        if args.input_manifest is not None
        else read_event_samples(args.input_csv)
    )
    builder = build_strict_centred_dataset if args.strict_centred else build_dataset
    manifest_rows, invalid_rows = builder(
        rows=samples,
        output_dir=args.output_dir,
        half_context_seconds=args.time_context_seconds,
        max_frequency_hz=args.max_frequency_hz,
        image_width_px=args.image_width_px,
        image_height_px=args.image_height_px,
        write_overlays=not args.skip_overlays,
    )
    write_csv(args.output_dir / "multispecies_event_dataset_manifest.csv", manifest_rows)
    write_csv(args.output_dir / "species_sample_counts.csv", species_counts(manifest_rows))
    write_report(
        path=args.output_dir / "dataset_construction_report.md",
        manifest_rows=manifest_rows,
        invalid_rows=invalid_rows,
        image_width_px=args.image_width_px,
        image_height_px=args.image_height_px,
        half_context_seconds=args.time_context_seconds,
        max_frequency_hz=args.max_frequency_hz,
    )
    write_quality_checklist(args.output_dir / "quality_review_checklist.md")
    if invalid_rows:
        write_csv(args.output_dir / "missing_or_invalid_events.csv", invalid_rows)
    print(f"Read {len(samples)} candidate event row(s)")
    print(f"Generated {len(manifest_rows)} standardised sample(s)")
    print(f"Missing/invalid rows: {len(invalid_rows)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
