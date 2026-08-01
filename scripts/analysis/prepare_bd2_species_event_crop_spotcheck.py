"""Prepare event-centred crop previews for BD2 species pilot candidates.

This no-inference script reads existing annotation boxes and WAV audio to make
manual inspection crops. Clean crops contain no GT overlay. Diagnostic overlays
are written to separate human-review-only folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_prep.prepare_agent_spectrogram_inputs import apply_grid_style  # noqa: E402
from scripts.analysis.prepare_bd2_species_visual_spotcheck import (  # noqa: E402
    DEFAULT_CANDIDATE_CSV,
    DEFAULT_MAX_DB,
    DEFAULT_MIN_DB,
    GRID_STYLE,
    CandidateRow,
    load_audio,
    load_candidates,
    portable,
    resolve_annotation_path,
    resolve_audio_path,
    slugify,
    spectrogram_image,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/analysis_reports/bd2_species_event_crop_spotcheck"
)
DEFAULT_MAX_EVENTS_PER_CANDIDATE = 3
DEFAULT_TIME_MARGIN_SECONDS = 0.1
DEFAULT_FREQUENCY_MARGIN_HZ = 10_000.0


@dataclass(frozen=True)
class EventRecord:
    event_uuid: str
    event_rank: int
    start_time: float
    low_frequency: float
    end_time: float
    high_frequency: float
    tag_values: tuple[str, ...]


@dataclass(frozen=True)
class CropBounds:
    time_start: float
    time_end: float
    frequency_low_hz: float
    frequency_high_hz: float


def event_recording_stem(candidate: CandidateRow) -> str:
    return f"{candidate.example_index:02d}_{slugify(candidate.recording_path.rsplit('.', 1)[0])}"


def clean_crop_path(output_dir: Path, candidate: CandidateRow, event: EventRecord) -> Path:
    return (
        output_dir
        / slugify(candidate.species)
        / "clean_event_crops"
        / f"{event_recording_stem(candidate)}_event_{event.event_rank:03d}_clean_crop.png"
    )


def two_panel_path(output_dir: Path, candidate: CandidateRow, event: EventRecord) -> Path:
    return (
        output_dir
        / slugify(candidate.species)
        / "two_panel_previews"
        / f"{event_recording_stem(candidate)}_event_{event.event_rank:03d}_two_panel.png"
    )


def diagnostic_crop_path(output_dir: Path, candidate: CandidateRow, event: EventRecord) -> Path:
    return (
        output_dir
        / slugify(candidate.species)
        / "gt_crop_overlays_human_review_only"
        / f"{event_recording_stem(candidate)}_event_{event.event_rank:03d}_gt_diagnostic_crop.png"
    )


def valid_box(coordinates: list[Any]) -> bool:
    if len(coordinates) != 4:
        return False
    try:
        start, low, end, high = [float(value) for value in coordinates]
    except (TypeError, ValueError):
        return False
    return start < end and low < high and start >= 0 and low >= 0


def load_event_records_for_candidate(candidate: CandidateRow) -> tuple[list[EventRecord], int]:
    """Load sorted species-matching event records and count invalid boxes."""
    annotation_path = resolve_annotation_path(candidate)
    if not annotation_path.is_file():
        return [], 0
    data = json.loads(annotation_path.read_text(encoding="utf-8")).get("data", {})
    tags = {tag["id"]: tag for tag in data.get("tags", [])}
    recordings = {recording["uuid"]: recording for recording in data.get("recordings", [])}
    events = {event["uuid"]: event for event in data.get("sound_events", [])}
    records: list[EventRecord] = []
    invalid_count = 0
    for annotation in data.get("sound_event_annotations", []):
        event = events.get(annotation.get("sound_event"))
        if not event:
            continue
        recording = recordings.get(event.get("recording"), {})
        if recording.get("path") != candidate.recording_path:
            continue
        tag_values = tuple(
            tags[tag_id]["value"]
            for tag_id in annotation.get("tags", [])
            if tag_id in tags
        )
        species_values = tuple(
            tags[tag_id]["value"]
            for tag_id in annotation.get("tags", [])
            if tag_id in tags and tags[tag_id].get("key") == "dwc:scientificName"
        )
        if candidate.species not in species_values:
            continue
        coordinates = event.get("geometry", {}).get("coordinates", [])
        if not valid_box(coordinates):
            invalid_count += 1
            continue
        start, low, end, high = [float(value) for value in coordinates]
        records.append(
            EventRecord(
                event_uuid=str(event.get("uuid", "")),
                event_rank=0,
                start_time=start,
                low_frequency=low,
                end_time=end,
                high_frequency=high,
                tag_values=tag_values,
            )
        )
    records.sort(key=lambda event: (event.start_time, event.end_time, event.low_frequency))
    ranked = [
        EventRecord(
            event_uuid=event.event_uuid,
            event_rank=index,
            start_time=event.start_time,
            low_frequency=event.low_frequency,
            end_time=event.end_time,
            high_frequency=event.high_frequency,
            tag_values=event.tag_values,
        )
        for index, event in enumerate(records, start=1)
    ]
    return ranked, invalid_count


def select_representative_events(
    events: list[EventRecord],
    *,
    max_events: int = DEFAULT_MAX_EVENTS_PER_CANDIDATE,
) -> list[EventRecord]:
    """Choose first/middle/last events for compact manual inspection."""
    if max_events <= 0:
        raise ValueError("max_events must be positive")
    if len(events) <= max_events:
        return events
    if max_events == 1:
        return [events[len(events) // 2]]
    indices = sorted(
        {
            round(position * (len(events) - 1) / (max_events - 1))
            for position in range(max_events)
        }
    )
    return [events[index] for index in indices]


def crop_bounds_for_event(
    event: EventRecord,
    *,
    duration_seconds: float,
    nyquist_hz: float,
    time_margin_seconds: float = DEFAULT_TIME_MARGIN_SECONDS,
    frequency_margin_hz: float = DEFAULT_FREQUENCY_MARGIN_HZ,
) -> CropBounds:
    """Return clamped event-centred crop bounds."""
    return CropBounds(
        time_start=max(0.0, event.start_time - time_margin_seconds),
        time_end=min(duration_seconds, event.end_time + time_margin_seconds),
        frequency_low_hz=max(0.0, event.low_frequency - frequency_margin_hz),
        frequency_high_hz=min(nyquist_hz, event.high_frequency + frequency_margin_hz),
    )


def plot_matrix(
    ax,
    *,
    image: np.ndarray,
    extent: list[float],
    title: str,
    xlim: tuple[float, float],
    ylim_hz: tuple[float, float],
    duration_seconds: float,
    displayed_max_freq_hz: float,
) -> None:
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim_hz[0] / 1000, ylim_hz[1] / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    apply_grid_style(
        ax,
        grid_style=GRID_STYLE,
        duration_seconds=max(0.001, xlim[1] - xlim[0]),
        displayed_max_freq_hz=displayed_max_freq_hz,
    )


def draw_gt_box(ax, event: EventRecord) -> None:
    rect = Rectangle(
        (event.start_time, event.low_frequency / 1000),
        event.end_time - event.start_time,
        (event.high_frequency - event.low_frequency) / 1000,
        fill=False,
        edgecolor="lime",
        linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(
        event.start_time,
        event.high_frequency / 1000,
        f"GT {event.event_rank}",
        color="yellow",
        fontsize=7,
        va="bottom",
        ha="left",
    )


def save_clean_crop(
    *,
    image: np.ndarray,
    extent: list[float],
    candidate: CandidateRow,
    event: EventRecord,
    bounds: CropBounds,
    duration_seconds: float,
    nyquist_hz: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_matrix(
        ax,
        image=image,
        extent=extent,
        title=f"{candidate.species} event crop",
        xlim=(bounds.time_start, bounds.time_end),
        ylim_hz=(bounds.frequency_low_hz, bounds.frequency_high_hz),
        duration_seconds=duration_seconds,
        displayed_max_freq_hz=bounds.frequency_high_hz - bounds.frequency_low_hz,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_diagnostic_crop(
    *,
    image: np.ndarray,
    extent: list[float],
    candidate: CandidateRow,
    event: EventRecord,
    bounds: CropBounds,
    duration_seconds: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_matrix(
        ax,
        image=image,
        extent=extent,
        title=f"{candidate.species} GT diagnostic crop",
        xlim=(bounds.time_start, bounds.time_end),
        ylim_hz=(bounds.frequency_low_hz, bounds.frequency_high_hz),
        duration_seconds=duration_seconds,
        displayed_max_freq_hz=bounds.frequency_high_hz - bounds.frequency_low_hz,
    )
    draw_gt_box(ax, event)
    ax.text(
        0.01,
        0.99,
        "GT DIAGNOSTIC OVERLAY - NOT MODEL INPUT",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="yellow",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_two_panel(
    *,
    image: np.ndarray,
    extent: list[float],
    candidate: CandidateRow,
    bounds: CropBounds,
    duration_seconds: float,
    nyquist_hz: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_matrix(
        axes[0],
        image=image,
        extent=extent,
        title="Full context",
        xlim=(0.0, duration_seconds),
        ylim_hz=(0.0, min(nyquist_hz, 120_000.0)),
        duration_seconds=duration_seconds,
        displayed_max_freq_hz=min(nyquist_hz, 120_000.0),
    )
    plot_matrix(
        axes[1],
        image=image,
        extent=extent,
        title="Event-centred crop",
        xlim=(bounds.time_start, bounds.time_end),
        ylim_hz=(bounds.frequency_low_hz, bounds.frequency_high_hz),
        duration_seconds=duration_seconds,
        displayed_max_freq_hz=bounds.frequency_high_hz - bounds.frequency_low_hz,
    )
    fig.suptitle(f"{candidate.species} - {candidate.recording_path}", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_candidate(
    candidate: CandidateRow,
    output_dir: Path,
    *,
    max_events_per_candidate: int,
) -> list[dict[str, Any]]:
    audio_path = resolve_audio_path(candidate)
    events, invalid_count = load_event_records_for_candidate(candidate)
    selected_events = select_representative_events(events, max_events=max_events_per_candidate)
    if not audio_path.is_file():
        return [
            {
                "species": candidate.species,
                "example_index": candidate.example_index,
                "recording_path": candidate.recording_path,
                "event_uuid": "",
                "source_event_rank": "",
                "audio_exists": "false",
                "crop_written": "false",
                "diagnostic_overlay_written": "false",
                "two_panel_written": "false",
                "invalid_gt_box_count_for_candidate": invalid_count,
                "error": "audio_missing",
            }
        ]

    audio, sample_rate = load_audio(audio_path)
    image, extent = spectrogram_image(audio, sample_rate)
    duration_seconds = len(audio) / sample_rate
    nyquist_hz = sample_rate / 2
    rows: list[dict[str, Any]] = []

    for event in selected_events:
        bounds = crop_bounds_for_event(
            event,
            duration_seconds=duration_seconds,
            nyquist_hz=nyquist_hz,
        )
        clean_path = clean_crop_path(output_dir, candidate, event)
        overlay_path = diagnostic_crop_path(output_dir, candidate, event)
        panel_path = two_panel_path(output_dir, candidate, event)
        error = ""
        crop_written = overlay_written = panel_written = False
        try:
            save_clean_crop(
                image=image,
                extent=extent,
                candidate=candidate,
                event=event,
                bounds=bounds,
                duration_seconds=duration_seconds,
                nyquist_hz=nyquist_hz,
                output_path=clean_path,
            )
            crop_written = True
            save_diagnostic_crop(
                image=image,
                extent=extent,
                candidate=candidate,
                event=event,
                bounds=bounds,
                duration_seconds=duration_seconds,
                output_path=overlay_path,
            )
            overlay_written = True
            save_two_panel(
                image=image,
                extent=extent,
                candidate=candidate,
                bounds=bounds,
                duration_seconds=duration_seconds,
                nyquist_hz=nyquist_hz,
                output_path=panel_path,
            )
            panel_written = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "species": candidate.species,
                "example_index": candidate.example_index,
                "dataset": candidate.dataset,
                "source": candidate.source,
                "recording_path": candidate.recording_path,
                "audio_path": portable(audio_path),
                "audio_exists": "true",
                "source_species_label": candidate.species,
                "source_event_rank": event.event_rank,
                "event_uuid": event.event_uuid,
                "event_start_time_seconds": event.start_time,
                "event_end_time_seconds": event.end_time,
                "event_low_frequency_hz": event.low_frequency,
                "event_high_frequency_hz": event.high_frequency,
                "event_duration_ms": round((event.end_time - event.start_time) * 1000, 6),
                "event_bandwidth_hz": round(event.high_frequency - event.low_frequency, 6),
                "candidate_event_count": candidate.event_count,
                "available_gt_event_count": len(events),
                "invalid_gt_box_count_for_candidate": invalid_count,
                "audio_duration_seconds": round(duration_seconds, 6),
                "sample_rate_hz": sample_rate,
                "crop_time_start_seconds": round(bounds.time_start, 6),
                "crop_time_end_seconds": round(bounds.time_end, 6),
                "crop_low_frequency_hz": round(bounds.frequency_low_hz, 6),
                "crop_high_frequency_hz": round(bounds.frequency_high_hz, 6),
                "quality_hint": candidate.quality_hint,
                "difficulty_prior": candidate.difficulty_prior,
                "clean_crop_path": portable(clean_path) if crop_written else "",
                "two_panel_preview_path": portable(panel_path) if panel_written else "",
                "diagnostic_overlay_path": portable(overlay_path) if overlay_written else "",
                "crop_written": str(crop_written).lower(),
                "two_panel_written": str(panel_written).lower(),
                "diagnostic_overlay_written": str(overlay_written).lower(),
                "error": error,
            }
        )
    if not selected_events:
        rows.append(
            {
                "species": candidate.species,
                "example_index": candidate.example_index,
                "dataset": candidate.dataset,
                "source": candidate.source,
                "recording_path": candidate.recording_path,
                "audio_path": portable(audio_path),
                "audio_exists": "true",
                "source_species_label": candidate.species,
                "source_event_rank": "",
                "event_uuid": "",
                "candidate_event_count": candidate.event_count,
                "available_gt_event_count": len(events),
                "invalid_gt_box_count_for_candidate": invalid_count,
                "crop_written": "false",
                "two_panel_written": "false",
                "diagnostic_overlay_written": "false",
                "error": "no_valid_gt_events",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "species",
        "example_index",
        "dataset",
        "source",
        "recording_path",
        "source_event_rank",
        "event_uuid",
        "event_start_time_seconds",
        "event_end_time_seconds",
        "event_low_frequency_hz",
        "event_high_frequency_hz",
        "event_duration_ms",
        "event_bandwidth_hz",
        "crop_time_start_seconds",
        "crop_time_end_seconds",
        "crop_low_frequency_hz",
        "crop_high_frequency_hz",
        "clean_crop_path",
        "two_panel_preview_path",
        "diagnostic_overlay_path",
    ]
    ordered = [field for field in preferred if field in fieldnames] + [
        field for field in fieldnames if field not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_species[row["species"]].append(row)
    lines = [
        "# BD2 Multi-Species Event-Crop Spot-Check Package",
        "",
        "This package is no-inference. Clean event crops and two-panel previews are generated from WAV audio plus existing GT box coordinates. Diagnostic GT crop overlays are stored separately and must not be used as model inputs.",
        "",
        "## Overall status",
        "",
        f"- Review rows: {len(rows)}",
        f"- Clean crops written: {sum(row.get('crop_written') == 'true' for row in rows)}",
        f"- Two-panel previews written: {sum(row.get('two_panel_written') == 'true' for row in rows)}",
        f"- Diagnostic overlays written: {sum(row.get('diagnostic_overlay_written') == 'true' for row in rows)}",
        f"- Missing audio rows: {sum(row.get('audio_exists') != 'true' for row in rows)}",
        f"- Rows with errors: {sum(bool(row.get('error')) for row in rows)}",
        "",
        "## Crops per species",
        "",
        "| Species | Crop rows | Clean crops | Candidate recordings | Invalid GT boxes | Event-duration range ms | Filename hints |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for species, species_rows in sorted(by_species.items()):
        durations = [
            float(row["event_duration_ms"])
            for row in species_rows
            if row.get("event_duration_ms") not in {"", None}
        ]
        duration_range = (
            f"{min(durations):.2f}-{max(durations):.2f}"
            if durations
            else "n/a"
        )
        invalid_boxes = sum(int(row.get("invalid_gt_box_count_for_candidate") or 0) for row in species_rows)
        candidates = {(row["example_index"], row["recording_path"]) for row in species_rows}
        hints = Counter(row.get("quality_hint", "") for row in species_rows)
        lines.append(
            f"| {species} | {len(species_rows)} | {sum(row.get('crop_written') == 'true' for row in species_rows)} | {len(candidates)} | {invalid_boxes} | {duration_range} | {dict(hints)} |"
        )
    lines.extend(
        [
            "",
            "## Initial interpretation",
            "",
            "- Event-centred crops should be more useful than full-window spectrograms for manual species-classification review because each image focuses on individual call geometry rather than dense whole-recording context.",
            "- Dense species/candidates are still represented by only up to three events per recording; this is a review package, not a full benchmark export.",
            "- Filename hints remain triage metadata only. Visual clean/noisy judgements should be added manually after inspecting the clean crops and two-panel previews.",
            "- If crops look too small, increase the time/frequency margins in the script; if they include too much neighbouring activity, reduce margins or choose less dense recordings.",
            "",
            "## Review workflow",
            "",
            "1. Inspect `clean_event_crops/*_clean_crop.png` for model-style event appearance.",
            "2. Inspect `two_panel_previews/*_two_panel.png` to verify local crop context without GT overlay.",
            "3. Use `gt_crop_overlays_human_review_only/*_gt_diagnostic_crop.png` only for human label-quality checks.",
            "4. Record final include/exclude decisions in `event_crop_review.csv` or a copied review sheet.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-events-per-candidate", type=int, default=DEFAULT_MAX_EVENTS_PER_CANDIDATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidate_csv)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.extend(
            process_candidate(
                candidate,
                args.output_dir,
                max_events_per_candidate=args.max_events_per_candidate,
            )
        )
    write_csv(args.output_dir / "event_crop_review.csv", rows)
    write_summary(args.output_dir / "event_crop_summary.md", rows)
    print(f"Wrote {len(rows)} event-crop review row(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
