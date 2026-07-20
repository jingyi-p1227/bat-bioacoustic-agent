"""Prepare no-inference spectrogram previews for BD2 species candidates.

This script reads candidate rows, source annotation JSON, and WAV audio in a
read-only manner. It writes clean preview images and separate diagnostic
ground-truth overlays for human review only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import make_spectrogram, to_decibels  # noqa: E402
from prepare_agent_spectrogram_inputs import (  # noqa: E402
    DEFAULT_MAX_FREQ_HZ,
    apply_grid_style,
)


DEFAULT_CANDIDATE_CSV = (
    REPO_ROOT
    / "outputs/analysis_reports/bd2_species_pilot_benchmark/pilot_candidate_examples.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/analysis_reports/bd2_species_pilot_visual_spotcheck"
)
DATASET_ROOT = REPO_ROOT.parent / "batdetect2_outputs/datasets"
DEFAULT_MIN_DB = -130.0
DEFAULT_MAX_DB = 0.0
GRID_STYLE = "grid_v2"


@dataclass(frozen=True)
class CandidateRow:
    species: str
    example_index: int
    dataset: str
    source: str
    recording_path: str
    event_count: int
    duration_seconds: float | None
    quality_hint: str
    difficulty_prior: str


def slugify(value: str) -> str:
    """Return a stable filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def parse_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_candidates(path: Path) -> list[CandidateRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                CandidateRow(
                    species=row["species"],
                    example_index=int(row["example_index"]),
                    dataset=row["dataset"],
                    source=row["source"],
                    recording_path=row["recording_path"],
                    event_count=int(row["event_count"]),
                    duration_seconds=parse_float(row.get("duration_seconds")),
                    quality_hint=row.get("quality_hint", ""),
                    difficulty_prior=row.get("difficulty_prior", ""),
                )
            )
    return rows


def resolve_audio_path(candidate: CandidateRow) -> Path:
    """Resolve the source WAV for one candidate without copying it."""
    if candidate.dataset == "australia":
        return DATASET_ROOT / "australia/audio" / candidate.recording_path
    if candidate.dataset == "uk":
        return (
            DATASET_ROOT
            / "uk/sources"
            / candidate.source
            / "audio"
            / candidate.recording_path
        )
    raise ValueError(f"Unsupported dataset: {candidate.dataset}")


def resolve_annotation_path(candidate: CandidateRow) -> Path:
    if candidate.dataset == "australia":
        return DATASET_ROOT / "australia/annotations.json"
    if candidate.dataset == "uk":
        return DATASET_ROOT / "uk/sources" / candidate.source / "annotations.json"
    raise ValueError(f"Unsupported dataset: {candidate.dataset}")


def output_stem(candidate: CandidateRow) -> str:
    return f"{candidate.example_index:02d}_{slugify(candidate.recording_path.rsplit('.', 1)[0])}"


def clean_preview_path(output_dir: Path, candidate: CandidateRow) -> Path:
    return (
        output_dir
        / slugify(candidate.species)
        / f"{output_stem(candidate)}_clean_preview.png"
    )


def diagnostic_overlay_path(output_dir: Path, candidate: CandidateRow) -> Path:
    return (
        output_dir
        / slugify(candidate.species)
        / "diagnostic_overlays_human_review_only"
        / f"{output_stem(candidate)}_gt_diagnostic_overlay.png"
    )


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT.parent.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def spectrogram_image(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, list[float]]:
    spec, stft = make_spectrogram(audio, sample_rate)
    spec_db = np.clip(to_decibels(spec), DEFAULT_MIN_DB, DEFAULT_MAX_DB)
    image = (spec_db - DEFAULT_MIN_DB) / (DEFAULT_MAX_DB - DEFAULT_MIN_DB)
    extent = list(stft.extent(len(audio)))
    extent[2] /= 1000
    extent[3] /= 1000
    return image, extent


def save_spectrogram(
    *,
    audio_path: Path,
    output_path: Path,
    title: str,
    boxes: list[list[float]] | None = None,
    diagnostic_overlay: bool = False,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> tuple[float, int]:
    audio, sample_rate = load_audio(audio_path)
    image, extent = spectrogram_image(audio, sample_rate)
    duration = len(audio) / sample_rate
    displayed_max_freq_hz = min(max_freq_hz, sample_rate / 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )
    ax.set_xlim(0, duration)
    ax.set_ylim(0, displayed_max_freq_hz / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    apply_grid_style(
        ax,
        grid_style=GRID_STYLE,
        duration_seconds=duration,
        displayed_max_freq_hz=displayed_max_freq_hz,
    )

    if boxes:
        for index, (start, low, end, high) in enumerate(boxes, start=1):
            low_khz = low / 1000
            high_khz = high / 1000
            rect = Rectangle(
                (start, low_khz),
                end - start,
                high_khz - low_khz,
                fill=False,
                linewidth=1.0,
                edgecolor="lime",
                linestyle="-",
            )
            ax.add_patch(rect)
            ax.text(
                start,
                high_khz,
                str(index),
                color="yellow",
                fontsize=7,
                va="bottom",
                ha="left",
            )
    if diagnostic_overlay:
        ax.text(
            0.01,
            0.99,
            "DIAGNOSTIC GT OVERLAY - NOT MODEL INPUT",
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
    return duration, sample_rate


def annotation_events_for_recording(candidate: CandidateRow) -> list[list[float]]:
    """Return event boxes for candidate species and recording path."""
    annotation_path = resolve_annotation_path(candidate)
    if not annotation_path.is_file():
        return []
    data = json.loads(annotation_path.read_text(encoding="utf-8")).get("data", {})
    tags = {tag["id"]: tag for tag in data.get("tags", [])}
    recordings = {recording["uuid"]: recording for recording in data.get("recordings", [])}
    events = {event["uuid"]: event for event in data.get("sound_events", [])}
    boxes: list[list[float]] = []
    for annotation in data.get("sound_event_annotations", []):
        event = events.get(annotation.get("sound_event"))
        if not event:
            continue
        recording = recordings.get(event.get("recording"), {})
        if recording.get("path") != candidate.recording_path:
            continue
        species_values = [
            tags[tag_id]["value"]
            for tag_id in annotation.get("tags", [])
            if tag_id in tags and tags[tag_id].get("key") == "dwc:scientificName"
        ]
        if candidate.species not in species_values:
            continue
        coords = event.get("geometry", {}).get("coordinates", [])
        if len(coords) == 4:
            boxes.append([float(value) for value in coords])
    boxes.sort(key=lambda box: (box[0], box[2], box[1], box[3]))
    return boxes


def review_recommendation(row: dict[str, Any]) -> str:
    if row["audio_exists"] != "true":
        return "exclude_missing_audio"
    count = int(row["event_count"])
    hint = row["quality_hint"]
    if hint == "hard_hint":
        return "manual_review_hard_or_noisy_candidate"
    if count >= 25:
        return "good_dense_candidate"
    if count >= 8:
        return "good_moderate_candidate"
    return "manual_review_low_event_count"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_candidates(candidates: list[CandidateRow], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        audio_path = resolve_audio_path(candidate)
        preview_path = clean_preview_path(output_dir, candidate)
        overlay_path = diagnostic_overlay_path(output_dir, candidate)
        boxes = annotation_events_for_recording(candidate)
        audio_exists = audio_path.is_file()
        audio_duration = None
        sample_rate = None
        preview_written = False
        overlay_written = False
        error = ""
        if audio_exists:
            try:
                audio_duration, sample_rate = save_spectrogram(
                    audio_path=audio_path,
                    output_path=preview_path,
                    title=f"{candidate.species} candidate {candidate.example_index:02d}",
                )
                preview_written = True
                _, _ = save_spectrogram(
                    audio_path=audio_path,
                    output_path=overlay_path,
                    title=f"{candidate.species} candidate {candidate.example_index:02d} GT diagnostic",
                    boxes=boxes,
                    diagnostic_overlay=True,
                )
                overlay_written = True
            except Exception as exc:  # keep going so one bad WAV does not stop review package
                error = f"{type(exc).__name__}: {exc}"

        row = {
            "species": candidate.species,
            "example_index": candidate.example_index,
            "dataset": candidate.dataset,
            "source": candidate.source,
            "recording_path": candidate.recording_path,
            "audio_path": portable(audio_path),
            "audio_exists": str(audio_exists).lower(),
            "event_count": candidate.event_count,
            "annotation_event_count": len(boxes),
            "duration_seconds_candidate_csv": candidate.duration_seconds,
            "audio_duration_seconds": round(audio_duration, 6) if audio_duration else "",
            "sample_rate_hz": sample_rate or "",
            "quality_hint": candidate.quality_hint,
            "difficulty_prior": candidate.difficulty_prior,
            "clean_preview_path": portable(preview_path) if preview_written else "",
            "diagnostic_overlay_path": portable(overlay_path) if overlay_written else "",
            "preview_written": str(preview_written).lower(),
            "diagnostic_overlay_written": str(overlay_written).lower(),
            "error": error,
        }
        row["auto_review_recommendation"] = review_recommendation(row)
        rows.append(row)
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_species[row["species"]].append(row)

    lines = [
        "# BD2 Multi-Species Pilot Visual Spot-Check Package",
        "",
        "This package is no-inference. Clean preview images are generated from WAV audio only. Diagnostic GT overlays are stored separately for human review and must not be used as model inputs.",
        "",
        "## Overall status",
        "",
        f"- Candidate rows: {len(rows)}",
        f"- Clean previews written: {sum(row['preview_written'] == 'true' for row in rows)}",
        f"- Diagnostic overlays written: {sum(row['diagnostic_overlay_written'] == 'true' for row in rows)}",
        f"- Missing audio files: {sum(row['audio_exists'] != 'true' for row in rows)}",
        "",
        "## Species review table",
        "",
        "| Species | Candidates | Previews | Events | Filename hints | Provisional recommendation |",
        "|---|---:|---:|---:|---|---|",
    ]
    for species, species_rows in sorted(by_species.items()):
        hint_counts = Counter(row["quality_hint"] for row in species_rows)
        rec_counts = Counter(row["auto_review_recommendation"] for row in species_rows)
        recommendation = (
            "suitable for initial 10-example pilot"
            if all(row["audio_exists"] == "true" for row in species_rows)
            else "needs missing-audio replacement"
        )
        lines.append(
            f"| {species} | {len(species_rows)} | {sum(row['preview_written'] == 'true' for row in species_rows)} | {sum(int(row['event_count']) for row in species_rows)} | {dict(hint_counts)} | {recommendation}; {dict(rec_counts)} |"
        )

    lines.extend(
        [
            "",
            "## Initial recommendations",
            "",
            "- Keep `Rhinolophus hipposideros` and `Rhinolophus ferrumequinum` as likely easier species candidates.",
            "- Keep `Myotis daubentonii`, `Myotis nattereri`, and `Myotis mystacinus` as harder species candidates, but manually inspect dense examples because high event counts may be visually crowded.",
            "- Keep `Plecotus auritus` as the explicit challenging species candidate.",
            "- Include `Pipistrellus pipistrellus` as a moderate UK control candidate if the review set needs a non-Myotis/Rhinolophus comparator.",
            "- Treat `Ozimops petersi` as the existing benchmark anchor by default. It can remain in the multi-species pilot for continuity, but dissertation comparisons should keep it separable from the new UK species benchmark.",
            "",
            "## Clean/noisy caveat",
            "",
            "The current clean/noisy labels are filename-derived triage hints only. They are not visual judgements and should be corrected during manual inspection of the preview images.",
            "",
            "## Review workflow",
            "",
            "1. Open each species folder and inspect `*_clean_preview.png` first.",
            "2. Use `diagnostic_overlays_human_review_only/*_gt_diagnostic_overlay.png` only to check whether labelled boxes cover the visible calls.",
            "3. Mark final selections in `species_candidate_review.csv`; do not use diagnostic overlays as model input.",
            "",
        ]
    )
    (output_dir / "species_candidate_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidate_csv)
    rows = process_candidates(candidates, args.output_dir)
    write_csv(args.output_dir / "species_candidate_review.csv", rows)
    write_summary(args.output_dir, rows)
    print(f"Wrote {len(rows)} candidate review row(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
