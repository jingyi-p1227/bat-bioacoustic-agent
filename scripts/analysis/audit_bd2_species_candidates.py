"""Audit local BatDetect2 datasets for multi-species pilot candidates.

The script reads annotation JSON files only. It does not inspect audio content,
copy WAV files, run inference, or modify source datasets.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT.parent / "batdetect2_outputs/datasets"
OUTPUT_DIR = REPO_ROOT / "outputs/analysis_reports/bd2_species_pilot_benchmark"
REFERENCE_DIR = REPO_ROOT / "docs/acoustic_reference_library"
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
class RecordingSpeciesStats:
    dataset: str
    source: str
    species: str
    recording_path: str
    event_count: int
    duration_seconds: float | None
    quality_hint: str


def annotation_paths() -> list[tuple[str, str, Path]]:
    paths: list[tuple[str, str, Path]] = []
    paths.append(("australia", "australia", DATASET_ROOT / "australia/annotations.json"))
    for path in sorted((DATASET_ROOT / "uk/sources").glob("*/annotations.json")):
        paths.append(("uk", path.parent.name, path))
    return paths


def quality_hint_from_name(name: str) -> str:
    """Return a lightweight filename-derived inspection hint, not a GT label."""
    lower = name.lower()
    if any(term in lower for term in ("clean", "good")):
        return "clean_hint"
    if any(term in lower for term in ("mix", "overlap", "noisy", "low", "small", "border")):
        return "hard_hint"
    if any(term in lower for term in ("feed", "soc", "social", "return", "circ", "rel", "check", "_rc", "a_rc")):
        return "context_or_quality_hint"
    return "unmarked"


def difficulty_label(species: str) -> str:
    if species.startswith("Rhinolophus "):
        return "likely_easier"
    if species.startswith("Myotis "):
        return "likely_harder"
    if species == "Plecotus auritus":
        return "challenging"
    return "moderate_or_unknown"


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def load_annotation_summary(dataset: str, source: str, path: Path) -> tuple[list[dict[str, Any]], list[RecordingSpeciesStats], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", {})
    tags = {tag["id"]: tag for tag in data.get("tags", [])}
    recordings = {recording["uuid"]: recording for recording in data.get("recordings", [])}
    events = {event["uuid"]: event for event in data.get("sound_events", [])}
    annotations = data.get("sound_event_annotations", [])
    if not events or not annotations:
        return [], [], {
            "dataset": dataset,
            "source": source,
            "annotation_path": str(path.relative_to(REPO_ROOT.parent)),
            "status": "no_event_boxes",
            "recordings": len(recordings),
            "events": len(events),
        }

    species_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    recording_species_counts: Counter[tuple[str, str]] = Counter()
    call_type_counter: Counter[str] = Counter()
    for annotation in annotations:
        event = events.get(annotation.get("sound_event"))
        if not event:
            continue
        species_values = [
            tags[tag_id]["value"]
            for tag_id in annotation.get("tags", [])
            if tag_id in tags and tags[tag_id].get("key") == "dwc:scientificName"
        ]
        call_types = [
            tags[tag_id]["value"]
            for tag_id in annotation.get("tags", [])
            if tag_id in tags and tags[tag_id].get("key") == "soundevent:call_type"
        ]
        for call_type in call_types:
            call_type_counter[call_type] += 1
        for species in species_values:
            species_events[species].append(event)
            recording = recordings.get(event.get("recording"), {})
            recording_species_counts[(species, recording.get("path", ""))] += 1

    species_rows: list[dict[str, Any]] = []
    for species, species_event_list in sorted(species_events.items()):
        durations = []
        bandwidths = []
        frequency_centers = []
        event_recordings = set()
        quality_counter: Counter[str] = Counter()
        for event in species_event_list:
            coords = event.get("geometry", {}).get("coordinates", [])
            if len(coords) == 4:
                start, low, end, high = [safe_float(value) for value in coords]
                if start is not None and end is not None:
                    durations.append(max(0.0, end - start) * 1000)
                if low is not None and high is not None:
                    bandwidths.append(max(0.0, high - low))
                    frequency_centers.append((low + high) / 2)
            recording = recordings.get(event.get("recording"), {})
            recording_path = recording.get("path", "")
            if recording_path:
                event_recordings.add(recording_path)
                quality_counter[quality_hint_from_name(recording_path)] += 1
        species_rows.append(
            {
                "dataset": dataset,
                "source": source,
                "species": species,
                "event_count": len(species_event_list),
                "recording_count": len(event_recordings),
                "difficulty_prior": difficulty_label(species),
                "median_duration_ms": median(durations),
                "median_bandwidth_hz": median(bandwidths),
                "median_frequency_center_hz": median(frequency_centers),
                "call_type_counts": json.dumps(dict(call_type_counter), sort_keys=True),
                "quality_hint_counts": json.dumps(dict(quality_counter), sort_keys=True),
                "annotation_path": str(path.relative_to(REPO_ROOT.parent)),
            }
        )

    recording_rows = [
        RecordingSpeciesStats(
            dataset=dataset,
            source=source,
            species=species,
            recording_path=recording_path,
            event_count=count,
            duration_seconds=safe_float(
                next(
                    (
                        recording.get("duration")
                        for recording in recordings.values()
                        if recording.get("path") == recording_path
                    ),
                    None,
                )
            ),
            quality_hint=quality_hint_from_name(recording_path),
        )
        for (species, recording_path), count in sorted(recording_species_counts.items())
    ]
    skipped = {
        "dataset": dataset,
        "source": source,
        "annotation_path": str(path.relative_to(REPO_ROOT.parent)),
        "status": "counted",
        "recordings": len(recordings),
        "events": len(events),
    }
    return species_rows, recording_rows, skipped


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_pilot_examples(recording_rows: list[RecordingSpeciesStats]) -> list[dict[str, Any]]:
    by_species: dict[str, list[RecordingSpeciesStats]] = defaultdict(list)
    for row in recording_rows:
        if row.species in TARGET_SPECIES:
            by_species[row.species].append(row)

    selected: list[dict[str, Any]] = []
    for species in TARGET_SPECIES:
        candidates = sorted(
            by_species.get(species, []),
            key=lambda row: (
                0 if row.quality_hint != "unmarked" else 1,
                -row.event_count,
                row.dataset,
                row.source,
                row.recording_path,
            ),
        )
        for index, row in enumerate(candidates[:10], start=1):
            selected.append(
                {
                    "species": species,
                    "example_index": index,
                    "dataset": row.dataset,
                    "source": row.source,
                    "recording_path": row.recording_path,
                    "event_count": row.event_count,
                    "duration_seconds": row.duration_seconds,
                    "quality_hint": row.quality_hint,
                    "difficulty_prior": difficulty_label(species),
                    "selection_note": "initial_candidate_no_audio_copied",
                }
            )
    return selected


def aggregate_species_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        species = row["species"]
        target = grouped.setdefault(
            species,
            {
                "species": species,
                "total_event_count": 0,
                "total_recording_count": 0,
                "datasets_sources": set(),
                "difficulty_prior": difficulty_label(species),
            },
        )
        target["total_event_count"] += int(row["event_count"])
        target["total_recording_count"] += int(row["recording_count"])
        target["datasets_sources"].add(f"{row['dataset']}:{row['source']}")
    output = []
    for item in grouped.values():
        output.append(
            {
                **{key: value for key, value in item.items() if key != "datasets_sources"},
                "datasets_sources": "; ".join(sorted(item["datasets_sources"])),
            }
        )
    return sorted(output, key=lambda row: (-row["total_event_count"], row["species"]))


def create_reference_library_structure(selected_species: list[str]) -> None:
    for directory in (
        REFERENCE_DIR / "raw_sources",
        REFERENCE_DIR / "full_text",
        REFERENCE_DIR / "evidence_chunks",
        REFERENCE_DIR / "species_cards",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch()

    readme = [
        "# Acoustic Reference Library",
        "",
        "Draft structure for species-level acoustic evidence used in future benchmark and prompt design.",
        "",
        "## Directories",
        "",
        "- `raw_sources/`: PDFs or source documents retained outside generated model inputs.",
        "- `full_text/`: extracted full text or human-readable source notes.",
        "- `evidence_chunks/`: small source-grounded claims with citation/provenance.",
        "- `species_cards/`: structured species acoustic cards. Do not fill acoustic ranges unless supported by a verified source.",
        "",
        "## Current status",
        "",
        "This is a scaffold only. The dataset audit supplies labelled-event availability, not literature-verified acoustic parameters.",
        "",
    ]
    (REFERENCE_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")

    template = [
        "# Species Acoustic Card: <species>",
        "",
        "- scientific_name: `<species>`",
        "- common_name: TODO",
        "- evidence_status: TODO verified literature needed",
        "- dataset_availability: TODO generated by dataset audit",
        "- expected_call_shape: TODO",
        "- expected_frequency_range_hz: TODO",
        "- expected_duration_ms: TODO",
        "- confusion_species: TODO",
        "- annotation_guidance: TODO",
        "- source_evidence_ids: []",
        "- limitations: Do not use this card as model context until evidence fields are populated from verified sources.",
        "",
    ]
    (REFERENCE_DIR / "species_cards/TEMPLATE.md").write_text("\n".join(template), encoding="utf-8")

    for species in selected_species:
        filename = species.lower().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "") + ".md"
        path = REFERENCE_DIR / "species_cards" / filename
        if not path.exists():
            lines = [
                f"# Species Acoustic Card: {species}",
                "",
                f"- scientific_name: `{species}`",
                "- evidence_status: dataset-count scaffold only",
                "- dataset_availability: see `outputs/analysis_reports/bd2_species_pilot_benchmark/species_counts.csv`",
                "- expected_call_shape: TODO verified literature/source evidence required",
                "- expected_frequency_range_hz: TODO verified literature/source evidence required",
                "- expected_duration_ms: TODO verified literature/source evidence required",
                "- confusion_species: TODO",
                "- annotation_guidance: TODO",
                "- source_evidence_ids: []",
                "- limitations: This card does not yet contain literature-verified acoustic parameters.",
                "",
            ]
            path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    species_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    file_status_rows: list[dict[str, Any]],
) -> None:
    selected_species = list(TARGET_SPECIES)
    top_rows = [row for row in aggregate_rows if row["species"] in selected_species]
    lines = [
        "# BD2 Species Candidate Audit",
        "",
        "This no-inference audit inspected local BatDetect2-style annotation JSON files for Australia and UK datasets. It did not read WAV samples beyond filenames, and did not modify source data.",
        "",
        "## Files inspected",
        "",
    ]
    for row in file_status_rows:
        lines.append(
            f"- `{row['annotation_path']}`: {row['status']}, recordings={row['recordings']}, event_boxes={row['events']}"
        )
    lines.extend(
        [
            "",
            "## Candidate species summary",
            "",
            "| Species | Events | Recordings | Difficulty prior | Sources |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in top_rows:
        lines.append(
            f"| {row['species']} | {row['total_event_count']} | {row['total_recording_count']} | {row['difficulty_prior']} | {row['datasets_sources']} |"
        )
    lines.extend(
        [
            "",
            "## Easy / hard candidates",
            "",
            "- Likely easier: `Rhinolophus hipposideros`, `Rhinolophus ferrumequinum`, because the user flagged Rhinolophus as likely easier and the local UK Rhinolophus source has dense labelled event boxes.",
            "- Likely harder: `Myotis daubentonii`, `Myotis nattereri`, `Myotis mystacinus`, because Myotis species are explicitly requested as harder candidates and have many labelled examples across UK sources.",
            "- Challenging: `Plecotus auritus`, explicitly requested as challenging and available in UK sources.",
            "- Existing anchor: `Ozimops petersi`, already used in the current Ozimops benchmark and available in Australia annotations.",
            "",
            "## Draft pilot benchmark",
            "",
            "Initial recommendation: 8 species x about 10 recording-level candidates per species. This is a candidate pool, not a frozen benchmark. The next step should inspect spectrogram quality and prevent source-recording leakage before clipping.",
            "",
            "Selected species:",
        ]
    )
    for species in selected_species:
        lines.append(f"- {species}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `species_counts.csv`: source-level species/event counts.",
            "- `species_counts_aggregated.csv`: species totals across inspected Australia/UK annotations.",
            "- `pilot_candidate_examples.csv`: deterministic candidate recording list, up to 10 per selected species.",
            "- `docs/acoustic_reference_library/`: draft library structure for raw sources, extracted text, evidence chunks, and structured species cards.",
            "",
            "## Caveats",
            "",
            "- Filename hints such as `clean`, `mix`, `overlap`, or `low` are not ground truth; they are only triage cues for manual spectrogram inspection.",
            "- Species acoustic cards currently contain TODO placeholders. Acoustic ranges should be populated only from verified literature or checked dataset evidence.",
            "- Some UK annotation sources contain recordings/tasks but no sound-event boxes; these are reported as `no_event_boxes` and excluded from event counts.",
        ]
    )
    (OUTPUT_DIR / "bd2_species_pilot_benchmark_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    species_rows: list[dict[str, Any]] = []
    recording_rows: list[RecordingSpeciesStats] = []
    file_status_rows: list[dict[str, Any]] = []
    for dataset, source, path in annotation_paths():
        if not path.is_file():
            continue
        rows, recordings, status = load_annotation_summary(dataset, source, path)
        species_rows.extend(rows)
        recording_rows.extend(recordings)
        file_status_rows.append(status)

    aggregate_rows = aggregate_species_rows(species_rows)
    selected_rows = select_pilot_examples(recording_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "species_counts.csv", species_rows)
    write_csv(OUTPUT_DIR / "species_counts_aggregated.csv", aggregate_rows)
    write_csv(OUTPUT_DIR / "annotation_file_status.csv", file_status_rows)
    write_csv(OUTPUT_DIR / "pilot_candidate_examples.csv", selected_rows)
    create_reference_library_structure(list(TARGET_SPECIES))
    write_report(species_rows, aggregate_rows, selected_rows, file_status_rows)
    print(f"Wrote BD2 species audit to {OUTPUT_DIR}")
    print(f"Wrote acoustic reference scaffold to {REFERENCE_DIR}")


if __name__ == "__main__":
    main()
