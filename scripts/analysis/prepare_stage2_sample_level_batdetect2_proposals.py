"""Prepare sample-level BatDetect2 proposals for Stage 2 multispecies windows.

The Stage 1 classification images are centred 0.300 s event windows. This
script exports the matching silence-padded audio windows, optionally runs
BatDetect2 in an isolated environment, converts detector output into a compact
proposal schema, and evaluates proposal-only localisation against the single
target event in each sample.

No ground-truth boxes are used as proposals. Ground truth is used only for the
final proposal-only audit.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_stage1_gt_event_classification_dataset/"
    "stage1_manifest.csv"
)
DEFAULT_V2_MANIFEST = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_event_level_dataset_v2_centred/"
    "multispecies_event_dataset_manifest.csv"
)
DEFAULT_TOOL_DIR = (
    REPO_ROOT / "outputs/tool_outputs/batdetect2_multispecies_stage1_windows"
)
DEFAULT_ANALYSIS_DIR = (
    REPO_ROOT
    / "outputs/analysis_reports/multispecies_classification/"
    "stage2_sample_level_batdetect2_proposal_audit"
)
WINDOW_SECONDS = 0.300
HALF_CONTEXT_SECONDS = 0.150
DEFAULT_MIN_DET_PROB = 0.30
DEFAULT_DETECTION_THRESHOLD = 0.01
DEFAULT_CHUNK_SIZE = 2.0
ALLOWED_SPECIES = (
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
class WindowSpec:
    sample_id: str
    anonymous_sample_id: str
    species: str
    source_dataset: str
    source_recording: str
    source_recording_id: str
    split_group: str
    event_index: int
    event_start_time: float
    event_end_time: float
    event_low_freq: float
    event_high_freq: float
    requested_start_time: float
    requested_end_time: float
    actual_audio_start_time: float
    actual_audio_end_time: float
    left_padding_seconds: float
    right_padding_seconds: float
    sample_rate_hz: int
    audio_duration_seconds: float
    original_audio_path: Path

    @property
    def local_gt_start(self) -> float:
        return round(self.event_start_time - self.requested_start_time, 6)

    @property
    def local_gt_end(self) -> float:
        return round(self.event_end_time - self.requested_start_time, 6)


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT.parent).as_posix()
        except ValueError:
            return path.as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (REPO_ROOT, REPO_ROOT.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def infer_audio_path(stage1_row: dict[str, str]) -> Path:
    source_dataset = stage1_row["source_dataset"]
    source_recording = stage1_row["source_recording"]
    if source_dataset == "australia":
        return REPO_ROOT.parent / "batdetect2_outputs/datasets/australia/audio" / source_recording
    source = stage1_row["source_recording_id"].split("_", 2)[1]
    return (
        REPO_ROOT.parent
        / "batdetect2_outputs/datasets/uk/sources"
        / source
        / "audio"
        / source_recording
    )


def build_window_specs(
    *,
    stage1_rows: list[dict[str, str]],
    v2_rows: list[dict[str, str]],
) -> list[WindowSpec]:
    """Join Stage 1 rows to V2 window metadata and return exact window specs."""

    v2_by_sample_id = {row["sample_id"]: row for row in v2_rows}
    specs: list[WindowSpec] = []
    for row in stage1_rows:
        v2 = v2_by_sample_id.get(row["sample_id"], {})
        event_start = _float(row, "event_start_time")
        event_end = _float(row, "event_end_time")
        requested_start = float(v2.get("requested_start_time") or ((event_start + event_end) / 2 - HALF_CONTEXT_SECONDS))
        requested_end = float(v2.get("requested_end_time") or (requested_start + WINDOW_SECONDS))
        audio_duration = float(v2.get("audio_duration_seconds") or 0.0)
        original_audio = resolve_path(v2["original_audio_path"]) if v2.get("original_audio_path") else infer_audio_path(row)
        specs.append(
            WindowSpec(
                sample_id=row["sample_id"],
                anonymous_sample_id=row["anonymous_sample_id"],
                species=row["species"],
                source_dataset=row["source_dataset"],
                source_recording=row["source_recording"],
                source_recording_id=row["source_recording_id"],
                split_group=row["split_group"],
                event_index=_int(row, "event_index"),
                event_start_time=event_start,
                event_end_time=event_end,
                event_low_freq=_float(row, "event_low_freq"),
                event_high_freq=_float(row, "event_high_freq"),
                requested_start_time=round(requested_start, 6),
                requested_end_time=round(requested_end, 6),
                actual_audio_start_time=float(v2.get("actual_audio_start_time") or max(0.0, requested_start)),
                actual_audio_end_time=float(v2.get("actual_audio_end_time") or max(0.0, requested_end)),
                left_padding_seconds=float(v2.get("left_padding_seconds") or max(0.0, -requested_start)),
                right_padding_seconds=float(v2.get("right_padding_seconds") or max(0.0, requested_end - audio_duration)),
                sample_rate_hz=_int(v2, "sample_rate_hz") if v2.get("sample_rate_hz") else 0,
                audio_duration_seconds=audio_duration,
                original_audio_path=original_audio,
            )
        )
    return specs


def load_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def padded_audio_window(audio: np.ndarray, sample_rate: int, spec: WindowSpec) -> np.ndarray:
    start_sample = int(round(spec.actual_audio_start_time * sample_rate))
    end_sample = int(round(spec.actual_audio_end_time * sample_rate))
    left_pad = int(round(spec.left_padding_seconds * sample_rate))
    right_pad = int(round(spec.right_padding_seconds * sample_rate))
    segment = audio[start_sample:end_sample]
    if left_pad:
        segment = np.concatenate([np.zeros(left_pad, dtype=np.float32), segment])
    if right_pad:
        segment = np.concatenate([segment, np.zeros(right_pad, dtype=np.float32)])
    expected_samples = int(round(WINDOW_SECONDS * sample_rate))
    if len(segment) < expected_samples:
        segment = np.concatenate([segment, np.zeros(expected_samples - len(segment), dtype=np.float32)])
    elif len(segment) > expected_samples:
        segment = segment[:expected_samples]
    return np.asarray(segment, dtype=np.float32)


def audio_window_path(audio_dir: Path, spec: WindowSpec) -> Path:
    return audio_dir / f"{spec.anonymous_sample_id}.wav"


def export_audio_windows(specs: list[WindowSpec], audio_dir: Path) -> list[dict[str, Any]]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, tuple[np.ndarray, int]] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        status = "success"
        notes = ""
        output_path = audio_window_path(audio_dir, spec)
        try:
            if spec.original_audio_path not in cache:
                cache[spec.original_audio_path] = load_mono_audio(spec.original_audio_path)
            audio, sample_rate = cache[spec.original_audio_path]
            segment = padded_audio_window(audio, sample_rate, spec)
            sf.write(output_path, segment, sample_rate)
            duration = len(segment) / sample_rate
        except Exception as exc:  # pragma: no cover - exercised through status output
            status = "failed"
            notes = f"{type(exc).__name__}: {exc}"
            sample_rate = spec.sample_rate_hz
            duration = 0.0
        rows.append(
            {
                "sample_id": spec.sample_id,
                "anonymous_sample_id": spec.anonymous_sample_id,
                "species": spec.species,
                "source_recording": spec.source_recording,
                "source_recording_id": spec.source_recording_id,
                "event_index": spec.event_index,
                "event_start_time": spec.event_start_time,
                "event_end_time": spec.event_end_time,
                "local_gt_start_time": spec.local_gt_start,
                "local_gt_end_time": spec.local_gt_end,
                "event_low_freq": spec.event_low_freq,
                "event_high_freq": spec.event_high_freq,
                "requested_start_time": spec.requested_start_time,
                "requested_end_time": spec.requested_end_time,
                "left_padding_seconds": spec.left_padding_seconds,
                "right_padding_seconds": spec.right_padding_seconds,
                "sample_rate_hz": sample_rate,
                "window_duration_seconds": round(duration, 6),
                "original_audio_path": portable(spec.original_audio_path),
                "audio_window_path": portable(output_path),
                "export_status": status,
                "notes": notes,
            }
        )
    return rows


def finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def convert_raw_event(
    raw: dict[str, Any],
    proposal_id: str,
    min_det_prob: float,
    window_duration: float = WINDOW_SECONDS,
) -> dict[str, Any] | None:
    det_prob = finite_float(raw.get("det_prob"), "det_prob")
    if det_prob < min_det_prob:
        return None
    start = max(0.0, finite_float(raw["start_time"], "start_time"))
    end = min(window_duration, finite_float(raw["end_time"], "end_time"))
    low = finite_float(raw["low_freq"], "low_freq")
    high = finite_float(raw["high_freq"], "high_freq")
    class_prob = finite_float(raw.get("class_prob", 0.0), "class_prob")
    if start >= end or low >= high or low < 0:
        raise ValueError("invalid BatDetect2 proposal geometry")
    return {
        "proposal_id": proposal_id,
        "start_time": round(start, 6),
        "end_time": round(end, 6),
        "start_time_seconds": round(start, 6),
        "end_time_seconds": round(end, 6),
        "low_freq": low,
        "high_freq": high,
        "low_frequency_hz": low,
        "high_frequency_hz": high,
        "det_prob": det_prob,
        "class_prob": class_prob,
        "label": str(raw.get("class") or ""),
        "event": str(raw.get("event") or ""),
        "source": "batdetect2",
    }


def convert_raw_payload(
    *,
    sample_id: str,
    anonymous_sample_id: str,
    raw_payload: dict[str, Any],
    min_det_prob: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_events = raw_payload.get("annotation")
    if not isinstance(raw_events, list):
        raise ValueError("BatDetect2 raw payload must contain annotation list")
    events: list[dict[str, Any]] = []
    invalid_count = 0
    below_threshold_count = 0
    for raw in raw_events:
        if not isinstance(raw, dict):
            invalid_count += 1
            continue
        try:
            converted = convert_raw_event(raw, "pending", min_det_prob)
            if converted is None:
                below_threshold_count += 1
            else:
                events.append(converted)
        except (KeyError, ValueError):
            invalid_count += 1
    events.sort(key=lambda e: (e["start_time_seconds"], e["end_time_seconds"], -e["det_prob"]))
    for index, event in enumerate(events, start=1):
        event["proposal_id"] = f"bd2_{index:03d}"
    payload = {
        "sample_id": sample_id,
        "anonymous_sample_id": anonymous_sample_id,
        "proposal_source": "batdetect2",
        "proposal_threshold": min_det_prob,
        "window_duration_seconds": WINDOW_SECONDS,
        "coordinate_frame": "local_window_0p000_to_0p300_seconds",
        "events": events,
    }
    summary = {
        "sample_id": sample_id,
        "anonymous_sample_id": anonymous_sample_id,
        "proposal_count": len(events),
        "raw_event_count": len(raw_events),
        "below_threshold_count": below_threshold_count,
        "invalid_dropped_count": invalid_count,
        "mean_det_prob": round(sum(e["det_prob"] for e in events) / len(events), 6) if events else "",
    }
    return payload, summary


def convert_raw_outputs(
    *,
    specs: list[WindowSpec],
    raw_dir: Path,
    proposal_dir: Path,
    min_det_prob: float,
) -> list[dict[str, Any]]:
    proposal_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        raw_path = raw_dir / f"{spec.anonymous_sample_id}.json"
        if not raw_path.is_file():
            summaries.append(
                {
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "proposal_count": 0,
                    "raw_event_count": 0,
                    "below_threshold_count": "",
                    "invalid_dropped_count": "",
                    "mean_det_prob": "",
                    "notes": "raw_batdetect2_missing",
                }
            )
            continue
        payload, summary = convert_raw_payload(
            sample_id=spec.sample_id,
            anonymous_sample_id=spec.anonymous_sample_id,
            raw_payload=json.loads(raw_path.read_text(encoding="utf-8")),
            min_det_prob=min_det_prob,
        )
        (proposal_dir / f"{spec.anonymous_sample_id}_batdetect2_proposals.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary["notes"] = ""
        summaries.append(summary)
    return summaries


def temporal_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    start = max(float(a["start_time"]), float(b["start_time"]))
    end = min(float(a["end_time"]), float(b["end_time"]))
    intersection = max(0.0, end - start)
    union = max(float(a["end_time"]), float(b["end_time"])) - min(float(a["start_time"]), float(b["start_time"]))
    return intersection / union if union > 0 else 0.0


def frequency_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    low = max(float(a["low_freq"]), float(b["low_freq"]))
    high = min(float(a["high_freq"]), float(b["high_freq"]))
    intersection = max(0.0, high - low)
    union = max(float(a["high_freq"]), float(b["high_freq"])) - min(float(a["low_freq"]), float(b["low_freq"]))
    return intersection / union if union > 0 else 0.0


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    time_intersection = max(0.0, min(float(a["end_time"]), float(b["end_time"])) - max(float(a["start_time"]), float(b["start_time"])))
    freq_intersection = max(0.0, min(float(a["high_freq"]), float(b["high_freq"])) - max(float(a["low_freq"]), float(b["low_freq"])))
    intersection = time_intersection * freq_intersection
    area_a = (float(a["end_time"]) - float(a["start_time"])) * (float(a["high_freq"]) - float(a["low_freq"]))
    area_b = (float(b["end_time"]) - float(b["start_time"])) * (float(b["high_freq"]) - float(b["low_freq"]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def gt_box(spec: WindowSpec) -> dict[str, Any]:
    return {
        "start_time": spec.local_gt_start,
        "end_time": spec.local_gt_end,
        "low_freq": spec.event_low_freq,
        "high_freq": spec.event_high_freq,
    }


def proposal_box(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_time": float(event["start_time_seconds"]),
        "end_time": float(event["end_time_seconds"]),
        "low_freq": float(event["low_frequency_hz"]),
        "high_freq": float(event["high_frequency_hz"]),
    }


def best_temporal_match(gt: dict[str, Any], proposals: list[dict[str, Any]], threshold: float) -> tuple[dict[str, Any] | None, float]:
    candidates = [(event, temporal_iou(gt, proposal_box(event))) for event in proposals]
    candidates = [item for item in candidates if item[1] >= threshold]
    if not candidates:
        return None, 0.0
    return max(candidates, key=lambda item: (item[1], float(item[0]["det_prob"])))


def best_start_match(gt: dict[str, Any], proposals: list[dict[str, Any]], tolerance_seconds: float) -> tuple[dict[str, Any] | None, float]:
    candidates = [
        (event, abs(float(event["start_time_seconds"]) - float(gt["start_time"])))
        for event in proposals
        if abs(float(event["start_time_seconds"]) - float(gt["start_time"])) <= tolerance_seconds
    ]
    if not candidates:
        return None, math.inf
    return min(candidates, key=lambda item: (item[1], -float(item[0]["det_prob"])))


def evaluate_proposals(
    *,
    specs: list[WindowSpec],
    proposal_dir: Path,
) -> dict[str, Any]:
    matched_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    per_species_counts: dict[str, Counter[str]] = defaultdict(Counter)
    metrics = {
        "temporal_iou_0p3": Counter(),
        "temporal_iou_0p1": Counter(),
        "start_time_10ms": Counter(),
    }
    matched_iou_values: dict[str, list[float]] = {"time": [], "frequency": [], "box": []}
    proposal_available = 0
    for spec in specs:
        proposal_path = proposal_dir / f"{spec.anonymous_sample_id}_batdetect2_proposals.json"
        proposals: list[dict[str, Any]] = []
        if proposal_path.is_file():
            proposals = json.loads(proposal_path.read_text(encoding="utf-8")).get("events", [])
        if proposals:
            proposal_available += 1
        gt = gt_box(spec)
        match_03, time_iou_03 = best_temporal_match(gt, proposals, 0.3)
        if match_03:
            metrics["temporal_iou_0p3"]["TP"] += 1
            per_species_counts[spec.species]["TP"] += 1
            matched_proposal_ids = {str(match_03["proposal_id"])}
            pbox = proposal_box(match_03)
            freq_iou = frequency_iou(gt, pbox)
            bx_iou = box_iou(gt, pbox)
            matched_iou_values["time"].append(time_iou_03)
            matched_iou_values["frequency"].append(freq_iou)
            matched_iou_values["box"].append(bx_iou)
            matched_rows.append(
                {
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "species": spec.species,
                    "proposal_id": match_03["proposal_id"],
                    "det_prob": match_03["det_prob"],
                    "time_iou": round(time_iou_03, 6),
                    "frequency_iou": round(freq_iou, 6),
                    "box_iou": round(bx_iou, 6),
                    "gt_start_time": gt["start_time"],
                    "gt_end_time": gt["end_time"],
                    "proposal_start_time": match_03["start_time_seconds"],
                    "proposal_end_time": match_03["end_time_seconds"],
                }
            )
        else:
            metrics["temporal_iou_0p3"]["FN"] += 1
            per_species_counts[spec.species]["FN"] += 1
            matched_proposal_ids = set()
            missed_rows.append(
                {
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "species": spec.species,
                    "gt_start_time": gt["start_time"],
                    "gt_end_time": gt["end_time"],
                    "proposal_count": len(proposals),
                    "reason": "no_temporal_iou_0p3_match",
                }
            )
        for event in proposals:
            if str(event["proposal_id"]) not in matched_proposal_ids:
                metrics["temporal_iou_0p3"]["FP"] += 1
                fp_rows.append(
                    {
                        "sample_id": spec.sample_id,
                        "anonymous_sample_id": spec.anonymous_sample_id,
                        "species": spec.species,
                        "proposal_id": event["proposal_id"],
                        "start_time": event["start_time_seconds"],
                        "end_time": event["end_time_seconds"],
                        "low_freq": event["low_frequency_hz"],
                        "high_freq": event["high_frequency_hz"],
                        "det_prob": event["det_prob"],
                    }
                )
        for name, threshold in (("temporal_iou_0p1", 0.1),):
            match, _score = best_temporal_match(gt, proposals, threshold)
            metrics[name]["TP" if match else "FN"] += 1
            metrics[name]["FP"] += max(0, len(proposals) - (1 if match else 0))
        start_match, _ = best_start_match(gt, proposals, 0.01)
        metrics["start_time_10ms"]["TP" if start_match else "FN"] += 1
        metrics["start_time_10ms"]["FP"] += max(0, len(proposals) - (1 if start_match else 0))
        status_rows.append(
            {
                "sample_id": spec.sample_id,
                "anonymous_sample_id": spec.anonymous_sample_id,
                "species": spec.species,
                "proposal_file": portable(proposal_path),
                "proposal_file_exists": str(proposal_path.is_file()).lower(),
                "proposal_count": len(proposals),
                "has_proposal": str(bool(proposals)).lower(),
                "matched_iou_0p3": str(bool(match_03)).lower(),
            }
        )
    aggregate: dict[str, Any] = {}
    for name, counts in metrics.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate[name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "F1": f1,
        }
    aggregate["sample_count"] = len(specs)
    aggregate["samples_with_at_least_one_proposal"] = proposal_available
    aggregate["proposal_missing_rate"] = 1 - (proposal_available / len(specs) if specs else 0.0)
    aggregate["mean_time_iou"] = mean(matched_iou_values["time"])
    aggregate["mean_frequency_iou"] = mean(matched_iou_values["frequency"])
    aggregate["mean_box_iou"] = mean(matched_iou_values["box"])
    per_species_rows: list[dict[str, Any]] = []
    for species in ALLOWED_SPECIES:
        counts = per_species_counts[species]
        tp = counts["TP"]
        fn = counts["FN"]
        support = tp + fn
        per_species_rows.append(
            {
                "species": species,
                "support": support,
                "matched_iou_0p3": tp,
                "missed_iou_0p3": fn,
                "proposal_recall_iou_0p3": tp / support if support else 0.0,
            }
        )
    return {
        "aggregate": aggregate,
        "per_species_rows": per_species_rows,
        "status_rows": status_rows,
        "matched_rows": matched_rows,
        "missed_rows": missed_rows,
        "false_positive_rows": fp_rows,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows = []
    for protocol in ("temporal_iou_0p3", "temporal_iou_0p1", "start_time_10ms"):
        row = {"protocol": protocol, **aggregate[protocol]}
        rows.append(row)
    write_csv(path, rows)


def write_report(
    *,
    path: Path,
    aggregate: dict[str, Any],
    per_species_rows: list[dict[str, Any]],
    min_det_prob: float,
) -> None:
    lines = [
        "# Stage 2 Sample-Level BatDetect2 Proposal Audit",
        "",
        "## Scope",
        "",
        "This no-VLM audit exported the same 0.300 s silence-padded local audio windows used for the Stage 1 event-level classification images, ran BatDetect2 on those windows, converted detector outputs into local-window proposal JSON files, and evaluated proposal-only localisation against the target event in each sample.",
        "",
        "Ground-truth boxes were not used as proposals. They were used only for the final proposal-only evaluation.",
        "",
        f"- Samples: {aggregate['sample_count']}",
        f"- Proposal threshold: det_prob >= {min_det_prob}",
        f"- Samples with at least one proposal: {aggregate['samples_with_at_least_one_proposal']}",
        f"- Proposal missing rate: {aggregate['proposal_missing_rate']:.3f}",
        "",
        "## Aggregate Proposal-Only Localisation",
        "",
        "| Protocol | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in ("temporal_iou_0p3", "temporal_iou_0p1", "start_time_10ms"):
        row = aggregate[protocol]
        lines.append(
            f"| {protocol} | {row['TP']} | {row['FP']} | {row['FN']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['F1']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Mean temporal IoU over IoU>=0.3 matches: {aggregate['mean_time_iou']:.3f}",
            f"- Mean frequency IoU over IoU>=0.3 matches: {aggregate['mean_frequency_iou']:.3f}",
            f"- Mean 2D box IoU over IoU>=0.3 matches: {aggregate['mean_box_iou']:.3f}",
            "",
            "## Per-Species Recall",
            "",
            "| Species | Support | Matched | Missed | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in per_species_rows:
        lines.append(
            f"| {row['species']} | {row['support']} | {row['matched_iou_0p3']} | "
            f"{row['missed_iou_0p3']} | {row['proposal_recall_iou_0p3']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These proposal files are suitable for a true Stage 2 proposal-constrained VLM run if proposal availability and recall are sufficient for the target protocol. Low recall species should be treated cautiously because a proposal-constrained model cannot verify or refine candidate events that the detector never proposes.",
            "",
            "The next VLM run should start with the deterministic pilot80 subset if the audit shows broad proposal availability. Full240 is appropriate only if parse/runtime cost is acceptable and proposal missing rates are not concentrated in a few species.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batdetect2(
    *,
    specs: list[WindowSpec],
    audio_dir: Path,
    raw_dir: Path,
    device_name: str,
    detection_threshold: float,
    chunk_size: float,
) -> dict[str, Any]:
    import torch
    from batdetect2 import api
    from batdetect2.detector.parameters import DEFAULT_MODEL_PATH
    from batdetect2.utils.detector_utils import save_results_to_file

    raw_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    started_model = time.perf_counter()
    model, params = api.load_model(DEFAULT_MODEL_PATH, device=device)
    model_load_seconds = time.perf_counter() - started_model
    config = api.get_config(
        **{
            **params,
            "detection_threshold": detection_threshold,
            "time_expansion": 1,
            "spec_slices": False,
            "chunk_size": chunk_size,
            "spec_features": False,
            "cnn_features": False,
            "quiet": True,
        }
    )
    file_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for spec in specs:
        audio_path = audio_window_path(audio_dir, spec)
        output_stem = raw_dir / spec.anonymous_sample_id
        file_started = time.perf_counter()
        try:
            result = api.process_file(str(audio_path), model, config=config)
            save_results_to_file(result, str(output_stem))
            file_rows.append(
                {
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "audio_path": portable(audio_path),
                    "status": "success",
                    "runtime_seconds": time.perf_counter() - file_started,
                    "raw_event_count": len(result["pred_dict"].get("annotation", [])),
                    "raw_json_path": portable(Path(str(output_stem) + ".json")),
                    "raw_csv_path": portable(Path(str(output_stem) + ".csv")),
                }
            )
        except Exception as exc:
            file_rows.append(
                {
                    "sample_id": spec.sample_id,
                    "anonymous_sample_id": spec.anonymous_sample_id,
                    "audio_path": portable(audio_path),
                    "status": "failed",
                    "runtime_seconds": time.perf_counter() - file_started,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    successes = [row for row in file_rows if row["status"] == "success"]
    metadata_row = {
        "command": " ".join(sys.argv),
        "batdetect2_version": metadata.version("batdetect2"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "model_path": str(DEFAULT_MODEL_PATH),
        "model_name": "Net2DFast_UK_same",
        "model_load_seconds": model_load_seconds,
        "detection_threshold": detection_threshold,
        "chunk_size": chunk_size,
        "total_files": len(file_rows),
        "successful_files": len(successes),
        "failed_files": len(file_rows) - len(successes),
        "total_runtime_seconds": time.perf_counter() - started,
        "files": file_rows,
    }
    (raw_dir / "run_metadata.json").write_text(
        json.dumps(metadata_row, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-manifest", type=Path, default=DEFAULT_STAGE1_MANIFEST)
    parser.add_argument("--v2-manifest", type=Path, default=DEFAULT_V2_MANIFEST)
    parser.add_argument("--tool-dir", type=Path, default=DEFAULT_TOOL_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--min-det-prob", type=float, default=DEFAULT_MIN_DET_PROB)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detection-threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD)
    parser.add_argument("--chunk-size", type=float, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--run-batdetect2", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_dir = args.tool_dir / "audio_windows"
    raw_dir = args.tool_dir / "raw_batdetect2"
    proposal_dir = args.tool_dir / "proposals"
    specs = build_window_specs(
        stage1_rows=load_csv_rows(args.stage1_manifest),
        v2_rows=load_csv_rows(args.v2_manifest),
    )
    if not args.skip_export:
        export_rows = export_audio_windows(specs, audio_dir)
        write_csv(args.tool_dir / "audio_window_manifest.csv", export_rows)
        print(f"Exported audio windows: {sum(row['export_status'] == 'success' for row in export_rows)}/{len(export_rows)}")
    if args.run_batdetect2:
        run_metadata = run_batdetect2(
            specs=specs,
            audio_dir=audio_dir,
            raw_dir=raw_dir,
            device_name=args.device,
            detection_threshold=args.detection_threshold,
            chunk_size=args.chunk_size,
        )
        print(
            "BatDetect2 inference: "
            f"{run_metadata['successful_files']}/{run_metadata['total_files']} succeeded"
        )
    if not args.skip_convert:
        conversion_rows = convert_raw_outputs(
            specs=specs,
            raw_dir=raw_dir,
            proposal_dir=proposal_dir,
            min_det_prob=args.min_det_prob,
        )
        write_csv(args.tool_dir / "proposal_summary.csv", conversion_rows)
        print(f"Converted proposal files: {sum(not row.get('notes') for row in conversion_rows)}/{len(conversion_rows)}")
    if not args.skip_evaluate:
        result = evaluate_proposals(specs=specs, proposal_dir=proposal_dir)
        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        (args.analysis_dir / "aggregate_summary.json").write_text(
            json.dumps(result["aggregate"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_metrics_csv(args.analysis_dir / "proposal_only_metrics.csv", result["aggregate"])
        write_csv(args.analysis_dir / "per_species_proposal_metrics.csv", result["per_species_rows"])
        write_csv(args.analysis_dir / "sample_level_proposal_status.csv", result["status_rows"])
        write_csv(args.analysis_dir / "matched_proposals.csv", result["matched_rows"])
        write_csv(args.analysis_dir / "missed_events.csv", result["missed_rows"])
        write_csv(args.analysis_dir / "false_positive_proposals.csv", result["false_positive_rows"])
        write_report(
            path=args.analysis_dir / "proposal_audit_report.md",
            aggregate=result["aggregate"],
            per_species_rows=result["per_species_rows"],
            min_det_prob=args.min_det_prob,
        )
        print(json.dumps(result["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
