"""Plot ground-truth boxes for every recording in the pilot subset.

This script reads the pilot manifest, loads each EventResult ground-truth JSON,
and saves a static spectrogram image with ground-truth boxes overlaid.
It does not call or run the agent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from main import EventResult, make_spectrogram, plot_events_on_spectrogram


DEFAULT_MANIFEST = Path("ground_truth/pilot_subset_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/pilot_ground_truth")


def read_mono_audio(audio_path: str | Path) -> tuple[Path, np.ndarray, int]:
    """Read an absolute or relative WAV path and convert multichannel audio to mono."""
    resolved_audio_path = Path(audio_path).expanduser().resolve()
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_audio_path}")

    audio, sr = sf.read(str(resolved_audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return resolved_audio_path, audio, sr


def load_event_result(path: str | Path) -> EventResult:
    """Load one EventResult JSON file."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return EventResult.model_validate(payload)


def plot_ground_truth(
    event_result: EventResult,
    output_dir: Path,
    hide_labels: bool = False,
    frequency_max_hz: float | None = None,
    max_labels: int | None = None,
) -> Path:
    """Create and save one ground-truth box overlay image."""
    audio_path, audio, sr = read_mono_audio(event_result.audio_path)
    spec, stft = make_spectrogram(audio, sr)
    fig = plot_events_on_spectrogram(
        spec,
        audio,
        stft,
        sr,
        event_result.events,
        hide_labels=hide_labels,
        frequency_max_hz=frequency_max_hz,
        title_mode="ground truth",
        max_labels=max_labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}_ground_truth_boxes.png"
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def resolve_manifest_path(path: str | Path) -> Path:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    return manifest_path.resolve()


def resolve_ground_truth_path(manifest_path: Path, row: dict) -> Path:
    """Resolve a manifest row's ground_truth_json path relative to the project root."""
    ground_truth_path = Path(row["ground_truth_json"])
    if ground_truth_path.is_absolute():
        return ground_truth_path

    project_root = manifest_path.parent.parent
    return project_root / ground_truth_path


def plot_pilot_ground_truth(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    hide_labels: bool = False,
    frequency_max_hz: float | None = None,
    max_labels: int | None = None,
) -> list[Path]:
    """Plot all pilot ground-truth overlays and return saved image paths."""
    manifest_path = resolve_manifest_path(manifest_path)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = manifest_path.parent.parent / output_dir

    saved_paths = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            event_result = load_event_result(resolve_ground_truth_path(manifest_path, row))
            saved_paths.append(
                plot_ground_truth(
                    event_result,
                    output_dir,
                    hide_labels=hide_labels,
                    frequency_max_hz=frequency_max_hz,
                    max_labels=max_labels,
                )
            )

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ground-truth boxes for the pilot subset."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to ground_truth/pilot_subset_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for <audio_stem>_ground_truth_boxes.png images.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Draw boxes without text labels.",
    )
    parser.add_argument(
        "--frequency-max-hz",
        type=float,
        default=None,
        help="Optional maximum y-axis frequency in Hz, for example 100000.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Optional maximum number of text labels to draw per image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = plot_pilot_ground_truth(
        args.manifest,
        args.output_dir,
        hide_labels=args.hide_labels,
        frequency_max_hz=args.frequency_max_hz,
        max_labels=args.max_labels,
    )
    print(f"Saved {len(saved_paths)} pilot ground-truth plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
