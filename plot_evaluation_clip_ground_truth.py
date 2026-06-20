"""Plot evaluation-set clips with ground-truth time-frequency boxes overlaid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.patches import Rectangle

from main import make_spectrogram, to_decibels


DEFAULT_MIN_DB = -130.0
DEFAULT_MAX_DB = 0.0
DEFAULT_MAX_FREQ_HZ = 120_000.0


def spectrogram_to_image(
    spec: np.ndarray,
    *,
    min_db: float = DEFAULT_MIN_DB,
    max_db: float = DEFAULT_MAX_DB,
) -> np.ndarray:
    """Convert a spectrogram to a normalized grayscale image."""
    spec_db = to_decibels(spec)
    spec_db = np.clip(spec_db, min_db, max_db)
    return (spec_db - min_db) / (max_db - min_db)


def read_mono_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and convert multichannel audio to mono."""
    audio, sample_rate = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def resolve_clip_paths(eval_dir: str | Path, clip_id: str) -> tuple[Path, Path]:
    """Resolve and validate one evaluation clip's audio and ground-truth paths."""
    eval_dir = Path(eval_dir).expanduser()
    audio_path = eval_dir / "audio" / f"{clip_id}.wav"
    ground_truth_path = eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"

    missing_paths = [
        path
        for path in (audio_path, ground_truth_path)
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Clip id {clip_id!r} is incomplete or missing: {missing}")

    return audio_path, ground_truth_path


def load_ground_truth(ground_truth_path: str | Path) -> dict:
    """Load one clip-level ground-truth JSON file."""
    with Path(ground_truth_path).open(encoding="utf-8") as f:
        return json.load(f)


def available_clip_ids(eval_dir: str | Path) -> list[str]:
    """Return sorted clip IDs available under an evaluation-set audio directory."""
    audio_dir = Path(eval_dir).expanduser() / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Evaluation audio directory not found: {audio_dir}")
    return sorted(path.stem for path in audio_dir.glob("*.wav"))


def plot_clip_ground_truth(
    *,
    eval_dir: str | Path,
    clip_id: str,
    output_dir: str | Path | None = None,
    min_db: float = DEFAULT_MIN_DB,
    max_freq_hz: float | None = DEFAULT_MAX_FREQ_HZ,
) -> Path:
    """Save one spectrogram image with clip-level ground-truth boxes."""
    eval_dir = Path(eval_dir).expanduser()
    audio_path, ground_truth_path = resolve_clip_paths(eval_dir, clip_id)
    output_dir = Path(output_dir).expanduser() if output_dir else eval_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(ground_truth_path)
    events = ground_truth.get("events", [])
    audio, sample_rate = read_mono_audio(audio_path)
    spec, stft = make_spectrogram(audio, sample_rate)
    image = spectrogram_to_image(spec, min_db=min_db)

    extent = list(stft.extent(len(audio)))
    extent[2] = extent[2] / 1000
    extent[3] = extent[3] / 1000

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )

    duration = len(audio) / sample_rate
    ax.set_xlim(0, duration)
    y_max_hz = min(max_freq_hz, sample_rate / 2) if max_freq_hz else sample_rate / 2
    ax.set_ylim(0, y_max_hz / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(f"{clip_id} ground truth | events: {len(events)}")
    ax.grid(color="cyan", linewidth=0.4, alpha=0.35)

    for index, event in enumerate(events, start=1):
        start_time = float(event["start_time"])
        end_time = float(event["end_time"])
        low_khz = float(event["low_frequency"]) / 1000
        high_khz = float(event["high_frequency"]) / 1000
        width = end_time - start_time
        height = high_khz - low_khz
        if width <= 0 or height <= 0:
            continue

        rect = Rectangle(
            (start_time, low_khz),
            width,
            height,
            fill=False,
            edgecolor="lime",
            linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.text(
            start_time,
            high_khz,
            str(index),
            color="lime",
            fontsize=7,
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
        )

    output_path = output_dir / f"{clip_id}_gt_overlay.png"
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot evaluation clips with ground-truth boxes overlaid."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation set directory containing audio/ and ground_truth/.",
    )
    parser.add_argument("--clip-id", help="Optional single clip id, for example OP_001.")
    parser.add_argument("--all", action="store_true", help="Plot all clips.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <eval-dir>/figures.",
    )
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument(
        "--max-freq",
        type=float,
        default=DEFAULT_MAX_FREQ_HZ,
        help="Maximum displayed frequency in Hz. Defaults to 120000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all == bool(args.clip_id):
        raise SystemExit("Provide exactly one of --clip-id or --all.")

    clip_ids = available_clip_ids(args.eval_dir) if args.all else [args.clip_id]
    saved_paths = [
        plot_clip_ground_truth(
            eval_dir=args.eval_dir,
            clip_id=clip_id,
            output_dir=args.output_dir,
            min_db=args.min_db,
            max_freq_hz=args.max_freq,
        )
        for clip_id in clip_ids
    ]

    print(f"Saved {len(saved_paths)} ground-truth overlay plot(s):")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
