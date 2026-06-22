"""Generate clean spectrogram images for agent evaluation inputs.

This script reads evaluation-set audio only. It does not load ground-truth JSON
or ground-truth overlay figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from main import make_spectrogram, to_decibels


DEFAULT_OUTPUT_DIR = Path("outputs/agent_inputs/prompt_v2_small_pilot")
DEFAULT_MIN_DB = -130.0
DEFAULT_MAX_DB = 0.0
DEFAULT_MAX_FREQ_HZ = 120_000.0


def resolve_audio_path(eval_dir: str | Path, clip_id: str) -> Path:
    """Resolve one evaluation clip WAV and fail clearly when it is missing."""
    audio_path = Path(eval_dir).expanduser() / "audio" / f"{clip_id}.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio for clip id {clip_id!r} was not found: {audio_path}"
        )
    return audio_path


def read_mono_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and convert multichannel audio to mono."""
    audio, sample_rate = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def spectrogram_to_image(
    spec: np.ndarray,
    *,
    min_db: float = DEFAULT_MIN_DB,
    max_db: float = DEFAULT_MAX_DB,
) -> np.ndarray:
    """Convert a spectrogram to a normalized grayscale image."""
    if min_db >= max_db:
        raise ValueError("min_db must be less than max_db")
    spec_db = to_decibels(spec)
    spec_db = np.clip(spec_db, min_db, max_db)
    return (spec_db - min_db) / (max_db - min_db)


def save_clean_spectrogram(
    *,
    eval_dir: str | Path,
    clip_id: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_db: float = DEFAULT_MIN_DB,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> Path:
    """Generate and save one clean spectrogram without annotation overlays."""
    if max_freq_hz <= 0:
        raise ValueError("max_freq_hz must be greater than 0")

    audio_path = resolve_audio_path(eval_dir, clip_id)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = read_mono_audio(audio_path)
    spec, stft = make_spectrogram(audio, sample_rate)
    image = spectrogram_to_image(spec, min_db=min_db)

    extent = list(stft.extent(len(audio)))
    extent[2] /= 1000
    extent[3] /= 1000

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
    ax.set_ylim(0, min(max_freq_hz, sample_rate / 2) / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(clip_id)
    ax.grid(color="cyan", linewidth=0.4, alpha=0.35)

    output_path = output_dir / f"{clip_id}_spectrogram.png"
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def parse_clip_list(value: str) -> list[str]:
    """Parse a comma-separated clip list while preserving input order."""
    clip_ids = [clip_id.strip() for clip_id in value.split(",") if clip_id.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare clean evaluation spectrograms for agent input."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation set directory containing audio/<clip_id>.wav.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--clip-id", help="Single clip id, for example OP_001.")
    selection.add_argument(
        "--clip-list",
        help="Comma-separated clip ids, for example OP_001,OP_010.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for clean spectrogram PNG files.",
    )
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument(
        "--max-freq",
        type=float,
        default=DEFAULT_MAX_FREQ_HZ,
        help="Maximum displayed frequency in Hz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_ids = (
        [args.clip_id]
        if args.clip_id is not None
        else parse_clip_list(args.clip_list)
    )

    saved_paths = [
        save_clean_spectrogram(
            eval_dir=args.eval_dir,
            clip_id=clip_id,
            output_dir=args.output_dir,
            min_db=args.min_db,
            max_freq_hz=args.max_freq,
        )
        for clip_id in clip_ids
    ]

    print(f"Saved {len(saved_paths)} clean agent spectrogram input(s):")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
