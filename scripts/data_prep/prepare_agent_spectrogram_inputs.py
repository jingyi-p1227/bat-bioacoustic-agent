"""Generate clean spectrogram images for agent evaluation inputs.

This script reads evaluation-set audio only. It does not load ground-truth JSON
or ground-truth overlay figures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.ticker import MultipleLocator

from main import (
    DEFAULT_FREQUENCY_MAJOR_STEP,
    DEFAULT_FREQUENCY_MINOR_STEP,
    DEFAULT_TIME_MAJOR_STEP,
    DEFAULT_TIME_MINOR_STEP,
    choose_readable_grid_steps,
    make_spectrogram,
    to_decibels,
)


DEFAULT_OUTPUT_DIR = Path("outputs/agent_inputs/prompt_v2_small_pilot")
DEFAULT_MIN_DB = -130.0
DEFAULT_MAX_DB = 0.0
DEFAULT_MAX_FREQ_HZ = 120_000.0
GRID_STYLES = ("grid_v1", "grid_v2")


@dataclass(frozen=True)
class AgentGridStyle:
    """Display-grid parameters for clean agent input images."""

    name: str
    mode: str
    description: str


GRID_STYLE_DEFINITIONS = {
    "grid_v1": AgentGridStyle(
        name="grid_v1",
        mode="fixed",
        description=(
            "Fixed project-default grid: 0.5 s major / 0.1 s minor time "
            "steps and 10 kHz major / 5 kHz minor frequency steps."
        ),
    ),
    "grid_v2": AgentGridStyle(
        name="grid_v2",
        mode="auto",
        description=(
            "Readable auto grid: major/minor steps are selected from the "
            "visible time and frequency spans."
        ),
    ),
}


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


def resolve_all_clip_ids(eval_dir: str | Path) -> list[str]:
    """Return all evaluation clip ids from audio/*.wav in stable order."""
    audio_dir = Path(eval_dir).expanduser() / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Evaluation audio directory not found: {audio_dir}")
    clip_ids = [path.stem for path in sorted(audio_dir.glob("*.wav"))]
    if not clip_ids:
        raise ValueError(f"No WAV files found in {audio_dir}")
    return clip_ids


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


def apply_grid_style(
    ax,
    *,
    grid_style: str,
    duration_seconds: float,
    displayed_max_freq_hz: float,
) -> None:
    """Apply one named clean-input grid style."""
    if grid_style not in GRID_STYLE_DEFINITIONS:
        raise ValueError(f"Unsupported grid_style: {grid_style}")

    if grid_style == "grid_v1":
        time_major_step = DEFAULT_TIME_MAJOR_STEP
        time_minor_step = DEFAULT_TIME_MINOR_STEP
        frequency_major_step_khz = DEFAULT_FREQUENCY_MAJOR_STEP / 1000
        frequency_minor_step_khz = DEFAULT_FREQUENCY_MINOR_STEP / 1000
    else:
        steps = choose_readable_grid_steps(
            time_span=duration_seconds,
            frequency_span=displayed_max_freq_hz,
        )
        time_major_step = steps.time_major_step
        time_minor_step = steps.time_minor_step
        frequency_major_step_khz = steps.frequency_major_step / 1000
        frequency_minor_step_khz = steps.frequency_minor_step / 1000

    ax.xaxis.set_major_locator(MultipleLocator(time_major_step))
    ax.xaxis.set_minor_locator(MultipleLocator(time_minor_step))
    ax.yaxis.set_major_locator(MultipleLocator(frequency_major_step_khz))
    ax.yaxis.set_minor_locator(MultipleLocator(frequency_minor_step_khz))
    ax.tick_params(axis="both", which="major", labelsize=8, length=4)
    ax.tick_params(axis="both", which="minor", labelsize=0, length=2)
    ax.grid(which="major", color="cyan", linewidth=0.7, alpha=0.65)
    ax.grid(which="minor", color="cyan", linewidth=0.35, alpha=0.35)


def save_clean_spectrogram(
    *,
    eval_dir: str | Path,
    clip_id: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_db: float = DEFAULT_MIN_DB,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
    grid_style: str = "grid_v1",
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
    apply_grid_style(
        ax,
        grid_style=grid_style,
        duration_seconds=duration,
        displayed_max_freq_hz=min(max_freq_hz, sample_rate / 2),
    )

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
    selection.add_argument(
        "--all",
        action="store_true",
        help="Generate inputs for every WAV in <eval-dir>/audio.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for clean spectrogram PNG files.",
    )
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument(
        "--grid-style",
        choices=GRID_STYLES,
        default="grid_v1",
        help="Named clean spectrogram grid style.",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=DEFAULT_MAX_FREQ_HZ,
        help="Maximum displayed frequency in Hz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        clip_ids = resolve_all_clip_ids(args.eval_dir)
    elif args.clip_id is not None:
        clip_ids = [args.clip_id]
    else:
        clip_ids = parse_clip_list(args.clip_list)

    saved_paths = [
        save_clean_spectrogram(
            eval_dir=args.eval_dir,
            clip_id=clip_id,
            output_dir=args.output_dir,
            min_db=args.min_db,
            max_freq_hz=args.max_freq,
            grid_style=args.grid_style,
        )
        for clip_id in clip_ids
    ]

    print(f"Saved {len(saved_paths)} clean agent spectrogram input(s):")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
