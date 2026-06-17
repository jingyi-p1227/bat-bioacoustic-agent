"""Generate spectrogram images to compare min_db display ranges.

This is an experiment-only script. It does not change the main visualization
defaults or annotation pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main import OUTPUT_DIR, make_spectrogram, read_mono_audio, to_decibels


MIN_DB_VALUES = [-80, -100, -120, -130]


def spectrogram_to_image(spec: np.ndarray, min_db: float, max_db: float = 0) -> np.ndarray:
    """Convert a spectrogram to a normalized image using a chosen dB range."""

    spec_db = to_decibels(spec)
    spec_db = np.clip(spec_db, min_db, max_db)
    return (spec_db - min_db) / (max_db - min_db)


def save_spectrogram_view(
    spec: np.ndarray,
    audio: np.ndarray,
    sr: int,
    stft,
    output_path: Path,
    min_db: float,
    start_time: float | None = None,
    end_time: float | None = None,
    low_frequency: float | None = None,
    high_frequency: float | None = None,
    title_suffix: str = "",
) -> None:
    """Save one spectrogram PNG for the requested display range and view."""

    fig, ax = plt.subplots(figsize=(12, 6))
    image = spectrogram_to_image(spec, min_db=min_db)
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=stft.extent(len(audio)),
        cmap="gray",
        interpolation="bilinear",
    )

    duration = len(audio) / sr
    if start_time is not None and end_time is not None:
        ax.set_xlim(max(0, start_time), min(duration, end_time))
    if low_frequency is not None and high_frequency is not None:
        ax.set_ylim(max(0, low_frequency), min(sr / 2, high_frequency))

    ax.set_title(f"pseudo_petersi_001 | min_db={min_db} | {title_suffix}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(color="cyan", linewidth=0.5, alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def generate_min_db_comparison(
    audio_path: str = "pseudo_petersi_001.wav",
    output_dir: Path = OUTPUT_DIR / "min_db_comparison",
    min_db_values: list[int] | None = None,
) -> list[Path]:
    """Generate full and zoomed comparison images for several min_db values."""

    values = min_db_values or MIN_DB_VALUES
    resolved_audio_path, audio, sr = read_mono_audio(audio_path)
    spec, stft = make_spectrogram(audio, sr)

    saved_paths: list[Path] = []
    for min_db in values:
        full_output = output_dir / f"{resolved_audio_path.stem}_full_min_db_{abs(min_db)}.png"
        save_spectrogram_view(
            spec,
            audio,
            sr,
            stft,
            full_output,
            min_db=min_db,
            title_suffix="full view",
        )
        saved_paths.append(full_output)

        zoom_output = output_dir / f"{resolved_audio_path.stem}_zoom_0_4s_20_100khz_min_db_{abs(min_db)}.png"
        save_spectrogram_view(
            spec,
            audio,
            sr,
            stft,
            zoom_output,
            min_db=min_db,
            start_time=0.0,
            end_time=4.0,
            low_frequency=20_000,
            high_frequency=100_000,
            title_suffix="0-4 s, 20-100 kHz",
        )
        saved_paths.append(zoom_output)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate min_db comparison spectrogram images."
    )
    parser.add_argument("--audio-path", default="pseudo_petersi_001.wav")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "min_db_comparison")
    parser.add_argument(
        "--min-db-values",
        default=",".join(str(value) for value in MIN_DB_VALUES),
        help="Comma-separated min_db values, for example -80,-100,-120,-130.",
    )
    args = parser.parse_args()

    values = [int(value.strip()) for value in args.min_db_values.split(",") if value.strip()]
    saved_paths = generate_min_db_comparison(
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        min_db_values=values,
    )
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
