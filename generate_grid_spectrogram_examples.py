"""Generate grid-overlay spectrogram examples for VLM localisation tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from main import (
    OUTPUT_DIR,
    make_spectrogram,
    plot_spectrogram_with_grid,
    read_mono_audio,
)


def save_grid_examples(
    audio_path: str,
    output_dir: Path = OUTPUT_DIR / "grid",
    time_major_step: float = 0.5,
    time_minor_step: float = 0.1,
    frequency_major_step: float = 10000,
    frequency_minor_step: float = 5000,
) -> list[Path]:
    """Save full and zoomed grid-overlay spectrogram PNGs."""
    resolved_audio_path, audio, sr = read_mono_audio(audio_path)
    spec, local_stft = make_spectrogram(audio, sr)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    common_grid_kwargs = {
        "show_grid": True,
        "time_major_step": time_major_step,
        "time_minor_step": time_minor_step,
        "frequency_major_step": frequency_major_step,
        "frequency_minor_step": frequency_minor_step,
    }

    full_fig = plot_spectrogram_with_grid(
        spec,
        audio,
        local_stft,
        sr,
        preset="full_grid",
        **common_grid_kwargs,
    )
    full_output = output_dir / f"{resolved_audio_path.stem}_full_grid.png"
    full_fig.savefig(full_output, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(full_fig)
    saved_paths.append(full_output)

    zoom_fig = plot_spectrogram_with_grid(
        spec,
        audio,
        local_stft,
        sr,
        start_time=0.0,
        end_time=4.0,
        low_frequency=20000,
        high_frequency=100000,
        preset="short_event_grid",
        **common_grid_kwargs,
    )
    zoom_output = output_dir / f"{resolved_audio_path.stem}_zoom_short_event_grid.png"
    zoom_fig.savefig(zoom_output, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(zoom_fig)
    saved_paths.append(zoom_output)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full and zoomed spectrograms with coordinate grids."
    )
    parser.add_argument("--audio-path", default="pseudo_petersi_001.wav")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "grid")
    parser.add_argument("--time-major-step", type=float, default=0.5)
    parser.add_argument("--time-minor-step", type=float, default=0.1)
    parser.add_argument("--frequency-major-step", type=float, default=10000)
    parser.add_argument("--frequency-minor-step", type=float, default=5000)
    args = parser.parse_args()

    saved_paths = save_grid_examples(
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        time_major_step=args.time_major_step,
        time_minor_step=args.time_minor_step,
        frequency_major_step=args.frequency_major_step,
        frequency_minor_step=args.frequency_minor_step,
    )

    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
