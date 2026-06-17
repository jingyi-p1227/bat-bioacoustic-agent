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


def save_grid_comparison(
    audio_path: str,
    output_dir: Path = OUTPUT_DIR / "grid_comparison",
    time_major_step: float = 0.5,
    time_minor_step: float = 0.1,
    frequency_major_step: float = 10000,
    frequency_minor_step: float = 5000,
) -> list[Path]:
    """Save fixed-grid and auto-grid comparison spectrogram PNGs."""
    resolved_audio_path, audio, sr = read_mono_audio(audio_path)
    spec, local_stft = make_spectrogram(audio, sr)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    fixed_grid_kwargs = {
        "show_grid": True,
        "grid_step_mode": "fixed",
        "time_major_step": time_major_step,
        "time_minor_step": time_minor_step,
        "frequency_major_step": frequency_major_step,
        "frequency_minor_step": frequency_minor_step,
    }
    auto_grid_kwargs = {
        "show_grid": True,
        "grid_step_mode": "auto",
    }

    views = [
        (
            "full",
            {},
        ),
        (
            "zoom_0_4s_20_100khz",
            {
                "start_time": 0.0,
                "end_time": 4.0,
                "low_frequency": 20000,
                "high_frequency": 100000,
            },
        ),
    ]

    for view_name, view_kwargs in views:
        for mode_name, grid_kwargs in [
            ("fixed_grid", fixed_grid_kwargs),
            ("auto_grid", auto_grid_kwargs),
        ]:
            fig = plot_spectrogram_with_grid(
                spec,
                audio,
                local_stft,
                sr,
                preset=f"{view_name}_{mode_name}",
                **view_kwargs,
                **grid_kwargs,
            )
            output_path = (
                output_dir / f"{resolved_audio_path.stem}_{view_name}_{mode_name}.png"
            )
            fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full and zoomed spectrograms with coordinate grids."
    )
    parser.add_argument("--audio-path", default="pseudo_petersi_001.wav")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "grid")
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate fixed-grid vs auto-grid comparison images.",
    )
    parser.add_argument("--time-major-step", type=float, default=0.5)
    parser.add_argument("--time-minor-step", type=float, default=0.1)
    parser.add_argument("--frequency-major-step", type=float, default=10000)
    parser.add_argument("--frequency-minor-step", type=float, default=5000)
    args = parser.parse_args()

    save_function = save_grid_comparison if args.comparison else save_grid_examples
    saved_paths = save_function(
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
