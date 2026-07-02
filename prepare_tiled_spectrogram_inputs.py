"""Generate clean, original-coordinate spectrogram tiles for P6 experiments.

The script reads evaluation WAV files only. It does not read ground truth,
predictions, overlays, or diagnostic figures.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from main import make_spectrogram
from prepare_agent_spectrogram_inputs import (
    DEFAULT_MAX_FREQ_HZ,
    DEFAULT_MIN_DB,
    apply_grid_style,
    parse_clip_list,
    read_mono_audio,
    resolve_audio_path,
    spectrogram_to_image,
)


DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_OUTPUT_DIR = Path("outputs/agent_inputs/p6_tiled_spectrograms")
DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")
GRID_STYLE = "grid_v2"


@dataclass(frozen=True)
class TileSetting:
    """One named tile duration and overlap configuration."""

    name: str
    tile_duration_seconds: float
    overlap_seconds: float


TILE_SETTINGS = (
    TileSetting("tile_0p5_overlap_0p1", 0.5, 0.1),
    TileSetting("tile_0p25_overlap_0p05", 0.25, 0.05),
)


@dataclass(frozen=True)
class TileWindow:
    """A sample-accurate tile window within one source clip."""

    tile_index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TileManifestRow:
    """Portable metadata for one generated tile image."""

    clip_id: str
    tile_id: str
    tile_start_seconds: float
    tile_end_seconds: float
    image_path: str
    original_audio_path: str
    tile_duration: float
    overlap: float
    grid_style: str


MANIFEST_FIELDS = tuple(TileManifestRow.__dataclass_fields__)


def generate_tile_windows(
    *,
    total_samples: int,
    sample_rate_hz: int,
    tile_duration_seconds: float,
    overlap_seconds: float,
) -> list[TileWindow]:
    """Generate covering windows using sample indices as the source of truth."""
    if total_samples <= 0:
        raise ValueError("total_samples must be greater than 0")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be greater than 0")
    if tile_duration_seconds <= 0:
        raise ValueError("tile_duration_seconds must be greater than 0")
    if overlap_seconds < 0 or overlap_seconds >= tile_duration_seconds:
        raise ValueError("overlap_seconds must satisfy 0 <= overlap < tile duration")

    tile_samples = round(tile_duration_seconds * sample_rate_hz)
    overlap_samples = round(overlap_seconds * sample_rate_hz)
    step_samples = tile_samples - overlap_samples
    if tile_samples <= 0 or step_samples <= 0:
        raise ValueError("tile settings are too small for the sample rate")

    windows: list[TileWindow] = []
    start_sample = 0
    while start_sample < total_samples:
        end_sample = min(start_sample + tile_samples, total_samples)
        windows.append(
            TileWindow(
                tile_index=len(windows) + 1,
                start_sample=start_sample,
                end_sample=end_sample,
                start_seconds=start_sample / sample_rate_hz,
                end_seconds=end_sample / sample_rate_hz,
            )
        )
        if end_sample == total_samples:
            break
        start_sample += step_samples
    return windows


def format_time_token(seconds: float) -> str:
    """Format seconds for a stable, filesystem-friendly image name."""
    return f"{seconds:.6f}".replace(".", "p")


def tile_image_name(clip_id: str, window: TileWindow) -> str:
    """Return a filename containing clip, tile index, and source coordinates."""
    return (
        f"{clip_id}_tile_{window.tile_index:03d}"
        f"_start_{format_time_token(window.start_seconds)}"
        f"_end_{format_time_token(window.end_seconds)}.png"
    )


def portable_path(path: Path, *, base_dir: Path) -> str:
    """Return a POSIX relative path when the path is beneath base_dir."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def save_tile_image(
    *,
    tile_audio,
    sample_rate_hz: int,
    clip_id: str,
    window: TileWindow,
    output_path: Path,
    min_db: float,
    max_freq_hz: float,
) -> None:
    """Save one clean tile with an x-axis in original clip coordinates."""
    spec, stft = make_spectrogram(tile_audio, sample_rate_hz)
    image = spectrogram_to_image(spec, min_db=min_db)

    extent = list(stft.extent(len(tile_audio)))
    extent[0] += window.start_seconds
    extent[1] += window.start_seconds
    extent[2] /= 1000
    extent[3] /= 1000

    displayed_max_freq_hz = min(max_freq_hz, sample_rate_hz / 2)
    visible_duration = window.end_seconds - window.start_seconds
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )
    ax.set_xlim(window.start_seconds, window.end_seconds)
    ax.set_ylim(0, displayed_max_freq_hz / 1000)
    ax.set_xlabel("Time (s, original clip coordinates)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(clip_id)
    apply_grid_style(
        ax,
        grid_style=GRID_STYLE,
        duration_seconds=visible_duration,
        displayed_max_freq_hz=displayed_max_freq_hz,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def write_tile_manifest(rows: list[TileManifestRow], manifest_path: Path) -> None:
    """Write deterministic tile metadata as CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def build_tiled_inputs(
    *,
    eval_dir: Path,
    output_dir: Path,
    clip_ids: list[str],
    settings: tuple[TileSetting, ...] = TILE_SETTINGS,
    min_db: float = DEFAULT_MIN_DB,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
    overwrite: bool = False,
    base_dir: Path | None = None,
) -> list[TileManifestRow]:
    """Generate all requested tiled images and their combined manifest rows."""
    if max_freq_hz <= 0:
        raise ValueError("max_freq_hz must be greater than 0")
    base_dir = Path.cwd() if base_dir is None else base_dir
    manifest_path = output_dir / "tile_manifest.csv"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. Use --overwrite to replace it."
        )

    planned: list[tuple[Path, str, TileSetting, TileWindow, object, int, Path]] = []
    for clip_id in clip_ids:
        audio_path = resolve_audio_path(eval_dir, clip_id)
        audio, sample_rate_hz = read_mono_audio(audio_path)
        for setting in settings:
            windows = generate_tile_windows(
                total_samples=len(audio),
                sample_rate_hz=sample_rate_hz,
                tile_duration_seconds=setting.tile_duration_seconds,
                overlap_seconds=setting.overlap_seconds,
            )
            for window in windows:
                image_path = output_dir / setting.name / tile_image_name(clip_id, window)
                if image_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"Tile image already exists: {image_path}. Use --overwrite."
                    )
                planned.append(
                    (
                        image_path,
                        clip_id,
                        setting,
                        window,
                        audio,
                        sample_rate_hz,
                        audio_path,
                    )
                )

    rows: list[TileManifestRow] = []
    for image_path, clip_id, setting, window, audio, sample_rate_hz, audio_path in planned:
        tile_audio = audio[window.start_sample : window.end_sample]
        save_tile_image(
            tile_audio=tile_audio,
            sample_rate_hz=sample_rate_hz,
            clip_id=clip_id,
            window=window,
            output_path=image_path,
            min_db=min_db,
            max_freq_hz=max_freq_hz,
        )
        rows.append(
            TileManifestRow(
                clip_id=clip_id,
                tile_id=f"{setting.name}_tile_{window.tile_index:03d}",
                tile_start_seconds=round(window.start_seconds, 6),
                tile_end_seconds=round(window.end_seconds, 6),
                image_path=portable_path(image_path, base_dir=base_dir),
                original_audio_path=portable_path(audio_path, base_dir=base_dir),
                tile_duration=round(window.end_seconds - window.start_seconds, 6),
                overlap=setting.overlap_seconds,
                grid_style=GRID_STYLE,
            )
        )

    write_tile_manifest(rows, manifest_path)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean grid_v2 spectrogram tiles for P6."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--clip-list",
        default=",".join(DEFAULT_CLIP_IDS),
        help="Comma-separated evaluation clip ids.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ_HZ)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_tiled_inputs(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        clip_ids=parse_clip_list(args.clip_list),
        min_db=args.min_db,
        max_freq_hz=args.max_freq,
        overwrite=args.overwrite,
    )

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        setting_name = row.tile_id.rsplit("_tile_", 1)[0]
        key = (row.clip_id, setting_name)
        counts[key] = counts.get(key, 0) + 1

    print(f"Generated {len(rows)} clean tile image(s).")
    for (clip_id, setting_name), count in sorted(counts.items()):
        print(f"{clip_id} | {setting_name} | {count}")
    print(f"Manifest: {args.output_dir / 'tile_manifest.csv'}")


if __name__ == "__main__":
    main()

