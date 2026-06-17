"""Split a WAV recording into fixed-duration clips with timeline metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_OUTPUT_AUDIO_DIR = Path("audio/clips")
DEFAULT_MANIFEST_DIR = Path("outputs/clip_manifests")


@dataclass(frozen=True)
class ClipMetadata:
    source_audio_path: str
    clip_index: int
    clip_start_time_seconds: float
    clip_end_time_seconds: float
    sample_rate_hz: int
    start_sample: int
    end_sample: int
    output_clip_path: str


def _portable_path(path: Path, base_dir: Path | None = None) -> str:
    """Return a POSIX path relative to base_dir when possible."""
    base = (base_dir or Path.cwd()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_mono_wav(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, int(sample_rate)


def split_audio_file(
    input_audio_path: str | Path,
    clip_duration_seconds: float,
    output_audio_dir: str | Path = DEFAULT_OUTPUT_AUDIO_DIR,
    manifest_dir: str | Path = DEFAULT_MANIFEST_DIR,
    overwrite: bool = False,
) -> tuple[list[ClipMetadata], Path]:
    """Split a WAV file into fixed-duration clips and write a JSONL manifest."""
    if clip_duration_seconds <= 0:
        raise ValueError("clip_duration_seconds must be greater than 0")

    source_path = Path(input_audio_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {source_path}")

    audio, sample_rate = _read_mono_wav(source_path)
    if len(audio) == 0:
        raise ValueError("Input audio file is empty")

    clip_length_samples = int(round(clip_duration_seconds * sample_rate))
    if clip_length_samples <= 0:
        raise ValueError("clip_duration_seconds is too short for the sample rate")

    source_stem = source_path.stem
    clip_dir = Path(output_audio_dir) / source_stem
    manifest_path = Path(manifest_dir) / f"{source_stem}_clips.jsonl"

    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    clip_ranges = list(enumerate(range(0, len(audio), clip_length_samples)))
    output_clip_paths = [
        clip_dir / f"{source_stem}_clip_{clip_index:04d}.wav"
        for clip_index, _ in clip_ranges
    ]

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest_path}. "
            "Use --overwrite to replace it."
        )

    for output_clip_path in output_clip_paths:
        if output_clip_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing clip: {output_clip_path}. "
                "Use --overwrite to replace it."
            )

    metadata_rows: list[ClipMetadata] = []
    for clip_index, start_sample in clip_ranges:
        end_sample = min(start_sample + clip_length_samples, len(audio))
        clip_audio = audio[start_sample:end_sample]
        output_clip_path = output_clip_paths[clip_index]

        sf.write(str(output_clip_path), clip_audio, sample_rate)
        metadata_rows.append(
            ClipMetadata(
                source_audio_path=_portable_path(source_path),
                clip_index=clip_index,
                clip_start_time_seconds=start_sample / sample_rate,
                clip_end_time_seconds=end_sample / sample_rate,
                sample_rate_hz=sample_rate,
                start_sample=start_sample,
                end_sample=end_sample,
                output_clip_path=_portable_path(output_clip_path),
            )
        )

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    return metadata_rows, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a WAV recording into fixed-duration clips."
    )
    parser.add_argument("input_audio_path", help="Path to the source WAV file.")
    parser.add_argument(
        "--clip-duration-seconds",
        type=float,
        required=True,
        help="Fixed clip duration in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing clip files and manifest.",
    )
    parser.add_argument(
        "--output-audio-dir",
        type=Path,
        default=DEFAULT_OUTPUT_AUDIO_DIR,
        help="Root directory for generated clips.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory for the JSONL clip manifest.",
    )
    args = parser.parse_args()

    metadata_rows, manifest_path = split_audio_file(
        input_audio_path=args.input_audio_path,
        clip_duration_seconds=args.clip_duration_seconds,
        output_audio_dir=args.output_audio_dir,
        manifest_dir=args.manifest_dir,
        overwrite=args.overwrite,
    )

    print(f"Wrote {len(metadata_rows)} clips.")
    print(f"Manifest: {manifest_path}")
    for row in metadata_rows:
        print(row.output_clip_path)


if __name__ == "__main__":
    main()
