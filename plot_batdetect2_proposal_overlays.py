"""Plot human-only GT and BatDetect2 proposal diagnostic overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from main import make_spectrogram
from prepare_agent_spectrogram_inputs import (
    DEFAULT_MAX_FREQ_HZ,
    DEFAULT_MIN_DB,
    apply_grid_style,
    read_mono_audio,
    spectrogram_to_image,
)


DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_PROPOSAL_DIR = Path(
    "outputs/tool_outputs/batdetect2_proposals/representative6"
)
DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")


def parse_clip_ids(value: str) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def proposal_overlay_path(output_dir: Path, clip_id: str) -> Path:
    return output_dir / f"{clip_id}_batdetect2_proposal_diagnostic.png"


def draw_event_box(ax, event: dict, *, proposal: bool, index: int) -> None:
    if proposal:
        start = float(event["start_time_seconds"])
        end = float(event["end_time_seconds"])
        low = float(event["low_frequency_hz"]) / 1000
        high = float(event["high_frequency_hz"]) / 1000
        color, linestyle, prefix = "orange", "--", "P"
    else:
        start = float(event["start_time"])
        end = float(event["end_time"])
        low = float(event["low_frequency"]) / 1000
        high = float(event["high_frequency"]) / 1000
        color, linestyle, prefix = "lime", "-", "G"
    if start >= end or low >= high:
        return
    ax.add_patch(
        Rectangle(
            (start, low),
            end - start,
            high - low,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=1.5,
        )
    )
    ax.text(
        start,
        high,
        f"{prefix}{index}",
        color=color,
        fontsize=6.5,
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
    )


def plot_proposal_overlay(
    *,
    eval_dir: Path,
    proposal_dir: Path,
    output_dir: Path,
    clip_id: str,
    min_db: float = DEFAULT_MIN_DB,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> Path:
    audio_path = eval_dir / "audio" / f"{clip_id}.wav"
    gt_path = eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"
    proposal_path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
    for path in (audio_path, gt_path, proposal_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required diagnostic input not found: {path}")

    gt_events = json.loads(gt_path.read_text(encoding="utf-8"))["events"]
    proposals = json.loads(proposal_path.read_text(encoding="utf-8"))["events"]
    audio, sample_rate = read_mono_audio(audio_path)
    spec, stft = make_spectrogram(audio, sample_rate)
    image = spectrogram_to_image(spec, min_db=min_db)
    extent = list(stft.extent(len(audio)))
    extent[2] /= 1000
    extent[3] /= 1000
    duration = len(audio) / sample_rate
    displayed_max = min(max_freq_hz, sample_rate / 2)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )
    ax.set_xlim(0, duration)
    ax.set_ylim(0, displayed_max / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(
        f"DIAGNOSTIC ONLY - {clip_id} | GT={len(gt_events)} | "
        f"BatDetect2 proposals={len(proposals)}"
    )
    apply_grid_style(
        ax,
        grid_style="grid_v2",
        duration_seconds=duration,
        displayed_max_freq_hz=displayed_max,
    )
    for index, event in enumerate(gt_events, start=1):
        draw_event_box(ax, event, proposal=False, index=index)
    for index, event in enumerate(proposals, start=1):
        draw_event_box(ax, event, proposal=True, index=index)
    ax.legend(
        handles=[
            Line2D([0], [0], color="lime", linewidth=1.5, label="Ground truth"),
            Line2D([0], [0], color="orange", linestyle="--", linewidth=1.5, label="BatDetect2 proposal"),
        ],
        loc="upper right",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = proposal_overlay_path(output_dir, clip_id)
    fig.savefig(output_path, format="PNG", bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clip-list", default=",".join(DEFAULT_CLIP_IDS))
    parser.add_argument("--min-db", type=float, default=DEFAULT_MIN_DB)
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ_HZ)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.proposal_dir / "diagnostic_overlays"
    paths = [
        plot_proposal_overlay(
            eval_dir=args.eval_dir,
            proposal_dir=args.proposal_dir,
            output_dir=output_dir,
            clip_id=clip_id,
            min_db=args.min_db,
            max_freq_hz=args.max_freq,
        )
        for clip_id in parse_clip_ids(args.clip_list)
    ]
    print(f"Saved {len(paths)} diagnostic overlay(s):")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

