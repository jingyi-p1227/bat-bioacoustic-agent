"""Generate clean linear-frequency PCEN spectrograms and visual diagnostics.

Model-input images are derived from WAV files only. Ground truth is read only
for separately stored diagnostic comparison sheets and is never drawn on the
clean PCEN inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

from main import make_spectrogram
from prepare_agent_spectrogram_inputs import (
    DEFAULT_MAX_FREQ_HZ,
    apply_grid_style,
    parse_clip_list,
    read_mono_audio,
    resolve_audio_path,
    spectrogram_to_image,
)


DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_BASELINE_DIR = Path("outputs/agent_inputs/prompt_v2_full_grid_v2")
DEFAULT_OUTPUT_DIR = Path("outputs/agent_inputs/p6_pcen_spectrograms")
DEFAULT_CLIP_IDS = ("OP_001", "OP_003", "OP_004", "OP_010", "OP_016", "OP_045")
GRID_STYLE = "grid_v2"
REPRESENTATION = "linear_frequency_pcen"


@dataclass(frozen=True)
class PCENParameters:
    """Fixed parameters for the first linear-frequency PCEN ablation."""

    alpha: float = 0.98
    delta: float = 2.0
    root: float = 0.5
    epsilon: float = 1e-6
    time_constant_seconds: float = 0.4
    initial_smoother: str = "frequency_median"
    display_lower_percentile: float = 1.0
    display_upper_percentile: float = 99.5


DEFAULT_PCEN_PARAMETERS = PCENParameters()


@dataclass(frozen=True)
class PCENManifestRow:
    """Portable metadata for one clean PCEN model input."""

    clip_id: str
    image_path: str
    original_audio_path: str
    representation: str
    grid_style: str
    pcen_parameters: str
    duration_seconds: float


MANIFEST_FIELDS = tuple(PCENManifestRow.__dataclass_fields__)


def pcen_image_path(output_dir: Path, clip_id: str) -> Path:
    """Return the stable clean PCEN image destination."""
    return output_dir / f"{clip_id}_pcen_grid_v2.png"


def comparison_contact_sheet_path(output_dir: Path, clip_id: str) -> Path:
    """Return the clean dB-versus-PCEN contact-sheet destination."""
    return output_dir / "contact_sheets" / f"{clip_id}_db_vs_pcen_contact_sheet.png"


def gt_diagnostic_contact_sheet_path(output_dir: Path, clip_id: str) -> Path:
    """Return the explicitly diagnostic-only comparison destination."""
    return (
        output_dir
        / "contact_sheets_gt_diagnostic_only"
        / f"{clip_id}_db_vs_pcen_gt_diagnostic_contact_sheet.png"
    )


def _portable_path(path: Path, base_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_pcen_parameters(parameters: PCENParameters) -> None:
    """Reject parameter combinations that cannot produce stable PCEN values."""
    if not 0 <= parameters.alpha <= 1:
        raise ValueError("PCEN alpha must be between 0 and 1")
    if parameters.delta < 0:
        raise ValueError("PCEN delta must be non-negative")
    if not 0 < parameters.root <= 1:
        raise ValueError("PCEN root must be in (0, 1]")
    if parameters.epsilon <= 0 or parameters.time_constant_seconds <= 0:
        raise ValueError("PCEN epsilon and time constant must be positive")
    if not (
        0
        <= parameters.display_lower_percentile
        < parameters.display_upper_percentile
        <= 100
    ):
        raise ValueError("PCEN display percentiles must be ordered within [0, 100]")


def linear_frequency_pcen(
    power_spectrogram: np.ndarray,
    *,
    hop_seconds: float,
    parameters: PCENParameters = DEFAULT_PCEN_PARAMETERS,
) -> np.ndarray:
    """Apply PCEN along time while preserving the linear frequency bins."""
    validate_pcen_parameters(parameters)
    if power_spectrogram.ndim != 2 or power_spectrogram.size == 0:
        raise ValueError("power_spectrogram must be a non-empty 2D array")
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    magnitude = np.sqrt(np.maximum(power_spectrogram, 0.0))
    smoothing = 1.0 - math.exp(-hop_seconds / parameters.time_constant_seconds)
    smoother = np.empty_like(magnitude, dtype=np.float64)
    initial = np.median(magnitude, axis=1)
    smoother[:, 0] = (
        (1.0 - smoothing) * initial + smoothing * magnitude[:, 0]
    )
    for frame_index in range(1, magnitude.shape[1]):
        smoother[:, frame_index] = (
            (1.0 - smoothing) * smoother[:, frame_index - 1]
            + smoothing * magnitude[:, frame_index]
        )

    normalized = magnitude / np.power(
        parameters.epsilon + smoother, parameters.alpha
    )
    return np.power(normalized + parameters.delta, parameters.root) - math.pow(
        parameters.delta, parameters.root
    )


def pcen_to_image(
    pcen: np.ndarray,
    *,
    parameters: PCENParameters = DEFAULT_PCEN_PARAMETERS,
) -> np.ndarray:
    """Normalize PCEN values for display using recorded robust percentiles."""
    validate_pcen_parameters(parameters)
    finite = pcen[np.isfinite(pcen)]
    if finite.size == 0:
        raise ValueError("PCEN array has no finite values")
    lower, upper = np.percentile(
        finite,
        [
            parameters.display_lower_percentile,
            parameters.display_upper_percentile,
        ],
    )
    if upper <= lower:
        return np.zeros_like(pcen, dtype=np.float64)
    return np.clip((pcen - lower) / (upper - lower), 0.0, 1.0)


def _plot_matrix(
    ax,
    *,
    image: np.ndarray,
    extent: list[float],
    clip_id: str,
    duration_seconds: float,
    displayed_max_freq_hz: float,
    title: str,
) -> None:
    ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )
    ax.set_xlim(0, duration_seconds)
    ax.set_ylim(0, displayed_max_freq_hz / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title or clip_id)
    apply_grid_style(
        ax,
        grid_style=GRID_STYLE,
        duration_seconds=duration_seconds,
        displayed_max_freq_hz=displayed_max_freq_hz,
    )


def save_clean_pcen_spectrogram(
    *,
    audio: np.ndarray,
    sample_rate_hz: int,
    clip_id: str,
    output_path: Path,
    max_freq_hz: float,
    parameters: PCENParameters,
) -> tuple[np.ndarray, np.ndarray, object, list[float]]:
    """Save one clean PCEN input and return matrices for diagnostics."""
    power_spec, stft = make_spectrogram(audio, sample_rate_hz)
    pcen = linear_frequency_pcen(
        power_spec,
        hop_seconds=stft.hop / sample_rate_hz,
        parameters=parameters,
    )
    pcen_image = pcen_to_image(pcen, parameters=parameters)
    extent = list(stft.extent(len(audio)))
    extent[2] /= 1000
    extent[3] /= 1000
    duration_seconds = len(audio) / sample_rate_hz
    displayed_max_freq_hz = min(max_freq_hz, sample_rate_hz / 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_matrix(
        ax,
        image=pcen_image,
        extent=extent,
        clip_id=clip_id,
        duration_seconds=duration_seconds,
        displayed_max_freq_hz=displayed_max_freq_hz,
        title=clip_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return power_spec, pcen_image, stft, extent


def save_side_by_side_contact_sheet(
    *,
    baseline_path: Path,
    pcen_path: Path,
    clip_id: str,
    output_path: Path,
) -> None:
    """Place the existing clean dB input beside the clean PCEN input."""
    with Image.open(baseline_path) as baseline_source:
        baseline = baseline_source.convert("RGB")
    with Image.open(pcen_path) as pcen_source:
        pcen = pcen_source.convert("RGB")

    margin = 18
    header_height = 54
    label_height = 24
    panel_width = max(baseline.width, pcen.width)
    panel_height = max(baseline.height, pcen.height)
    sheet = Image.new(
        "RGB",
        (panel_width * 2 + margin * 3, header_height + label_height + panel_height + margin),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 14), f"{clip_id}: clean dB grid_v2 vs PCEN grid_v2", fill="black")
    draw.text((margin, header_height), "Current dB grid_v2", fill="black")
    draw.text(
        (panel_width + margin * 2, header_height),
        "Linear-frequency PCEN grid_v2",
        fill="black",
    )
    y = header_height + label_height
    sheet.paste(baseline, (margin, y))
    sheet.paste(pcen, (panel_width + margin * 2, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    baseline.close()
    pcen.close()


def load_gt_events(eval_dir: Path, clip_id: str) -> list[dict]:
    """Read GT only for diagnostic-only comparison figures."""
    path = eval_dir / "ground_truth" / f"{clip_id}_ground_truth.json"
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError(f"Ground-truth events must be a list: {path}")
    return events


def _draw_gt_boxes(ax, events: list[dict]) -> None:
    for event_index, event in enumerate(events, start=1):
        start = float(event["start_time"])
        end = float(event["end_time"])
        low = float(event["low_frequency"]) / 1000
        high = float(event["high_frequency"]) / 1000
        if start >= end or low >= high:
            continue
        ax.add_patch(
            Rectangle(
                (start, low),
                end - start,
                high - low,
                fill=False,
                edgecolor="lime",
                linewidth=1.5,
            )
        )
        ax.text(
            start,
            high,
            str(event_index),
            color="lime",
            fontsize=7,
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
        )


def save_gt_diagnostic_comparison(
    *,
    clip_id: str,
    audio: np.ndarray,
    sample_rate_hz: int,
    power_spec: np.ndarray,
    pcen_image: np.ndarray,
    stft,
    extent: list[float],
    events: list[dict],
    output_path: Path,
    max_freq_hz: float,
) -> None:
    """Save a human-only dB/PCEN comparison with identical GT boxes."""
    db_image = spectrogram_to_image(power_spec)
    duration_seconds = len(audio) / sample_rate_hz
    displayed_max_freq_hz = min(max_freq_hz, sample_rate_hz / 2)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), squeeze=False)
    for ax, image, title in (
        (axes[0, 0], db_image, "Current dB grid_v2 + GT"),
        (axes[0, 1], pcen_image, "Linear-frequency PCEN grid_v2 + GT"),
    ):
        _plot_matrix(
            ax,
            image=image,
            extent=extent,
            clip_id=clip_id,
            duration_seconds=duration_seconds,
            displayed_max_freq_hz=displayed_max_freq_hz,
            title=title,
        )
        _draw_gt_boxes(ax, events)
    fig.suptitle(f"DIAGNOSTIC ONLY - {clip_id} - do not use as model input")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="PNG", bbox_inches="tight")
    plt.close(fig)


def write_pcen_manifest(rows: list[PCENManifestRow], output_path: Path) -> None:
    """Write deterministic PCEN model-input metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_visual_report(
    *,
    output_path: Path,
    clip_ids: list[str],
    parameters: PCENParameters,
) -> None:
    """Write the initial visual sanity-check record for P6B.1."""
    parameter_json = json.dumps(asdict(parameters), sort_keys=True)
    output_path.write_text(
        "\n".join(
            [
                "# P6B.1 PCEN Visual Sanity Check",
                "",
                "## Scope",
                "",
                "This stage compares the current dB `grid_v2` representation with a linear-frequency PCEN representation. No model inference was run. Clean PCEN images are derived from WAV files only; GT comparison sheets are diagnostic-only and must not be used as model inputs.",
                "",
                "## Parameters",
                "",
                f"```json\n{parameter_json}\n```",
                "",
                "PCEN is applied to the magnitude derived from the existing linear-frequency STFT. It does not use a mel projection, so the original time-frequency geometry and Hz mapping are preserved.",
                "",
                "## Pilot Clips",
                "",
                ", ".join(f"`{clip_id}`" for clip_id in clip_ids),
                "",
                "## Visual Findings",
                "",
                "### Call visibility",
                "",
                "PCEN makes the main 30-40 kHz call structures brighter and more locally distinct in `OP_001`, `OP_003`, `OP_010`, and `OP_045`. `OP_004` also gains contrast, although its stronger calls appear broader and more saturated. `OP_016` is mixed: several calls are easier to notice, but the strong call near 0.1 s becomes visually enlarged and saturated, which may make a tight frequency box harder to estimate.",
                "",
                "### Background and echoes",
                "",
                "PCEN suppresses some slowly varying horizontal energy, but it does not act as a general denoiser in these images. Fine high-frequency texture becomes much brighter, and repeated structures below 20 kHz are strongly enhanced. Echo-like or artefactual structures may therefore become more salient and could increase false positives.",
                "",
                "### Geometry",
                "",
                "Time and frequency axes are unchanged, and the same GT boxes align correctly in both diagnostic panels. PCEN therefore preserves the coordinate geometry required for bounding-box annotation. However, intensity compression changes the apparent edge and width of some strong calls, especially in `OP_004` and `OP_016`, so geometry preservation does not guarantee equally reliable visual box placement.",
                "",
                "### Improved and worsened cases",
                "",
                "The clearest apparent improvements are `OP_001`, `OP_003`, `OP_010`, and `OP_045`, where target-band calls stand out more strongly. `OP_004` is ambiguous because contrast improves while saturation increases. `OP_016` is the main possible regression: target calls are visible, but enhanced clutter and the broadened strong event may make separation and tight localisation harder rather than easier.",
                "",
                "### Initial decision",
                "",
                "PCEN is technically suitable for a controlled representative-six test with `qwen3.6:latest` once that model is available. The pilot is justified by the clearer target-band contrast, but it must explicitly test false positives and box IoU because enhanced noise and saturation are credible failure modes. PCEN should remain an alternative input condition and must not replace the dB baseline without quantitative evidence.",
                "",
                "## Safety Note",
                "",
                "Only `<output-dir>/<clip_id>_pcen_grid_v2.png` files are intended as model inputs. Files under `contact_sheets_gt_diagnostic_only/` contain ground truth and are for human inspection only.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_pcen_inputs(
    *,
    eval_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    clip_ids: list[str],
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
    parameters: PCENParameters = DEFAULT_PCEN_PARAMETERS,
    overwrite: bool = False,
    base_dir: Path | None = None,
) -> list[PCENManifestRow]:
    """Generate clean PCEN images, comparisons, diagnostics, and a manifest."""
    validate_pcen_parameters(parameters)
    if max_freq_hz <= 0:
        raise ValueError("max_freq_hz must be positive")
    base_dir = Path.cwd() if base_dir is None else base_dir
    manifest_path = output_dir / "pcen_manifest.csv"
    report_path = output_dir / "p6b1_pcen_visual_sanity_check.md"
    planned_outputs = [manifest_path, report_path]
    for clip_id in clip_ids:
        planned_outputs.extend(
            [
                pcen_image_path(output_dir, clip_id),
                comparison_contact_sheet_path(output_dir, clip_id),
                gt_diagnostic_contact_sheet_path(output_dir, clip_id),
            ]
        )
    existing = [path for path in planned_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Use --overwrite to replace it."
        )

    parameter_json = json.dumps(
        asdict(parameters), sort_keys=True, separators=(",", ":")
    )
    rows: list[PCENManifestRow] = []
    for clip_id in clip_ids:
        audio_path = resolve_audio_path(eval_dir, clip_id)
        baseline_path = baseline_dir / f"{clip_id}_spectrogram.png"
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline grid_v2 image not found: {baseline_path}")
        audio, sample_rate_hz = read_mono_audio(audio_path)
        output_path = pcen_image_path(output_dir, clip_id)
        power_spec, pcen_image, stft, extent = save_clean_pcen_spectrogram(
            audio=audio,
            sample_rate_hz=sample_rate_hz,
            clip_id=clip_id,
            output_path=output_path,
            max_freq_hz=max_freq_hz,
            parameters=parameters,
        )
        save_side_by_side_contact_sheet(
            baseline_path=baseline_path,
            pcen_path=output_path,
            clip_id=clip_id,
            output_path=comparison_contact_sheet_path(output_dir, clip_id),
        )
        save_gt_diagnostic_comparison(
            clip_id=clip_id,
            audio=audio,
            sample_rate_hz=sample_rate_hz,
            power_spec=power_spec,
            pcen_image=pcen_image,
            stft=stft,
            extent=extent,
            events=load_gt_events(eval_dir, clip_id),
            output_path=gt_diagnostic_contact_sheet_path(output_dir, clip_id),
            max_freq_hz=max_freq_hz,
        )
        rows.append(
            PCENManifestRow(
                clip_id=clip_id,
                image_path=_portable_path(output_path, base_dir),
                original_audio_path=_portable_path(audio_path, base_dir),
                representation=REPRESENTATION,
                grid_style=GRID_STYLE,
                pcen_parameters=parameter_json,
                duration_seconds=round(len(audio) / sample_rate_hz, 6),
            )
        )

    write_pcen_manifest(rows, manifest_path)
    write_visual_report(
        output_path=report_path,
        clip_ids=clip_ids,
        parameters=parameters,
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean PCEN inputs and visual sanity comparisons."
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--clip-list",
        default=",".join(DEFAULT_CLIP_IDS),
        help="Comma-separated evaluation clip ids.",
    )
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQ_HZ)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_ids = parse_clip_list(args.clip_list)
    rows = build_pcen_inputs(
        eval_dir=args.eval_dir,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        clip_ids=clip_ids,
        max_freq_hz=args.max_freq,
        overwrite=args.overwrite,
    )
    print(f"Generated {len(rows)} clean PCEN spectrogram(s).")
    for row in rows:
        print(f"{row.clip_id} | {row.image_path}")
    print(f"Manifest: {args.output_dir / 'pcen_manifest.csv'}")
    print(f"Report: {args.output_dir / 'p6b1_pcen_visual_sanity_check.md'}")


if __name__ == "__main__":
    main()
