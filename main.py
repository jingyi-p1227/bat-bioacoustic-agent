from pydantic import BaseModel, Field, model_validator
from io import BytesIO
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import soundfile as sf
from dotenv import load_dotenv
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from pydantic_ai import Agent, BinaryContent, ToolReturn
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from scipy.signal import ShortTimeFFT


load_dotenv()

DEFAULT_AUDIO_DIR = Path("audio")
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", str(DEFAULT_AUDIO_DIR))).expanduser()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

ANNOTATION_DIR = Path("annotations")
ANNOTATION_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class OllamaSafeOpenAIChatModel(OpenAIChatModel):
    """Avoid `content: null` assistant messages that some Ollama models reject."""

    class _MapModelResponseContext(OpenAIChatModel._MapModelResponseContext):
        def _into_message_param(self):
            message_param = super()._into_message_param()
            if message_param.get("content") is None:
                message_param["content"] = "Calling local tool."
            return message_param


model = OllamaSafeOpenAIChatModel(
    model_name="qwen3-vl:latest",
    provider=OllamaProvider(base_url=f"{OLLAMA_HOST}/v1"),
)

# 定义单个事件框
class SpectrogramEvent(BaseModel):
    event_id: str
    start_time_seconds: float
    end_time_seconds: float
    low_frequency_hz: float
    high_frequency_hz: float
    label: str
    confidence: float
    evidence: str
    tools_used: list[str]
    human_review_needed: bool
    review_reason: str

# 定义整个事件结果结构 schema: audio_path+events+notes
class EventResult(BaseModel):
    audio_path: str
    events: list[SpectrogramEvent]
    notes: str = ""


class TemporalEvent(BaseModel):
    """Temporal-only event annotation without frequency boundaries."""

    event_id: str
    start_time_seconds: float
    end_time_seconds: float
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str
    tools_used: list[str]
    human_review_needed: bool
    review_reason: str

    @model_validator(mode="after")
    def validate_time_range(self) -> "TemporalEvent":
        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than or equal to start_time_seconds")
        return self


class TemporalEventResult(BaseModel):
    """Structured temporal-only annotation result."""

    audio_path: str
    events: list[TemporalEvent]
    notes: str = ""


agent = Agent(
    model=model,
    output_type=EventResult | str,
    instructions=(
        "You are a careful bioacoustic annotation assistant. "
        "Your goal is to imitate a human annotator working on a spectrogram. "
        "If the user asks only to generate a spectrogram, or explicitly says 'Do not annotate', "
        "only call generate_spectrogram(audio_path), then stop with a brief plain-text confirmation. "
        "In that case, do not call zoom_spectrogram, do not propose candidate boxes, and do not "
        "return EventResult, final_result, JSON, or events. "
        "Use this local tool pipeline: "
        "1) call generate_spectrogram(audio_path), "
        "2) call zoom_spectrogram(audio_path, start_time, end_time, low_frequency, high_frequency, preset) "
        "when a candidate event needs closer inspection, "
        "3) propose time-frequency bounding boxes for visible acoustic events, "
        "4) return structured JSON events, "
        "5) call save_annotations(audio_path, events) when the user asks to save results. "
        "If the user asks to visualise saved predictions, use plot_predicted_boxes(audio_path, annotation_json_path). "
        "The audio may contain birds, bats, insects, amphibians, mammals, noise, or unknown sounds. "
        "Do not assume it is a bat. "
        "Avoid unsupported species-level claims. "
        "If evidence is weak, use labels like 'unknown animal sound', 'possible insect sound', "
        "or 'possible bird call'. "
        "For every event, output: event_id, start_time_seconds, end_time_seconds, "
        "low_frequency_hz, high_frequency_hz, label, confidence, evidence, "
        "tools_used, human_review_needed, and review_reason. "
        "Include generate_spectrogram and any zoom_spectrogram calls in tools_used. "
        "If you cannot localise an event reliably, say so instead of inventing precise boxes. "

        "Never write a literal tool_call JSON in your final answer. "
        "Actually use the generate_spectrogram tool before proposing boxes. "
        "If the spectrogram is not available, return an empty events list and set notes to "
        "'spectrogram unavailable'. "
        "Only propose boxes for events that are visibly grounded in the generated spectrogram. "
        "Use conservative confidence. Do not use confidence above 0.8 in this pilot. "
        "Do not create broad full-spectrum background-noise boxes unless background noise is the main visible event. "

        "Focus on discrete acoustic events that a human annotator would draw boxes around. "
        "Do not create bounding boxes for continuous background noise unless explicitly requested. "
        "Avoid very large boxes spanning tens or hundreds of seconds unless the acoustic event is truly continuous. "
        "Prefer localised candidate boxes around visible bursts, calls, pulses, or modulated structures. "
        "If only background noise is visible, return an empty event list and explain that no discrete animal sound event was found. "
        "Return events in valid JSON when the user asks for experimental output. "
    ),
)


def make_spectrogram(audio: np.ndarray, sr: int):
    # Use the actual sample rate of the audio file.
    nperseg = min(1024, max(128, len(audio) // 10))
    noverlap = nperseg // 2

    stft = ShortTimeFFT.from_window(
        "hann",
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    spec = stft.spectrogram(audio)
    return spec, stft


def _list_audio_paths() -> list[Path]:
    audio_files = AUDIO_DIR.glob("*.[wW][aA][vV]")
    return list(audio_files)


@agent.tool_plain
async def list_audio_files() -> str:
    """List audio files in the configured AUDIO_DIR."""
    audio_files = _list_audio_paths()
    if not audio_files:
        return f"No WAV audio files found in {AUDIO_DIR.resolve()}."
    return "Available WAV audio files:\n" + "\n".join(
        f"- {audio_file.name}" for audio_file in audio_files
    )


def resolve_audio_path(path: str | Path) -> Path:
    """Resolve an audio filename safely inside AUDIO_DIR."""
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (AUDIO_DIR / candidate).resolve()

    audio_root = AUDIO_DIR.resolve()
    if audio_root not in resolved.parents and resolved != audio_root:
        raise ValueError(f"Path must be inside AUDIO_DIR: {audio_root}")
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    return resolved


def read_mono_audio(path: str | Path) -> tuple[Path, np.ndarray, int]:
    audio_path = resolve_audio_path(path)
    audio, sr = sf.read(str(audio_path))

    # Convert stereo audio to mono.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio_path, audio, sr


def to_decibels(x, ref=np.max, eps=1e-20):
    if callable(ref):
        ref = ref(x)

    if ref == 0:
        return x

    return 20 * np.log10(np.maximum(x, eps) / ref)


def to_image(x, min_db=-80, max_db=0, eps=1e-20):
    x = to_decibels(x, eps=eps)
    x = np.clip(x, min_db, max_db)
    x = (x - min_db) / (max_db - min_db)
    return x


def plot_spectrogram_with_grid(
    spec,
    audio,
    stft: ShortTimeFFT,
    sr: int,
    grid=True,
    show_grid: bool = False,
    time_major_step: float = 0.5,
    time_minor_step: float = 0.1,
    frequency_major_step: float = 10000,
    frequency_minor_step: float = 5000,
    start_time: float | None = None,
    end_time: float | None = None,
    low_frequency: float | None = None,
    high_frequency: float | None = None,
    preset: str = "full",
) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    extent = stft.extent(len(audio))
    spec_img = to_image(spec)

    ax.imshow(spec_img, aspect="auto", origin="lower", extent=extent, cmap="gray")

    duration = len(audio) / sr
    ax.set_title(
        f"Spectrogram ({preset}) | duration: {duration:.2f}s | sample rate: {sr} Hz"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if start_time is not None and end_time is not None:
        ax.set_xlim(max(0, start_time), min(duration, end_time))
    if low_frequency is not None and high_frequency is not None:
        ax.set_ylim(max(0, low_frequency), min(sr / 2, high_frequency))

    if show_grid:
        for name, step in {
            "time_major_step": time_major_step,
            "time_minor_step": time_minor_step,
            "frequency_major_step": frequency_major_step,
            "frequency_minor_step": frequency_minor_step,
        }.items():
            if step <= 0:
                raise ValueError(f"{name} must be greater than 0")
        ax.xaxis.set_major_locator(MultipleLocator(time_major_step))
        ax.xaxis.set_minor_locator(MultipleLocator(time_minor_step))
        ax.yaxis.set_major_locator(MultipleLocator(frequency_major_step))
        ax.yaxis.set_minor_locator(MultipleLocator(frequency_minor_step))
        ax.tick_params(axis="both", which="major", labelsize=8, length=4)
        ax.tick_params(axis="both", which="minor", labelsize=0, length=2)
        ax.grid(which="major", color="cyan", linewidth=0.7, alpha=0.65)
        ax.grid(which="minor", color="cyan", linewidth=0.35, alpha=0.35)
    elif grid:
        ax.grid(color="red", linewidth=0.5, alpha=0.4)

    return fig


def plot_events_on_spectrogram(
    spec,
    audio,
    stft: ShortTimeFFT,
    sr: int,
    events: list[SpectrogramEvent],
    hide_labels: bool = False,
    frequency_max_hz: float | None = None,
    title_mode: str = "prediction",
    max_labels: int | None = None,
) -> Figure:
    preset = "ground truth boxes" if title_mode == "ground truth" else "predicted boxes"
    fig = plot_spectrogram_with_grid(
        spec,
        audio,
        stft,
        sr,
        high_frequency=frequency_max_hz,
        low_frequency=0 if frequency_max_hz is not None else None,
        preset=preset,
    )
    ax = fig.axes[0]

    labels_drawn = 0
    for event in events:
        width = event.end_time_seconds - event.start_time_seconds
        height = event.high_frequency_hz - event.low_frequency_hz
        if width <= 0 or height <= 0:
            continue

        rect = Rectangle(
            (event.start_time_seconds, event.low_frequency_hz),
            width,
            height,
            fill=False,
            edgecolor="lime",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        if hide_labels:
            continue
        if max_labels is not None and labels_drawn >= max_labels:
            continue
        ax.text(
            event.start_time_seconds,
            event.high_frequency_hz,
            f"{event.label} ({event.confidence:.2f})",
            color="lime",
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 2, "edgecolor": "none"},
        )
        labels_drawn += 1

    return fig


def figure_to_image(fig: Figure) -> BytesIO:
    buffer = BytesIO()
    fig.savefig(buffer, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    return buffer


@agent.tool_plain
async def generate_spectrogram(
    audio_path: str,
    show_grid: bool = False,
    time_major_step: float = 0.5,
    time_minor_step: float = 0.1,
    frequency_major_step: float = 10000,
    frequency_minor_step: float = 5000,
) -> ToolReturn:
    """Generate a spectrogram image for an audio file in AUDIO_DIR.

    Args:
        audio_path: Audio filename, for example "clip_001.wav".
        show_grid: If true, overlay major/minor coordinate grid lines.
        time_major_step: Major time grid spacing in seconds.
        time_minor_step: Minor time grid spacing in seconds.
        frequency_major_step: Major frequency grid spacing in Hz.
        frequency_minor_step: Minor frequency grid spacing in Hz.
    """
    resolved_audio_path, audio, sr = read_mono_audio(audio_path)

    spec, local_stft = make_spectrogram(audio, sr)
    fig = plot_spectrogram_with_grid(
        spec,
        audio,
        local_stft,
        sr,
        show_grid=show_grid,
        time_major_step=time_major_step,
        time_minor_step=time_minor_step,
        frequency_major_step=frequency_major_step,
        frequency_minor_step=frequency_minor_step,
        preset="full",
    )
    buffer = figure_to_image(fig)

    duration = len(audio) / sr
    grid_text = " Coordinate grid enabled." if show_grid else ""

    summary = (
        f"Generated spectrogram for {resolved_audio_path.name}. "
        f"Duration: {duration:.2f}s. Sample rate: {sr} Hz. "
        "Use the time and frequency axes to propose bounding boxes."
        f"{grid_text}"
    )

    return ToolReturn(
        return_value=summary,
        content=[
            summary,
            BinaryContent(
                buffer.read(),
                media_type="image/png",
                identifier=(
                    f"{resolved_audio_path.stem}_spectrogram"
                    f"{'_grid' if show_grid else ''}.png"
                ),
            ),
        ],
    )


@agent.tool_plain
async def zoom_spectrogram(
    audio_path: str,
    start_time: float,
    end_time: float,
    low_frequency: float,
    high_frequency: float,
    preset: str = "custom",
    show_grid: bool = False,
    time_major_step: float = 0.5,
    time_minor_step: float = 0.1,
    frequency_major_step: float = 10000,
    frequency_minor_step: float = 5000,
) -> ToolReturn:
    """Generate a zoomed spectrogram image for a candidate time-frequency region.

    Args:
        audio_path: Audio filename, for example "clip_001.wav".
        start_time: Zoom start time in seconds.
        end_time: Zoom end time in seconds.
        low_frequency: Lowest zoom frequency in Hz.
        high_frequency: Highest zoom frequency in Hz.
        preset: Short name for the zoom purpose, for example "candidate_event".
        show_grid: If true, overlay major/minor coordinate grid lines.
        time_major_step: Major time grid spacing in seconds.
        time_minor_step: Minor time grid spacing in seconds.
        frequency_major_step: Major frequency grid spacing in Hz.
        frequency_minor_step: Minor frequency grid spacing in Hz.
    """
    if start_time >= end_time:
        raise ValueError("start_time must be less than end_time")
    if low_frequency >= high_frequency:
        raise ValueError("low_frequency must be less than high_frequency")

    resolved_audio_path, audio, sr = read_mono_audio(audio_path)
    spec, local_stft = make_spectrogram(audio, sr)
    fig = plot_spectrogram_with_grid(
        spec,
        audio,
        local_stft,
        sr,
        start_time=start_time,
        end_time=end_time,
        low_frequency=low_frequency,
        high_frequency=high_frequency,
        show_grid=show_grid,
        time_major_step=time_major_step,
        time_minor_step=time_minor_step,
        frequency_major_step=frequency_major_step,
        frequency_minor_step=frequency_minor_step,
        preset=preset,
    )
    buffer = figure_to_image(fig)
    grid_text = " Coordinate grid enabled." if show_grid else ""

    summary = (
        f"Generated zoomed spectrogram for {resolved_audio_path.name}: "
        f"{start_time:.3f}-{end_time:.3f}s, "
        f"{low_frequency:.1f}-{high_frequency:.1f} Hz, preset={preset}."
        f"{grid_text}"
    )

    return ToolReturn(
        return_value=summary,
        content=[
            summary,
            BinaryContent(
                buffer.read(),
                media_type="image/png",
                identifier=(
                    f"{resolved_audio_path.stem}_{preset}_zoom"
                    f"{'_grid' if show_grid else ''}.png"
                ),
            ),
        ],
    )


def event_to_dict(event: BaseModel | dict) -> dict:
    if isinstance(event, BaseModel):
        return event.model_dump()
    return event


def read_event_result(annotation_json_path: str | Path) -> EventResult:
    json_path = Path(annotation_json_path).expanduser()
    if not json_path.is_absolute():
        json_path = Path.cwd() / json_path
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    return EventResult.model_validate(payload)


@agent.tool_plain
async def save_annotations(audio_path: str, events: list[SpectrogramEvent]) -> str:
    """Save proposed spectrogram bounding-box events as JSON.

    Args:
        audio_path: Audio filename, for example "clip_001.wav".
        events: A list of structured event objects.
    """
    resolved_audio_path = resolve_audio_path(audio_path)
    output_path = ANNOTATION_DIR / f"{resolved_audio_path.stem}_events.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audio_path": str(resolved_audio_path),
        "events": [event_to_dict(event) for event in events],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return f"Saved {len(events)} events to {output_path}"


@agent.tool_plain
async def plot_predicted_boxes(
    audio_path: str,
    annotation_json_path: str,
    hide_labels: bool = False,
    frequency_max_hz: float | None = None,
    title_mode: str = "prediction",
    max_labels: int | None = None,
) -> str:
    """Save a spectrogram image with predicted event boxes overlaid.

    Args:
        audio_path: Audio filename, for example "clip_001.wav".
        annotation_json_path: Path to a JSON file containing an EventResult payload.
        hide_labels: If true, draw boxes without text labels.
        frequency_max_hz: Optional maximum y-axis frequency in Hz.
        title_mode: Either "prediction" or "ground truth".
        max_labels: Optional maximum number of text labels to draw.
    """
    resolved_audio_path, audio, sr = read_mono_audio(audio_path)
    event_result = read_event_result(annotation_json_path)

    spec, local_stft = make_spectrogram(audio, sr)
    fig = plot_events_on_spectrogram(
        spec,
        audio,
        local_stft,
        sr,
        event_result.events,
        hide_labels=hide_labels,
        frequency_max_hz=frequency_max_hz,
        title_mode=title_mode,
        max_labels=max_labels,
    )

    output_path = OUTPUT_DIR / f"{resolved_audio_path.stem}_predicted_boxes.png"
    fig.savefig(output_path, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return str(output_path)


app = agent.to_web()

if __name__ == "__main__":
    import asyncio

    async def main():
        print("Demo script started")
        print("Audio dir:", AUDIO_DIR)
        print("Audio dir exists:", AUDIO_DIR.exists())

        audio_files = _list_audio_paths()
        print("Number of audio files:", len(audio_files))
        print("Audio files:")
        for f in audio_files:
            print("-", f)

        if not audio_files:
            print("No audio files found.")
            return

        first_audio = audio_files[0]

        print("\nTesting spectrogram generation...")
        spectrogram = await generate_spectrogram(first_audio.name)
        print("Spectrogram generated:", type(spectrogram))

        print("\nAsking agent to annotate...")
        result = await agent.run(
            f"""
            Analyze the audio file {first_audio.name} using only the spectrogram evidence.

            Do not use the filename as evidence.
            Do not assume the taxonomic group.
            If the evidence is insufficient, say so.
            Return structured JSON events. Save them only if there are discrete events.
            """
        )

        print("\nAgent result:")
        print(result.output)

    asyncio.run(main())
