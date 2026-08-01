# Toy Audio Agent V1

Single-agent bioacoustic strong-label annotation prototype.

## Project Goal

This project explores a local tool-use workflow for strong-label bioacoustic annotation. The V1 agent inspects spectrograms, proposes cautious time-frequency event boxes, saves structured event JSON, visualises predicted boxes, and supports simple box-level evaluation against ground truth.

The prototype is intentionally small. It does not use external classifiers yet; labels are spectrogram-grounded proposals intended for human review.

## Current Implemented Tools

- `list_audio_files()`: lists `.wav` files in `AUDIO_DIR`.
- `generate_spectrogram(audio_path)`: generates a full spectrogram image from a local audio file.
- `zoom_spectrogram(audio_path, start_time, end_time, low_frequency, high_frequency, preset)`: generates a zoomed spectrogram view for a candidate time-frequency region.
- Both spectrogram tools can optionally render a coordinate grid with `show_grid=True`, plus configurable major/minor time and frequency steps.
- `save_annotations(audio_path, events)`: saves structured predicted events to `annotations/<audio_stem>_events.json`.
- `plot_predicted_boxes(audio_path, annotation_json_path)`: overlays saved predicted boxes on the spectrogram and writes `outputs/<audio_stem>_predicted_boxes.png`.
- `convert_aoef_to_eventresult.py`: converts one AOEF/Whombat recording into the demo EventResult JSON schema.
- `evaluation.py`: local script for comparing predicted boxes with ground-truth boxes from JSON or CSV.

Each predicted event contains:

```text
event_id
start_time_seconds
end_time_seconds
low_frequency_hz
high_frequency_hz
label
confidence
evidence
tools_used
human_review_needed
review_reason
```

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- An Ollama server with `qwen3-vl:latest`

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you are running Ollama locally, install it from [`ollama.com`](https://ollama.com/) and pull the model:

```bash
ollama pull qwen3-vl:latest
```

## Setup

Create the local virtual environment and install dependencies:

```bash
uv sync
```

The app reads these environment variables:

```bash
OLLAMA_HOST=http://localhost:11434
AUDIO_DIR=<path_to_folder_with_audio>
```

The app also loads a local `.env` file automatically. If unset, defaults are:

```bash
OLLAMA_HOST=http://localhost:11434
AUDIO_DIR=audio/
```

## Run The Demo

Start the web app:

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 7932 --reload --reload-include main.py
```

Then open:

```text
http://127.0.0.1:7932
```

Run the fresh zoom-guided prompt experiment without web chat history:

```bash
uv run python run_zoom_guided_prompt.py
```

Generate full and zoomed grid-overlay spectrogram examples:

```bash
uv run python scripts/visualization/generate_grid_spectrogram_examples.py
```

The grid examples are saved to:

```text
outputs/grid/
```

## Example Prompts

`generate_spectrogram`:

```text
Generate a spectrogram for sample_001.wav. Do not annotate.
```

`zoom_spectrogram`:

```text
Use zoom_spectrogram on sample_001.wav from 100 to 150 seconds and 1000 to 3000 Hz with preset candidate_event. Do not annotate.
```

Grid-overlay examples:

```text
Generate a spectrogram for pseudo_petersi_001.wav with show_grid=True, time_major_step=0.5, time_minor_step=0.1, frequency_major_step=10000, and frequency_minor_step=5000. Do not annotate.
```

```text
Use zoom_spectrogram on pseudo_petersi_001.wav from 0 to 4 seconds and 20000 to 100000 Hz with preset short_event and show_grid=True. Do not annotate.
```

`save_annotations`:

```text
Analyze sample_001.wav using spectrogram evidence only. Propose discrete time-frequency event boxes, return structured JSON events, and save them with save_annotations.
```

`plot_predicted_boxes`:

```text
Use plot_predicted_boxes for sample_001.wav with annotations/sample_001_events.json.
```

## Local Python Example

You can also call the visualisation tool directly:

```python
import asyncio
from main import plot_predicted_boxes

output_path = asyncio.run(
    plot_predicted_boxes("sample_001.wav", "annotations/sample_001_events.json")
)
print(output_path)
```

The figure is saved to:

```text
outputs/<audio_stem>_predicted_boxes.png
```

## Run Evaluation

Use `evaluation.py` to compare predicted events with ground-truth boxes:

```bash
uv run python evaluation.py annotations/sample_001_events.json ground_truth/sample_001_ground_truth.csv
```

Ground truth can be JSON or CSV. It should use the same box columns:

```text
event_id,start_time_seconds,end_time_seconds,low_frequency_hz,high_frequency_hz,label
```

The evaluator computes:

- time overlap
- frequency overlap
- 2D box IoU
- predicted box count
- ground-truth box count
- matched event count
- false positives
- missed events

The summary is saved to:

```text
outputs/evaluation_summary.csv
```

## V1 Status

Current checkpoint: V1 pipeline stable.

- Local spectrogram and annotation tools work.
- AOEF-to-EventResult ground truth conversion works.
- Ground-truth visualisation with predicted-box overlay works.
- Evaluation self-test passes when using `ground_truth/pseudo_petersi_001_ground_truth.json` as both prediction and ground truth:
  - `predicted_count = 19`
  - `ground_truth_count = 19`
  - `false_positives = 0`
  - `missed_events = 0`
  - `mean_iou = 1.0`

## Current Limitations

- No external classifier is integrated.
- Labels are cautious visual/spectrogram-based proposals, not verified species identifications.
- The agent may still require human review for uncertain boxes.
- Evaluation uses simple greedy IoU-threshold matching.
- There is no interactive annotation UI.
- The workflow currently targets `.wav` files in `AUDIO_DIR`.
- Ground-truth format handling is intentionally minimal.

## Next Steps

- Integrate external bioacoustic classifiers as optional evidence providers.
- Compare agent-proposed boxes with classifier detections.
- Add richer evaluation outputs, such as per-event match tables and label agreement.
- Extend from a single-agent prototype to a multi-agent workflow, for example:
  - spectrogram inspection agent
  - classifier-evidence agent
  - annotation adjudication agent
  - human-review triage agent

## Repository map

- `annotations/`, `audio/`, `ground_truth/`: source and evaluation data
- `prompts/`: versioned agent prompts
- `docs/`: protocols, audits, knowledge notes and dissertation drafts
- `outputs/`: generated inputs, predictions, evaluations and reports
- `experiments/registry.yaml`: experiment registry and freeze status
- `src/toy_audio_agent/`: reusable package code for new development
- `scripts/`: new experiment and analysis entry points
- `tests/`: unit and integration tests

Legacy root-level experiment scripts are retained to preserve reproducibility
of frozen experiments.
