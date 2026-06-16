# Experiment Log — Toy Audio Agent

## Project context

Dissertation topic:

**Can AI agents annotate animal sounds reliably?**

This project explores whether an AI agent can produce useful strong-label bioacoustic annotations from spectrograms. The current goal is not high accuracy yet, but to identify reliability issues and build a stable local evaluation pipeline.

---

## Early reliability findings

Initial tests used Drosophila / fly song examples and a visual-language agent viewing generated spectrograms.

Main issues observed:

- **Role prompt bias:** a bat-specialist prompt caused a fly recording to be misclassified as bat / Myotis with high confidence.
- **Metadata leakage:** informative filenames such as `Dmel_male.wav` could strongly influence predictions.
- **Overconfident misclassification:** anonymised fly recordings were sometimes given specific but incorrect animal labels.
- **Better calibration with stricter prompts:** instructions such as “do not use filename,” “do not assume taxonomic group,” and “say uncertain when evidence is weak” reduced confident hallucination.
- **Spectrogram axis uncertainty:** the agent sometimes appeared to misread frequency ranges, suggesting that axis formatting and explicit sampling-rate information matter.

Takeaway:

```text
The agent can produce plausible bioacoustic interpretations, but reliability depends strongly on prompt design, metadata control, uncertainty calibration, and spectrogram visualisation.
```

---

## V1 pipeline status

Date: 2026-06-06

Current checkpoint:

```text
V1 pipeline stable
```

Implemented local workflow:

```text
audio_path
→ generate_spectrogram
→ optional zoom_spectrogram
→ agent proposes EventResult events
→ save_annotations
→ plot_predicted_boxes
→ evaluation.py
```

Current local tools/scripts:

- `generate_spectrogram(audio_path)`
- `zoom_spectrogram(audio_path, start_time, end_time, low_frequency, high_frequency, preset)`
- `save_annotations(audio_path, events)`
- `plot_predicted_boxes(audio_path, annotation_json_path)`
- `convert_aoef_to_eventresult.py`
- `evaluation.py`

Stable checks completed:

- local tools work
- AOEF-to-EventResult ground truth conversion works
- ground-truth visualisation works
- evaluation self-test passes

Evaluation self-test:

```text
prediction file:    ground_truth/pseudo_petersi_001_ground_truth.json
ground-truth file:  ground_truth/pseudo_petersi_001_ground_truth.json

predicted_count = 19
ground_truth_count = 19
false_positives = 0
missed_events = 0
mean_iou = 1.0
```

---

## EventResult schema

The local demo uses a lightweight JSON schema:

```text
audio_path
events
notes
```

Each event contains:

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

AOEF / Whombat conversion notes:

- AOEF `BoundingBox` coordinate order is:

```text
[start_time_seconds, low_frequency_hz, end_time_seconds, high_frequency_hz]
```

- `label` is extracted from annotation tags, prioritising `dwc:scientificName`, then broader class/genus/family/order tags.
- Converted ground truth is saved as:

```text
ground_truth/<audio_stem>_ground_truth.json
```

---

## Pilot subset construction

Date: 2026-06-06

A 9-recording pilot subset was selected from the BatDetect2 strong-label dataset and converted from AOEF / Whombat-style annotations to the local EventResult ground-truth JSON schema.

Subset composition:

- 3 clear positive recordings
- 2 dense/complex positive recordings
- 2 noisy recordings
- 2 negative empty recordings

Totals:

- 9 recordings
- 250 ground-truth events

Verification:

- all 9 audio paths exist
- all 9 ground-truth JSON files exist
- JSON event counts match the manifest
- total pilot events = 250

Manifest:

```text
ground_truth/pilot_subset_manifest.csv
```

---

## Pilot ground-truth visualisation

Date: 2026-06-08

Checkpoint:

```text
Pilot ground-truth visualisation passed
```

Verification:

- 9 / 9 recordings plotted successfully
- filenames match `pilot_subset_manifest.csv`
- default full plots look normal
- frequency-cropped plots look normal
- hide-label / max-label plots improve dense-case readability
- negative_empty recordings show no boxes as expected
- no obvious coordinate mismatch found

Output directory:

```text
outputs/pilot_ground_truth/
```

---

## Baseline agent prediction — pseudo_petersi_001.wav

Date: 2026-06-08

### Setup

- Recording: `pseudo_petersi_001.wav`
- Species label in ground truth: `Ozimops petersi`
- Ground-truth events: 19
- Agent tools used: `generate_spectrogram` only
- No external classifier
- No ground truth shown to the agent

### Agent output

The agent predicted 1 candidate event:

- time: 1.5-1.7 s
- frequency: 20-30 kHz
- label: possible bat echolocation call
- confidence: 0.7
- human_review_needed: true

### Evaluation

- predicted_count = 1
- ground_truth_count = 19
- matched_events = 0
- false_positives = 1
- missed_events = 19
- mean_iou = 0.0

### Interpretation

The baseline full-spectrogram single-agent setup severely under-detected short bat echolocation calls. The predicted box did not overlap sufficiently with any ground-truth event. This suggests that full-spectrogram visual inspection alone is insufficient for reliable strong-label bat annotation.

---

## Prompt-guided agent prediction — pseudo_petersi_001.wav

Date: 2026-06-08

Compared with the naive baseline, the prompt-guided run provided stronger task priors: the agent was told that the recording may contain multiple short bat echolocation calls and that the relevant search band was approximately 25-90 kHz.

The agent produced 15 candidate events, compared with 1 event in the naive baseline. However, all predicted boxes used the same broad 25-90 kHz frequency range, suggesting that the model used the prompt-provided search band as the box boundary rather than estimating tight frequency limits from the spectrogram.

Evaluation against 19 ground-truth events:

- predicted_count = 15
- ground_truth_count = 19
- matched_events = 0
- mean_iou = 0.0

Interpretation:

Prompt guidance changed the quantity of candidate outputs, but did not produce reliable strong-label localisation.

---

## Zoom-guided run attempt

Date: 2026-06-08

The agent was instructed to use both `generate_spectrogram` and `zoom_spectrogram` before annotation. However, the run failed before producing JSON output due to an Ollama / Pydantic-AI message-format error:

```text
invalid message content type: <nil>
```

Interpretation:

This suggests that the current chat-based tool-calling interface is unstable for multi-image / multi-tool visual annotation workflows.

Engineering implication:

The next implementation step may need to separate deterministic image generation from agent reasoning, for example by pre-generating full and zoomed spectrogram images before sending a simpler prompt to the model.

---

## Gemma4 coordinate-prompt evaluation

Date: 2026-06-10

### Input

- Audio: `pseudo_petersi_001.wav`
- Ground truth: 19 AOEF-derived bat call annotations
- Image: grid-overlay zoomed spectrogram, 0-4 s, 20-100 kHz

### Model

- Backend: HPC Ollama
- Model: `gemma4:31b`
- Prompt mode: `annotate-coordinates`

### Output

- The model returned valid JSON with real spectrogram coordinates in seconds and Hz.
- A minimal schema repair was applied: `review_reason: null` was converted to an empty string.
- The repaired EventResult contained 7 predicted events.

### Evaluation

- predicted_count = 7
- ground_truth_count = 19
- matched_events = 1
- false_positives = 6
- missed_events = 18
- mean_iou for matched event = 0.616

### Interpretation

This is the first successful end-to-end general-purpose VLM result in the pipeline. The model produced real time-frequency candidate boxes and achieved one valid match against the ground truth. However, recall remains low, with 18 of 19 ground-truth calls missed.

### Visual inspection note

Visual inspection shows that `gemma4:31b` predicts boxes in approximately the correct frequency band, but it under-samples the repeated pulse sequence. The model appears to select a few representative pulses rather than annotating every visible short call. This explains the low recall despite one matched prediction.

### Grid-guided annotation summary

Using a grid-overlay zoomed spectrogram and a coordinate-specific prompt, `gemma4:31b` successfully produced candidate bat-call annotations in real time-frequency coordinates. After a minimal schema repair that did not alter any coordinates, the baseline coordinate run returned 7 predicted events.

At an IoU threshold of 0.5, the baseline coordinate run matched 1 of 19 ground-truth events. A high-recall prompt increased the number of proposals from 7 to 11, but still matched only 1 event at IoU 0.5.

At more relaxed thresholds, the high-recall run showed better approximate localisation:

- IoU 0.1: matches increased from 2 to 5
- IoU 0.25: matches increased from 2 to 3
- IoU 0.5: matches remained 1

This suggests that the high-recall prompt helped the model identify more approximately correct regions, but did not improve precise strong-label localisation. Prompting changed proposal behaviour and coarse coverage, while accurate time-frequency boundary estimation remained a major limitation.

### Model comparison update

| Model | Prompt/Input | Valid EventResult | Matched | Main failure |
| --- | --- | ---: | ---: | --- |
| qwen3-vl full spectrogram | naive | yes | 0/19 | under-detection |
| qwen3-vl prompt-guided | full spectrogram | yes | 0/19 | coarse boxes |
| qwen3-vl zoom/grid | annotate | no | - | reasoning text |
| gemma4:31b grid | annotate | no | - | image-space `box_2d` |
| gemma4:31b grid | annotate-coordinates | yes after minimal repair | **1/19** | low recall |

---

## Current limitations

- No external classifier is integrated yet.
- No BatDetect2 inference is run yet.
- No multi-agent workflow yet.
- No interactive annotation UI.
- Evaluation uses simple greedy IoU matching.
- Agent predictions still need systematic testing on the pilot subset.

---

## Next steps

Short term:

- Run the agent on the 9-recording pilot subset.
- Save predictions as EventResult JSON.
- Visualise predicted boxes against spectrograms.
- Evaluate predictions against converted ground truth.
- Record failure modes: false positives, missed calls, poor localisation, label overreach, and uncertainty calibration.

Medium term:

- Integrate an external bioacoustic classifier as an additional evidence source.
- Compare agent-only predictions with classifier-assisted predictions.
- Add richer evaluation outputs, such as per-event match tables and label agreement.

Longer term:

- Extend from a single-agent prototype to a multi-agent workflow:
  - spectrogram inspection agent
  - classifier-evidence agent
  - annotation adjudication agent
  - human-review triage agent
