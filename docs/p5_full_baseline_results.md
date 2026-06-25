# P5 Full Baseline Results: Prompt V2 Fixed-View Evaluation

## Purpose

This document summarises the P5 full-baseline experiments for the `Ozimops petersi` Evaluation Set V1.

The purpose of P5 was to move beyond the six-clip smoke test and evaluate prompt-v2 single-agent strong labelling on the full 45-clip benchmark. The main questions were:

1. How much does the spectrogram grid style affect VLM-based strong labelling?
2. Which VLM backend is the strongest fixed-view baseline?
3. What failure modes remain before moving to adaptive grid/zoom workflows?

All experiments used clean spectrogram inputs generated from WAV audio only. Ground-truth overlays and diagnostic figures were not used as model inputs.

---

## Evaluation Set

Evaluation set:

```text
outputs/evaluation_sets/ozimops_petersi_v1/
```

Summary:

```text
Species: Ozimops petersi
Number of clips: 45
Ground-truth event instances: 198
Clip duration: mostly 1 second, with partial final clips where applicable
Task: event-level strong labelling of individual bat echolocation calls
```

The task is not whole-clip classification. Each model prediction is evaluated as a structured event-level annotation with a time-frequency bounding box:

```text
[start_time_seconds, low_frequency_hz, end_time_seconds, high_frequency_hz]
```

---

## Prompt and Evaluation Protocol

Prompt:

```text
prompts/prompt_v2_bat_strong_label.md
```

Evaluation protocol:

```text
docs/evaluation_protocol_v1_ozi_petersi.md
```

The evaluation protocol was not changed during P5.

Matching was performed independently per clip using confidence-ordered, one-to-one greedy temporal matching. A prediction was eligible to match a ground-truth event when:

```text
temporal IoU >= 0.3
```

For each matched pair, the evaluation computed:

```text
time_iou
frequency_iou
box_iou
start_time_error
end_time_error
low_frequency_error
high_frequency_error
```

Aggregate metrics included:

```text
precision
recall
F1
mean_time_iou
mean_frequency_iou
mean_box_iou
box_iou >= 0.3 count
box_iou >= 0.5 count
```

---

## P5A: Grid-Style Full Baseline

### Goal

P5A tested whether spectrogram grid design affects VLM-based strong labelling.

The same model, prompt, evaluation set, and evaluation protocol were used. The only experimental variable was the clean spectrogram grid style.

Model:

```text
qwen3.6:latest
```

Grid styles:

```text
grid_v1
grid_v2
```

### Grid Definitions

`grid_v1` is the fixed project-default grid:

```text
Time major grid: 0.5 s
Time minor grid: 0.1 s
Frequency major grid: 10 kHz
Frequency minor grid: 5 kHz
```

`grid_v2` is an auto-readable grid:

```text
Grid steps are selected from the visible time span and frequency span using the existing readable-grid helper.
For the 1-second evaluation clips, this produces denser and more readable overview grids than grid_v1.
```

Both grid styles used clean spectrograms only, with no ground-truth boxes, event labels, prediction boxes, or diagnostic information.

### Input Directories

```text
outputs/agent_inputs/prompt_v2_full_grid_v1/
outputs/agent_inputs/prompt_v2_full_grid_v2/
```

### Run Directories

```text
outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v1/
outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/
```

### Results

| Model          | Grid    | Clips | GT events | Predictions |  TP |  FP |  FN | Precision | Recall |    F1 | Mean box IoU |
| -------------- | ------- | ----: | --------: | ----------: | --: | --: | --: | --------: | -----: | ----: | -----------: |
| qwen3.6:latest | grid_v1 |    45 |       198 |         233 |  57 | 176 | 141 |     0.245 |  0.288 | 0.265 |        0.289 |
| qwen3.6:latest | grid_v2 |    45 |       198 |         247 | 134 | 113 |  64 |     0.543 |  0.677 | 0.602 |        0.332 |

### Interpretation

The grid-style experiment showed that spectrogram visualisation has a large effect on VLM-based strong labelling.

Under the same model and prompt, changing from `grid_v1` to `grid_v2` improved F1 from `0.265` to `0.602`. This improvement was driven by both higher recall and higher precision:

```text
grid_v1: TP=57, FP=176, FN=141
grid_v2: TP=134, FP=113, FN=64
```

This indicates that the VLM was not only sensitive to the underlying spectrogram content, but also to the visual interface used to present the spectrogram. The grid is therefore not merely a cosmetic plotting choice. It functions as part of the annotation tool interface by helping the model estimate time-frequency coordinates.

Based on this result, `grid_v2` was selected as the default fixed-view spectrogram style for subsequent prompt-v2 baselines.

---

## P5B: Full Model Baseline With Grid V2

### Goal

P5B compared the two selected stronger VLM backends under the best fixed-view grid style from P5A.

Models:

```text
qwen3.6:latest
gemma4:31b
```

Grid style:

```text
grid_v2
```

Prompt and evaluation protocol were kept fixed.

### Run Directories

```text
outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/
outputs/agent_runs/prompt_v2_full_gemma4_31b_grid_v2/
```

### Results

| Model          | Grid    | Clips | Parse failures | GT events | Predictions |  TP |  FP | FN | Precision | Recall |    F1 |     Mean time IoU | Mean frequency IoU | Mean box IoU |     Box IoU >= 0.3 |     Box IoU >= 0.5 |
| -------------- | ------- | ----: | -------------: | --------: | ----------: | --: | --: | -: | --------: | -----: | ----: | ----------------: | -----------------: | -----------: | ----------------: | ----------------: |
| qwen3.6:latest | grid_v2 |    45 |              0 |       198 |         247 | 134 | 113 | 64 |     0.543 |  0.677 | 0.602 |             0.581 |              0.526 |        0.332 |                74 |                23 |
| gemma4:31b     | grid_v2 |    45 |              2 |       198 |         244 | 113 | 131 | 85 |     0.463 |  0.571 | 0.511 |             0.609 |              0.516 |        0.339 |                65 |                24 |

The full comparison summary is saved at:

```text
outputs/agent_runs/prompt_v2_full_model_comparison.csv
```

The gemma4 evaluation summary is saved at:

```text
outputs/agent_runs/prompt_v2_full_gemma4_31b_grid_v2/evaluation/aggregate_summary.json
```

The qwen3.6 evaluation summary is saved at:

```text
outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/
```

### Parse Failures

`qwen3.6:latest` had no parse failures.

`gemma4:31b` produced two parse failures:

```text
OP_014: JSONDecodeError: Expecting ',' delimiter: line 27 column 6
OP_023: JSONDecodeError: Expecting ',' delimiter: line 51 column 6
```

For both failed clips, placeholder prediction JSON files with empty `events` lists were written so that evaluation could continue.

---

## Model Comparison Interpretation

### qwen3.6:latest

`qwen3.6:latest` is the strongest current fixed-view baseline.

It achieved:

```text
TP=134
FP=113
FN=64
Precision=0.543
Recall=0.677
F1=0.602
```

Compared with `gemma4:31b`, it detected more true events, missed fewer ground-truth events, and produced a higher overall F1 score. This makes it the primary fixed-view baseline for subsequent experiments.

Its main limitation is that many predictions still fail to reach high box-IoU thresholds. The model often finds approximate event locations, but time-frequency boxes may remain too wide, shifted, or insufficiently tight for high-quality strong labelling.

### gemma4:31b

`gemma4:31b` remains a useful comparison baseline.

It achieved:

```text
TP=113
FP=131
FN=85
Precision=0.463
Recall=0.571
F1=0.511
Mean box IoU=0.339
```

Its mean box IoU is slightly higher than qwen3.6 in the recorded comparison, suggesting that when it successfully matches events, it can sometimes produce tighter boxes. However, it misses more events, produces more false positives, and had two JSON parse failures.

Therefore, `gemma4:31b` is retained as a comparison model, but not selected as the primary fixed-view baseline.

### Archived Weak Baseline: qwen3-vl:latest

The earlier six-clip smoke test with `qwen3-vl:latest` showed much weaker performance:

```text
TP=7
FP=26
FN=26
F1=0.212
Mean box IoU=0.222
```

It is retained only as an early weak local baseline and is not used for further main iteration.

---

## Main Findings

### 1. Grid design has a major effect on VLM annotation performance

The improvement from `grid_v1` to `grid_v2` shows that spectrogram visualisation is a core part of the annotation workflow.

This supports the broader project idea that AI agents may require the right tools and visual interfaces to annotate animal sounds reliably.

### 2. qwen3.6:latest is the best current fixed-view baseline

Across the full 45-clip evaluation set, `qwen3.6:latest + grid_v2` achieved the best overall event-level detection performance.

### 3. gemma4:31b is useful but weaker overall

`gemma4:31b` is still worth keeping because it provides a different error profile and may produce relatively good boxes for successfully matched events. However, its lower recall, lower F1, and parse failures make it less suitable as the main baseline.

### 4. Fixed-view annotation is still not fully reliable

Even the best fixed-view baseline, `qwen3.6 + grid_v2`, still has substantial false positives and false negatives:

```text
FP=113
FN=64
```

This indicates that a single full-view spectrogram is not sufficient for consistently reliable strong labelling across all clips.

### 5. Remaining errors motivate adaptive grid/zoom workflows

The remaining failure modes are likely related to:

```text
dense call sequences
boundary-truncated calls
coordinate precision
over-wide boxes
weak or partially visible calls
visual ambiguity in full-clip overview spectrograms
```

These are the cases where an adaptive-view agent may help by selecting grid styles, requesting zoomed views, and refining time-frequency boxes.

---

## P5C Per-Clip Adaptive Zoom Analysis

The adaptive zoom prototype improved the representative-six subset overall, but the per-clip results show that the benefit was uneven. Adaptive zoom improved difficult clips such as `OP_003`, `OP_010`, and `OP_016`, with the largest gain on `OP_003`, where F1 increased from `0.222` to `0.750`. This suggests that zoomed spectrogram views can help with boundary-near or visually ambiguous calls.

However, adaptive zoom also degraded some clips that were already strong under the fixed-view baseline. `OP_001` decreased from `0.909` to `0.667`, and `OP_004` decreased from `1.000` to `0.833`. `OP_045` remained perfect, but the planner still requested two zoom views, indicating unnecessary tool use on an easy clip.

Overall, adaptive zoom improved aggregate F1 from `0.606` to `0.667`, mainly by reducing false positives and slightly reducing false negatives. Total false positives decreased by 4 and false negatives decreased by 1. Mean box IoU also improved from `0.376` to `0.400`.

These results suggest that adaptive viewing is useful, but the current view-planning strategy is too eager to request zooms. The next iteration should use a gated adaptive-zoom policy, where zoom is requested only for dense, boundary-truncated, weak, or uncertain regions, while clear and well-separated calls should be annotated directly from the overview.

---

## Current Baseline Decision

For future experiments:

```text
Primary fixed-view baseline:
qwen3.6:latest + grid_v2

Comparison fixed-view baseline:
gemma4:31b + grid_v2

Archived weak baseline:
qwen3-vl:latest
```

`grid_v2` should be used as the default clean spectrogram overview style unless a later experiment explicitly tests another visualisation.

---

## Next Step

The next planned stage is an adaptive grid/zoom workflow.

This should be treated as a tool-assisted agent baseline rather than a direct fixed-view baseline.

Proposed workflow:

```text
Round 1: View planning
- model inspects overview spectrogram
- model chooses preferred grid style
- model optionally requests up to 3 zoom windows

Round 2: Final annotation
- script generates requested zoom views
- model receives overview + zoom images
- model returns final annotation JSON
```

Initial adaptive-view testing should begin with the six representative clips:

```text
OP_001
OP_010
OP_045
OP_003
OP_004
OP_016
```

The key question is whether adaptive zoom improves difficult cases where fixed-view models find approximate locations but fail to produce tight boxes, especially dense and boundary-stress clips such as OP_016.
