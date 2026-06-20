# Evaluation Protocol V1: Ozimops petersi Single-Agent Strong-Labelling Benchmark

## 1. Purpose

This document defines the evaluation protocol for the first single-agent
strong-labelling benchmark on the *Ozimops petersi* evaluation set.

The purpose of this protocol is to evaluate whether a general-purpose AI agent
can generate reliable event-level annotations for individual bat echolocation
calls from spectrogram evidence. The task is not whole-clip classification.
Instead, the agent must produce structured sound-event annotations with
time-frequency bounding boxes.

This protocol should be treated as frozen for Evaluation Set V1. After this
point, the ground-truth files, clip selection, and evaluation rules should not
be changed unless a new evaluation-set version is created.

## 2. Evaluation Set

Evaluation set directory:

```text
outputs/evaluation_sets/ozimops_petersi_v1/
```

Main components:

```text
audio/
ground_truth/
figures/
manifest.csv
```

Summary:

| Item | Value |
| --- | ---: |
| Source species | *Ozimops petersi* |
| Source recordings | 10 |
| Generated clips | 45 |
| Unique source events | 191 |
| Clip-level event instances | 198 |
| Clip duration | Mostly 1 second, with partial final clips where applicable |

The evaluation set contains positive clips only. Every clip contains at least
one target event. Therefore, this benchmark primarily evaluates event
detection, localisation, and strong-label quality, rather than false-positive
behaviour on fully empty clips.

## 3. Task Definition

For each input clip, the agent receives a spectrogram image generated from the
corresponding audio file.

The agent must identify every visible bat echolocation call and return one
structured annotation per individual call.

Each predicted event should include:

```text
event_id
start_time_seconds
end_time_seconds
low_frequency_hz
high_frequency_hz
label
confidence
evidence or notes
human_review_needed
review_reason
```

The required strong-label box format is:

```text
[start_time, low_frequency, end_time, high_frequency]
```

where:

- `start_time` and `end_time` are in seconds relative to the current clip.
- `low_frequency` and `high_frequency` are in Hz.

## 4. Annotation Standard

The target annotation unit is one individual bat echolocation call.

The expected annotation behaviour is:

- Annotate every visible echolocation call.
- Draw one tight time-frequency bounding box per individual call.
- Annotate only the main harmonic.
- Do not include echoes, reverberation, background noise, or unrelated
  artefacts.
- Do not merge adjacent calls into a single broad box.
- If a call is weak but still appears to be a genuine echolocation call,
  annotate it.
- For boundary-truncated events, annotate only the visible part inside the
  current clip.

The agent should not hallucinate call regions outside the visible spectrogram or
outside the clip time range.

## 5. Input Format

Each clip has:

```text
audio/<clip_id>.wav
ground_truth/<clip_id>_ground_truth.json
figures/<clip_id>_gt_overlay.png
```

Example:

```text
audio/OP_001.wav
ground_truth/OP_001_ground_truth.json
figures/OP_001_gt_overlay.png
```

The ground-truth JSON contains event-level annotations. Each event includes
clip-relative time-frequency coordinates and source-event metadata.

## 6. Output Format

The agent output should be valid structured JSON.

Recommended format:

```json
{
  "clip_id": "OP_001",
  "events": [
    {
      "event_id": "pred_001",
      "start_time_seconds": 0.273,
      "end_time_seconds": 0.302,
      "low_frequency_hz": 31000,
      "high_frequency_hz": 35100,
      "label": "Ozimops petersi",
      "confidence": 0.82,
      "evidence": "Short bright echolocation call visible around 31-35 kHz.",
      "human_review_needed": false,
      "review_reason": ""
    }
  ]
}
```

The evaluation script should record parsing errors separately. Invalid or
missing fields should not silently pass.

## 7. Geometry Validation

Before matching predictions to ground truth, each predicted box should be
validated.

A valid prediction must satisfy:

```text
0 <= start_time_seconds < end_time_seconds <= clip_duration
0 <= low_frequency_hz < high_frequency_hz
```

Predictions that fall partly outside the clip may be clipped to the valid clip
range for overlap calculation, but they should be flagged with a geometry
warning.

Predictions with non-positive duration or non-positive frequency span after
clipping are invalid. Invalid predictions should be counted as false positives
and recorded as schema or geometry failures.

## 8. Matching Strategy

Evaluation is performed independently for each clip.

Because this project evaluates strong-label annotation rather than only call
presence, two complementary matching views are used:

1. Temporal detection matching
2. Strong-label box-quality evaluation

### 8.1 Temporal Detection Matching

Temporal detection matching asks whether the agent found the individual call in
time.

For each predicted event and ground-truth event, compute temporal IoU:

```text
time_iou = overlap_duration / union_duration
```

A prediction is eligible to match a ground-truth event if:

```text
time_iou >= 0.3
```

Matching is one-to-one:

- one prediction can match at most one ground-truth event;
- one ground-truth event can match at most one prediction.

Predictions are sorted by confidence in descending order when confidence is
available. If confidence is missing or not numeric, predictions are evaluated in
output order.

For each prediction, match it to the unmatched ground-truth event with the
highest temporal IoU, provided the temporal IoU threshold is satisfied.

Matched predictions are temporal true positives. Unmatched predictions are false
positives. Unmatched ground-truth events are false negatives.

### 8.2 Strong-Label Box-Quality Evaluation

For each temporally matched pair, compute:

```text
time_iou
frequency_iou
box_iou
start_time_error
end_time_error
low_frequency_error
high_frequency_error
```

Frequency IoU is computed using the frequency interval:

```text
frequency_iou = frequency_overlap / frequency_union
```

Box IoU is computed over the two-dimensional time-frequency rectangle:

```text
box_iou = intersection_area / union_area
```

This separates the question "did the agent find the call?" from the question
"did the agent draw a good strong-label box?"

A prediction can be temporally correct but still have poor frequency
localisation. This should be explicitly reported rather than hidden.

## 9. Metrics

### 9.1 Event Detection Metrics

Using temporal detection matching:

```text
TP = number of matched predicted events
FP = number of unmatched predicted events
FN = number of unmatched ground-truth events
```

Report:

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

If the denominator is zero, the metric should be reported as undefined rather
than silently set to zero.

### 9.2 Localisation Metrics

For matched pairs, report:

```text
mean_time_iou
median_time_iou
mean_frequency_iou
median_frequency_iou
mean_box_iou
median_box_iou
mean_absolute_start_time_error
mean_absolute_end_time_error
mean_absolute_low_frequency_error
mean_absolute_high_frequency_error
```

The main localisation metric for strong-labelling quality is `box_iou`, but
time and frequency components should also be reported separately.

### 9.3 Strict Strong-Label Metrics

In addition to temporal matching, report stricter strong-label success rates:

```text
box_iou >= 0.3
box_iou >= 0.5
```

For each threshold, compute:

```text
strict_box_precision
strict_box_recall
strict_box_f1
```

These strict metrics should not replace temporal detection metrics. They should
be interpreted as stronger evidence that the agent produced usable
time-frequency annotations.

### 9.4 Label Metrics

Because Evaluation Set V1 is single-species, species classification is not the
primary target.

Still, report:

```text
exact_label_accuracy_on_matched_events
generic_bat_rate
unknown_or_uncertain_label_rate
```

Recommended interpretation:

- `"Ozimops petersi"` = exact species label.
- `"Bat"` or `"Chiroptera"` = acceptable generic bat detection but not exact
  species classification.
- other species labels = label error.

This allows the evaluation to distinguish localisation ability from
species-identification confidence.

## 10. Truncation Handling

Some source events cross one-second clip boundaries. These are represented as
truncated clip-level event instances.

Ground-truth events may include:

```text
is_truncated_by_clip_boundary
truncation_side
```

Possible `truncation_side` values:

```text
none
left
right
both
```

Evaluation Set V1 contains 14 truncated clip-level event instances,
corresponding to 7 source events crossing clip boundaries.

Truncated events are included in the main evaluation. They should not be
excluded by default.

For truncated events:

- The ground-truth box represents only the visible portion inside the current
  clip.
- The agent should annotate only the visible portion.
- The agent should not extend the prediction outside the clip boundary.
- Boundary cases should also be reported as a separate subset.

Report metrics separately for:

```text
non_truncated_events
left_truncated_events
right_truncated_events
all_truncated_events
```

This allows boundary-related failure modes to be analysed explicitly.

## 11. Failure Categories

During manual failure analysis, errors should be assigned to one or more
categories.

Recommended categories:

```text
missed_call
false_positive
merged_calls
split_call
over_wide_frequency_box
under_wide_frequency_box
over_wide_time_box
under_wide_time_box
boundary_truncation_error
label_error
low_confidence_correct
high_confidence_wrong
invalid_json
invalid_geometry
```

Definitions:

| Category | Definition |
| --- | --- |
| `missed_call` | A ground-truth event has no matched prediction. |
| `false_positive` | A prediction has no matched ground-truth event. |
| `merged_calls` | One predicted box covers multiple ground-truth calls. |
| `split_call` | Multiple predicted boxes correspond to one ground-truth call. |
| `over_wide_frequency_box` | The prediction includes much more frequency range than the ground truth. |
| `under_wide_frequency_box` | The prediction misses a substantial part of the ground-truth frequency range. |
| `over_wide_time_box` | The prediction extends too far before or after the ground-truth call. |
| `under_wide_time_box` | The prediction covers only part of the ground-truth call. |
| `boundary_truncation_error` | The prediction misses or incorrectly extends a call at the left or right clip boundary. |
| `label_error` | The predicted label is inconsistent with the ground-truth target label. |
| `invalid_json` | The output cannot be parsed as valid JSON. |
| `invalid_geometry` | A predicted box has impossible or invalid coordinates. |

## 12. Representative Examples

The following clips are selected as representative examples for prompt
development and failure analysis:

| clip_id | role |
| --- | --- |
| `OP_001` | canonical multi-event example |
| `OP_010` | dense multi-event / separation example |
| `OP_045` | simple clean / partial-final-clip example |
| `OP_003` | right-truncated boundary example |
| `OP_004` | left-truncated boundary example |
| `OP_016` | dense boundary-stress example |

No weak/noisy example is assigned in Evaluation Set V1 because manual quality
labels have not yet been added.

## 13. Reporting Outputs

Each evaluation run should produce:

```text
aggregate_summary.json
per_clip_metrics.csv
matched_events.csv
unmatched_predictions.csv
missed_ground_truth_events.csv
failure_notes.md
```

Recommended fields for `matched_events.csv`:

```text
clip_id
prediction_id
ground_truth_event_id
time_iou
frequency_iou
box_iou
start_time_error
end_time_error
low_frequency_error
high_frequency_error
predicted_label
ground_truth_label
confidence
truncation_side
failure_categories
```

Recommended fields for `per_clip_metrics.csv`:

```text
clip_id
num_ground_truth_events
num_predictions
tp
fp
fn
precision
recall
f1
mean_time_iou
mean_frequency_iou
mean_box_iou
num_truncated_events
notes
```

## 14. Reproducibility Requirements

Each evaluation run should record:

```text
evaluation_set_version
git_commit_hash
prompt_version
model_name
model_provider_or_backend
temperature_or_decoding_settings
spectrogram_generation_settings
date_time
```

Spectrogram settings should include:

```text
min_db
max_frequency
STFT settings if applicable
image resolution
whether grid lines were shown
```

For Evaluation Set V1, the default GT overlay visualisation uses:

```text
min_db = -130
frequency axis in kHz
time axis in seconds
maximum displayed frequency approximately 120 kHz
```

## 15. Frozen Protocol Statement

Evaluation Set V1 and this protocol are frozen before running the first formal
prompt-v2 single-agent baseline.

Allowed after freezing:

- run agents;
- evaluate predictions;
- add result files;
- add failure-analysis notes;
- add new prompt versions.

Not allowed without creating a new evaluation-set version:

- modifying ground-truth JSON files;
- changing clip segmentation;
- removing or adding clips;
- changing the primary matching rule;
- changing truncation labels;
- changing representative-example roles.

If any of these changes are necessary, create a new version such as:

```text
ozimops_petersi_v2
evaluation_protocol_v2_ozi_petersi.md
```

## 16. Interpretation

This protocol is designed to answer the following question:

> Can a single AI agent generate reliable strong-label annotations for
> individual *Ozimops petersi* echolocation calls from spectrogram evidence?

The most important distinction is between:

- detecting that a call exists; and
- drawing a high-quality time-frequency box for that call.

Therefore, temporal detection metrics and strong-label box-quality metrics must
both be reported.

A successful agent should achieve high recall without excessive false positives,
avoid merging adjacent calls, localise the main harmonic tightly, handle
boundary-truncated calls, and produce structured outputs that can support human
review or downstream annotation workflows.
