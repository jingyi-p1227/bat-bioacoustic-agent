# Evaluation Set V1: Ozimops petersi

## Purpose

This controlled evaluation set tests whether a single-agent workflow can create
reliable strong labels for individual bat echolocation calls. It provides
short, consistently structured audio clips with call-level time-frequency
ground truth for a spectrogram-only baseline.

## Source Dataset

The set is derived from the BatDetect2 Australia dataset. The source folder
passed to `build_evaluation_set.py` must contain:

```text
annotations.json
audio/*.wav
```

The primary species is *Ozimops petersi*. The selected source recordings are
`pseudo_petersi_001.wav` through `pseudo_petersi_010.wav`, giving 10 source
recordings in total.

## Annotation Standard

Each individual echolocation call is represented by one tight time-frequency
bounding box:

```text
[start_time, low_frequency, end_time, high_frequency]
```

Boxes should:

- contain one individual call;
- avoid echoes, reverberations, and artefacts;
- cover only the main harmonic;
- include genuine low-SNR echolocation calls when visible.

Species uncertainty may be represented with a generic `Bat` label in future
sets. This set uses *Ozimops petersi* labels throughout.

## Generated Evaluation Set

The generated directory has the following structure:

```text
ozimops_petersi_v1/
├── audio/
├── ground_truth/
└── manifest.csv
```

It contains:

| Item | Count |
| --- | ---: |
| Source recordings | 10 |
| One-second clips | 45 |
| Unique source events | 191 |
| Clip-level event instances | 198 |

The event-instance count is larger than the unique-event count because seven
source events cross one-second clip boundaries. Each is included in both
overlapping clip-level ground-truth files, with its original UUID preserved and
its local time range clipped to the relevant clip.

The automatically assigned scenarios are:

| Scenario | Clips |
| --- | ---: |
| `multi_event` | 18 |
| `positive` | 27 |

## Boundary Truncation

Every clip-level event records whether clipping changed its visible time range:

- `is_truncated_by_clip_boundary`: `true` when either boundary truncates the
  source event.
- `truncation_side`: `none`, `left`, `right`, or `both`.

`left` means the source event starts before the clip. `right` means it ends
after the clip. The current distribution is:

| Truncation side | Event instances |
| --- | ---: |
| `none` | 184 |
| `left` | 7 |
| `right` | 7 |
| `both` | 0 |

All written time values are rounded to six decimal places.

## Manifest Fields

| Field | Meaning |
| --- | --- |
| `clip_id` | Stable sequential identifier such as `OP_001`. |
| `clip_path` | Relative path to the generated WAV. |
| `ground_truth_path` | Relative path to the clip-level ground-truth JSON. |
| `source_recording` | Original BatDetect2 Australia recording filename. |
| `source_start_time` | Clip start on the original recording timeline. |
| `source_end_time` | Clip end on the original recording timeline. |
| `clip_duration` | Actual clip duration, including a final partial clip. |
| `species` | Target species for the evaluation set. |
| `has_target_event` | Whether the clip contains at least one target event. |
| `num_gt_events` | Number of clip-level ground-truth event instances. |
| `event_density` | `zero`, `low`, `medium`, or `high`. |
| `auto_scenario` | Automatically derived activity scenario. |
| `manual_scenario` | Reserved for later spectrogram review. |
| `notes` | Free-text notes for later curation. |

## Reproducible Build

```bash
uv run python build_evaluation_set.py \
  --dataset-dir "/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/batdetect2_outputs/datasets/australia" \
  --output-dir outputs/evaluation_sets/ozimops_petersi_v1 \
  --species "Ozimops petersi" \
  --source-prefix "pseudo_petersi_" \
  --clip-seconds 1.0 \
  --overwrite
```

The source dataset remains external to the repository. Generated WAV files and
evaluation outputs are excluded from Git.

## Verification Summary

```text
Clips: 45
Unique source events: 191
Clip event instances: 198
All paths relative: true
Verification errors: none
```

## Limitations

- This is a positive, single-species evaluation set.
- It does not yet include negative clips.
- It does not test multi-species generalisation.
- Manual scenarios such as `clear_call`, `weak_call`, `noisy`, or `borderline`
  have not yet been assigned. These can be added after spectrogram spot-checking.

## Next Steps

1. Spot-check generated spectrograms with ground-truth boxes.
2. Select representative annotation examples for prompt v2.
3. Use this set for the spectrogram-only single-agent baseline evaluation.
