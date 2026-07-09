# P5 Next Experiment Plan: Full Prompt-V2 Baseline

## Purpose

P5 extends the prompt-v2 experiment from the six representative smoke-test clips to the full `Ozimops petersi` Evaluation Set V1.

The goal is to establish stronger model baselines for single-agent bat echolocation strong labelling, using the frozen evaluation set and frozen evaluation protocol.

## Current State

Completed stages:

* Evaluation Set V1 built and documented.
* Ground-truth overlay visualisation implemented.
* Representative examples selected.
* Evaluation protocol frozen.
* Prompt v2 written.
* Six-clip prompt-v2 smoke tests completed.
* Model comparison completed.

Selected baseline models:

* Primary baseline: `qwen3.6:latest`
* Comparison baseline: `gemma4:31b`
* Archived weak baseline: `qwen3-vl:latest`

## Main P5 Question

Can stronger VLM backends generate reliable strong-label annotations for individual `Ozimops petersi` echolocation calls across the full 45-clip evaluation set?

## Experiment Design

### Inputs

Evaluation set:

```text
outputs/evaluation_sets/ozimops_petersi_v1/
```

Prompt:

```text
prompts/prompt_v2_bat_strong_label.md
```

Models:

```text
qwen3.6:latest
gemma4:31b
```

Input images should be clean spectrograms only. Ground-truth overlay figures must not be used as model input.

### Output directories

Recommended output directories:

```text
outputs/agent_runs/prompt_v2_full_qwen3_6_latest/
outputs/agent_runs/prompt_v2_full_gemma4_31b/
```

Each run should contain:

```text
<clip_id>_raw_response.txt
<clip_id>_predictions.json
<clip_id>_parse_error.txt, if applicable
evaluation/
```

The evaluation folder should contain:

```text
aggregate_summary.json
per_clip_metrics.csv
matched_events.csv
unmatched_predictions.csv
missed_ground_truth_events.csv
failure_notes_template.md
diagnostic_figures/
failure_analysis.md
```

## Step-by-Step Plan

### Step 1: Generate clean spectrogram inputs for all 45 clips

Use the existing clean spectrogram input-preparation script.

Expected output:

```text
outputs/agent_inputs/prompt_v2_full/
```

Each clip should have one clean spectrogram image without GT boxes.

### Step 2: Run full prompt-v2 baseline with qwen3.6

Run all 45 clips with:

```text
model_name = qwen3.6:latest
```

Requirements:

* Use the same prompt v2.
* Use clean spectrogram inputs only.
* Save raw responses and parsed JSON predictions.
* Continue running if individual clips fail.
* Record parse failures explicitly.
* Store model metadata in each prediction file.

### Step 3: Evaluate qwen3.6 full run

Use the frozen evaluation protocol:

* temporal matching threshold: time IoU ≥ 0.3
* confidence-ordered greedy one-to-one matching
* report temporal detection metrics
* report time, frequency, and box IoU
* report strict box IoU thresholds at 0.3 and 0.5
* report truncation subset performance

### Step 4: Run full prompt-v2 baseline with gemma4:31b

Repeat the same full 45-clip run with:

```text
model_name = gemma4:31b
```

Do not change prompt, input images, or evaluation protocol.

### Step 5: Evaluate gemma4 full run

Use the same evaluation script and output structure.

### Step 6: Generate model comparison summary

Create a comparison summary across:

* qwen3-vl early weak baseline, six-clip reference only
* gemma4:31b full run
* qwen3.6:latest full run

For the full-run comparison, report at least:

| model | clips | GT events | predictions | TP | FP | FN | precision | recall | F1 | mean time IoU | mean frequency IoU | mean box IoU | box IoU ≥ 0.3 | box IoU ≥ 0.5 |
| ----- | ----: | --------: | ----------: | -: | -: | -: | --------: | -----: | -: | ------------: | -----------------: | -----------: | ------------: | ------------: |

### Step 7: Generate diagnostic overlays

For each full model run, generate diagnostic overlays.

At minimum, generate overlays for the six representative clips:

* OP_001
* OP_010
* OP_045
* OP_003
* OP_004
* OP_016

If time allows, generate overlays for all clips or for the worst-performing clips by F1 / box IoU.

### Step 8: Failure analysis

For each model, analyse:

* missed calls
* false positives
* poor frequency localisation
* over-wide boxes
* under-wide boxes
* merged calls
* split calls
* boundary-truncation failures
* dense-case failures
* invalid JSON or geometry failures

Special attention should be paid to:

* boundary-truncated events
* dense clips
* clips where prediction count is close to GT count but IoU remains poor
* clips where one model succeeds and the other fails

## Success Criteria

P5 is complete when:

* qwen3.6 full 45-clip predictions are generated.
* gemma4 full 45-clip predictions are generated.
* Both runs are evaluated with the frozen protocol.
* Model comparison summary is generated.
* Diagnostic overlays are generated for representative clips.
* A short failure-analysis report is written.
* The results support a decision about whether to proceed to prompt v3, zoom-guided workflow, or full-scale evaluation/reporting.

## Expected Outcomes

Based on the six-clip smoke test, expected patterns are:

* `qwen3.6:latest` may achieve higher recall and F1.
* `gemma4:31b` may produce tighter boxes for matched events but miss more calls.
* Both models may still struggle with dense boundary-stress cases.
* Boundary-truncated calls and precise frequency localisation are likely to remain important failure modes.

## Decision After P5

After P5, choose one of the following paths:

### Path A: Prompt v3

Use this if errors are mostly instruction-sensitive, such as:

* over-wide boxes
* ignoring boundary calls despite visible evidence
* merging adjacent calls
* inconsistent confidence use

### Path B: Zoom-guided or multi-step workflow

Use this if errors are mostly visual-resolution or localisation-sensitive, such as:

* the model sees the call but cannot draw tight coordinates
* dense clips remain hard
* boundary cases require local zoom
* predictions are close but fail IoU thresholds

### Path C: Full report-ready baseline

Use this if one model performs consistently well enough across the 45 clips to support a stable baseline analysis.
