# Adaptive Grid and Zoom Workflow Design

## Purpose

This note defines a future two-round agent workflow for `Ozimops petersi` strong labelling. It extends the prompt-v2 baseline design without changing the frozen evaluation protocol.

The goal is to let the model first inspect a clean overview spectrogram, choose the most readable grid style, and request a small number of zoom views before producing final event-level annotations.

## Constraints

* Prediction must not read ground-truth JSON files.
* Model inputs must be clean spectrograms generated from WAV audio only.
* Ground-truth overlays and diagnostic figures must never be used as model inputs.
* Prompt v2 annotation schema remains the final output schema.
* The frozen evaluation protocol is unchanged.
* A clip may request at most 3 zoom windows.
* Zoom requests must stay within the clip duration and displayed frequency range.
* Only the final annotation JSON is evaluated.

## Grid Styles

Two overview grid styles are used for fixed prompt-v2 baselines:

* `grid_v1`: fixed project-default grid with 0.5 s major / 0.1 s minor time steps and 10 kHz major / 5 kHz minor frequency steps.
* `grid_v2`: readable auto grid where major/minor steps are chosen from the visible time and frequency span.

These are clean input images only: clip id, time axis in seconds, frequency axis in kHz, spectrogram, and grid.

## Round 1: View Planning

The model receives one or more clean overview spectrograms for the same clip, for example `grid_v1` and `grid_v2`.

The task is not final annotation. The model should decide which overview is easier to use and whether zoom windows are needed.

Expected planning output:

```json
{
  "clip_id": "OP_001",
  "preferred_grid": "grid_v2",
  "zoom_requests": [
    {
      "start_time_seconds": 0.20,
      "end_time_seconds": 0.45,
      "low_frequency_hz": 20000,
      "high_frequency_hz": 60000,
      "reason": "Dense calls need clearer separation."
    }
  ],
  "reason": "grid_v2 has readable time spacing for this clip."
}
```

Validation rules:

* `preferred_grid` must be `grid_v1` or `grid_v2`.
* `zoom_requests` length must be 0 to 3.
* `start_time_seconds >= 0`.
* `end_time_seconds <= clip_duration_seconds`.
* `start_time_seconds < end_time_seconds`.
* `low_frequency_hz >= 0`.
* `high_frequency_hz <= displayed_max_frequency_hz`.
* `low_frequency_hz < high_frequency_hz`.

Invalid zoom requests should be clipped or rejected by the orchestration script before image generation.

## Round 2: Final Annotation

The script generates the requested clean zoom images from WAV audio only.

The model receives:

* the preferred clean overview image;
* up to 3 clean zoom images;
* clip id;
* clip duration;
* frequency-axis unit reminder;
* the same prompt-v2 final annotation instructions.

The model returns final annotation JSON using the prompt-v2 schema:

```json
{
  "clip_id": "OP_001",
  "events": [
    {
      "event_id": "pred_001",
      "start_time_seconds": 0.12,
      "end_time_seconds": 0.15,
      "low_frequency_hz": 30000,
      "high_frequency_hz": 42000,
      "label": "Ozimops petersi",
      "confidence": 0.85,
      "evidence": "Short visible call in the main harmonic band.",
      "human_review_needed": false,
      "review_reason": ""
    }
  ]
}
```

## Evaluation

Only the final Round 2 annotation JSON is evaluated.

The existing frozen protocol is reused:

* temporal IoU threshold: 0.3;
* confidence-ordered greedy one-to-one matching;
* temporal detection metrics;
* time, frequency, and box IoU;
* strict box IoU counts at 0.3 and 0.5;
* missed boundary-truncated events reported separately through existing failure categories.

## Expected Benefits

This workflow should help when the model can see calls but struggles to localise tight boxes from a single overview image. It is most relevant for:

* dense multi-event clips;
* left- or right-boundary truncated calls;
* clips where timing is close but frequency boxes are poor;
* cases where one grid style makes time or frequency alignment easier.

## Minimal Implementation Path

1. Run full fixed-grid baselines for `grid_v1` and `grid_v2`.
2. Compare aggregate and per-clip performance.
3. Identify clips where grid choice changes outcome.
4. Implement a planning-only script that outputs validated zoom requests.
5. Generate clean zoom images from WAV audio.
6. Run Round 2 final annotation on overview plus zoom images.
7. Evaluate only the final annotations with the frozen protocol.
