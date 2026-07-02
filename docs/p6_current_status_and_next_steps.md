# P6 Current Status and Next Steps

## Motivation

P6 extends the completed single-agent strong-labelling baselines by testing whether controlled visual tools can improve event detection and time-frequency localisation. The current framing is **tool-augmented single-agent annotation**, not a multi-agent system. The model remains responsible for the final structured annotation, while preprocessing tools provide alternative views of the same WAV audio.

## Current Status

### P6A: Tiled Spectrogram Workflow

Completed:

- Generated clean overlapping spectrogram tiles for the six representative clips.
- Prepared `0.5 s / 0.1 s overlap` and `0.25 s / 0.05 s overlap` conditions.
- Preserved original clip coordinates on every tile axis.
- Generated clean and GT-diagnostic contact sheets.
- Confirmed that every GT event is visible in at least one tile and has at least one complete tile view.
- Implemented `merge_tiled_predictions.py` with geometry validation, provenance, clip-boundary handling, and confidence-ordered box-IoU NMS.

### P6B: PCEN Spectrogram Workflow

Completed:

- Generated clean linear-frequency PCEN spectrograms for the same six clips.
- Preserved the original time-frequency geometry and `grid_v2` axes.
- Generated clean dB-versus-PCEN contact sheets.
- Generated separately stored GT diagnostic comparisons for human inspection only.
- Recorded all PCEN parameters in a reproducible manifest.
- Produced an initial visual sanity-check report.

### Model Status

No P6 model inference has been run. `qwen3.6:latest` is not currently available locally, and HPC access is temporarily unavailable.

`qwen3-vl` should not be used as a substitute. Its earlier performance was substantially weaker, so results would not be directly comparable with the established qwen3.6 + `grid_v2` baseline.

## Visual Sanity-Check Findings

### Tiled Views

- The `0.5 s / 0.1 s overlap` condition is simpler, retains more context, requires fewer model views, and is recommended for the first tiled pilot.
- The `0.25 s / 0.05 s overlap` condition makes individual calls in `OP_016` visually larger and clearer.
- The 0.25 s condition also creates more overlapping views, increasing inference cost and duplicate-prediction risk.
- Both settings cover all GT events in the representative-six subset.

### PCEN

- PCEN makes target-band calls more visually distinct in clips such as `OP_001`, `OP_003`, `OP_010`, and `OP_045`.
- It preserves the time-frequency coordinate system needed for bounding-box annotation.
- It is mixed for `OP_016`: some calls become more visible, but noise, low-frequency structures, and strong-call saturation are also enhanced.
- PCEN should therefore be tested as an alternative representation, not treated as a proven denoising improvement.

## Recommended Next Model Pilot

Once `qwen3.6:latest` is available, run a controlled comparison on the same six representative clips:

```text
OP_001
OP_003
OP_004
OP_010
OP_016
OP_045
```

Conditions:

1. Original full-view dB spectrogram with `grid_v2`.
2. `0.5 s / 0.1 s overlap` tiled spectrograms, merged back to clip-level predictions.
3. Full-view PCEN-enhanced spectrogram with `grid_v2`.

Keep the model, prompt, GT, evaluation protocol, and matching thresholds unchanged. Retain raw tile predictions and merge provenance so improvements can be separated from duplicate-suppression effects.

The `0.25 s` tiled condition should remain a targeted follow-up, particularly for `OP_016`, rather than being included automatically in the first comparison.

## Decision Criteria

The next condition should not be selected from aggregate F1 alone. Compare:

- false-positive reduction;
- false-negative reduction;
- whether true events are recovered on `OP_016`;
- mean temporal, frequency, and box IoU;
- strict box-IoU counts at `0.3` and `0.5`;
- duplicate predictions introduced by overlapping tiles;
- regressions on clean or already strong clips such as `OP_045`;
- number of model views and preprocessing/merge complexity.

A useful tool should improve difficult cases or box quality without obtaining the gain through an unacceptable increase in false positives, duplicate events, or workflow cost.

## Questions for Santiago

1. Should the first qwen3.6 preprocessing pilot test tiled spectrograms or PCEN first?
2. Should the `0.25 s` tiled condition be tested only on `OP_016`, or on all six representative clips?
3. Should BatDetect2 be treated as a proposal tool that supplies candidate regions for final single-agent review?
4. Should a second species be added now to test generalisation, or only after the preprocessing condition is selected on `Ozimops petersi`?
5. Can Santiago help check local qwen3.6 availability or restore access to the HPC inference environment?

## Immediate Next Step

Keep the P6 inputs and merge utilities frozen until qwen3.6 access is restored. Then run the three-condition representative-six comparison before considering any full 45-clip preprocessing experiment.
