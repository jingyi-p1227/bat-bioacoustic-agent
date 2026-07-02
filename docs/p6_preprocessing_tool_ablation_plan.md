# P6 Preprocessing-Tool Ablation Plan

## Research Question

Can controlled spectrogram preprocessing or smaller visual views improve event detection and time-frequency localisation beyond the current `qwen3.6:latest` + `grid_v2` baseline, especially for dense and boundary-stress clips, without increasing false positives or introducing coordinate errors?

## Motivation From P5

P5 established two important findings:

1. Visual presentation matters. `grid_v2` substantially outperformed `grid_v1` under the same prompt and model.
2. The gated overview-first workflow improved full-set performance, but the P5F control showed that most of the gain came from the workflow rather than zoom evidence.

The next experiment should therefore test whether alternative signal representations provide genuinely useful visual evidence. The purpose is not to add tools for their own sake. Each tool must demonstrate a measurable benefit under the frozen evaluation protocol and must preserve a traceable mapping back to source time and frequency coordinates.

## Fixed Experimental Controls

Unless a pilot reveals a blocking incompatibility, keep the following fixed:

- model: `qwen3.6:latest`;
- prompt: `prompts/prompt_v2_bat_strong_label.md`;
- evaluation set and ground truth: unchanged;
- matching protocol and thresholds: unchanged;
- baseline display: current dB spectrogram with `grid_v2`;
- model inputs: WAV-derived clean views only, with no GT or diagnostic overlays;
- run metadata: representation name, parameters, input images, and coordinate mapping recorded.

## Candidate Tools

### 1. Current dB Spectrogram + Grid V2

Use the existing clean `grid_v2` spectrogram as the control condition. All ablation results must be compared against this representation, not against an older grid or a different model run.

### 2. PCEN-Enhanced Spectrogram

Test a per-channel energy normalization style representation intended to improve local contrast under changing background energy. Record all PCEN parameters. Preserve the same time and frequency axes so predicted boxes remain directly evaluable.

Primary hypothesis: weak calls become easier to distinguish from slowly varying background energy.

Primary risk: enhancement may distort relative visual intensity or amplify artefacts, increasing false positives.

### 3. Denoised or Band-Pass Spectrogram

Test a conservative bat-relevant band-pass or documented denoising transformation. Keep the unprocessed overview available for qualitative comparison, and do not remove frequency regions without recording the exact filter limits.

Primary hypothesis: suppressing irrelevant low-frequency or broadband noise reduces false positives and improves frequency-box tightness.

Primary risk: genuine low-SNR call energy may be removed, increasing false negatives or narrowing boxes incorrectly.

### 4. 0.5-Second Tiled Spectrogram

Split each one-second clip into non-overlapping or explicitly defined 0.5 s views. Each tile must retain source clip offsets, and predictions must be mapped back to clip-relative time before evaluation.

Primary hypothesis: larger on-image call geometry improves counting and localisation in dense regions.

Primary risk: calls crossing tile boundaries may be duplicated, truncated, or missed.

### 5. 0.25-Second Tiled Spectrogram

Test a finer crop for difficult cases only after the 0.5 s pilot. Coordinate mapping and boundary reconciliation must use the same deterministic rules as the 0.5 s condition.

Primary hypothesis: individual calls become visually larger and easier to bound precisely.

Primary risk: loss of temporal context, increased number of model views, duplicate detections, and higher runtime.

### 6. Optional MFCC-Derived Auxiliary Representation

MFCC-derived features may be explored as an auxiliary context or classification-oriented view, but not as the primary strong-labelling input. MFCCs compress spectral detail and do not directly preserve the rectangular time-frequency geometry required by this benchmark.

## Recommended Pilot Clips

| Clip | Role in the pilot |
| --- | --- |
| `OP_001` | Canonical multi-event reference |
| `OP_003` | Right-boundary case that improved under adaptive viewing |
| `OP_004` | Paired left-boundary case |
| `OP_010` | Dense multi-event and call-separation case |
| `OP_016` | Dense boundary-stress hard failure |
| `OP_045` | Clean success and partial-final-clip control |

These six clips provide a compact but intentionally non-random diagnostic pilot. Results from them must not be reported as full benchmark performance.

## Primary Metrics

- precision;
- recall;
- F1;
- `mean_time_iou`;
- `mean_frequency_iou`;
- `mean_box_iou`;
- count of matched events with `box_iou >= 0.3`;
- count of matched events with `box_iou >= 0.5`.

Also record TP, FP, FN, parse failures, number of model views, and preprocessing/runtime cost. The latter values are needed to judge whether a small quality gain justifies a more complex tool workflow.

## Key Questions

1. Does the tool reduce false positives?
2. Does it reduce false negatives?
3. Does it improve temporal, frequency, and two-dimensional box quality?
4. Does it improve `OP_016`-like dense boundary-stress cases?
5. Does it preserve strong performance on clean cases such as `OP_045`?
6. Do tiled views create duplicate or boundary-truncated predictions?
7. Is any improvement due to the representation itself rather than a changed prompt or evaluation procedure?

## Pilot Procedure

1. Generate each representation from the same six WAV clips with deterministic parameters.
2. Inspect generated images for valid axes, duration, frequency range, and nonblank content.
3. Run the same model and prompt once per controlled representation condition.
4. Save raw responses, parsed predictions, parameters, and errors in separate run directories.
5. Evaluate with the frozen protocol.
6. Produce per-clip deltas against the existing `grid_v2` fixed-view baseline.
7. Review diagnostic overlays for `OP_003`, `OP_004`, `OP_010`, `OP_016`, and `OP_045`.
8. Separate representation improvements from coordinate-remapping or duplicate-merging effects.

## Interpretation Rules

- Do not select a tool from F1 alone. Inspect FP, FN, and box localisation together.
- A reduction in predictions is not automatically an improvement; confirm that true calls were retained.
- Better temporal matching with worse frequency IoU indicates incomplete strong-labelling improvement.
- Improvements on `OP_016` must correspond to recovered GT events, not only fewer unsupported predictions.
- Regressions on clean or boundary-paired clips must be reported explicitly.
- Tiled-view predictions must be deduplicated using a documented rule before comparison; raw and reconciled counts should both be retained.
- Do not compare conditions that use different prompts, models, GT, or matching thresholds as if preprocessing were the only variable.
- Treat the six-clip pilot as hypothesis generation, not a final benchmark.

## Decision Criteria for a Full 45-Clip Run

Expand a representation to all 45 clips only when it meets all of the following:

1. All six pilot clips complete without unresolved parsing or coordinate-mapping failures.
2. Aggregate pilot F1 improves over, or remains close to, `grid_v2` while box localisation improves materially.
3. The condition does not gain recall through an unacceptable increase in false positives.
4. It shows a plausible benefit on at least one difficult clip, especially `OP_016`, `OP_003`, or `OP_010`.
5. It does not materially degrade the clean control `OP_045` or both sides of the `OP_003`/`OP_004` boundary pair.
6. Runtime, image count, and implementation complexity remain compatible with a reproducible single-agent workflow.
7. Visual inspection confirms that gains are associated with clearer call evidence rather than plotting artefacts or evaluation leakage.

If no candidate meets these criteria, retain `grid_v2` and the gated overview-first workflow as the primary baseline, and document preprocessing as a negative or inconclusive experiment.

## Expected Outputs

For each candidate condition, produce:

```text
outputs/agent_inputs/p6_<representation>_representative6/
outputs/agent_runs/p6_<representation>_qwen3_6_representative6/
outputs/agent_runs/p6_<representation>_qwen3_6_representative6/evaluation/
```

The final P6 comparison should include an aggregate table, per-clip deltas, preprocessing parameters, diagnostic overlays, and a short decision note explaining whether the condition should advance to the full set.
