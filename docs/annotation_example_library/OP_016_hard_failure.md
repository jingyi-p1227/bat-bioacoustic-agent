# Annotation Example: OP_016 Hard Failure

## clip_id

`OP_016`

## case_type

`dense_boundary_stress`, `hard_failure`

## why_this_example_matters

This clip combines seven events with both left- and right-truncated calls. It is the strongest compact stress case for testing dense-event recall, boundary handling, and local box quality.

## relevant_figures

- GT reference: `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_016_gt_overlay.png`
- Fixed qwen3.6 grid_v2: `outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_016_diagnostic_overlay.png`
- Representative adaptive zoom: `outputs/agent_runs/prompt_v2_adaptive_zoom_qwen3_6_representative6/evaluation/diagnostic_figures/OP_016_diagnostic_overlay.png`
- P5E gated adaptive: `outputs/agent_runs/prompt_v2_gated_adaptive_zoom_qwen3_6_full/evaluation/diagnostic_figures/OP_016_diagnostic_overlay.png`
- P5F gated overview-only: `outputs/agent_runs/prompt_v2_gated_overview_only_qwen3_6_full/evaluation/diagnostic_figures/OP_016_diagnostic_overlay.png`

## model_behaviour

This was the only clip that requested zoom in the full P5E/P5F planning runs. Under P5E it remained unresolved with `TP=0`, `FP=2`, and `FN=7`. Zoom reduced false positives relative to the overview-only control, but did not recover true events.

## prompt_lesson

Additional instructions alone may not solve cases where the visual representation does not expose separable evidence. The model should still avoid producing unsupported boxes and should mark uncertainty explicitly.

## tool_lesson

This is the primary P6 stress test for PCEN-like enhancement, denoising or band-pass processing, and 0.5 s or 0.25 s tiles. Any apparent improvement must be checked for duplicate tile predictions and coordinate-remapping errors.

## notes_for_report

Use to show the limit of the current workflow. It prevents the report from overstating the value of zoom and provides a concrete motivation for preprocessing ablations.

