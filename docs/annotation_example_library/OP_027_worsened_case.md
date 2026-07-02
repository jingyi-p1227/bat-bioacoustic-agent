# Annotation Example: OP_027 Worsened Case

## clip_id

`OP_027`

## case_type

`workflow_regression`, `right_boundary_truncation`

## why_this_example_matters

Full-set analysis identified this clip as the largest worsening relative to the fixed-view baseline. It is important for understanding when a gated or multi-stage procedure changes a previously useful prediction into a worse result.

## relevant_figures

- GT reference: `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_027_gt_overlay.png`
- Fixed qwen3.6 grid_v2: `outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_027_diagnostic_overlay.png`
- P5E gated adaptive: `outputs/agent_runs/prompt_v2_gated_adaptive_zoom_qwen3_6_full/evaluation/diagnostic_figures/OP_027_diagnostic_overlay.png`
- P5F gated overview-only: `outputs/agent_runs/prompt_v2_gated_overview_only_qwen3_6_full/evaluation/diagnostic_figures/OP_027_diagnostic_overlay.png`

## model_behaviour

The gated full-set workflow worsened this clip relative to fixed qwen3.6 + grid_v2. The exact event-level mechanism should be confirmed by comparing matched, missed, and unmatched boxes in the diagnostic overlays before assigning a narrower failure label.

## prompt_lesson

An overview-sufficiency check should not cause the final annotator to discard clearly visible calls or replace tight boxes with more conservative but inaccurate geometry.

## tool_lesson

Preprocessing should be evaluated for regressions as well as aggregate gains. This clip is a useful guard case: a tool that helps `OP_016` but substantially harms `OP_027` may not be suitable for full-set use.

## notes_for_report

Use as the counterexample to aggregate P5 improvement. Complete the card with exact per-event observations after manual side-by-side review.

