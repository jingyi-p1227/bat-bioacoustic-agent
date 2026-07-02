# Annotation Example: OP_045 Clean Success

## clip_id

`OP_045`

## case_type

`clean_success`, `partial_final_clip`

## why_this_example_matters

This is a simple, clean sanity-check case from a partial final clip. It helps verify that the workflow handles a clip shorter than one second without inventing content outside the visible time range.

## relevant_figures

- GT reference: `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_045_gt_overlay.png`
- Fixed qwen3.6 grid_v2: `outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_045_diagnostic_overlay.png`
- P5E gated adaptive: `outputs/agent_runs/prompt_v2_gated_adaptive_zoom_qwen3_6_full/evaluation/diagnostic_figures/OP_045_diagnostic_overlay.png`
- P5F gated overview-only: `outputs/agent_runs/prompt_v2_gated_overview_only_qwen3_6_full/evaluation/diagnostic_figures/OP_045_diagnostic_overlay.png`

## model_behaviour

The representative-six adaptive experiment retained perfect event-level performance on this clip, but the ungated planner requested two unnecessary zoom views. It is therefore both a clean success case and an example of avoidable tool use.

## prompt_lesson

Clear, well-separated calls should be annotated directly from the overview. The model must respect the actual clip duration, especially for partial final clips.

## tool_lesson

A preprocessing or zoom tool should not be invoked merely because it is available. A view-sufficiency gate should keep this case overview-only unless the representation is visibly degraded.

## notes_for_report

Use as a clean qualitative reference and as a contrast to `OP_016`. It is useful for explaining why tool efficiency matters in addition to annotation accuracy.

