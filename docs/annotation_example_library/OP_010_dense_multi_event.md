# Annotation Example: OP_010 Dense Multi-Event

## clip_id

`OP_010`

## case_type

`dense_multi_event`, `call_separation`

## why_this_example_matters

This clip contains seven non-truncated ground-truth events. It tests recall, one-call-per-box separation, and whether adjacent calls are merged into broad annotations.

## relevant_figures

- GT reference: `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_010_gt_overlay.png`
- Fixed qwen3.6 grid_v1: `outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v1/evaluation/diagnostic_figures/OP_010_diagnostic_overlay.png`
- Fixed qwen3.6 grid_v2: `outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_010_diagnostic_overlay.png`
- Full gemma4 grid_v2: `outputs/agent_runs/prompt_v2_full_gemma4_31b_grid_v2/evaluation/diagnostic_figures/OP_010_diagnostic_overlay.png`
- P5E gated adaptive: `outputs/agent_runs/prompt_v2_gated_adaptive_zoom_qwen3_6_full/evaluation/diagnostic_figures/OP_010_diagnostic_overlay.png`

## model_behaviour

The clip improved in the representative adaptive experiment and is a useful success case for comparing event separation across grid styles, models, and gated workflows.

## prompt_lesson

Dense activity requires deliberate counting and one tight annotation per visible call. Adjacent calls must not be merged, and weak calls should not be skipped solely because stronger neighbours are present.

## tool_lesson

Use this clip to test whether denser grids, enhanced contrast, or short tiles improve separation without increasing duplicate predictions across tile boundaries.

## notes_for_report

This is the preferred dense-call example for a presentation because it is challenging but still interpretable. Contrast it with the unresolved stress case `OP_016`.

