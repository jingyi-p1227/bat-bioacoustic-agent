# Annotation Example: OP_003 Boundary Improved

## clip_id

`OP_003`

## case_type

`right_boundary_truncation`, `workflow_improvement`

## why_this_example_matters

This clip contains a call truncated by the right clip boundary. It tests whether the model annotates only visible evidence and avoids extending a box beyond the clip.

## relevant_figures

- GT reference: `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_003_gt_overlay.png`
- Fixed qwen3.6 grid_v2: `outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_003_diagnostic_overlay.png`
- Representative adaptive zoom: `outputs/agent_runs/prompt_v2_adaptive_zoom_qwen3_6_representative6/evaluation/diagnostic_figures/OP_003_diagnostic_overlay.png`
- P5E gated adaptive: `outputs/agent_runs/prompt_v2_gated_adaptive_zoom_qwen3_6_full/evaluation/diagnostic_figures/OP_003_diagnostic_overlay.png`
- P5F gated overview-only: `outputs/agent_runs/prompt_v2_gated_overview_only_qwen3_6_full/evaluation/diagnostic_figures/OP_003_diagnostic_overlay.png`

## model_behaviour

In the representative-six adaptive experiment, F1 increased from `0.222` under fixed view to `0.750`. Full-run analysis also identified this as a useful boundary case for comparing fixed and gated workflows.

## prompt_lesson

Boundary calls need an explicit rule: annotate the visible portion inside the clip and do not infer the hidden continuation.

## tool_lesson

Closer views may help with boundary-near evidence, but P5F shows that the overview-first decision procedure itself can also improve cautious annotation. This case should be used to separate representation benefit from workflow benefit.

## notes_for_report

Pair with `OP_004`, which contains the corresponding left-boundary continuation, to explain boundary-aware clipping and evaluation.

