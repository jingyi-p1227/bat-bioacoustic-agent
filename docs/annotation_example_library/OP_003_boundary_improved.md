# OP_003: Right-Boundary Recall Improvement

## Why It Matters

`OP_003` is a useful positive result for tiled preprocessing, while remaining a guard case for boundary handling. It demonstrates that improved overall recall does not imply recovery of a truncated event.

## Ground-Truth Pattern

- Five events in one second.
- Four complete events and one event truncated at the right clip boundary.

## Model and Tool Behaviour

| Method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Fixed qwen3.6 + grid_v2 | 1 | 3 | 4 | 0.222 |
| 0.5 s tiled qwen3.6 | 4 | 1 | 1 | 0.800 |
| PCEN qwen3.6 | 3 | 1 | 2 | 0.667 |
| BatDetect2 proposal-only | 3 | 1 | 2 | 0.667 |

The tiled condition recovers three additional complete calls, but the remaining missed GT event is the right-boundary truncation.

## Main Failure Mode

`right_boundary_miss`, alongside low fixed-view recall.

## System Design Lesson

Tiling can expose otherwise missed calls, but boundary candidates require explicit handling. A boundary-aware proposal or review step is still needed even when aggregate recall improves.

## Useful Figures

- [GT overlay](../../outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_003_gt_overlay.png)
- [Fixed grid_v2](../../outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_003_diagnostic_overlay.png)
- [0.5 s tiled](../../outputs/agent_runs/p6_tiled_qwen3_6_tile_0p5_overlap_0p1/diagnostic_figures/OP_003_diagnostic_overlay.png)
- [PCEN](../../outputs/agent_runs/p6_pcen_qwen3_6_representative6/diagnostic_figures/OP_003_diagnostic_overlay.png)
- [Proposal-only](../../outputs/agent_runs/p6_batdetect2_proposal_only_representative6/diagnostic_figures/OP_003_diagnostic_overlay.png)

## Dissertation-Ready Interpretation

> On OP_003, 0.5 s tiling increased event-level F1 from 0.222 to 0.800 by recovering complete calls missed in the overview. The remaining false negative was the right-boundary event, indicating that local views improve visibility but do not replace explicit boundary-aware annotation logic.
