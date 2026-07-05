# OP_010: Dense but Separable Multi-Event Sequence

## Why It Matters

`OP_010` is the best dense success case for testing call counting and one-box-per-call separation without boundary truncation. It contrasts with the much shorter and harder calls in `OP_016`.

## Ground-Truth Pattern

- Seven complete events in one second.
- No truncated events.
- Calls are dense but remain temporally separable.

## Model and Tool Behaviour

| Method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Fixed qwen3.6 + grid_v2 | 5 | 2 | 2 | 0.714 |
| 0.5 s tiled qwen3.6 | 7 | 1 | 0 | 0.933 |
| PCEN qwen3.6 | 6 | 1 | 1 | 0.857 |
| Proposal-only / metadata-assisted / Policy B | 6 | 1 | 1 | 0.857 |

The 0.5 s tiled condition achieves full recall but retains one unmatched prediction, consistent with overlap-driven duplicate risk.

## Main Failure Mode

`missed_dense_calls` under the overview and `duplicate_or_false_positive_candidate` after tiling.

## System Design Lesson

Local views can improve dense-call recall, but merged tile predictions require duplicate suppression and provenance. This case is suitable for checking that one visible call produces one final event.

## Useful Figures

- [GT overlay](../../outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_010_gt_overlay.png)
- [Fixed grid_v2](../../outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_010_diagnostic_overlay.png)
- [0.5 s tiled](../../outputs/agent_runs/p6_tiled_qwen3_6_tile_0p5_overlap_0p1/diagnostic_figures/OP_010_diagnostic_overlay.png)
- [PCEN](../../outputs/agent_runs/p6_pcen_qwen3_6_representative6/diagnostic_figures/OP_010_diagnostic_overlay.png)
- [Proposal-only](../../outputs/agent_runs/p6_batdetect2_proposal_only_representative6/diagnostic_figures/OP_010_diagnostic_overlay.png)

## Dissertation-Ready Interpretation

> OP_010 demonstrates the recall benefit of tiled views on a dense but separable sequence: the 0.5 s condition matched all seven GT events, compared with five under the fixed overview. The remaining unmatched tiled prediction highlights the need for duplicate-aware merging when overlapping local views are used.
