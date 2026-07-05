# OP_001: Canonical Multi-Event and Tool-Regression Case

## Why It Matters

`OP_001` is a canonical overview case where the fixed VLM is already strong. It prevents tool selection from being judged only on difficult clips: tiling, detector assistance, and validation can all regress a good baseline.

## Ground-Truth Pattern

- Five complete, non-truncated events in one second.
- Moderate multi-event activity without a clip-boundary call.

## Model and Tool Behaviour

| Method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Fixed qwen3.6 + grid_v2 | 5 | 1 | 0 | 0.909 |
| PCEN qwen3.6 | 5 | 1 | 0 | 0.909 |
| 0.5 s tiled qwen3.6 | 3 | 2 | 2 | 0.600 |
| Proposal-only | 3 | 4 | 2 | 0.500 |
| Metadata-assisted | 2 | 4 | 3 | 0.364 |
| Policy B | 3 | 3 | 2 | 0.545 |

## Main Failure Mode

`unnecessary_tool_regression`. Additional proposal or tiled context does not guarantee improvement and can alter correct overview geometry.

## System Design Lesson

A tool-use gate should preserve strong overview predictions and invoke extra processing only when evidence is insufficient. Aggregate gains need clean/canonical regression checks.

## Useful Figures

- [GT overlay](../../outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_001_gt_overlay.png)
- [Fixed grid_v2](../../outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_001_diagnostic_overlay.png)
- [0.5 s tiled](../../outputs/agent_runs/p6_tiled_qwen3_6_tile_0p5_overlap_0p1/diagnostic_figures/OP_001_diagnostic_overlay.png)
- [Metadata-assisted](../../outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/diagnostic_figures/OP_001_diagnostic_overlay.png)
- [Proposal-preserving validator](../../outputs/agent_runs/p6_proposal_preserving_validator_representative6/diagnostic_figures/OP_001_diagnostic_overlay.png)

## Dissertation-Ready Interpretation

> OP_001 acts as a regression guard for tool augmentation. The fixed overview achieved F1 0.909, whereas 0.5 s tiling fell to 0.600 and metadata-assisted refinement to 0.364. This result supports conditional tool invocation rather than assuming that additional views or detector metadata are universally beneficial.
