# OP_016: Dense Short-Call and Proposal-Timing Stress Case

## Why It Matters

`OP_016` is the clearest compact example of why tool-augmented annotation needs provenance and validation. A fixed VLM misses the sequence, BatDetect2 supplies accurate candidate timing, unconstrained refinement damages that timing, and deterministic preservation recovers it.

## Ground-Truth Pattern

- Seven events in a one-second clip.
- Five short interior calls plus one left-truncated and one right-truncated event.
- The interior GT calls are approximately 7-10 ms long.

## Model and Tool Behaviour

| Method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Fixed qwen3.6 + grid_v2 | 0 | 7 | 7 | 0.000 |
| 0.5 s tiled qwen3.6 | 2 | 7 | 5 | 0.250 |
| 0.25 s tiled qwen3.6 | 1 | 6 | 6 | 0.143 |
| BatDetect2 proposal-only | 6 | 0 | 1 | 0.923 |
| Metadata-assisted qwen3.6 | 1 | 5 | 6 | 0.154 |
| P6E.2 proposal preservation | 6 | 0 | 1 | 0.923 |
| P6E.4 Policy B | 6 | 0 | 1 | 0.923 |

The unconstrained VLM shifted five good detector proposals and broke their temporal matches. Preservation restores all six detector-supported events. The left-boundary event remains missing because neither the detector nor VLM proposed it.

## Main Failure Mode

`harmful_proposal_refinement`, with a separate `boundary_miss`. The difficult part is not only visual detection; it is preventing a language model from replacing a precise detector prior with plausible but displaced geometry.

## System Design Lesson

Detector proposals should remain immutable references with explicit provenance. Refinements need deviation checks, while missing boundary events require a separate proposal-generation or review mechanism. Smaller tiles alone do not solve this case.

## Useful Figures

- [GT overlay](../../outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_016_gt_overlay.png)
- [Fixed grid_v2](../../outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_016_diagnostic_overlay.png)
- [0.5 s tiled](../../outputs/agent_runs/p6_tiled_qwen3_6_tile_0p5_overlap_0p1/diagnostic_figures/OP_016_diagnostic_overlay.png)
- [Proposal-only](../../outputs/agent_runs/p6_batdetect2_proposal_only_representative6/diagnostic_figures/OP_016_diagnostic_overlay.png)
- [Unconstrained metadata-assisted](../../outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/diagnostic_figures/OP_016_diagnostic_overlay.png)
- [Policy B](../../outputs/agent_runs/p6_timing_rule_ablation_representative6/policy_b_anchored_expansion/diagnostic_figures/OP_016_diagnostic_overlay.png)

## Dissertation-Ready Interpretation

> OP_016 demonstrates that external detector timing can be substantially more reliable than unconstrained VLM geometry for dense, millisecond-scale calls. BatDetect2 recovered six of seven events with no false positives, whereas free-form VLM refinement displaced five correct proposals. Deterministic preservation restored detector performance, but the unresolved left-boundary event shows that validation cannot recover candidates absent from both tools.
