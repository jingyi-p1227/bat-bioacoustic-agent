# OP_045: Clean Partial Clip and Source-Proposal Extent Failure

## Why It Matters

`OP_045` separates representation quality from detector geometry quality. The fixed VLM solves the clip, while PCEN and BatDetect2-based conditions fail. It is the strongest evidence that proposal preservation cannot repair a proposal that is wrong under the chosen annotation standard.

## Ground-Truth Pattern

- Three well-separated events.
- Partial final clip spanning approximately 0.826 s.
- No boundary-truncated GT events.
- GT intervals are approximately 53-58 ms long.

## Model and Tool Behaviour

| Method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Fixed qwen3.6 + grid_v2 | 3 | 0 | 0 | 1.000 |
| PCEN qwen3.6 | 0 | 3 | 3 | 0.000 |
| BatDetect2 proposal-only | 0 | 3 | 3 | 0.000 |
| P6E.4 Policy B | 0 | 3 | 3 | 0.000 |

BatDetect2 identifies three plausible call locations, but its approximately 12-14 ms intervals are too short to reach the temporal-IoU match threshold against the wider GT boxes. Metadata assistance retains those short intervals, and preservation therefore cannot help.

## Main Failure Mode

`source_proposal_extent_failure`. PCEN is a separate representation regression on a clip solved by the original dB overview.

## System Design Lesson

Proposal-preserving validation needs an exception for evidence-backed extent expansion. A detector box should be treated as a candidate core, not automatically as the full annotation extent. Preprocessing tools must also be checked against clean-case regressions.

## Useful Figures

- [GT overlay](../../outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_045_gt_overlay.png)
- [Fixed grid_v2](../../outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_045_diagnostic_overlay.png)
- [PCEN](../../outputs/agent_runs/p6_pcen_qwen3_6_representative6/diagnostic_figures/OP_045_diagnostic_overlay.png)
- [Proposal-only](../../outputs/agent_runs/p6_batdetect2_proposal_only_representative6/diagnostic_figures/OP_045_diagnostic_overlay.png)
- [Metadata-assisted](../../outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/diagnostic_figures/OP_045_diagnostic_overlay.png)
- [Policy B](../../outputs/agent_runs/p6_timing_rule_ablation_representative6/policy_b_anchored_expansion/diagnostic_figures/OP_045_diagnostic_overlay.png)

## Dissertation-Ready Interpretation

> OP_045 shows that preserving detector geometry is only useful when the source proposal is compatible with the annotation standard. Although BatDetect2 located all three candidate regions, its intervals were much shorter than the GT extents and produced no matches. This failure requires evidence-based duration expansion rather than stronger preservation.
