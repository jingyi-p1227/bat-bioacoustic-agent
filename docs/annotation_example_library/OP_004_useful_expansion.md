# OP_004: Useful VLM Expansion and Over-Conservative Validation

## Why It Matters

`OP_004` shows that VLM refinement can be genuinely useful. It is the development-set counterexample to unconditional proposal preservation and motivated Policy B's anchored-expansion rule.

## Ground-Truth Pattern

- Six events in one second.
- One left-truncated event continuing the right-boundary call from `OP_003`.
- Five complete events, including a call whose GT extent is wider than its detector proposal.

## Model and Tool Behaviour

| Method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Fixed qwen3.6 + grid_v2 | 6 | 0 | 0 | 1.000 |
| Metadata-assisted qwen3.6 | 5 | 0 | 1 | 0.909 |
| P6E.2 full preservation | 4 | 1 | 2 | 0.727 |
| P6E.4 Policy B | 5 | 0 | 1 | 0.909 |

The VLM expanded one under-wide proposal from approximately 12.2 ms to 20 ms. Full preservation reverted that useful change; Policy B retained it as an anchored moderate expansion.

## Main Failure Mode

`over_conservative_proposal_preservation`.

## System Design Lesson

Validation must distinguish harmful translation from plausible asymmetric expansion. Proposal geometry is a prior, not always the final annotation extent. The perfect fixed-view result also warns that the tool-assisted workflow is not automatically superior.

## Useful Figures

- [GT overlay](../../outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_004_gt_overlay.png)
- [Fixed grid_v2](../../outputs/agent_runs/prompt_v2_full_qwen3_6_grid_v2/evaluation/diagnostic_figures/OP_004_diagnostic_overlay.png)
- [Metadata-assisted](../../outputs/agent_runs/p6_batdetect2_metadata_assisted_qwen3_6_representative6/diagnostic_figures/OP_004_diagnostic_overlay.png)
- [P6E.2 full preservation](../../outputs/agent_runs/p6_proposal_preserving_validator_representative6/diagnostic_figures/OP_004_diagnostic_overlay.png)
- [Policy B](../../outputs/agent_runs/p6_timing_rule_ablation_representative6/policy_b_anchored_expansion/diagnostic_figures/OP_004_diagnostic_overlay.png)

## Dissertation-Ready Interpretation

> OP_004 demonstrates that detector preservation can itself introduce error. A moderate VLM duration expansion converted an under-wide proposal into a match, but full proposal restoration removed this gain. Policy B recovered the event by allowing an anchored expansion, illustrating why proposal translation and proposal extent refinement should be validated separately.
