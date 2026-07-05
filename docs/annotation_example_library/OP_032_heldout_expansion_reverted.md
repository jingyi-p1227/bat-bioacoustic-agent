# OP_032: Held-Out Useful Expansion Reverted

## Why It Matters

`OP_032` is the held-out counterexample that prevents Policy B's representative-six improvement from being over-generalised. It shows that a fixed 6 ms anchoring tolerance can still reject useful duration refinement.

## Ground-Truth Pattern

- Three events in one second.
- One left-truncated boundary event.
- Two complete events with GT durations of approximately 47-50 ms.

## Model and Tool Behaviour

| Method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| P6E.5 proposal-only | 0 | 3 | 3 | 0.000 |
| P6E.5 metadata-assisted qwen3.6 | 2 | 0 | 1 | 0.800 |
| P6E.5 Policy B | 1 | 1 | 2 | 0.400 |

The VLM expands two under-wide detector proposals into matches. Policy B keeps one expansion but restores the second because its start moves by 9.5 ms, converting a TP into an FP.

## Main Failure Mode

`held_out_useful_expansion_reverted`, plus an unresolved left-boundary event.

## System Design Lesson

Absolute anchoring thresholds do not generalise reliably across call duration and proposal quality. Validation should use duration-normalized displacement and explicit extent reasoning rather than one fixed millisecond threshold.

## Useful Figures

- [GT overlay](../../outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_032_gt_overlay.png)
- [Proposal-only](../../outputs/agent_runs/p6e5_batdetect2_proposal_only_heldout/diagnostic_figures/OP_032_diagnostic_overlay.png)
- [Metadata-assisted](../../outputs/agent_runs/p6e5_batdetect2_metadata_assisted_heldout/diagnostic_figures/OP_032_diagnostic_overlay.png)
- [Policy B](../../outputs/agent_runs/p6e5_policy_b_anchored_validator_heldout/diagnostic_figures/OP_032_diagnostic_overlay.png)

## Dissertation-Ready Interpretation

> OP_032 provides held-out evidence that a rule tuned on representative examples can suppress beneficial refinement. Unconstrained metadata assistance recovered two events from under-wide detector proposals, whereas Policy B reverted one expansion and halved F1 from 0.800 to 0.400. The case motivates duration-normalized validation and explicit treatment of proposal extent.
