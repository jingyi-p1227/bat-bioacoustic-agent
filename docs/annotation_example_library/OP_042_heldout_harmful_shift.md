# OP_042: Held-Out Harmful Rigid Shift Repaired

## Why It Matters

`OP_042` is positive held-out evidence for deterministic validation. Unlike `OP_032`, it contains the kind of near-rigid VLM displacement that Policy B was designed to reverse.

## Ground-Truth Pattern

- Five complete, non-truncated events in one second.
- Calls are distributed across the clip with no boundary event.

## Model and Tool Behaviour

| Method | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| P6E.5 proposal-only | 5 | 0 | 0 | 1.000 |
| P6E.5 metadata-assisted qwen3.6 | 4 | 1 | 1 | 0.800 |
| P6E.5 Policy B | 5 | 0 | 0 | 1.000 |

The VLM shifts the first good proposal by approximately 30.5 ms at the start and 35.6 ms at the end. Policy B recognises the same-direction translation and restores proposal timing.

## Main Failure Mode

`harmful_rigid_translation` under unconstrained VLM refinement.

## System Design Lesson

Deterministic guards are useful when the refinement behaves like a displaced copy of a strong detector proposal. The contrast with `OP_032` shows why translation and extent expansion need different rules.

## Useful Figures

- [GT overlay](../../outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_042_gt_overlay.png)
- [Proposal-only](../../outputs/agent_runs/p6e5_batdetect2_proposal_only_heldout/diagnostic_figures/OP_042_diagnostic_overlay.png)
- [Metadata-assisted](../../outputs/agent_runs/p6e5_batdetect2_metadata_assisted_heldout/diagnostic_figures/OP_042_diagnostic_overlay.png)
- [Policy B](../../outputs/agent_runs/p6e5_policy_b_anchored_validator_heldout/diagnostic_figures/OP_042_diagnostic_overlay.png)

## Dissertation-Ready Interpretation

> OP_042 shows that deterministic proposal preservation can generalise to a held-out harmful shift. Unconstrained VLM refinement displaced one correct proposal and reduced F1 to 0.800; Policy B restored the original detector timing and recovered perfect event-level performance. This success is specific to rigid translation and does not justify the same rule for extent expansion.
