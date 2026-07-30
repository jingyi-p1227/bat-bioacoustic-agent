# Multi-Agent Stage 2C Pilot Summary

## Scope

This package consolidates the qwen3.6 and GPT-5.6 Sol multi-agent Stage 2C classification pilots for the dissertation-facing results directory. Both pilots used the same 24 selected-proposal samples, label-safe centred crop images, and deterministic nearest-centre BatDetect2 proposal coordinates. The agents were instructed to preserve proposal coordinates and perform forced-choice species classification only. Ground-truth species labels were used only after prediction for evaluation.

No new inference was run for this consolidation, and raw model response files are not copied into `docs/`.

## Main Same-Sample Results

| Method | Samples | Parse status | Accuracy | Macro-F1 | Balanced accuracy | Notes |
|---|---:|---|---:|---:|---:|---|
| qwen3.6 single-agent Stage 2C | 24 | 24 valid predictions | 0.000 | 0.000 | 0.000 | Same hard-sample subset used for multi-agent comparison. |
| qwen3.6 multi-agent Stage 2C | 24 | A1 24/24, A2 24/24, A3 24/24 | 0.042 | 0.014 | 0.042 | Barely improved over qwen3.6 single-agent. |
| GPT-5.6 Sol single-agent Stage 2C | 24 | 24 valid predictions | 0.542 | 0.486 | 0.542 | Much stronger underlying classifier on the same samples. |
| GPT-5.6 Sol multi-agent Stage 2C | 24 | A1 21/24, A2 21/24, A3 22/24 | 0.542 | 0.488 | 0.542 | Did not materially improve over GPT-5.6 Sol single-agent. |

## Per-Species Multi-Agent Results

| Species | qwen3.6 F1 | GPT-5.6 Sol F1 | GPT-5.6 Sol recall | Comment |
|---|---:|---:|---:|---|
| Rhinolophus hipposideros | 0.000 | 0.857 | 1.000 | GPT substantially stronger than qwen on this subset. |
| Rhinolophus ferrumequinum | 0.000 | 0.500 | 0.333 | GPT substantially stronger than qwen on this subset. |
| Myotis daubentonii | 0.000 | 0.444 | 0.667 | Still difficult; confusion remains concentrated here. |
| Myotis nattereri | 0.000 | 0.500 | 0.333 | Still difficult; confusion remains concentrated here. |
| Myotis mystacinus | 0.000 | 0.000 | 0.000 | Still difficult; confusion remains concentrated here. |
| Plecotus auritus | 0.000 | 0.000 | 0.000 | Still difficult; confusion remains concentrated here. |
| Pipistrellus pipistrellus | 0.111 | 0.600 | 1.000 | GPT recognised all true examples but also over-predicted the label. |
| Ozimops petersi | 0.000 | 1.000 | 1.000 | GPT recovered all three pilot samples. |

## Review and Uncertainty Signal

The qwen3.6 reviewer/adjudicator did not provide useful uncertainty signalling. It marked no cases for human review, while final accuracy remained 0.042. This means the critic/adjudicator structure did not identify the difficult or incorrect qwen cases.

GPT-5.6 Sol produced a more plausible review signal: 7 of 24 cases were marked uncertain or for human review. However, this signal was imperfect. Human-review cases had accuracy 0.429, while non-review cases had accuracy 0.667, so review flags were somewhat enriched for error/difficulty but not strong enough to act as a reliable automatic quality filter.

## Interpretation

The pilots show that multi-agent decomposition alone is not sufficient. qwen3.6 remained weak even when split into classifier, reviewer, and adjudicator roles, improving only from 0.000 to 0.042 accuracy on the selected hard subset. GPT-5.6 Sol was much stronger than qwen3.6, but its multi-agent result matched the single-agent GPT result at 0.542 accuracy, with only a tiny macro-F1 difference (0.488 versus 0.486).

The main bottleneck is therefore underlying model capability for species-level acoustic interpretation, not the presence or absence of a multi-agent wrapper. Multi-agent workflows may still be useful for human-in-the-loop annotation support, uncertainty triage, and proposal review, but this pilot does not support claiming that decomposition improves forced-choice species accuracy by itself.

## Relation to Full240 Results

The full240 single-agent Stage 2C qwen3.6 run had matched species accuracy 0.109 and joint F1 0.097. The full OpenRouter GPT-5.6 Sol comparison found much stronger species classification and joint performance than qwen3.6, reinforcing that model capability dominates the classification bottleneck.

## Source Files

- qwen3.6 multi-agent: `outputs/analysis_reports/multi_agent/qwen3_6_stage2c_pilot24/`
- GPT-5.6 Sol multi-agent: `outputs/analysis_reports/openrouter_model_comparison/gpt_5_6_sol_multi_agent_stage2c_pilot24_same_samples/`
- GPT-5.6 Sol single-agent comparison: `outputs/analysis_reports/openrouter_model_comparison/gpt_5_6_sol_uk_node_final_comparison/`
- qwen3.6 Stage 2C full240: `outputs/analysis_reports/multispecies_classification/qwen3_6_stage2c_nearest_centre_proposal_classification_full240/`
