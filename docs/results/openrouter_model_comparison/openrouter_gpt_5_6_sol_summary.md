# OpenRouter GPT-5.6 Sol Model Comparison Summary

## Scope

This summary consolidates the completed OpenRouter `openai/gpt-5.6-sol`
comparison into the dissertation-facing results package. No new inference was
run for this consolidation step.

The comparison used two frozen workflows:

- Full45 `Ozimops petersi` single-agent localisation with clean grid_v2
  spectrograms and BatDetect2 proposal metadata.
- Full240 multi-species Stage 2C selected-proposal classification, where the
  nearest-centre BatDetect2 proposal geometry was preserved and the model only
  classified the selected event.

## Main Results

GPT-5.6 Sol did not beat qwen3.6 on the primary localisation protocol. Under
temporal IoU >= 0.3, GPT-5.6 Sol achieved F1 `0.768`, compared with `0.785` for
the qwen3.6 proposal-constrained VLM condition.

GPT-5.6 Sol was very strong on onset-sensitive localisation. Under the 10 ms
start-time proximity protocol, it achieved F1 `0.957`, higher than the
comparison conditions in the current table.

GPT-5.6 Sol substantially improved multi-species species classification and
joint task performance. On matched selected proposals, species accuracy
increased to `0.597`, macro-F1 to `0.572`, and balanced accuracy to `0.611`.
Joint F1 increased to `0.531`, compared with `0.097` for qwen3.6 Stage 2C
full240.

## Species-Level Interpretation

The stronger model reduced the severe label-collapse pattern observed with
qwen3.6, but some species remained difficult. `Plecotus auritus` remained the
hardest class in the matched-proposal classification analysis, with very low
recall. The `Myotis` group still showed substantial confusion, especially for
`Myotis mystacinus` and `Myotis daubentonii`.

`Ozimops petersi` was much better recovered than in the qwen3.6 run, reaching
matched-proposal F1 `0.844`. This suggests that the selected-proposal workflow
can support species classification when the model has enough acoustic-pattern
recognition capacity, although performance remains uneven across taxa.

## Cost

The full OpenRouter comparison used `433,138` total tokens and cost about
`$4.61` under the project pricing assumption. This covered `45` localisation
calls and `235` classification API calls; five classification samples had no
selected proposal and therefore did not require a model call.

## Dissertation Interpretation

These results support a human-in-the-loop and agent-assisted annotation
workflow framing. A stronger frontier VLM improved species classification
substantially, but did not simply dominate every localisation metric. The best
localisation behaviour still depends on structured tool use, detector proposals,
and evaluation of onset-sensitive geometry. For multi-species classification,
the improvement from GPT-5.6 Sol indicates that model capability matters, but
remaining Myotis and Plecotus failures show that expert review and targeted
acoustic guidance remain necessary.

## Source Tables

- `localisation_comparison_table.csv`
- `classification_comparison_table.csv`
- `joint_task_comparison_table.csv`
- `token_cost_summary.csv`

