# Dissertation Paragraphs

To test whether the observed limitations of the local qwen3.6 workflow were
model-capability limited, the final localisation and multi-species
classification workflows were repeated with `openai/gpt-5.6-sol` through
OpenRouter. The comparison used the same frozen model-facing inputs and
structured output schemas as the qwen3.6 experiments: BatDetect2
proposal-constrained localisation for the full45 `Ozimops petersi` benchmark,
and nearest-centre selected-proposal classification for the full 240-sample
multi-species dataset. No ground-truth overlays, human-review overlays, raw
response files, or environment secrets were included in the dissertation-facing
package.

For localisation, GPT-5.6 Sol achieved temporal IoU >= 0.3 F1 `0.768`, which
did not exceed the qwen3.6 proposal-constrained VLM result of `0.785`. However,
the model was very strong under onset-sensitive evaluation, reaching 10 ms
start-time F1 `0.957`. This indicates that the frontier model was highly
effective at preserving or selecting event onsets, while the primary
time-overlap localisation score did not improve over the strongest qwen3.6
condition.

For multi-species classification, GPT-5.6 Sol substantially improved over
qwen3.6. On matched selected proposals, species accuracy increased to `0.597`,
macro-F1 to `0.572`, and balanced accuracy to `0.611`; joint F1 increased from
`0.097` for qwen3.6 Stage 2C full240 to `0.531`. `Ozimops petersi` was much
better recovered, although `Myotis` species and especially `Plecotus auritus`
remained difficult. These results support a human-in-the-loop, agent-assisted
annotation framing: stronger VLMs can improve acoustic interpretation, but
reliable deployment still depends on constrained tool use, detector proposals,
and expert or critic review for difficult taxa.

