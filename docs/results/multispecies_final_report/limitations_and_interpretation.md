# Limitations and Interpretation

## Interpretation

The multi-species experiments separate two subproblems: finding the target event within a short event-centred window, and assigning a species label to that event. The results show that these two subproblems behave very differently.

Localisation can be made strong in this controlled event-centred setup. BatDetect2 produced candidate regions for most samples, and a simple nearest-centre rule selected the target proposal effectively because each image window was constructed around the GT event centre. On full240, this produced IoU>=0.3 localisation F1 `0.888` and 10 ms start-time F1 `0.935`.

Species classification did not become reliable. Even when the target event was centred, marked with a neutral GT box, or selected by a detector proposal, qwen3.6 showed low species accuracy and strong label-collapse behaviour. Stage 1C was the best Stage 1 classification condition, but reached only `0.192` accuracy and `0.105` macro-F1. Stage 2C full240 had matched-proposal species accuracy `0.109` and joint F1 `0.097`.

## Limitations

- The nearest-centre selection rule is valid only because the Stage 1/Stage 2 windows are target-centred. It is not a general detector for long recordings.
- Stage 2C should therefore be interpreted as a decomposition experiment, not a deployable end-to-end long-recording workflow.
- Species guidance was compact and partly provisional; it was not fully literature-backed for every species.
- The model was qwen3.6, and results may not represent stronger commercial VLMs.
- Event-level spectrogram crops may not provide enough temporal, contextual, or call-sequence information for reliable species identification.
- Some species are intrinsically difficult to distinguish from isolated calls, especially the Myotis group.
- Ozimops petersi is included as an Australian benchmark anchor, while most other species are drawn from UK/European-labelled data; this complicates direct interpretation of species-level acoustic priors.
- BatDetect2 proposal quality is useful for localisation but does not itself solve species classification.

## Dissertation Interpretation

These results support a cautious conclusion: tool use can substantially improve localisation, but does not automatically solve higher-level biological classification. In this dataset, deterministic proposal selection produced strong target localisation, while VLM-based species classification remained weak even under GT-location conditions. This suggests that the limiting factor is not merely target localisation; it is the mapping from spectrogram morphology to fine-grained species identity.
