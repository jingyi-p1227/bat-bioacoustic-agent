# Final Multi-Species Classification and Joint-Task Results

## 1. Scope

This report summarises the multi-species event-level classification and joint localisation/classification experiments. The experiments used the label-safe Stage 1 event-level dataset and the Stage 2 sample-level BatDetect2 proposal workflow.

No GT overlays, human-review overlays, raw PDFs, image exemplars, or embedded species labels were used as model input in the reported model-facing conditions.

## 2. Dataset Construction

The Stage 1 dataset contains:

- Species: `8`
- Total samples: `240`
- Samples per species: `30`
- Model-facing image size: `800x600`
- Label-safe pass: `240/240`
- Target-centred pass: `240/240`
- Embedded label text detected: `0`

The included species were:

- `Rhinolophus hipposideros`
- `Rhinolophus ferrumequinum`
- `Myotis daubentonii`
- `Myotis nattereri`
- `Myotis mystacinus`
- `Plecotus auritus`
- `Pipistrellus pipistrellus`
- `Ozimops petersi`

Stage 1 used GT event location only to construct the input image. In Stage 1A, the target event was horizontally centred without a box. In Stage 1B and Stage 1C, a neutral box marked the target event location. The GT species label was never included in the model input.

## 3. Stage 1 Classification Results

| Condition | Input | Guidance | Accuracy | Macro-F1 | Balanced Accuracy | Parse Success |
|---|---|---|---:|---:|---:|---:|
| Stage 1A | GT-centred crop, no box | zero-shot | 0.129 | 0.051 | 0.129 | 1.000 |
| Stage 1B | GT-box marker | zero-shot | 0.138 | 0.074 | 0.138 | 1.000 |
| Stage 1C | GT-box marker | compact species/acoustic guidance | 0.192 | 0.105 | 0.192 | 1.000 |

The neutral box marker produced only a small improvement over the no-box zero-shot condition. Compact species/acoustic guidance produced the best Stage 1 result, but performance remained low.

### Species-Level Patterns

Stage 1C improved `Rhinolophus ferrumequinum` most clearly, reaching F1 `0.526`. However, `Rhinolophus hipposideros`, `Myotis nattereri`, `Myotis mystacinus`, `Plecotus auritus`, and `Ozimops petersi` had F1 `0.000` in Stage 1C. `Pipistrellus pipistrellus` had recall `1.000` but low precision `0.146`, indicating strong over-prediction.

The dominant failure pattern was label collapse into `Pipistrellus pipistrellus`, especially for Myotis, Plecotus, Ozimops, and some Rhinolophus examples.

## 4. Coarse-Taxonomy Re-Evaluation

The coarse mapping grouped species as:

- `Rhinolophus hipposideros` and `Rhinolophus ferrumequinum` -> `Rhinolophus`
- `Myotis daubentonii`, `Myotis nattereri`, and `Myotis mystacinus` -> `Myotis`
- `Plecotus auritus` -> `Plecotus`
- `Pipistrellus pipistrellus` -> `Pipistrellus`
- `Ozimops petersi` -> `Ozimops`

| Condition | Coarse Accuracy | Coarse Macro-F1 | Coarse Balanced Accuracy |
|---|---:|---:|---:|
| Stage 1A | 0.129 | 0.081 | 0.207 |
| Stage 1B | 0.142 | 0.104 | 0.210 |
| Stage 1C | 0.242 | 0.179 | 0.292 |

Coarse evaluation improved Stage 1C, especially for Rhinolophus-vs-other recognition. In Stage 1C, coarse Rhinolophus F1 was `0.621`, but Myotis F1 was only `0.021`, Plecotus F1 was `0.000`, and Ozimops F1 was `0.000`.

This suggests that the model did not merely struggle with fine-grained species separation inside Myotis; it also failed broader acoustic-label mapping for several groups.

## 5. Stage 2 Joint Results

Stage 2 evaluated joint localisation and species classification in the same 0.300 s event-centred windows.

### Stage 2 Scaffold Caveat

The earlier Stage 2 scaffold did not have true sample-level BatDetect2 proposals for all samples and is treated only as a workflow scaffold. The later Stage 2B and Stage 2C experiments used real sample-level BatDetect2 proposals generated for the event windows.

### Sample-Level BatDetect2 Proposal Audit

Raw BatDetect2 proposals were available for `235/240` samples. Proposal-only evaluation on full240 showed high recall but many false positives:

| Protocol | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Temporal IoU >= 0.3 | 211 | 992 | 29 | 0.175 | 0.879 | 0.292 |
| Temporal IoU >= 0.1 | 225 | 978 | 15 | 0.187 | 0.938 | 0.312 |
| 10 ms start-time | 224 | 979 | 16 | 0.186 | 0.933 | 0.310 |

This confirmed that BatDetect2 proposals contained useful target candidates, but raw proposal sets were too broad to use directly as final detections.

### Central Proposal Selection Baseline

Because each Stage 1/Stage 2 image window was centred on the GT event, a deterministic proposal-selection baseline could choose the proposal nearest to the known image centre time of `0.150 s`. This did not use GT boxes as proposals or species labels for selection.

| Scope | Rule | IoU>=0.3 F1 | IoU>=0.1 F1 | 10 ms F1 | Mean Time IoU | Mean Frequency IoU | Mean Box IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| pilot80 | nearest_to_centre | 0.879 | 0.904 | 0.955 | 0.648 | 0.744 | 0.505 |
| full240 | nearest_to_centre | 0.888 | 0.943 | 0.935 | 0.653 | 0.758 | 0.517 |

The nearest-centre baseline substantially reduced false positives and showed that localisation could be strong in this controlled event-centred setting.

### Stage 2B: True Proposal-Constrained VLM Pilot80

Stage 2B gave qwen3.6 all available BatDetect2 proposals and asked it to retain, reject, refine, and classify detections. This performed poorly because the VLM retained or generated too many detections.

| Protocol | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Temporal IoU >= 0.3 | 43 | 247 | 37 | 0.148 | 0.538 | 0.232 |
| Temporal IoU >= 0.1 | 48 | 242 | 32 | 0.166 | 0.600 | 0.259 |
| 10 ms start-time | 49 | 241 | 31 | 0.169 | 0.613 | 0.265 |

Species accuracy on matched detections was `0.233`, but joint F1 was only `0.054` because localisation precision was low.

### Stage 2C: Nearest-Centre Proposal Classification

Stage 2C decomposed the task. It preserved the deterministic nearest-centre proposal coordinates and asked qwen3.6 only to classify the selected proposal. The model was not allowed to refine the bounding box or add detections.

| Scope | IoU>=0.3 F1 | 10 ms F1 | Matched Species Accuracy | Matched Macro-F1 | Joint F1 |
|---|---:|---:|---:|---:|---:|
| pilot80 | 0.879 | 0.955 | 0.159 | 0.051 | 0.140 |
| full240 | 0.888 | 0.935 | 0.109 | 0.032 | 0.097 |

Stage 2C full240 achieved strong localisation:

- IoU>=0.3 TP/FP/FN: `211/24/29`
- IoU>=0.3 precision/recall/F1: `0.898/0.879/0.888`
- 10 ms start-time F1: `0.935`
- Mean time IoU: `0.653`
- Mean frequency IoU: `0.758`
- Mean 2D box IoU: `0.517`

However, species classification on matched proposals remained weak:

- Matched detections: `211`
- Species accuracy: `0.109`
- Macro-F1: `0.032`
- Balanced accuracy: `0.096`
- Joint correct: `23`
- Joint F1: `0.097`

Non-zero full240 joint recall was concentrated in `Pipistrellus pipistrellus` (`22/30`) and one `Plecotus auritus` case (`1/30`). Rhinolophus, Myotis, and Ozimops had zero joint recall.

## 6. Main Final Interpretation

The final multi-species result is a decomposition: BatDetect2 proposals plus nearest-centre selection solve localisation well in the event-centred setup, but qwen3.6 species classification remains weak.

Stage 2C full240 is the clearest result. It reached IoU>=0.3 localisation F1 `0.888`, showing that the selected detector proposals align well with the target event in these centred windows. But matched-proposal species accuracy was only `0.109`, and joint F1 was `0.097`. This means joint performance is mainly limited by species classification rather than localisation.

The Stage 1 experiments support the same interpretation. Even with GT target location provided by centring or a neutral box marker, zero-shot qwen3.6 did not reliably classify species. Compact guidance improved performance, but did not solve label collapse or recover Myotis and Ozimops.

## 7. Limitations

- Nearest-centre proposal selection is valid only because windows are target-centred. It is not a general detector for long recordings.
- The event-centred setup tests species classification under controlled localisation conditions, not open-ended multi-species detection in full recordings.
- Species guidance was partly provisional and not fully literature-backed for every species.
- qwen3.6 may not represent the upper bound of vision-language model capability.
- Event-level spectrogram crops may lack enough contextual information for reliable species identification.
- The Myotis group remains especially difficult.
- Ozimops petersi is an Australian benchmark anchor, while the other species come from UK/European-labelled data; this should be handled cautiously in interpretation.

## 8. Conclusion

The multi-species experiments show that tool-supported localisation and species classification should be treated as separate challenges. In this controlled setup, deterministic proposal selection provided strong localisation, but the VLM did not provide reliable fine-grained species labels. The final result therefore supports a cautious dissertation claim: detector tools can stabilise event localisation, but species-level acoustic interpretation requires stronger species knowledge, more suitable context, better models, or human/expert review.
