# P7A Results Draft: Single-Agent Tool-Use Experiments

> Draft status: dissertation-ready results narrative based on frozen project artifacts. Citation placeholders must be replaced with verified references before submission.

## 1. Overview of Experimental Aim

These experiments investigated whether a vision-language model (VLM) could produce event-level strong labels for individual bat echolocation calls, and whether external visual or detector-based tools could improve its predictions. The task required one time-frequency bounding box per visible call rather than a whole-clip species label. All conditions used the frozen event-matching protocol and were assessed using precision, recall, F1, temporal intersection over union (IoU), frequency IoU, and two-dimensional box IoU [TODO: cite strong-labelling and IoU evaluation literature].

The initial comparison used six representative clips containing 33 ground-truth events. These clips were selected to include clean, dense, and boundary-truncated cases. A separate ten-clip held-out set containing 43 ground-truth events was subsequently used to test whether a deterministic validation rule developed on the representative clips generalised. A targeted 0.25 s tiling experiment was conducted only on OP_016 and is therefore reported separately rather than ranked against the six-clip aggregates.

The experiments progressed through three stages. First, visual preprocessing conditions were compared with a fixed qwen3.6 overview baseline. Second, BatDetect2 outputs were introduced as proposal metadata rather than treated as ground truth. Third, deterministic post-processors were tested to constrain unsupported VLM changes to detector geometry. BatDetect2 taxonomy predictions were not evaluated as Australian species identifications; its outputs were treated as generic `bat_call` proposals [TODO: cite the official BatDetect2 paper/software record].

## 2. Fixed qwen3.6 Baseline

The fixed-view baseline used qwen3.6 with a clean dB spectrogram and the `grid_v2` display condition. On the representative six clips, it detected 20 of 33 ground-truth events, with 13 false positives and 13 false negatives. Precision, recall, and F1 were all 0.6061. Mean temporal IoU was 0.6031, mean frequency IoU was 0.5839, and mean box IoU was 0.3757. Fourteen matched events reached box IoU >= 0.3 and four reached box IoU >= 0.5.

| Method | TP | FP | FN | Precision | Recall | F1 | Time IoU | Frequency IoU | Box IoU | Box IoU >= 0.3 | Box IoU >= 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed qwen3.6 + grid_v2 | 20 | 13 | 13 | 0.6061 | 0.6061 | 0.6061 | 0.6031 | 0.5839 | 0.3757 | 14 | 4 |

Performance varied substantially by clip. The baseline achieved perfect event-level F1 on the simple partial clip OP_045 and also handled OP_004 well, but failed to match any of the seven dense short calls in OP_016. The relatively high mean box IoU compared with several later conditions indicates that, when the baseline found an event, its localisation could be useful. Its principal limitation was inconsistent call recovery across visually difficult clips.

## 3. Visual Preprocessing Experiments

### 3.1 0.5 s tiled spectrograms

Dividing each clip into overlapping 0.5 s views increased the number of true positives from 20 to 25 and reduced false negatives from 13 to 8. Recall increased from 0.6061 to 0.7576, producing an F1 of 0.7143. This was the highest recall among the representative-six conditions. However, mean frequency IoU fell from 0.5839 to 0.4933 and mean box IoU fell from 0.3757 to 0.3429. Only two matches reached box IoU >= 0.5, compared with four for the fixed baseline.

These results suggest that shorter views made candidate calls easier to notice but did not improve precise time-frequency extent estimation. Overlapping tiles also introduced duplicate-prediction risk, which required deterministic merging. Thus, tiling primarily improved event recovery rather than box quality.

### 3.2 Targeted 0.25 s tiling on OP_016

The 0.25 s condition was tested only on OP_016 because this dense boundary-stress case remained difficult. It produced one true positive, six false positives, and six false negatives, corresponding to precision, recall, and F1 of 0.1429. By comparison, the 0.5 s tiled condition produced two true positives, seven false positives, and five false negatives on the same clip, with F1 of 0.2500. The smaller tiles therefore did not resolve OP_016 and should not be interpreted as an aggregate result.

### 3.3 PCEN-enhanced spectrograms

PCEN preprocessing produced the same aggregate detection counts as the fixed baseline: 20 true positives, 13 false positives, and 13 false negatives, giving F1 of 0.6061. Localisation quality nevertheless declined. Mean temporal IoU decreased to 0.5706, mean frequency IoU to 0.5246, and mean box IoU to 0.3271. The number of matches with box IoU >= 0.3 fell from 14 to 10 [TODO: cite the verified PCEN source and its intended robustness properties].

PCEN also regressed OP_045 from three correctly matched events under the fixed baseline to zero true positives, three false positives, and three false negatives. Within this pilot, PCEN did not provide evidence of an aggregate advantage and could alter visual contrast in ways that harmed otherwise easy cases.

| Visual condition | TP | FP | FN | Precision | Recall | F1 | Time IoU | Frequency IoU | Box IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed grid_v2 | 20 | 13 | 13 | 0.6061 | 0.6061 | 0.6061 | 0.6031 | 0.5839 | 0.3757 |
| 0.5 s tiled | 25 | 12 | 8 | 0.6757 | 0.7576 | 0.7143 | 0.6655 | 0.4933 | 0.3429 |
| PCEN grid_v2 | 20 | 13 | 13 | 0.6061 | 0.6061 | 0.6061 | 0.5706 | 0.5246 | 0.3271 |

## 4. BatDetect2 Proposal-Tool Experiments

### 4.1 Proposal-only baseline

BatDetect2 proposals at `det_prob >= 0.30` were evaluated directly as generic bat-call boxes. On the representative six clips, proposal-only achieved 21 true positives, 11 false positives, and 12 false negatives, giving precision of 0.6562, recall of 0.6364, and F1 of 0.6462. Mean temporal, frequency, and box IoU were 0.4929, 0.6171, and 0.3317, respectively.

The detector was particularly effective on OP_016, where it recovered six of seven calls with no false positives (F1 = 0.9231). This contrasted with the fixed VLM and both tiled conditions. The result supports the use of BatDetect2 as a timing proposal tool, but not as an infallible source of annotation geometry. For example, its OP_045 intervals were too short relative to the project annotation standard, resulting in zero matched events.

### 4.2 Metadata-assisted VLM

The metadata-assisted condition supplied qwen3.6 with the clean `grid_v2` spectrogram and structured BatDetect2 proposal metadata. The model was instructed to verify, refine, reject, or add proposals. This unconstrained combination performed worse than either component baseline: it produced 17 true positives, 14 false positives, and 16 false negatives, yielding F1 of 0.5312. Mean box IoU fell to 0.2739, and no matches reached box IoU >= 0.5.

Event-level analysis showed that the model sometimes shifted already useful detector proposals. On OP_016, metadata-assisted qwen3.6 retained only one true positive and introduced five false positives, reducing F1 to 0.1538. The result demonstrates that supplying a useful external tool does not ensure useful tool use: unconstrained VLM refinement can damage accurate proposal geometry [TODO: cite literature on tool-using agents, verification, and uncertainty-aware control].

| Proposal condition | TP | FP | FN | Precision | Recall | F1 | Time IoU | Frequency IoU | Box IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BatDetect2 proposal-only | 21 | 11 | 12 | 0.6562 | 0.6364 | 0.6462 | 0.4929 | 0.6171 | 0.3317 |
| Metadata-assisted qwen3.6 | 17 | 14 | 16 | 0.5484 | 0.5152 | 0.5312 | 0.4264 | 0.6219 | 0.2739 |

## 5. Deterministic Validation Experiments

Three shadow-mode validators were applied to the existing metadata-assisted predictions without rerunning either model. The proposal-preserving validator restored full BatDetect2 geometry when VLM deviations exceeded fixed thresholds. The timing-preserving validator separated temporal and frequency decisions, allowing VLM frequency bounds to remain when shifts were not flagged. Policy B additionally allowed moderate duration expansion when one temporal boundary remained anchored, while preserving proposal timing for near-rigid translations.

Both the full proposal-preserving and timing-preserving validators achieved 22 true positives, nine false positives, and 11 false negatives (F1 = 0.6875). Their detection counts were identical, while the timing-only variant had slightly higher mean frequency IoU (0.6591 versus 0.6546) and slightly lower mean box IoU (0.3450 versus 0.3513).

Policy B achieved the strongest representative-six F1, with 23 true positives, eight false positives, and ten false negatives. Precision was 0.7419, recall was 0.6970, and F1 was 0.7188. It restored the six matched OP_016 calls while allowing a useful moderate expansion in OP_004. However, Policy B was developed using these representative cases; its result is therefore a development-set finding rather than evidence of generalisation.

| Validator | TP | FP | FN | Precision | Recall | F1 | Time IoU | Frequency IoU | Box IoU | Box IoU >= 0.3 | Box IoU >= 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full proposal-preserving | 22 | 9 | 11 | 0.7097 | 0.6667 | 0.6875 | 0.4989 | 0.6546 | 0.3513 | 10 | 4 |
| Timing-preserving | 22 | 9 | 11 | 0.7097 | 0.6667 | 0.6875 | 0.4989 | 0.6591 | 0.3450 | 10 | 4 |
| Policy B anchored | 23 | 8 | 10 | 0.7419 | 0.6970 | 0.7188 | 0.5053 | 0.6592 | 0.3497 | 12 | 4 |

## 6. Held-Out Validation

The frozen Policy B rule was then evaluated on ten held-out clips containing 43 ground-truth events. BatDetect2 proposal-only was the strongest held-out method, with 34 true positives, five false positives, and nine false negatives. It achieved precision of 0.8718, recall of 0.7907, and F1 of 0.8293.

Unconstrained metadata-assisted qwen3.6 found the same 34 true positives and missed the same nine events, but doubled the number of false positives from five to ten. Its F1 was therefore lower at 0.7816. Policy B did not change these aggregate detection counts and also achieved F1 of 0.7816. Moreover, its mean temporal, frequency, and box IoU values (0.4668, 0.5865, and 0.2705) were each slightly lower than those of the unconstrained condition (0.4776, 0.5902, and 0.2777).

| Held-out method | TP | FP | FN | Precision | Recall | F1 | Time IoU | Frequency IoU | Box IoU | Box IoU >= 0.3 | Box IoU >= 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposal-only | 34 | 5 | 9 | 0.8718 | 0.7907 | 0.8293 | 0.4938 | 0.6055 | 0.3187 | 13 | 6 |
| Metadata-assisted qwen3.6 | 34 | 10 | 9 | 0.7727 | 0.7907 | 0.7816 | 0.4776 | 0.5902 | 0.2777 | 10 | 4 |
| Policy B anchored | 34 | 10 | 9 | 0.7727 | 0.7907 | 0.7816 | 0.4668 | 0.5865 | 0.2705 | 10 | 4 |

Policy B repaired a harmful VLM shift in OP_042 but reverted a useful expansion in OP_032. These opposing effects explain why its representative-six gain did not transfer to the held-out aggregate. The held-out experiment therefore does not support expanding the current deterministic rule to all 45 clips.

## 7. Case-Study Summary

### 7.1 OP_016: dense short-call sequence

OP_016 contained seven dense, short-duration events, including boundary-truncated calls. The fixed VLM matched none of them. Tiling improved recovery only modestly: the 0.5 s condition obtained two true positives (F1 = 0.2500), while targeted 0.25 s tiling obtained one (F1 = 0.1429). BatDetect2 proposal-only recovered six calls without false positives (F1 = 0.9231), but unconstrained VLM refinement reduced this to one true positive, five false positives, and six false negatives. Both proposal preservation and Policy B restored six matches. The remaining left-boundary event was not recovered, showing that preservation can protect a good proposal but cannot create a missing one.

### 7.2 OP_045: source proposal extent failure

OP_045 was a simple partial-duration clip with three events. Fixed qwen3.6 achieved perfect event-level detection (3 TP, 0 FP, 0 FN), whereas PCEN, proposal-only, and Policy B each produced 0 TP, 3 FP, and 3 FN. The detector proposals were substantially shorter than the annotated event extents. Consequently, proposal preservation could not repair the case because the source geometry itself was unsuitable. OP_045 distinguishes harmful VLM deviation from detector extent failure and motivates evidence-based duration expansion rather than unconditional preservation.

### 7.3 OP_004: useful VLM expansion

On OP_004, unconstrained metadata-assisted qwen3.6 produced five true positives, no false positives, and one false negative (F1 = 0.9091). Full proposal preservation reduced performance to four true positives, one false positive, and two false negatives (F1 = 0.7273) by removing a useful duration expansion. Policy B allowed the anchored expansion and restored F1 to 0.9091. This development example motivated the distinction between a near-rigid shift and an expansion supported by one stable boundary.

### 7.4 OP_032: held-out useful expansion reverted

OP_032 exposed the limitation of that hand-designed distinction. Proposal-only produced no temporal matches (0 TP, 3 FP, 3 FN), while unconstrained VLM expansion recovered two events (2 TP, 0 FP, 1 FN; F1 = 0.8000). Policy B reverted one useful expansion, producing one true positive, one false positive, and two false negatives (F1 = 0.4000). This case provides direct held-out evidence that the fixed anchoring tolerance was too conservative for some under-extended proposals.

### 7.5 OP_042: held-out harmful shift repaired

For OP_042, proposal-only matched all five events with no errors. Unconstrained VLM refinement shifted one useful proposal out of the matching range, producing four true positives, one false positive, and one false negative (F1 = 0.8000). Policy B correctly restored the event and returned to perfect event-level F1. Thus, the validator addressed its intended failure mode in this clip, but this local success was offset by its behaviour on OP_032.

| Case | Central issue | Main result |
|---|---|---|
| OP_016 | Dense short calls; harmful proposal shifts | Proposal-only and validators reached F1 0.9231; unconstrained assistance fell to 0.1538. |
| OP_045 | Detector proposals too short | Fixed VLM reached F1 1.0000; proposal preservation could not repair source extent. |
| OP_004 | Useful anchored expansion | Policy B retained the useful VLM change on the development subset. |
| OP_032 | Useful held-out expansion reverted | Unconstrained F1 was 0.8000; Policy B reduced it to 0.4000. |
| OP_042 | Harmful held-out rigid shift | Policy B repaired the shift and restored F1 from 0.8000 to 1.0000. |

## 8. Main Findings

First, changing the visual representation did not produce a uniformly better annotator. The 0.5 s tiled condition improved recall and achieved a strong representative-six F1, but reduced frequency and box localisation quality. The finer 0.25 s view did not solve the targeted dense case, and PCEN did not improve aggregate detection while regressing an easy clip.

Second, BatDetect2 supplied useful timing priors. Its proposal-only condition was competitive on the representative clips and was the strongest held-out baseline. However, proposal quality was case-dependent: OP_016 benefited from accurate detector timing, whereas OP_045 exposed under-extended source boxes.

Third, unconstrained metadata assistance was unreliable. qwen3.6 could make useful expansions, but it could also move accurate detector proposals away from true events. The metadata-assisted condition performed below proposal-only in both representative and held-out comparisons.

Fourth, deterministic validation repaired identifiable failure modes but did not generalise reliably. Policy B achieved the best representative-six F1 of 0.7188 and repaired OP_016 and OP_004, yet it provided no held-out aggregate improvement and reduced localisation IoUs. The contrast between OP_032 and OP_042 shows why a single absolute timing threshold cannot consistently distinguish useful expansion from harmful displacement.

These findings support constrained and provenance-aware tool use, but not the full-set deployment of the current rule. The evidence remains limited to one species, a small development subset, and a ten-clip held-out validation set. Results should therefore be interpreted as controlled ablations and failure-mode evidence rather than a general benchmark of automated bat annotation [TODO: cite bioacoustic monitoring and human-in-the-loop annotation literature].

## 9. Transition to Discussion

The Results establish a central tension for tool-augmented bioacoustic annotation. Detector proposals can provide stronger temporal evidence than the VLM alone, but both unconditional acceptance and unconstrained refinement fail in different cases. Simple deterministic validators can protect good proposals, yet fixed thresholds cannot account for variation in call duration, boundary truncation, signal quality, and detector under-extension.

The Discussion should therefore examine three implications. First, validation should use duration-normalised deviations and explicit proposal provenance rather than only absolute millisecond thresholds. Second, extent expansion should require positive visual evidence and should be separated from proposal translation. Third, uncertain conflicts may require human review or a carefully evaluated critic/referee stage rather than increasingly complex hand-written rules [TODO: cite uncertainty calibration, human-in-the-loop, and critic/referee workflow literature]. These directions preserve the project's main evidence: reliable agentic annotation depends not merely on access to tools, but on constrained use, validation, and transparent handling of uncertainty.
