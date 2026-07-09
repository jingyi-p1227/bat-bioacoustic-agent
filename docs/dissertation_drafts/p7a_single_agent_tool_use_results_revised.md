# P7A Results: Single-Agent and Tool-Use Experiments

> Draft status: revised Results chapter based on frozen project artifacts. Citation placeholders require verified references before submission.

## 1. Experimental Scope and Reporting

The experiments assessed whether a vision-language model (VLM) could produce event-level strong labels for individual bat echolocation calls and whether visual preprocessing, BatDetect2 proposals, or deterministic validation could improve its predictions. Each predicted event was represented by a time-frequency bounding box. All conditions used the same frozen one-to-one event-matching protocol [TODO: cite strong-labelling and IoU evaluation literature].

The six clips OP_001, OP_003, OP_004, OP_010, OP_016, and OP_045 formed a **diagnostic development subset** containing 33 ground-truth events. They were selected to expose clean, dense, and boundary-truncated cases rather than to estimate population-level performance. The deterministic Policy B rule was developed from failure modes observed in this subset. A separate set of ten clips containing 43 ground-truth events was held out from rule development and used to assess generalisation.

Unless stated otherwise, precision, recall, and F1 are aggregate event-level metrics calculated from TP, FP, and FN counts pooled across all clips in the relevant subset. Mean IoU values are calculated over matched events. Results from the targeted 0.25 s experiment on OP_016 cover one clip only and are not directly ranked against six-clip aggregates.

## 2. Representative-Six Development Results

Table 1 reports the main results on the diagnostic development subset. Policy B had the highest F1 on this subset (0.7188), closely followed by 0.5 s tiled qwen3.6 (0.7143). These results reflect different error profiles: Policy B had higher precision, whereas the tiled condition had the highest recall. The fixed-view condition retained the highest mean box IoU among the listed representative-six methods.

**Table 1. Representative-six diagnostic development results.**

| Method | TP | FP | FN | Precision | Recall | F1 | Mean time IoU | Mean frequency IoU | Mean box IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed qwen3.6 + grid_v2 | 20 | 13 | 13 | 0.6061 | 0.6061 | 0.6061 | 0.6031 | 0.5839 | 0.3757 |
| 0.5 s tiled qwen3.6 | 25 | 12 | 8 | 0.6757 | 0.7576 | 0.7143 | 0.6655 | 0.4933 | 0.3429 |
| PCEN qwen3.6 | 20 | 13 | 13 | 0.6061 | 0.6061 | 0.6061 | 0.5706 | 0.5246 | 0.3271 |
| BatDetect2 proposal-only | 21 | 11 | 12 | 0.6562 | 0.6364 | 0.6462 | 0.4929 | 0.6171 | 0.3317 |
| Metadata-assisted qwen3.6 | 17 | 14 | 16 | 0.5484 | 0.5152 | 0.5312 | 0.4264 | 0.6219 | 0.2739 |
| Proposal-preserving validator | 22 | 9 | 11 | 0.7097 | 0.6667 | 0.6875 | 0.4989 | 0.6546 | 0.3513 |
| Timing-preserving validator | 22 | 9 | 11 | 0.7097 | 0.6667 | 0.6875 | 0.4989 | 0.6591 | 0.3450 |
| Policy B anchored validator | 23 | 8 | 10 | 0.7419 | 0.6970 | 0.7188 | 0.5053 | 0.6592 | 0.3497 |

> **Figure placeholder: Representative-six method comparison.** Suggested plot: precision, recall, and F1 by method, with development-subset status stated in the caption.

### 2.1 Fixed-view baseline

The fixed qwen3.6 baseline used a clean dB spectrogram with the `grid_v2` display condition. It detected 20 of 33 ground-truth events, with 13 false positives and 13 false negatives. Precision, recall, and F1 were each 0.6061. Its mean time, frequency, and box IoU values were 0.6031, 0.5839, and 0.3757, respectively.

Performance varied by clip. The baseline correctly matched all three OP_045 events and performed well on OP_004, but matched none of the seven dense short calls in OP_016. The aggregate result therefore concealed substantial variation between clean and difficult cases.

### 2.2 Visual preprocessing conditions

The 0.5 s tiled condition increased true positives from 20 to 25 and reduced false negatives from 13 to 8. Recall rose from 0.6061 to 0.7576, while F1 rose to 0.7143. Mean box IoU, however, decreased from 0.3757 to 0.3429, and mean frequency IoU decreased from 0.5839 to 0.4933. Within this development subset, shorter views increased event recovery without improving precise frequency localisation.

The targeted 0.25 s OP_016 experiment produced 1 TP, 6 FP, and 6 FN, with precision, recall, and F1 of 0.1429. On the same clip, the 0.5 s tiled condition produced 2 TP, 7 FP, and 5 FN, with F1 of 0.2500. The finer tiles therefore did not resolve this specific dense case.

PCEN produced the same pooled detection counts as the fixed baseline (20 TP, 13 FP, and 13 FN; F1 = 0.6061), but reduced mean time IoU to 0.5706, mean frequency IoU to 0.5246, and mean box IoU to 0.3271. It also changed OP_045 from 3 TP, 0 FP, and 0 FN under the fixed baseline to 0 TP, 3 FP, and 3 FN [TODO: cite the verified PCEN source].

### 2.3 BatDetect2 proposal conditions

BatDetect2 proposals with `det_prob >= 0.30` were evaluated as generic `bat_call` boxes; UK taxonomy labels were not treated as Australian species predictions [TODO: cite the official BatDetect2 paper or software record]. Proposal-only produced 21 TP, 11 FP, and 12 FN, giving F1 of 0.6462. Its mean frequency IoU was 0.6171, higher than the fixed baseline value of 0.5839, although its mean time and box IoUs were lower.

The metadata-assisted condition supplied qwen3.6 with both the clean spectrogram and structured proposal metadata. It produced 17 TP, 14 FP, and 16 FN, giving F1 of 0.5312. This was lower than both the fixed-view and proposal-only conditions. Event-level inspection showed that some accurate detector proposals were moved by the VLM and no longer matched ground truth.

### 2.4 Deterministic validators

The full proposal-preserving and timing-preserving validators each produced 22 TP, 9 FP, and 11 FN, giving F1 of 0.6875. Their pooled detection counts were identical. The timing-preserving variant had slightly higher mean frequency IoU (0.6591 versus 0.6546), while full proposal preservation had slightly higher mean box IoU (0.3513 versus 0.3450).

Policy B distinguished moderate anchored expansion from near-rigid temporal translation. It produced 23 TP, 8 FP, and 10 FN, with precision of 0.7419, recall of 0.6970, and F1 of 0.7188. This was the highest F1 on the representative development subset. Because the rule was designed in response to failures observed in these six clips, this result was treated as development evidence and was subsequently tested on held-out data.

## 3. Held-Out Generalisation Results

Table 2 reports the ten-clip held-out comparison. Proposal-only had the highest held-out F1 (0.8293). The metadata-assisted VLM found the same number of true events but produced five additional false positives. Applying the frozen Policy B rule did not change the pooled TP, FP, or FN counts and slightly reduced all three mean IoU values relative to unconstrained metadata assistance.

**Table 2. Ten-clip held-out generalisation results.**

| Method | TP | FP | FN | Precision | Recall | F1 | Mean time IoU | Mean frequency IoU | Mean box IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BatDetect2 proposal-only | 34 | 5 | 9 | 0.8718 | 0.7907 | 0.8293 | 0.4938 | 0.6055 | 0.3187 |
| Metadata-assisted qwen3.6 | 34 | 10 | 9 | 0.7727 | 0.7907 | 0.7816 | 0.4776 | 0.5902 | 0.2777 |
| Policy B anchored validator | 34 | 10 | 9 | 0.7727 | 0.7907 | 0.7816 | 0.4668 | 0.5865 | 0.2705 |

> **Figure placeholder: Held-out method comparison.** Suggested plot: pooled TP, FP, and FN alongside precision, recall, and F1 for the three held-out methods.

Policy B repaired a harmful VLM shift in OP_042 but reverted a useful expansion in OP_032. These opposing clip-level effects produced no aggregate held-out improvement. The held-out result therefore did not reproduce the gain observed on the diagnostic development subset.

## 4. Case Studies

### 4.1 OP_016: proposal degradation and recovery

OP_016 contained seven dense short-duration events, including boundary-truncated calls. Fixed-view qwen3.6 produced 0 TP, 7 FP, and 7 FN. The 0.5 s tiled condition increased recovery to 2 TP but retained 7 FP and 5 FN. BatDetect2 proposal-only produced 6 TP, 0 FP, and 1 FN (F1 = 0.9231), whereas unconstrained metadata-assisted qwen3.6 reduced performance to 1 TP, 5 FP, and 6 FN (F1 = 0.1538). Both proposal preservation and Policy B restored 6 TP, 0 FP, and 1 FN. The left-boundary event remained missing because it was absent from the source proposals.

> **Figure placeholder: OP_016 proposal degradation and recovery.** Suggested panels: proposal-only, unconstrained metadata-assisted prediction, and Policy B output.

### 4.2 OP_045: source proposal extent failure

OP_045 was a simple partial-duration clip containing three events. Fixed-view qwen3.6 matched all three events, whereas PCEN, proposal-only, and Policy B each produced 0 TP, 3 FP, and 3 FN. The BatDetect2 intervals were shorter than the annotation extents, so preserving their geometry did not recover temporal matches. This case differed from OP_016 because the principal error was already present in the source proposals.

### 4.3 OP_004: useful VLM expansion

On OP_004, metadata-assisted qwen3.6 produced 5 TP, 0 FP, and 1 FN (F1 = 0.9091). Full proposal preservation reduced this to 4 TP, 1 FP, and 2 FN (F1 = 0.7273) by removing a useful duration expansion. Policy B retained that anchored expansion and returned to 5 TP, 0 FP, and 1 FN. This development case informed the Policy B rule.

### 4.4 OP_032 and OP_042: contrasting held-out outcomes

In OP_032, proposal-only produced 0 TP, 3 FP, and 3 FN. Unconstrained VLM expansion recovered two events (2 TP, 0 FP, 1 FN; F1 = 0.8000), but Policy B reverted one useful expansion and reduced performance to 1 TP, 1 FP, and 2 FN (F1 = 0.4000).

In OP_042, proposal-only matched all five events. The unconstrained VLM shifted one proposal out of the matching range, producing 4 TP, 1 FP, and 1 FN (F1 = 0.8000). Policy B restored the shifted event and returned to 5 TP, 0 FP, and 0 FN. The two clips demonstrate opposite effects of the same frozen rule on held-out data.

> **Figure placeholder: OP_032 versus OP_042 validator contrast.** Suggested paired panels comparing proposal-only, unconstrained metadata assistance, and Policy B for each clip.

## 5. Results Summary

On the representative development subset, 0.5 s tiled views increased recall from 0.6061 to 0.7576 but reduced mean frequency and box IoU. PCEN did not change pooled F1 and reduced localisation metrics. BatDetect2 proposal-only outperformed unconstrained metadata-assisted qwen3.6, while the latter damaged several accurate proposals.

Policy B had the highest F1 on the representative development subset (0.7188), but this rule was derived from failures observed in those clips. On ten held-out clips, it did not improve the pooled detection counts of metadata-assisted qwen3.6 and slightly reduced mean IoU values. BatDetect2 proposal-only had the highest held-out F1 at 0.8293.

Across both subsets, results varied by failure type. Proposal preservation recovered OP_016 but could not correct the short source intervals in OP_045. Policy B retained a useful expansion in development clip OP_004, repaired a harmful held-out shift in OP_042, and reverted a useful held-out expansion in OP_032.

## 6. Appendix-Style Detailed Metrics

The strict box-IoU counts omitted from the main tables are reported below. They count matched events whose two-dimensional box IoU met each threshold.

| Scope | Method | Box IoU >= 0.3 | Box IoU >= 0.5 |
|---|---|---:|---:|
| Representative-six | Fixed qwen3.6 + grid_v2 | 14 | 4 |
| Representative-six | 0.5 s tiled qwen3.6 | 14 | 2 |
| Representative-six | PCEN qwen3.6 | 10 | 2 |
| Representative-six | BatDetect2 proposal-only | 6 | 4 |
| Representative-six | Metadata-assisted qwen3.6 | 6 | 0 |
| Representative-six | Proposal-preserving validator | 10 | 4 |
| Representative-six | Timing-preserving validator | 10 | 4 |
| Representative-six | Policy B anchored validator | 12 | 4 |
| OP_016 only | 0.25 s tiled qwen3.6 | 0 | 0 |
| Held-out ten | BatDetect2 proposal-only | 13 | 6 |
| Held-out ten | Metadata-assisted qwen3.6 | 10 | 4 |
| Held-out ten | Policy B anchored validator | 10 | 4 |

## 7. Discussion Points to Carry Forward

The broader implications of these results should be developed in the Discussion chapter rather than treated as additional Results. The principal points to carry forward are that detector proposals can provide useful timing evidence, unconstrained VLM refinement can alter correct geometry, and fixed deterministic rules can both repair and suppress useful changes. These observations motivate duration-normalised validation, explicit proposal provenance, and separate treatment of translation and extent expansion [TODO: cite tool-use and validation literature].

The Discussion should also consider uncertainty handling and escalation. Cases such as OP_032 and OP_042 indicate that the current fixed thresholds are not sufficient to decide reliably between preservation and refinement. Human review or a carefully evaluated critic/referee mechanism may be appropriate for unresolved conflicts, but these approaches were not tested in the experiments reported here [TODO: cite uncertainty calibration, human-in-the-loop, and critic/referee literature]. No claim is therefore made in this chapter about their effectiveness.
