# Final Single-Agent Localisation Results

## Scope

This report summarises the completed full-45 single-agent localisation experiments for the `Ozimops petersi` benchmark. All rows use frozen prediction outputs from `outputs/agent_runs/` and ground truth from the frozen evaluation set. No model inference, ground-truth editing, prompt editing, or prediction modification was performed while generating this report.

The comparison covers eight completed conditions:

- grid_v2 baseline;
- PCEN + grid_v2;
- 0.5s tiled spectrograms;
- BatDetect2 proposal-only;
- previous BatDetect2 proposal-constrained VLM;
- Walters PDF-backed generic acoustic guidance;
- source-recording-safe annotation exemplars;
- P14 conservative best-stack.

## Evaluation Protocols

Three detection protocols are reported:

1. **Temporal IoU >= 0.3**: primary event-level localisation protocol used for the main comparison.
2. **Temporal IoU >= 0.1**: relaxed temporal-overlap sensitivity protocol.
3. **10 ms start-time proximity**: onset-sensitive protocol following the cached BatDetect2-style start-time matching audit.

The main table reports TP, FP, FN, precision, recall, and F1 under temporal IoU >= 0.3. It also reports F1 under temporal IoU >= 0.1 and 10 ms start-time matching. Mean temporal IoU, mean frequency IoU, and mean 2D box IoU are calculated over matched pairs under the primary temporal IoU >= 0.3 protocol.

## Main Results

| Condition | TP | FP | FN | P | R | F1 @ IoU>=0.3 | F1 @ IoU>=0.1 | F1 @ 10ms | Time IoU | Freq IoU | Box IoU | Parse | Invalid boxes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid_v2 baseline | 134 | 113 | 64 | 0.543 | 0.677 | 0.602 | 0.701 | 0.333 | 0.581 | 0.526 | 0.332 | 45/45 | 0 |
| PCEN + grid_v2 | 128 | 148 | 70 | 0.464 | 0.646 | 0.540 | 0.692 | 0.426 | 0.554 | 0.463 | 0.271 | 45/45 | 0 |
| 0.5s tiled | 146 | 196 | 52 | 0.427 | 0.737 | 0.541 | 0.626 | 0.556 | 0.628 | 0.512 | 0.336 | 45/45 | 0 |
| BatDetect2 proposal-only | 145 | 62 | 53 | 0.700 | 0.732 | 0.716 | 0.884 | 0.884 | 0.458 | 0.576 | 0.283 | 45/45 | 0 |
| previous proposal-constrained VLM | 157 | 45 | 41 | 0.777 | 0.793 | 0.785 | 0.890 | 0.830 | 0.461 | 0.607 | 0.292 | 45/45 | 0 |
| Walters acoustic | 121 | 125 | 77 | 0.492 | 0.611 | 0.545 | 0.707 | 0.338 | 0.601 | 0.540 | 0.345 | 45/45 | 0 |
| annotation exemplars | 122 | 129 | 76 | 0.486 | 0.616 | 0.543 | 0.704 | 0.334 | 0.588 | 0.544 | 0.345 | 45/45 | 0 |
| P14 conservative best-stack | 144 | 50 | 54 | 0.742 | 0.727 | 0.735 | 0.929 | 0.913 | 0.460 | 0.601 | 0.296 | 45/45 | 0 |

## Interpretation

The primary best result under temporal IoU >= 0.3 is the **previous BatDetect2 proposal-constrained VLM** condition, with F1 = `0.785`. This improves over BatDetect2 proposal-only (`0.716`) and the grid_v2 baseline (`0.602`). The result supports the interpretation that the VLM works best as a proposal verifier/refiner rather than as a free-form detector.

The **P14 conservative best-stack** is not the best condition under temporal IoU >= 0.3, where its F1 is `0.735`. However, it is the strongest condition under the 10 ms start-time protocol, with F1 = `0.913`. This suggests that the conservative instruction successfully preserved proposal onset timing, but at the cost of lower broader temporal-overlap performance.

BatDetect2 proposal-only remains a strong baseline, especially under the 10 ms onset-sensitive protocol. It achieves F1 = `0.884` under 10 ms matching, showing that detector proposals provide strong timing priors even before VLM refinement.

PCEN + grid_v2 does not improve aggregate localisation over the grid_v2 baseline under the primary IoU >= 0.3 protocol. It should remain an onset-sensitive diagnostic representation rather than a replacement for the main visual input.

The 0.5s tiled condition increases recall under IoU >= 0.3 (`0.737`) but creates too many false positives (`196`), lowering F1 to `0.541`. Tiling is therefore not recommended as the full45 default.

Walters acoustic guidance and annotation exemplars do not improve aggregate localisation F1 over the grid_v2 baseline. Both conditions show relatively high matched-box quality among detected matches, but their lower TP counts and higher missed-event counts mean they should not be included in the final localisation stack without further redesign.

## Final Position

For dissertation reporting, the best primary localisation result is the previous BatDetect2 proposal-constrained VLM condition. The P14 conservative condition should be reported separately as an onset-preserving variant that gives the strongest 10 ms start-time result. The final single-agent localisation conclusion is that reliable performance comes from constrained tool use: BatDetect2 supplies strong candidate timing, and qwen3.6 is most useful when verifying and refining these proposals rather than independently detecting all calls.
