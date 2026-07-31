# Single-Agent Localisation Key Findings

- Primary best full45 result: previous BatDetect2 proposal-constrained VLM, F1 = `0.785` at temporal IoU >= 0.3.
- Strongest onset-sensitive result: P14 conservative best-stack, F1 = `0.913` under 10 ms start-time matching.
- BatDetect2 proposal-only is a strong detector baseline and provides reliable onset priors.
- qwen3.6 works best as a verifier/refiner of detector proposals, not as a free-form detector.
- PCEN + grid_v2 should remain diagnostic only for this benchmark.
- 0.5s tiling improves recall but causes too many false positives.
- Walters acoustic guidance and annotation exemplars do not improve aggregate full45 localisation F1.
- P14 trades lower primary IoU>=0.3 F1 for much stronger onset preservation.
