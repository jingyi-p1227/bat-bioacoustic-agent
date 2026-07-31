# Key Findings: Multi-Species Classification and Joint Task

1. The Stage 1 dataset was label-safe and balanced: 8 species, 240 samples, 30 samples per species, 800x600 model-facing images, and no embedded species-label leakage.

2. Zero-shot species classification was weak. Stage 1A achieved accuracy `0.129` and macro-F1 `0.051`; Stage 1B with a neutral GT box marker improved only slightly to accuracy `0.138` and macro-F1 `0.074`.

3. Compact species/acoustic guidance helped, but did not solve the task. Stage 1C reached accuracy `0.192`, macro-F1 `0.105`, and balanced accuracy `0.192`.

4. Coarse-taxonomy evaluation improved Stage 1C to coarse macro-F1 `0.179`, with better Rhinolophus recognition, but Myotis and Ozimops remained poorly recovered.

5. Raw BatDetect2 sample-level proposals had high recall but extremely high false-positive count: full240 IoU>=0.3 recall `0.879`, precision `0.175`, and F1 `0.292`.

6. The nearest-centre proposal rule solved localisation well in the event-centred setup. On full240 it achieved IoU>=0.3 F1 `0.888` and 10 ms start-time F1 `0.935`.

7. Stage 2B showed that letting qwen3.6 filter and refine all proposals was harmful: pilot80 IoU>=0.3 F1 was only `0.232`, with `247` false positives.

8. Stage 2C fixed proposal selection by preserving nearest-centre proposal coordinates. Full240 localisation remained strong, with IoU>=0.3 F1 `0.888`.

9. Species classification remained the main bottleneck. Stage 2C full240 matched-proposal species accuracy was only `0.109`, macro-F1 `0.032`, and joint F1 `0.097`.

10. The final interpretation is that event-centred proposal localisation can be made reliable with deterministic tooling, but qwen3.6 does not reliably perform fine-grained species identification from these event-level spectrogram crops.
