# PCEN and Audio Preprocessing

## Citation

**TODO:** add and verify the original or an authoritative PCEN citation. No PCEN paper is currently present in the repository, so authors, year, venue, and claims must not be inferred here.

## Summary

Per-channel energy normalization was tested in this project as an alternate spectrogram representation intended to change local contrast under varying background energy. The project generated PCEN images with recorded parameters and evaluated them under the same qwen3.6 prompt and matching protocol. On the representative six, PCEN did not improve aggregate F1 over the fixed dB baseline and reduced mean box IoU; OP_045 regressed from a fixed-view success to complete failure.

## Key Methodological Idea

Treat preprocessing as a controlled input intervention: preserve coordinate axes, record parameters, hold model/prompt/evaluation constant, and measure both gains and clean-case regressions.

## Relevance to This Dissertation

PCEN is the main non-tiled preprocessing ablation. It demonstrates that visual enhancement to a human observer does not automatically improve VLM localisation.

## How It Supports the Project Argument

The negative result supports evidence-based tool selection. A representation should become an agent tool only when it improves task metrics without unacceptable regressions.

## Dissertation Use

- **Methods:** PCEN generation and controlled input condition.
- **Results:** fixed dB versus PCEN comparison.
- **Discussion:** representation quality is model- and task-dependent.

## Local Evidence

- `outputs/agent_inputs/p6_pcen_spectrograms/pcen_manifest.csv`
- `outputs/agent_runs/p6_pcen_qwen3_6_representative6/evaluation/`
- `outputs/analysis_reports/p6_single_agent_tool_use_summary/p6_consolidated_metrics.csv`

## TODO

- Add the verified PCEN citation before describing its published motivation or formula.
- Check final parameter notation against the cited source.
