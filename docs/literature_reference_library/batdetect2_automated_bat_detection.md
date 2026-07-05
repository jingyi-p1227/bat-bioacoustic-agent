# BatDetect2 and Automated Bat Call Detection

## Citation

**TODO:** add the verified BatDetect2 software citation and, if applicable, the associated paper citation. The current repository confirms use of BatDetect2 version `1.3.1`, but does not contain a citable author list, year, venue, DOI, or software archive record.

## Summary

Within this project, BatDetect2 is an external automated detector that produces candidate call intervals, frequency bounds, detection probabilities, class probabilities, and UK-taxonomy labels. The dissertation deliberately uses these outputs as generic `bat_call` proposals because the evaluation species is Australian and the model taxonomy is not treated as reliable species truth. Held-out evaluation shows that proposal-only detection can be a strong baseline, while free-form VLM refinement can either improve under-wide proposals or damage accurate timing.

## Key Methodological Idea

Use a specialist detector to generate structured candidate regions and confidence metadata, then preserve provenance so downstream systems can accept, reject, or refine each proposal without confusing it with ground truth.

## Relevance to This Dissertation

BatDetect2 provides the concrete tool-use condition for P6D-P6E. It makes detector-only, VLM-only, unconstrained hybrid, and validated hybrid conditions directly comparable under the same event-level protocol.

## How It Supports the Project Argument

The results support a constrained-tool argument: specialist proposals offer useful timing priors, but neither detector geometry nor VLM refinement should be trusted unconditionally. Reliable annotation requires provenance, validation, uncertainty handling, and review.

## Dissertation Use

- **Introduction/Related work:** automated bat detection as a scalable alternative to manual screening.
- **Methods:** software version, threshold `det_prob >= 0.30`, generic-label treatment, and proposal conversion.
- **Results:** proposal-only and metadata-assisted comparisons.
- **Discussion:** taxonomy transfer limits and proposal-versus-final-label distinction.

## Local Evidence

- `outputs/tool_outputs/batdetect2_proposals/representative6/raw_batdetect2/run_metadata.json`
- `outputs/agent_runs/p6_batdetect2_proposal_only_representative6/evaluation/`
- `outputs/agent_runs/p6e5_batdetect2_proposal_only_heldout/evaluation/`

## TODO

- Obtain the official software/paper citation from an authoritative BatDetect2 source.
- Verify the model's intended taxonomy and training-data description before making literature claims.
