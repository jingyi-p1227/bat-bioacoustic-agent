# Evaluation Metrics, Calibration, and Uncertainty

## Citation

**TODO:** add verified sources for precision/recall/F1, intersection-over-union in localisation, probabilistic calibration, and uncertainty evaluation. No metric references are currently stored in the repository.

## Summary

The project evaluates one-to-one event matching at temporal IoU `>=0.30`, then reports TP, FP, FN, precision, recall, F1, mean temporal IoU, mean frequency IoU, mean box IoU, and strict box-IoU counts. Confidence and review fields are retained, but formal calibration metrics have not yet been computed. Literature citations are needed for general metric definitions and for any claim that a threshold or confidence is calibrated.

## Key Methodological Idea

Separate detection quality from localisation quality and confidence quality. F1 describes event retrieval, IoU describes geometry, and calibration requires its own comparison between stated confidence and observed outcomes.

## Relevance to This Dissertation

This separation explains several important results: tiled views improve recall while box localisation remains weaker; BatDetect2 can identify the right candidate region but fail the chosen temporal extent; and validators can preserve TP while changing box IoU.

## How It Supports the Project Argument

Reliable annotation cannot be reduced to one aggregate score. The system needs event-level errors, geometry metrics, confidence, provenance, and review decisions.

## Dissertation Use

- **Methods:** metric definitions, one-to-one matching, and thresholds.
- **Results:** detection/localisation trade-offs.
- **Discussion:** uncertainty, threshold sensitivity, and calibration limitations.

## Local Evidence

- `docs/evaluation_protocol_v1_ozi_petersi.md`
- `evaluation.py`
- `evaluate_prompt_v2_small_pilot.py`

## TODO

- Add canonical metric citations appropriate to the final discipline and terminology.
- Do not describe model confidence as calibrated unless a calibration analysis is added.
