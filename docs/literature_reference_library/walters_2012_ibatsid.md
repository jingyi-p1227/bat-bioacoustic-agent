# Literature Reference Card: Walters et al. 2012 (iBatsID)

## Citation

Walters, C. L., et al. (2012). "A continental-scale tool for acoustic identification of European bats."

Citation details, journal, volume, pages, and DOI: **to be verified and added from the publisher record before formal submission**.

## Short Summary

This paper presents a continental-scale approach to objective acoustic identification of European bats. It connects a standardized reference-call library with extracted call parameters and a hierarchical identification procedure, allowing acoustic observations to be classified at an appropriate taxonomic or call-type level. Its emphasis is not only on producing a label, but also on standardizing the evidence, handling uncertain cases, and making automated identification usable as a repeatable scientific tool.

## Relevance to This Project

The current project studies whether a vision-language model can produce event-level strong labels from spectrograms. Walters et al. provides a useful historical and methodological reference because it treats bat acoustic identification as a structured tool workflow grounded in reference examples, measurable call properties, and explicit decision boundaries.

It also helps position this project correctly. The goal is not to replace specialist acoustic methods with an unconstrained language model. The goal is to test whether a tool-augmented single agent can produce traceable candidate annotations while preserving uncertainty and supporting human review.

## Useful Concepts

### Objective and Standardized Bat Acoustic Identification

Acoustic identification should use repeatable rules and measurable evidence rather than undocumented visual judgement. For this project, prompt rules, preprocessing settings, model versions, and evaluation thresholds should remain explicit and reproducible.

### Reference Call Library

A curated reference library links representative calls to trusted labels and known recording context. This motivates maintaining a project-level annotation example library containing clean successes, dense-call cases, boundary cases, and known failures.

### Call-Type Hierarchy

Identification may be more reliable at different levels of specificity. A system should be able to distinguish an individual call event even when a fine-grained taxonomic label is uncertain. Future multi-species work should therefore separate event detection confidence from species-label confidence.

### Call Parameter Extraction

Time and frequency properties provide interpretable evidence for acoustic identification. In this project, the corresponding measurable outputs are event start/end time, lower/upper frequency bounds, temporal IoU, frequency IoU, and two-dimensional box IoU.

### Uncertainty and Threshold Handling

Automated identification should expose uncertainty and use thresholds deliberately. This supports retaining `confidence`, `human_review_needed`, and review reasons in model output, while treating matching and quality thresholds as evaluation configuration rather than hidden assumptions.

### Tool-Based Acoustic Identification

The paper frames identification as a practical tool supported by reference data and structured measurements. This is directly relevant to the project's overview-first workflow and proposed preprocessing tools, where the agent may inspect alternative representations without receiving ground-truth information.

## How It Informs This Project

### Annotation Example Library

Build a small, curated set of examples with provenance, case type, figures, model behaviour, and lessons. Examples should support analysis and prompt development without being confused with the frozen evaluation ground truth.

### Literature-Grounded Prompt Rules

Prompt instructions should reflect defensible annotation principles: one call per event, measurable time-frequency boundaries, explicit uncertainty, and no unsupported extrapolation beyond visible evidence.

### Pydantic Evaluators

Structured evaluators can check geometry, confidence ranges, threshold decisions, and review flags alongside domain metrics from `evaluation.py`. Literature-derived rules should be implemented as transparent evaluators, not folded into opaque aggregate scores.

### Preprocessing and Tool-Augmented Single-Agent Workflow

P6 should test whether alternate representations improve the visibility and localisation of calls. PCEN-like enhancement, denoising or band-pass views, and cropped/tiled spectrograms should be treated as controlled tools. Their value must be measured against the current `grid_v2` baseline using the frozen evaluation protocol.

## Follow-Up

- Verify the complete bibliographic citation from the publisher record.
- Extract the exact call hierarchy, feature set, and uncertainty procedure during a full paper review.
- Record which concepts become prompt rules, evaluator checks, or preprocessing hypotheses.

