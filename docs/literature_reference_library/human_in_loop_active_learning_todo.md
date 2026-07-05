# Human-in-the-Loop and Active-Learning Annotation

## Citation

**TODO:** add verified human-in-the-loop and active-learning references relevant to ecological or audio annotation. No suitable citation is currently available in local materials.

## Summary

This theme should explain how automated systems can prioritise uncertain, novel, boundary, or disagreement cases for review rather than treating every prediction equally. The current project already records `human_review_needed`, review reasons, confidence, detector provenance, and validator decisions, but it does not implement an active-learning loop. These fields create the scaffolding for future review triage without proving active-learning benefit.

## Key Methodological Idea

Allocate human attention using explicit uncertainty and disagreement signals, then preserve reviewed examples and decisions as auditable training or evaluation data.

## Relevance to This Dissertation

OP_016 boundary misses, OP_045 source-extent failures, and OP_032 validator disagreement are natural review candidates. New VLM-only events are also marked for review because they lack detector provenance.

## How It Supports the Project Argument

Human review is not merely a fallback after model failure; it is part of a reliable annotation system. A verified literature base would support the proposed transition from unconditional automation to selective review.

## Dissertation Use

- **Introduction/Related work:** human oversight in automated annotation.
- **Methods:** review flags and uncertainty fields.
- **Discussion/Future work:** active-learning or review-prioritisation loop.

## TODO

- Add one general active-learning reference and one domain-relevant annotation reference.
- Avoid claiming annotation-effort reduction until it is measured or supported by cited work.
