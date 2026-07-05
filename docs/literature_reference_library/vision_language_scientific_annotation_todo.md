# Vision-Language Models for Scientific Visual Annotation

## Citation

**TODO:** add verified sources on vision-language models, image-grounded structured output, and preferably scientific-image or spectrogram interpretation. The repository contains model run records but no citable VLM papers.

## Summary

The dissertation tests whether a vision-language model can interpret spectrograms as scientific images and return structured event geometry. The experiments show that the model can produce valid JSON and useful boxes, but performance depends on grid style, preprocessing, local views, tool metadata, and workflow constraints. A literature review is needed to distinguish general multimodal capabilities from evidence specifically established for scientific annotation.

## Key Methodological Idea

Combine an image representation with explicit coordinate conventions, a constrained output schema, and task-specific evaluation rather than relying on free-form visual description.

## Relevance to This Dissertation

Qwen3.6 is the primary VLM condition. Its strong fixed-view cases, tiled recall gains, and proposal-refinement failures define both the opportunity and limitation of spectrogram-based VLM annotation.

## How It Supports the Project Argument

The project argues that multimodal models are useful annotators only when their outputs are structured, evaluated, and constrained by domain tools. A verified VLM citation base is needed to position this contribution without overstating general scientific reliability.

## Dissertation Use

- **Related work:** multimodal structured prediction and scientific-image use.
- **Methods:** spectrogram input, prompt schema, and JSON validation.
- **Discussion:** visual grounding errors and coordinate uncertainty.

## TODO

- Add the official citation for the deployed model if permitted and available.
- Add peer-reviewed work on VLM scientific-image annotation or clearly label the gap.
