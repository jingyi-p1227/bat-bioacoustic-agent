# Tool-constrained Multimodal Agents for Bat Bioacoustic Annotation

**Disentangling Event Localisation and Species Classification**

This repository supports the MSc dissertation project:

> Can AI Agents Annotate Animal Sounds Reliably?

## Overview

Large-scale bioacoustic monitoring requires extensive expert annotation
of sound recordings. This project investigates whether tool-constrained
multimodal language model (MLLM)-based agents can support bat
bioacoustic annotation from spectrogram representations.

Rather than replacing specialist bioacoustic models, this work explores
how general-purpose multimodal models can be integrated with
domain-specific tools in a human-in-the-loop annotation workflow.

The annotation problem is separated into:

-   **Event localisation:** identifying the time-frequency boundaries of
    individual bat echolocation calls.
-   **Species classification:** assigning species identities from
    spectrogram evidence.

## Research Questions

**RQ1:** Can multimodal language model agents localise bat echolocation
events from spectrograms?

**RQ2:** Can multimodal language model agents classify bat species from
spectrogram evidence?

**RQ3:** How do specialist detection tools, stronger models and
multi-agent workflows affect annotation reliability?

## Method Overview

``` text
Audio recordings
        ↓
Spectrogram generation
        ↓
Specialist detector proposals (BatDetect2)
        ↓
Structured tool outputs
        ↓
MLLM interpretation and reasoning
        ↓
Annotation evaluation
```

The repository includes:

-   spectrogram preparation utilities;
-   BatDetect2 proposal processing;
-   structured MLLM annotation workflows;
-   localisation and classification evaluation;
-   failure analysis tools.

## Models

Primary model:

-   Qwen3.6 through Ollama-based inference.

Comparison model:

-   GPT-5.6 Sol through API-based inference.

## Repository Structure

``` text
src/          Core implementation
scripts/      Experiment runners and analysis utilities
configs/      Model and experiment configurations
prompts/      Prompt templates
experiments/  Experiment records
tests/        Automated tests
docs/         Documentation
```

## Installation

This project uses `uv` for environment management.

``` bash
uv sync
```

Run tests:

``` bash
uv run pytest
```

## Reproducibility

The repository provides:

-   experiment configurations;
-   prompt templates;
-   evaluation scripts;
-   analysis utilities.

Raw datasets and generated model outputs are not included.

## Key Results

-   Specialist detector proposals improved call localisation.
-   Species identification remained the main bottleneck.
-   Stronger model capability provided larger gains than adding agent
    roles.
-   Agent workflows are most promising as human-in-the-loop annotation
    assistants.

## Data Availability

The experiments use recordings derived from the BatDetect2 dataset (Mac Aodha et al., 2022).

The original recordings are not included in this repository. This repository contains selected implementation components for preprocessing, MLLM-based annotation workflows and evaluation.

Users should obtain the original dataset from the official source according to the dataset licence.

## Citation

Peng, J. (2026).

*Tool-constrained Multimodal Agents for Bat Bioacoustic Annotation:
Disentangling Event Localisation and Species Classification.*

MSc Dissertation, University College London.
