# Tool-constrained Multimodal Agents for Bat Bioacoustic Annotation

**Disentangling Event Localisation and Species Classification**

This repository supports the MSc dissertation project:

> Can AI Agents Annotate Animal Sounds Reliably?

## Overview

This project investigates whether multimodal large language model (MLLM)-based agents can support automated bat bioacoustic annotation from spectrogram representations.

Rather than replacing specialist bioacoustic detectors, the work explores tool-constrained agent workflows where dedicated detection models provide structured evidence and MLLMs perform interpretation, verification and classification.

The study separates bat annotation into two related tasks:

- **Event localisation:** identifying the time-frequency boundaries of individual echolocation calls.
- **Species classification:** assigning species identities from spectrogram evidence.

## Research Questions

**RQ1:** Can MLLM-based agents localise bat echolocation events from spectrograms?

**RQ2:** Can MLLM-based agents classify bat species from spectrogram evidence?

**RQ3:** How do specialist tools, stronger models and multi-agent workflows affect annotation reliability?

## Method Overview

The experimental workflow follows:

```text
Audio recordings
        ↓
Spectrogram generation
        ↓
Specialist detector proposals (BatDetect2)
        ↓
Structured tool outputs
        ↓
Multimodal model reasoning
        ↓
Validated annotations and evaluation
```

The repository contains:

- spectrogram preparation and preprocessing utilities;
- BatDetect2 proposal processing tools;
- structured MLLM annotation workflows;
- localisation, classification and joint-task evaluation scripts;
- failure analysis and visualisation utilities.

## Models

Experiments evaluate multimodal language models under different annotation settings.

Primary local model:

- Qwen3.6 through Ollama-based inference.

Additional comparison model:

- GPT-5.6 Sol through API-based inference.

Model configurations and experiment settings are stored in `configs/` where applicable. Credentials and private environment files are excluded.

## Repository Structure

```text
src/          Core implementation
scripts/      Experiment runners and analysis utilities
configs/      Model and experiment configurations
prompts/      Prompt templates
experiments/  Experiment records
results/      Generated summaries (when included)
tests/        Automated tests
docs/         Documentation and reproducibility notes
```

## Installation

The project uses `uv` for environment management.

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Some experiments require local model backends or API credentials, which should be configured separately.

## Reproducibility

The repository provides:

- experiment configurations;
- prompt templates;
- evaluation scripts;
- analysis utilities;
- documentation of experimental procedures.

Raw audio datasets, generated model outputs and large intermediate files are not included.

## Data Availability

Raw recordings are not redistributed in this repository. Users should obtain datasets from their original sources according to relevant access and licensing conditions.

The repository focuses on the annotation framework, evaluation pipeline and analysis workflow.

## Results

Experimental results and analysis notes are documented under:

```text
docs/
```

The final dissertation evaluates:

- direct MLLM-based call localisation;
- detector-assisted localisation refinement;
- multi-species classification;
- stronger-model comparison;
- multi-agent workflow analysis.

## Citation

Peng, J. (2026).  
*Tool-constrained Multimodal Agents for Bat Bioacoustic Annotation: Disentangling Event Localisation and Species Classification.*  
MSc Dissertation, University College London.
