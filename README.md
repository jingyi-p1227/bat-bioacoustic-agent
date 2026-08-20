# Tool-constrained Vision-Language Agents for Bat Bioacoustic Annotation

**Disentangling Event Localisation and Species Classification**

This repository supports the MSc dissertation project:

> Can AI Agents Annotate Animal Sounds Reliably?

## Overview

This repository contains code, prompts, configuration files and evaluation
utilities for an MSc dissertation investigating whether vision-language agents
can support bat bioacoustic annotation from spectrogram representations.

The project focuses on two related but distinct annotation problems:

- event localisation: identifying the time-frequency extent of bat
  echolocation calls;
- species classification: assigning species labels from spectrogram evidence.

The work studies tool-constrained agent workflows rather than fully autonomous
annotation. Specialist tools such as BatDetect2 are used as structured evidence
providers, while vision-language models are evaluated as interpreters,
verifiers and classifiers. Outputs are represented in structured formats and
assessed using reproducible evaluation scripts.

## Research Questions

**RQ1:** Can VLM-based agents localise bat echolocation events from
spectrograms?

**RQ2:** Can VLMs classify bat species from spectrogram evidence?

**RQ3:** How do stronger models and multi-agent workflows affect annotation
reliability?

## Method Overview

The experimental pipeline follows this general structure:

```text
Audio recordings
-> spectrogram generation
-> specialist detector proposals (BatDetect2)
-> structured tool output
-> vision-language model interpretation
-> validated annotation results
```

The repository includes tooling for:

- generating clean spectrogram inputs;
- converting and using BatDetect2 proposal metadata;
- running structured VLM annotation and classification prompts;
- evaluating event localisation, species classification and joint outcomes;
- analysing failures, proposal deviations and uncertainty behaviour.

## Models

The primary model used in the local experiments is:

- Qwen3.6, accessed through Ollama for local or tunnelled inference.

The comparison model used for stronger-model analysis is:

- GPT-5.6 Sol, accessed through API-based inference.

Model and experiment settings are recorded under `configs/` where applicable.
API keys and local environment files are not included in the repository.

## Repository Structure

```text
src/
scripts/
configs/
prompts/
tests/
docs/
experiments/
```

- `src/` contains reusable package code.
- `scripts/` contains experiment runners, evaluation entry points, analysis
  scripts, data-preparation utilities, diagnostic tools and visualisation
  scripts.
- `configs/` contains versioned experiment and model configuration files.
- `prompts/` contains prompt templates used in the annotation workflows.
- `tests/` contains the pytest test suite and small fixtures.
- `docs/` contains project documentation, audits, dissertation drafts,
  reference notes and public-facing result summaries.
- `experiments/` contains the experiment registry.

Some legacy root-level scripts are retained for reproducibility of earlier
frozen experiments.

## Installation

This project uses `uv` for Python environment management.

```bash
uv sync
```

Run the test suite with:

```bash
uv run pytest
```

Some inference scripts require a configured local model backend or API
credentials. These are expected to be provided through local environment
variables and are not stored in the repository.

## Reproducibility

The repository provides:

- experiment configurations;
- prompt templates;
- evaluation scripts;
- analysis utilities;
- dissertation-facing result summaries.

Detailed reproducibility notes are provided in:

```text
docs/reproducibility.md
```

Generated model inputs, raw model outputs and large experiment artifacts are
not intended to be committed directly to Git.

## Data Availability

Raw audio datasets are not redistributed in this repository. Original datasets
should be obtained from their source providers under the relevant access and
licensing conditions.

This repository provides the code, configurations, prompts and analysis
framework used for the dissertation. Where data cannot be redistributed, the
repository records the expected directory structure and processing scripts
needed to reproduce the experiments from locally available data.

Additional notes are provided in:

```text
docs/data_availability.md
```

## Results

Detailed experiment summaries and analysis reports are provided in:

```text
docs/results/
```

These summaries cover the final single-agent localisation experiments,
multi-species classification and joint-task experiments, stronger-model
comparison, and multi-agent pilot analysis.

## Citation

Peng, J. (2026).  
*Tool-constrained Vision-Language Agents for Bat Bioacoustic Annotation:
Disentangling Event Localisation and Species Classification.*  
MSc Dissertation, University College London.
