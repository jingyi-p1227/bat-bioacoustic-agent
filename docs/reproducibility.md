# Reproducibility Guide

This guide describes the repository-level steps needed to inspect and, where
the required external data and model access are available, reproduce the code,
prompts, evaluation logic and dissertation-facing summaries. Exact
reproduction may depend on access to external datasets and API-based models.

## Environment Setup

The project uses Python with dependencies managed by `uv`.

- Python version: see `.python-version` and `pyproject.toml`.
- Package manager: `uv`.
- Dependency specification: `pyproject.toml`.
- Lock file: `uv.lock`.

Clone the repository, then install the Python environment:

```bash
uv sync
```

## Running Tests

Run the test suite with:

```bash
uv run pytest
```

The tests cover core parsing, evaluation, data-preparation helpers, validator
logic and selected experiment utilities. They do not require access to private
API keys.

## Repository Organisation

The repository is organised around code, experiment entry points,
configuration, prompts, tests and documentation:

- `src/` contains reusable package code.
- `scripts/` contains experiment runners, evaluation entry points, analysis
  scripts, data-preparation utilities, diagnostics and visualisation tools.
- `configs/` contains versioned experiment and model configuration files where
  applicable.
- `prompts/` contains prompt templates that define agent behaviour.
- `tests/` contains the pytest test suite and small fixtures.
- `docs/` contains project documentation, audits, dissertation drafts,
  reference notes and public-facing result summaries.
- `experiments/` contains the experiment registry.

Some legacy root-level scripts are retained for reproducibility of frozen
experiments, while newer entry points are organised under `scripts/`.

## Data Requirements

Raw datasets are external and are not included in this public repository.
Original audio recordings and external annotation datasets should be obtained
from their original providers and used under the relevant access conditions.
Expected data paths should be configured locally for the relevant scripts.

Generated audio clips, spectrogram images, model inputs, diagnostic overlays
and intermediate outputs are also not tracked in Git. Relevant data-processing
and input-generation scripts are provided under:

```text
scripts/data_prep/
scripts/analysis/
scripts/visualization/
```

Dataset and benchmark construction decisions are documented in the dissertation
Methods section and in the dataset audit notes under `docs/`.

## Model Configuration

### Qwen3.6

Qwen3.6 experiments use local or tunnelled inference through Ollama. The
Ollama endpoint is configured outside the repository, typically with an
environment variable such as `OLLAMA_HOST`.

### GPT-5.6 Sol

GPT-5.6 Sol experiments use API-based inference. API credentials are required
and must be provided through local environment variables. Credentials are not
stored in the repository and should not be committed.

## Experiment Reproduction

Experiment reproduction is organised through configuration files, prompts,
evaluation scripts and analysis utilities:

- `configs/` defines experiment settings where applicable.
- `prompts/` defines agent behaviour for inference workflows.
- `experiments/registry.yaml` records selected experiment identifiers, status,
  model information and output locations.
- `scripts/evaluation/` calculates localisation, classification and joint-task
  metrics.
- `scripts/analysis/` generates summaries, comparison tables and diagnostic
  reports.

The evaluation code covers event localisation metrics, including temporal IoU,
frequency IoU, two-dimensional box IoU and start-time proximity; species
classification metrics, including accuracy, macro-F1, balanced accuracy,
per-species metrics and confusion matrices; and joint localisation plus
classification outcomes.

Evaluation and analysis entry points are mainly under:

```text
scripts/evaluation/
scripts/analysis/
src/toy_audio_agent/evaluation/
```

## Output Policy

Generated outputs are not tracked in Git. This includes raw model responses,
prediction files, generated spectrogram images, diagnostic figures and full
evaluation output directories.

Public dissertation-facing summaries are stored under:

```text
docs/results/
```

These summaries provide the main reported results without requiring large
generated artefacts to be committed to the repository.
