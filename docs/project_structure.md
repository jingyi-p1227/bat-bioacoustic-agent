# Project structure

## Source and configuration

- Root-level Python files are legacy experiment runners and reusable modules.
- `src/toy_audio_agent/` is the destination for newly reusable package code.
- `scripts/` contains new experiment, evaluation and analysis entry points.
- `prompts/` contains versioned prompts used for model inference.
- `configs/` contains model and experiment configurations.

## Data

- `annotations/` contains source or normalised annotations.
- `audio/` contains source audio and generated clips.
- `ground_truth/` contains evaluation ground truth.
- Existing data paths are retained for compatibility with frozen experiments.

## Runtime knowledge

- `docs/annotation_example_library/annotation_memory.jsonl` is currently a
  runtime annotation-memory store.
- `docs/literature_reference_library/verified_evidence_store.jsonl` is
  currently a runtime verified-evidence store.
- These paths are retained until all experiments are frozen.

## Generated outputs

- `outputs/evaluation_sets/` contains processed evaluation datasets.
- `outputs/agent_inputs/` contains generated model inputs.
- `outputs/agent_runs/` contains inference and evaluation outputs.
- `outputs/tool_outputs/` contains external tool outputs.
- `outputs/analysis_reports/` contains consolidated analyses.

## Documentation

- `docs/dataset_audits/` contains dataset audits.
- `docs/dissertation_drafts/` contains current manuscript drafts.
- `docs/experiment_notes/` will contain phase-specific experiment records.

## Migration policy

Before the final experiment freeze:
- do not move existing data;
- do not rename frozen output directories;
- do not move legacy runners;
- place all new reusable code under `src/`;
- place all new experiment entry points under `scripts/`.

After the final experiment freeze, legacy files may be migrated gradually
with compatibility wrappers and regression tests.