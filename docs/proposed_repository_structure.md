# Proposed Repository Structure

Date: 2026-07-31

This proposal is based on the no-move repository inventory audit. It does not change `docs/project_structure.md` yet.

## Current Structure Summary

- `src/toy_audio_agent/` exists but contains only a small subset of reusable code.
- `scripts/inference/`, `scripts/evaluation/`, and `scripts/analysis/` contain the newer organized experiment entry points.
- Many root-level Python files are legacy experiment scripts or reusable helpers that are still imported by tests and newer scripts.
- `outputs/` contains large generated artifacts and is correctly ignored.
- `docs/results/` contains dissertation-facing outputs, but the generic `results/` ignore rule means some present summaries are not tracked.

## Recommended Final Structure

```text
toy-audio-agent-main/
  README.md
  pyproject.toml
  uv.lock
  configs/
    experiments/
    models/
  docs/
    project_structure.md
    repository_inventory_audit.md
    results/
    dissertation_drafts/
    annotation_example_library/
    literature_reference_library/
    acoustic_reference_library/
    dataset_audits/
  experiments/
    registry.yaml
  prompts/
  scripts/
    inference/
    evaluation/
    analysis/
    data_prep/
    visualization/
    diagnostics/
    maintenance/
    legacy_root/
  src/toy_audio_agent/
    evaluation/
    experiments/
    schemas/
    retrieval/
    audio/
    plotting/
  tests/
    fixtures/
  outputs/        # ignored generated artifacts
```

## Recommended New Script Subdirectories

- `scripts/data_prep/`: dataset/input/proposal generation and conversion scripts.
- `scripts/visualization/`: plotting and diagnostic overlay/contact-sheet scripts.
- `scripts/diagnostics/`: model smoke tests, backend checks, and one-off diagnostic comparisons.
- `scripts/maintenance/`: repair, validation, deterministic post-processing, and repository audit utilities.
- `scripts/legacy_root/`: root-level historical scripts that are not yet worth fully migrating but should leave the root eventually.

## What To Move Now

Move nothing automatically. The safest immediate actions are Git-hygiene changes only:

1. Add `*.egg-info/` to `.gitignore`.
2. Remove tracked `toy_audio_agent.egg-info/` from Git after confirming no packaging reason to keep it.
3. Decide whether all `docs/results/` markdown/CSV summaries should be tracked, then add explicit `.gitignore` exceptions if yes.
4. Keep root-level legacy scripts in place until import wrappers or package modules exist.

## What To Leave Alone Until After Dissertation

- Frozen output directories under `outputs/`.
- Legacy root-level experiment runners that reproduce dissertation results.
- `main.py`, because it still contains the demo app plus shared models/audio helpers.
- `run_prompt_v2_small_pilot.py`, because newer scripts import helper functions from it.
- Evaluation protocol scripts whose outputs are cited in dissertation-facing reports.
- Prompt files under `prompts/`, unless a new experiment explicitly versions a new prompt.

## Safest Migration Order

1. **Git hygiene commit**: `.gitignore` fixes, remove package metadata, ensure `.env` and `outputs/` stay ignored.
2. **Documentation commit**: commit repository inventory, cleanup audit, and selected `docs/results/` summaries.
3. **Package reusable models/evaluators**: migrate `event_characterisation_models.py`, `event_characterisation_evaluators.py`, `event_characterisation_retrieval.py`, and `extract_event_characterisation_features.py` into `src/toy_audio_agent/` with compatibility wrappers.
4. **Move data-prep scripts**: migrate `build_*`, `convert_*`, `prepare_*`, and `split_audio_clips.py` into `scripts/data_prep/`; update tests.
5. **Move plotting scripts**: migrate `plot_*` and spectrogram example scripts into `scripts/visualization/`; update tests/docs.
6. **Move validation utilities**: migrate validator/post-processor scripts into `scripts/maintenance/`; update tests.
7. **Retire or archive legacy runners**: move old one-off root `run_*` scripts only after dissertation results are frozen and registry references are stable.

## Next Refactor Phase Commands

Recommended dry-run/verification commands before moving anything:

```bash
git status --short --branch
git ls-files "*.py"
git grep -n "from run_prompt_v2_small_pilot import"
git grep -n "from event_characterisation_"
git grep -n "import apply_" tests scripts src
uv run pytest -q
```

For each migration batch:

```bash
# 1. Move one small family of files.
# 2. Add compatibility imports or update tests.
uv run pytest -q
git diff --stat
git status --short
```

Avoid broad `git mv` batches until the relevant test imports and frozen experiment references are mapped.
