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

## Migration Batches Completed

The first low-risk migration batch moved selected utility scripts out of the
repository root:

- deterministic validators and repair helpers moved to `scripts/maintenance/`;
- plotting, overlay and contact-sheet utilities moved to `scripts/visualization/`;
- diagnostic comparison and smoke-test summary utilities moved to
  `scripts/diagnostics/`.

The second low-risk migration batch moved data-preparation scripts out of the
repository root:

- dataset construction scripts moved to `scripts/data_prep/`;
- AOEF, NPZ/WAV and BatDetect2 proposal conversion scripts moved to
  `scripts/data_prep/`;
- clean spectrogram, PCEN and tiled input-generation scripts moved to
  `scripts/data_prep/`.

The remaining root-level scripts are still treated as legacy-active and should
be migrated gradually.

## What To Move Next

The safest immediate actions after this batch are small, test-backed migrations
only:

1. Keep high-risk legacy-active runners in place until their dependencies are
   isolated.
2. Move another small family only after mapping tests and frozen experiment
   references.
3. Consider compatibility wrappers only for scripts that are still documented as
   root-level commands.

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
4. **Data-prep migration completed**: `build_*`, `convert_*`, `prepare_*`, and
   `split_audio_clips.py` now live under `scripts/data_prep/`.
5. **Migrate remaining low-risk diagnostics**: only after confirming no frozen
   historical command references need compatibility wrappers.
6. **Package reusable validation logic**: move reusable internals from
   maintenance scripts into `src/toy_audio_agent/` when they are needed by more
   than one active workflow.
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
