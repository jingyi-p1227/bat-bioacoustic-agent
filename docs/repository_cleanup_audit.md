# Repository Cleanup Audit

Date: 2026-07-31

This audit is a no-deletion, no-migration review of the current
`toy-audio-agent-main` repository before preparing it for GitHub. It is meant
to protect experimental reproducibility while identifying safe cleanup work.

## Current Snapshot

- Current branch: `chore/repository-structure`
- Working tree status at audit time: clean
- Repository size on disk: about `2.0G`
- `.git` size: about `5.0M`
- `.venv` size: about `426M`
- `outputs/` size: about `1.6G`
- `docs/` size: about `1.2M`
- `scripts/` size: about `1.4M`
- `src/` size: about `136K`
- `tests/` size: about `1.3M`

The repository is dominated by generated outputs and the local virtual
environment. The actual source, tests and dissertation-facing markdown/CSV
files are small.

## Current Structure

The current documented structure is:

- `src/toy_audio_agent/`: destination for reusable package code.
- `scripts/inference/`: newer inference entry points.
- `scripts/evaluation/`: newer evaluation entry points.
- `scripts/analysis/`: plotting, comparison and report scripts.
- `configs/`: versioned experiment/model configuration.
- `experiments/registry.yaml`: experiment registry.
- `outputs/manifest.csv`: important generated output index.
- root-level Python files: legacy experiment runners and reusable modules.

The migration policy in `docs/project_structure.md` says not to move legacy
runners or rename frozen output directories before the final experiment freeze.

## Git Ignore Findings

`.gitignore` currently ignores:

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `.DS_Store`
- `outputs/`
- `results/`
- `*.png`, `*.jpg`, `*.wav`, `*.mp3`
- `docs/acoustic_reference_library/raw_sources/*.pdf`
- `.env`

This is mostly appropriate for GitHub hygiene. The important caveat is the
generic `results/` rule. It also matches `docs/results/...`.

Observed consequence:

- `docs/results/openrouter_model_comparison/` and
  `docs/results/multi_agent_pilot/` are already tracked, so Git continues to
  see them despite the ignore rule.
- `docs/results/single_agent_final_report/` and
  `docs/results/multispecies_final_report/` are ignored and currently will not
  be added by normal `git add .`.

Recommendation:

- If dissertation-facing result summaries should be on GitHub, add an explicit
  exception to `.gitignore`, for example:

```gitignore
!docs/results/
!docs/results/**/
!docs/results/**/*.md
!docs/results/**/*.csv
```

- Keep `outputs/` ignored. Do not commit raw model runs, generated images,
  audio windows, large diagnostic outputs, or raw API responses.

## Tracked Generated or Build Artifacts

The following tracked files look like generated/build artifacts and should be
reviewed before pushing:

- `toy_audio_agent.egg-info/PKG-INFO`
- `toy_audio_agent.egg-info/SOURCES.txt`
- `toy_audio_agent.egg-info/dependency_links.txt`
- `toy_audio_agent.egg-info/requires.txt`
- `toy_audio_agent.egg-info/top_level.txt`
- `flies.npz`

Recommendation:

- Remove `toy_audio_agent.egg-info/` from Git unless there is a specific reason
  to keep package build metadata under version control. Add
  `*.egg-info/` or `toy_audio_agent.egg-info/` to `.gitignore`.
- Review `flies.npz`. If it is a small demo input required by tests or the web
  demo, keep it and document its purpose. If it is only a temporary conversion
  artifact, remove it from Git and regenerate it when needed.

## Ignored Local Artifacts

The following ignored local artifacts are safe cleanup candidates:

- `.DS_Store`
- `.pytest_cache/`
- `__pycache__/` directories
- `*.pyc`

`.venv/` is also ignored and large. It is safe to delete only if you are happy
to recreate it with `uv sync`. Because it is useful for local work, it does not
need to be deleted just to prepare a GitHub push.

Do not delete `.env`; it is correctly ignored and may contain local API keys.
It should never be committed or printed.

## Root-Level Legacy Scripts

There are `47` root-level Python files. Many are not dead code: tests and newer
scripts still import them directly. Examples:

- `run_prompt_v2_small_pilot.py` is reused by several newer inference scripts
  for JSON parsing and prompt-v2 helpers.
- `prepare_agent_spectrogram_inputs.py` is reused by visual spot-check scripts.
- `prepare_pcen_spectrogram_inputs.py` is reused by P8 PCEN runners.
- `merge_tiled_predictions.py` is covered by tests and supports tiled workflows.
- `event_characterisation_*` and `run_p7b1_knowledge_grounded_reasoning.py`
  are used by P7 tests and reasoning experiments.
- proposal validator scripts are covered by their own tests.

Recommendation:

- Do not bulk-delete root-level Python files.
- Treat root-level Python files as legacy-but-active unless both of these are
  true:
  1. no test imports them;
  2. no script, doc, registry entry, or frozen output report references them.
- If cleanup is desired, migrate them gradually to `scripts/` or
  `src/toy_audio_agent/` with compatibility wrappers and tests, instead of
  deleting them.

## Scripts Directory

The newer `scripts/` layout is healthy and already contains:

- `scripts/inference/`: 14 inference/smoke-test/model-run entry points.
- `scripts/evaluation/`: 7 evaluation entry points.
- `scripts/analysis/`: 11 analysis/report/dataset-prep entry points.

Recommendation:

- Keep this structure.
- New formal experiment scripts should continue to be placed here rather than
  at the repository root.
- The recently created multi-agent scripts are now tracked:
  - `scripts/inference/run_qwen_stage2c_multi_agent_pilot24.py`
  - `scripts/inference/run_openrouter_multi_agent_stage2c_pilot24.py`
  - `tests/test_run_qwen_stage2c_multi_agent_pilot24.py`
  - `tests/test_run_openrouter_multi_agent_stage2c_pilot24.py`

## Outputs and Frozen Results

`outputs/` is about `1.6G` and is ignored. This is correct for GitHub. It
contains frozen experiment outputs, raw responses, model inputs, generated
figures, audio windows, and analysis reports.

Recommendation:

- Do not move or delete frozen outputs during code cleanup.
- Do not try to commit `outputs/`.
- Keep only dissertation-facing summaries under `docs/results/` if you want
  the final conclusions in GitHub.
- If a result in `docs/results/` depends on an ignored `outputs/` file, keep
  the source path references in the markdown but expect external users to need
  the output archive separately.

## Documentation Findings

Tracked dissertation-facing result docs currently include:

- `docs/results/openrouter_model_comparison/`
- `docs/results/multi_agent_pilot/`

Ignored but likely important dissertation-facing docs currently include:

- `docs/results/single_agent_final_report/`
- `docs/results/multispecies_final_report/`

Recommendation:

- Decide whether all `docs/results/` should be committed.
- If yes, update `.gitignore` with explicit `docs/results` exceptions and add
  the ignored single-agent and multispecies final reports.
- If no, move only the final selected markdown/CSV summaries into a non-ignored
  docs path such as `docs/dissertation/results/`.

## Safe Cleanup Candidates

Safe to remove locally after a dry run:

- `.DS_Store`
- `.pytest_cache/`
- all `__pycache__/` directories
- all `*.pyc`

Safe to remove from Git after review:

- `toy_audio_agent.egg-info/`

Needs human decision:

- `flies.npz`
- whether `docs/results/single_agent_final_report/` and
  `docs/results/multispecies_final_report/` should be committed.

Do not delete:

- `outputs/` frozen experiment outputs without a separate archival decision.
- `ground_truth/`
- `annotations/`
- `audio/` or generated evaluation-set audio that supports reproducibility.
- root-level legacy scripts that are referenced by tests or newer scripts.
- `.env` contents; keep ignored and private.

## Proposed Cleanup Phases

### Phase 1: Protect GitHub Hygiene

1. Update `.gitignore` to ignore package build metadata:

```gitignore
*.egg-info/
```

2. Decide how to handle `docs/results/`.
3. Confirm `.env` remains ignored.
4. Confirm `outputs/` remains ignored.

### Phase 2: Remove Tracked Build Metadata

After review, remove the tracked egg-info files from Git:

```bash
git rm -r toy_audio_agent.egg-info
```

Then run tests.

### Phase 3: Local Cache Cleanup

Dry run first:

```bash
git clean -ndX
```

Then, only if the dry run looks safe, remove ignored cache artifacts. Do not
use this command if it would remove local files you still need.

### Phase 4: Legacy Script Review

For root-level scripts, create a mapping table:

- script name
- current tests that import it
- newer `scripts/` replacement, if any
- frozen experiment it supports
- recommendation: keep, migrate later, or delete candidate

Only delete a script after this mapping shows it has no active references.

### Phase 5: Verification

Run:

```bash
uv run pytest -q
git status --short
git diff --stat
```

For any migration of scripts/imports, run the full test suite. For
documentation-only changes, tests are optional.

### Phase 6: Commit and Push

Suggested commit split:

1. `docs: add repository cleanup audit`
2. `chore: ignore generated package metadata`
3. `chore: remove tracked package build metadata`
4. `docs: include dissertation-facing result summaries`
5. optional later commit: `chore: migrate legacy experiment helpers`

Push only after the working tree is clean except for intentional commits.

## Recommended Immediate Next Step

The lowest-risk next action is:

1. decide whether `docs/results/` should be included on GitHub;
2. update `.gitignore` accordingly;
3. remove `toy_audio_agent.egg-info/` from Git;
4. run `uv run pytest -q`;
5. commit this audit and the cleanup changes.

Do not delete legacy root-level scripts yet. They are still part of the tested
reproducibility surface.
