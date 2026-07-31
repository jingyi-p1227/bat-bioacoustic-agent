# Repository Inventory Audit

Date: 2026-07-31

This is a no-move, no-delete repository structure audit. It inventories the current tree and suggests a cleaner structure before any refactor. It does not treat generated `outputs/` contents as source code.

## Snapshot

- Branch: `refactor/organise-legacy-scripts`
- Tracked Python files: `135`
- Root-level tracked Python files: `47`
- Test files: `51`
- `scripts/` Python files: `32`
- `src/` Python files: `5`

## Excluded Large or Generated Directories

| Directory | Approx. size | Ignored by Git | Purpose |
|---|---:|---|---|
| `.git` | `5.1M` | no | Git object database and repository metadata. |
| `.venv` | `426M` | yes | Local uv/Python virtual environment. |
| `__pycache__` | `992K` | yes | Generated Python bytecode cache. |
| `.pytest_cache` | `48K` | no | Generated pytest cache. |
| `outputs` | `1.6G` | yes | Generated experiment outputs, model runs, inputs, reports and tool outputs. |
| `tests/__pycache__` | `1.0M` | yes | Generated Python bytecode cache for tests. |

## High-Level Repository Map

### Root-Level Directories

- `__pycache__/`: 
- `annotations/`: Legacy/source annotation files; currently being cleaned of obsolete early experiment data.
- `configs/`: Versioned experiment/model configuration.
- `docs/`: Project notes, dissertation drafts, audits, literature and result summaries.
- `experiments/`: Experiment registry.
- `outputs/`: Generated outputs, ignored by Git.
- `prompts/`: Prompt markdown for bbox/localisation workflows.
- `scripts/`: Newer organized experiment/evaluation/analysis entry points.
- `src/`: Reusable Python package code.
- `tests/`: Pytest suite and fixtures.
- `toy_audio_agent.egg-info/`: Generated package metadata; should not usually be tracked.

### Root-Level Files

- `README.md`
- `analyze_proposal_deviations.py`
- `apply_policy_b_anchored_validator.py`
- `apply_proposal_preserving_validator.py`
- `apply_timing_preserving_validator.py`
- `apply_timing_rule_ablation_validator.py`
- `build_evaluation_set.py`
- `build_event_characterisation_eval_dataset.py`
- `compare_min_db_spectrograms.py`
- `compare_vlm_backends.py`
- `convert_aoef_to_eventresult.py`
- `convert_batdetect2_proposals_to_predictions.py`
- `convert_npz_to_wav.py`
- `evaluate_prompt_v2_small_pilot.py`
- `evaluation.py`
- `event_characterisation_evaluators.py`
- `event_characterisation_models.py`
- `event_characterisation_retrieval.py`
- `experiment_log.md`
- `extract_event_characterisation_features.py`
- `flies.npz`
- `generate_grid_spectrogram_examples.py`
- `justfile`
- `main.py`
- `merge_tiled_predictions.py`
- `p7b1_reasoning_evaluators.py`
- `plot_batdetect2_proposal_overlays.py`
- `plot_evaluation_clip_ground_truth.py`
- `plot_pilot_ground_truth.py`
- `plot_prompt_v2_small_pilot_diagnostics.py`
- `plot_tiled_spectrogram_contact_sheets.py`
- `prepare_agent_spectrogram_inputs.py`
- `prepare_batdetect2_proposals.py`
- `prepare_pcen_spectrogram_inputs.py`
- `prepare_tiled_spectrogram_inputs.py`
- `pyproject.toml`
- `repair_eventresult_json.py`
- `run_batdetect2_proposal_inference.py`
- `run_event_characterisation_ablation.py`
- `run_library_assisted_bbox_ablation.py`
- `run_p7b1_knowledge_grounded_reasoning.py`
- `run_p7c_followup_checks.py`
- `run_prompt_v2_adaptive_zoom.py`
- `run_prompt_v2_batdetect2_assisted_pilot.py`
- `run_prompt_v2_gated_adaptive_zoom.py`
- `run_prompt_v2_small_pilot.py`
- `run_prompt_v2_tiled_pilot.py`
- `run_pydantic_evals.py`
- `run_zoom_guided_prompt.py`
- `split_audio_clips.py`
- `summarize_model_smoke_tests.py`
- `summarize_single_agent_tool_use_experiments.py`
- `uv.lock`

### `scripts/` Subdirectories

- `scripts/inference/`: 14 Python files
  - `check_openrouter_alternative_vlm_smoke.py`, `run_full45_localisation_condition.py`, `run_full45_pcen_confirmatory.py`, `run_openrouter_full_model_comparison.py`, `run_openrouter_multi_agent_stage2c_pilot24.py`, `run_openrouter_two_task_smoke_budget.py`, `run_p8c_pcen_grid_v2_full45.py`, `run_p9_light_walters_guidance.py`, `run_qwen_stage2c_multi_agent_pilot24.py`, `run_stage1a_multispecies_classification.py`, `run_stage1b_multispecies_classification.py`, `run_stage1c_multispecies_classification.py`, `run_stage2_joint_proposal_constrained_pilot80.py`, `run_stage2c_selected_proposal_classification.py`
- `scripts/evaluation/`: 6 Python files
  - `evaluate_multi_protocol_detection.py`, `evaluate_p8c_pcen_grid_v2.py`, `evaluate_p9_light_walters_guidance.py`, `evaluate_stage1a_multispecies_classification.py`, `evaluate_stage2_joint_pilot80.py`, `evaluate_stage2c_selected_proposal_classification.py`
- `scripts/analysis/`: 12 Python files
  - `audit_bd2_species_candidates.py`, `build_multispecies_event_level_dataset.py`, `build_multispecies_stage1_gt_event_classification_dataset.py`, `evaluate_stage2_central_proposal_selection.py`, `prepare_bd2_species_event_crop_spotcheck.py`, `prepare_bd2_species_visual_spotcheck.py`, `prepare_multispecies_v2_quality_review_contact_sheets.py`, `prepare_stage2_sample_level_batdetect2_proposals.py`, `report_p8_sensitivity_and_pcen.py`, `report_p8c_pcen_grid_v2.py`, `report_p9_light_walters_guidance.py`, `summarize_single_agent_full45_localisation.py`
- `scripts/maintenance/`: 0 Python files

### `src/`

- `src/toy_audio_agent/__init__.py`
- `src/toy_audio_agent/evaluation/__init__.py`
- `src/toy_audio_agent/evaluation/event_matching.py`
- `src/toy_audio_agent/experiments/__init__.py`
- `src/toy_audio_agent/experiments/p9_light.py`

### `tests/`

- `51` tracked Python test files.
- `tests/fixtures/` contains dedicated small test fixtures, including the pseudo-petersi temporal-evaluation ground truth fixture.

### `docs/`

- `docs/acoustic_reference_library/`
- `docs/annotation_example_library/`
- `docs/dataset_audits/`
- `docs/dissertation/`
- `docs/dissertation_drafts/`
- `docs/experiment_notes/`
- `docs/literature_reference_library/`
- `docs/protocols/`
- `docs/results/`

### `configs/`, `experiments/`, `prompts/`

- `configs/experiments/`: P8/P9 experiment configuration files.
- `configs/models/`: model configuration directory, currently present for future use.
- `experiments/registry.yaml`: tracked registry of selected frozen/completed experiments.
- `prompts/`: prompt-v2 markdown files for localisation/adaptive-view workflows.

### `audio/`, `annotations/`, `ground_truth/`, `outputs/`

- `audio/`: source/generated audio location when present; binary audio remains ignored by file extension rules.
- `annotations/`: legacy root annotation directory. Obsolete early files have been deleted from the working tree.
- `ground_truth/`: legacy root ground-truth directory. Obsolete early files have been deleted; tests now use `tests/fixtures/ground_truth/` for the small temporal fixture.
- `outputs/`: generated output tree, ignored by Git and excluded from detailed inventory.

## Tracked Python File Classification

| Category | Count |
|---|---:|
| dataset preparation script | 17 |
| formal analysis/report script | 8 |
| formal evaluation script | 7 |
| formal inference script | 14 |
| legacy experiment runner | 13 |
| reusable package code | 12 |
| test file | 51 |
| validation/repair utility | 7 |
| visualisation/plotting script | 6 |

| Path | Category |
|---|---|
| `analyze_proposal_deviations.py` | validation/repair utility |
| `apply_policy_b_anchored_validator.py` | validation/repair utility |
| `apply_proposal_preserving_validator.py` | validation/repair utility |
| `apply_timing_preserving_validator.py` | validation/repair utility |
| `apply_timing_rule_ablation_validator.py` | validation/repair utility |
| `build_evaluation_set.py` | dataset preparation script |
| `build_event_characterisation_eval_dataset.py` | dataset preparation script |
| `compare_min_db_spectrograms.py` | formal analysis/report script |
| `compare_vlm_backends.py` | legacy experiment runner |
| `convert_aoef_to_eventresult.py` | dataset preparation script |
| `convert_batdetect2_proposals_to_predictions.py` | dataset preparation script |
| `convert_npz_to_wav.py` | dataset preparation script |
| `evaluate_prompt_v2_small_pilot.py` | formal evaluation script |
| `evaluation.py` | reusable package code |
| `event_characterisation_evaluators.py` | reusable package code |
| `event_characterisation_models.py` | reusable package code |
| `event_characterisation_retrieval.py` | reusable package code |
| `extract_event_characterisation_features.py` | reusable package code |
| `generate_grid_spectrogram_examples.py` | visualisation/plotting script |
| `main.py` | reusable package code |
| `merge_tiled_predictions.py` | validation/repair utility |
| `p7b1_reasoning_evaluators.py` | reusable package code |
| `plot_batdetect2_proposal_overlays.py` | visualisation/plotting script |
| `plot_evaluation_clip_ground_truth.py` | visualisation/plotting script |
| `plot_pilot_ground_truth.py` | visualisation/plotting script |
| `plot_prompt_v2_small_pilot_diagnostics.py` | visualisation/plotting script |
| `plot_tiled_spectrogram_contact_sheets.py` | visualisation/plotting script |
| `prepare_agent_spectrogram_inputs.py` | dataset preparation script |
| `prepare_batdetect2_proposals.py` | dataset preparation script |
| `prepare_pcen_spectrogram_inputs.py` | dataset preparation script |
| `prepare_tiled_spectrogram_inputs.py` | dataset preparation script |
| `repair_eventresult_json.py` | validation/repair utility |
| `run_batdetect2_proposal_inference.py` | legacy experiment runner |
| `run_event_characterisation_ablation.py` | legacy experiment runner |
| `run_library_assisted_bbox_ablation.py` | legacy experiment runner |
| `run_p7b1_knowledge_grounded_reasoning.py` | legacy experiment runner |
| `run_p7c_followup_checks.py` | legacy experiment runner |
| `run_prompt_v2_adaptive_zoom.py` | legacy experiment runner |
| `run_prompt_v2_batdetect2_assisted_pilot.py` | legacy experiment runner |
| `run_prompt_v2_gated_adaptive_zoom.py` | legacy experiment runner |
| `run_prompt_v2_small_pilot.py` | legacy experiment runner |
| `run_prompt_v2_tiled_pilot.py` | legacy experiment runner |
| `run_pydantic_evals.py` | legacy experiment runner |
| `run_zoom_guided_prompt.py` | legacy experiment runner |
| `scripts/analysis/audit_bd2_species_candidates.py` | dataset preparation script |
| `scripts/analysis/build_multispecies_event_level_dataset.py` | dataset preparation script |
| `scripts/analysis/build_multispecies_stage1_gt_event_classification_dataset.py` | dataset preparation script |
| `scripts/analysis/evaluate_stage2_central_proposal_selection.py` | formal analysis/report script |
| `scripts/analysis/prepare_bd2_species_event_crop_spotcheck.py` | dataset preparation script |
| `scripts/analysis/prepare_bd2_species_visual_spotcheck.py` | dataset preparation script |
| `scripts/analysis/prepare_multispecies_v2_quality_review_contact_sheets.py` | dataset preparation script |
| `scripts/analysis/prepare_stage2_sample_level_batdetect2_proposals.py` | dataset preparation script |
| `scripts/analysis/report_p8_sensitivity_and_pcen.py` | formal analysis/report script |
| `scripts/analysis/report_p8c_pcen_grid_v2.py` | formal analysis/report script |
| `scripts/analysis/report_p9_light_walters_guidance.py` | formal analysis/report script |
| `scripts/analysis/summarize_single_agent_full45_localisation.py` | formal analysis/report script |
| `scripts/evaluation/evaluate_multi_protocol_detection.py` | formal evaluation script |
| `scripts/evaluation/evaluate_p8c_pcen_grid_v2.py` | formal evaluation script |
| `scripts/evaluation/evaluate_p9_light_walters_guidance.py` | formal evaluation script |
| `scripts/evaluation/evaluate_stage1a_multispecies_classification.py` | formal evaluation script |
| `scripts/evaluation/evaluate_stage2_joint_pilot80.py` | formal evaluation script |
| `scripts/evaluation/evaluate_stage2c_selected_proposal_classification.py` | formal evaluation script |
| `scripts/inference/check_openrouter_alternative_vlm_smoke.py` | formal inference script |
| `scripts/inference/run_full45_localisation_condition.py` | formal inference script |
| `scripts/inference/run_full45_pcen_confirmatory.py` | formal inference script |
| `scripts/inference/run_openrouter_full_model_comparison.py` | formal inference script |
| `scripts/inference/run_openrouter_multi_agent_stage2c_pilot24.py` | formal inference script |
| `scripts/inference/run_openrouter_two_task_smoke_budget.py` | formal inference script |
| `scripts/inference/run_p8c_pcen_grid_v2_full45.py` | formal inference script |
| `scripts/inference/run_p9_light_walters_guidance.py` | formal inference script |
| `scripts/inference/run_qwen_stage2c_multi_agent_pilot24.py` | formal inference script |
| `scripts/inference/run_stage1a_multispecies_classification.py` | formal inference script |
| `scripts/inference/run_stage1b_multispecies_classification.py` | formal inference script |
| `scripts/inference/run_stage1c_multispecies_classification.py` | formal inference script |
| `scripts/inference/run_stage2_joint_proposal_constrained_pilot80.py` | formal inference script |
| `scripts/inference/run_stage2c_selected_proposal_classification.py` | formal inference script |
| `split_audio_clips.py` | dataset preparation script |
| `src/toy_audio_agent/__init__.py` | reusable package code |
| `src/toy_audio_agent/evaluation/__init__.py` | reusable package code |
| `src/toy_audio_agent/evaluation/event_matching.py` | reusable package code |
| `src/toy_audio_agent/experiments/__init__.py` | reusable package code |
| `src/toy_audio_agent/experiments/p9_light.py` | reusable package code |
| `summarize_model_smoke_tests.py` | formal analysis/report script |
| `summarize_single_agent_tool_use_experiments.py` | formal analysis/report script |
| `tests/test_analyze_proposal_deviations.py` | test file |
| `tests/test_apply_policy_b_anchored_validator.py` | test file |
| `tests/test_apply_proposal_preserving_validator.py` | test file |
| `tests/test_apply_timing_preserving_validator.py` | test file |
| `tests/test_apply_timing_rule_ablation_validator.py` | test file |
| `tests/test_audio_clips.py` | test file |
| `tests/test_bd2_species_candidate_audit.py` | test file |
| `tests/test_build_evaluation_set.py` | test file |
| `tests/test_build_multispecies_event_level_dataset.py` | test file |
| `tests/test_build_multispecies_stage1_gt_event_classification_dataset.py` | test file |
| `tests/test_convert_batdetect2_proposals_to_predictions.py` | test file |
| `tests/test_evaluate_prompt_v2_small_pilot.py` | test file |
| `tests/test_evaluate_stage2_central_proposal_selection.py` | test file |
| `tests/test_evaluation_temporal.py` | test file |
| `tests/test_event_characterisation_evals.py` | test file |
| `tests/test_event_characterisation_retrieval.py` | test file |
| `tests/test_event_matching.py` | test file |
| `tests/test_extract_event_characterisation_features.py` | test file |
| `tests/test_merge_tiled_predictions.py` | test file |
| `tests/test_p7b1_reasoning.py` | test file |
| `tests/test_p8c_pcen_grid_v2.py` | test file |
| `tests/test_p9_light.py` | test file |
| `tests/test_plot_evaluation_clip_ground_truth.py` | test file |
| `tests/test_plot_prompt_v2_small_pilot_diagnostics.py` | test file |
| `tests/test_plot_tiled_spectrogram_contact_sheets.py` | test file |
| `tests/test_prepare_agent_spectrogram_inputs.py` | test file |
| `tests/test_prepare_batdetect2_proposals.py` | test file |
| `tests/test_prepare_bd2_species_event_crop_spotcheck.py` | test file |
| `tests/test_prepare_bd2_species_visual_spotcheck.py` | test file |
| `tests/test_prepare_multispecies_v2_quality_review_contact_sheets.py` | test file |
| `tests/test_prepare_pcen_spectrogram_inputs.py` | test file |
| `tests/test_prepare_stage2_sample_level_batdetect2_proposals.py` | test file |
| `tests/test_prepare_tiled_spectrogram_inputs.py` | test file |
| `tests/test_run_full45_localisation_condition.py` | test file |
| `tests/test_run_library_assisted_bbox_ablation.py` | test file |
| `tests/test_run_openrouter_multi_agent_stage2c_pilot24.py` | test file |
| `tests/test_run_p7c_followup_checks.py` | test file |
| `tests/test_run_prompt_v2_adaptive_zoom.py` | test file |
| `tests/test_run_prompt_v2_batdetect2_assisted_pilot.py` | test file |
| `tests/test_run_prompt_v2_gated_adaptive_zoom.py` | test file |
| `tests/test_run_prompt_v2_small_pilot.py` | test file |
| `tests/test_run_prompt_v2_tiled_pilot.py` | test file |
| `tests/test_run_qwen_stage2c_multi_agent_pilot24.py` | test file |
| `tests/test_spectrogram_grid.py` | test file |
| `tests/test_stage1a_multispecies_classification.py` | test file |
| `tests/test_stage2_joint_pilot80.py` | test file |
| `tests/test_stage2c_selected_proposal_classification.py` | test file |
| `tests/test_summarize_model_smoke_tests.py` | test file |
| `tests/test_summarize_single_agent_full45_localisation.py` | test file |
| `tests/test_summarize_single_agent_tool_use_experiments.py` | test file |
| `tests/test_temporal_schema.py` | test file |

## Root-Level Python Summary

The detailed root-level migration table is in `docs/root_level_script_inventory.csv`. Summary by migration risk:

| Risk | Count |
|---|---:|
| high | 4 |
| low | 21 |
| medium | 22 |

Highest-risk root-level files to avoid moving immediately:

- `main.py`: FastAPI demo app and shared pydantic/audio helpers are imported by legacy tools.
- `run_event_characterisation_ablation.py`: Experiment runner; preserve frozen reproducibility and update imports before moving.
- `run_p7b1_knowledge_grounded_reasoning.py`: Experiment runner; preserve frozen reproducibility and update imports before moving.
- `run_prompt_v2_small_pilot.py`: Utility functions are imported by multiple newer inference scripts.

## Tests Importing Root-Level Scripts Directly

| Test file | Root-level imports |
|---|---|
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_analyze_proposal_deviations.py` | `analyze_proposal_deviations.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_apply_policy_b_anchored_validator.py` | `apply_policy_b_anchored_validator.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_apply_proposal_preserving_validator.py` | `apply_proposal_preserving_validator.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_apply_timing_preserving_validator.py` | `apply_timing_preserving_validator.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_apply_timing_rule_ablation_validator.py` | `apply_timing_rule_ablation_validator.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_audio_clips.py` | `split_audio_clips.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_build_evaluation_set.py` | `build_evaluation_set.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_convert_batdetect2_proposals_to_predictions.py` | `convert_batdetect2_proposals_to_predictions.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_evaluate_prompt_v2_small_pilot.py` | `evaluate_prompt_v2_small_pilot.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_evaluation_temporal.py` | `evaluation.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_event_characterisation_evals.py` | `build_event_characterisation_eval_dataset.py`, `event_characterisation_evaluators.py`, `event_characterisation_models.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_event_characterisation_retrieval.py` | `build_event_characterisation_eval_dataset.py`, `event_characterisation_evaluators.py`, `event_characterisation_models.py`, `event_characterisation_retrieval.py`, `run_event_characterisation_ablation.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_extract_event_characterisation_features.py` | `event_characterisation_models.py`, `extract_event_characterisation_features.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_merge_tiled_predictions.py` | `merge_tiled_predictions.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_p7b1_reasoning.py` | `build_event_characterisation_eval_dataset.py`, `event_characterisation_models.py`, `event_characterisation_retrieval.py`, `extract_event_characterisation_features.py`, `p7b1_reasoning_evaluators.py`, `run_p7b1_knowledge_grounded_reasoning.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_p8c_pcen_grid_v2.py` | `prepare_pcen_spectrogram_inputs.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_plot_evaluation_clip_ground_truth.py` | `plot_evaluation_clip_ground_truth.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_plot_prompt_v2_small_pilot_diagnostics.py` | `plot_prompt_v2_small_pilot_diagnostics.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_plot_tiled_spectrogram_contact_sheets.py` | `plot_tiled_spectrogram_contact_sheets.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_prepare_agent_spectrogram_inputs.py` | `prepare_agent_spectrogram_inputs.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_prepare_batdetect2_proposals.py` | `prepare_batdetect2_proposals.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_prepare_pcen_spectrogram_inputs.py` | `prepare_pcen_spectrogram_inputs.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_prepare_tiled_spectrogram_inputs.py` | `prepare_tiled_spectrogram_inputs.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_run_library_assisted_bbox_ablation.py` | `event_characterisation_models.py`, `event_characterisation_retrieval.py`, `run_library_assisted_bbox_ablation.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_run_p7c_followup_checks.py` | `run_p7c_followup_checks.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_run_prompt_v2_adaptive_zoom.py` | `run_prompt_v2_adaptive_zoom.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_run_prompt_v2_batdetect2_assisted_pilot.py` | `run_prompt_v2_batdetect2_assisted_pilot.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_run_prompt_v2_gated_adaptive_zoom.py` | `run_prompt_v2_gated_adaptive_zoom.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_run_prompt_v2_small_pilot.py` | `run_prompt_v2_small_pilot.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_run_prompt_v2_tiled_pilot.py` | `merge_tiled_predictions.py`, `run_prompt_v2_tiled_pilot.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_spectrogram_grid.py` | `main.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_summarize_model_smoke_tests.py` | `summarize_model_smoke_tests.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_summarize_single_agent_tool_use_experiments.py` | `summarize_single_agent_tool_use_experiments.py` |
| `/Users/morn/Desktop/BIOS0057 EDS/ai agent demo/toy-audio-agent-main/tests/test_temporal_schema.py` | `main.py` |

## `docs/results/` Dissertation-Facing Summaries

| File | Tracked by Git |
|---|---|
| `docs/results/multi_agent_pilot/dissertation_paragraphs.md` | yes |
| `docs/results/multi_agent_pilot/gpt_multi_agent_summary.csv` | yes |
| `docs/results/multi_agent_pilot/multi_agent_pilot_summary.md` | yes |
| `docs/results/multi_agent_pilot/qwen_multi_agent_summary.csv` | yes |
| `docs/results/multi_agent_pilot/same_sample_comparison.csv` | yes |
| `docs/results/multi_agent_pilot/uncertainty_review_summary.csv` | yes |
| `docs/results/multispecies_final_report/dissertation_paragraphs.md` | no |
| `docs/results/multispecies_final_report/final_multispecies_report.md` | no |
| `docs/results/multispecies_final_report/key_findings.md` | no |
| `docs/results/multispecies_final_report/limitations_and_interpretation.md` | no |
| `docs/results/multispecies_final_report/multispecies_main_results_table.csv` | no |
| `docs/results/multispecies_final_report/stage1_classification_summary.csv` | no |
| `docs/results/multispecies_final_report/stage2_joint_summary.csv` | no |
| `docs/results/openrouter_model_comparison/classification_comparison_table.csv` | yes |
| `docs/results/openrouter_model_comparison/dissertation_paragraphs.md` | yes |
| `docs/results/openrouter_model_comparison/joint_task_comparison_table.csv` | yes |
| `docs/results/openrouter_model_comparison/localisation_comparison_table.csv` | yes |
| `docs/results/openrouter_model_comparison/openrouter_gpt_5_6_sol_summary.md` | yes |
| `docs/results/openrouter_model_comparison/token_cost_summary.csv` | yes |
| `docs/results/single_agent_final_report/single_agent_dissertation_paragraphs.md` | no |
| `docs/results/single_agent_final_report/single_agent_final_report.md` | no |
| `docs/results/single_agent_final_report/single_agent_key_findings.md` | no |
| `docs/results/single_agent_final_report/single_agent_main_results_table.csv` | no |

No raw response JSON files were found under `docs/results/`; the present files are markdown and CSV summaries.

## Git Hygiene Findings

- `.env` ignored: yes
- `outputs/` ignored: yes
- `docs/results/` exception configured: no
- `*.egg-info/` ignored: no
- `docs/acoustic_reference_library/raw_sources/*.pdf` ignored: yes

Tracked files that look generated, binary, or package-build related:

- `flies.npz`
- `toy_audio_agent.egg-info/PKG-INFO`
- `toy_audio_agent.egg-info/SOURCES.txt`
- `toy_audio_agent.egg-info/dependency_links.txt`
- `toy_audio_agent.egg-info/requires.txt`
- `toy_audio_agent.egg-info/top_level.txt`

No tracked raw API response JSON files were found by the simple filename scan.

## Immediate Audit Conclusions

- The codebase is already partly reorganized: newer work is under `scripts/` and `src/`, while many root-level scripts remain legacy-but-active.
- The root-level scripts should not be moved in bulk because tests and newer scripts import many of them directly.
- The safest first cleanup is Git hygiene: ignore/remove tracked package metadata and decide how to handle `docs/results/`.
- The next structural refactor should add `scripts/data_prep/`, `scripts/visualization/`, `scripts/diagnostics/`, and `scripts/legacy_root/` before moving files gradually.
