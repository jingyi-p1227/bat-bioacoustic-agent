# Annotation Example Library

## Purpose

This library is the qualitative companion to the `Ozimops petersi` strong-labelling benchmark. It links visible time-frequency patterns to measured model and tool behaviour, so aggregate metrics can be interpreted through concrete annotation cases.

The cards are analysis references only. Ground-truth and diagnostic overlays must never be used as model inputs.

## Selection Method

Cases were selected from the frozen representative-six set, the P6E.5 held-out set, and the consolidated P6 case highlights. Selection prioritised:

- clean or canonical successes;
- dense multi-event sequences;
- left- and right-boundary truncation;
- detector timing strengths and source-proposal failures;
- useful VLM refinements and harmful VLM shifts;
- preprocessing and deterministic-validator regressions.

All metrics come from existing evaluator CSV/JSON files and `outputs/analysis_reports/p6_single_agent_tool_use_summary/p6_case_highlights.csv`. Figure paths point to existing repository artifacts.

## Uses

1. **Error analysis:** connect TP/FP/FN changes to visible geometry.
2. **Prompt and tool refinement:** identify when overview, tiles, detector proposals, or validation rules help or harm.
3. **Dissertation writing:** provide stable, evidence-backed case studies and figure shortlists.
4. **Future example guidance:** define curated examples without leaking benchmark answers into normal prediction runs.

## Case Index

| Clip | Primary role | Central lesson |
|---|---|---|
| [OP_001](OP_001_canonical_multi_event.md) | Canonical multi-event overview case | Extra tools can regress a strong fixed-view result. |
| [OP_003](OP_003_boundary_improved.md) | Right-boundary and tiled-recall case | Tiling improves recall but still misses the truncated event. |
| [OP_004](OP_004_useful_expansion.md) | Useful duration expansion | Over-conservative timing preservation can remove a true improvement. |
| [OP_010](OP_010_dense_multi_event.md) | Dense but separable sequence | 0.5 s tiling achieves full recall with one duplicate/FP. |
| [OP_016](OP_016_hard_failure.md) | Dense short-call stress case | Detector timing is strong; unconstrained VLM refinement is harmful. |
| [OP_032](OP_032_heldout_expansion_reverted.md) | Held-out source-extent failure | A fixed 6 ms rule wrongly reverts a useful VLM expansion. |
| [OP_042](OP_042_heldout_harmful_shift.md) | Held-out harmful rigid shift | Policy B can correctly restore a displaced detector proposal. |
| [OP_045](OP_045_clean_success.md) | Clean partial clip and detector failure | Preservation cannot repair proposals that are intrinsically too short. |

An additional legacy guard card, [OP_027](OP_027_worsened_case.md), is retained for P5 workflow-regression analysis.

## Curation Rules

- Do not modify or duplicate ground-truth JSON files.
- Keep paths repository-relative.
- Name the exact run when reporting behaviour.
- Separate observed metrics from interpretation.
- Do not invent clean/noisy labels without documented visual review.
- Treat cards as reporting references, not extra benchmark labels.
