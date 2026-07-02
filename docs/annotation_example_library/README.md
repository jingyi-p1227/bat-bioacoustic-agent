# Annotation Example Library

## Purpose

This library collects representative examples from the `Ozimops petersi` strong-labelling benchmark. It complements aggregate metrics by preserving concrete evidence about when the annotation workflow succeeds, fails, or changes under different visual tools.

The library should include:

- clean and easy success cases;
- dense multi-event cases;
- left- and right-boundary cases;
- hard failures and ambiguous calls;
- false-positive and missed-call examples;
- grid-sensitive or preprocessing-sensitive cases;
- cases that improve or worsen under gated and adaptive workflows.

## Intended Uses

1. **Prompt refinement:** translate recurring failures into precise, testable annotation instructions.
2. **Qualitative analysis:** connect aggregate metrics to visible event-level behaviour.
3. **Example-guided annotation:** provide a future curated reference set without embedding evaluation answers in ordinary prediction inputs.
4. **Tool design:** identify which cases may benefit from PCEN-like enhancement, denoising, band-pass filtering, cropping, or tiling.
5. **Reporting:** maintain a stable shortlist of figures for supervisor meetings and final project documentation.

## Initial Cards

| Clip | Primary role |
| --- | --- |
| `OP_045` | Clean success and partial-final-clip sanity check |
| `OP_003` | Right-boundary case improved by adaptive/gated workflows |
| `OP_010` | Dense multi-event separation case |
| `OP_016` | Dense boundary-stress hard failure |
| `OP_027` | Case that worsened relative to the fixed-view baseline |

## Card Fields

Every card should maintain the following fields:

- `clip_id`
- `case_type`
- `why_this_example_matters`
- `relevant_figures`
- `model_behaviour`
- `prompt_lesson`
- `tool_lesson`
- `notes_for_report`

## Curation Rules

- Do not modify or duplicate ground-truth JSON files.
- Keep all figure paths relative to the repository root.
- Record the run name whenever describing model behaviour.
- Separate observed evidence from interpretation.
- Do not use diagnostic or GT-overlay figures as model inputs.
- Treat these cards as analysis references, not additional benchmark labels.

