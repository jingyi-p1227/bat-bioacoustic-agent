# Literature Reference Library

## Purpose

This library maps literature needs to the UCL35 dissertation argument: reliable agentic bioacoustic annotation requires specialist tools, structured outputs, validation, uncertainty handling, and human or critic review.

Only Walters et al. has partial bibliographic information in the current project materials. Every other citation gap is marked `TODO`; these cards are writing plans, not substitutes for verified sources.

## Citation Status

| Note | Status | Primary themes |
|---|---|---|
| [BatDetect2](batdetect2_automated_bat_detection.md) | **TODO citation** | Automated detection, proposal tools |
| [Walters et al. 2012](walters_2012_ibatsid.md) | **Partial citation; verify details** | Bat identification, reference calls, uncertainty |
| [Bioacoustic monitoring](bioacoustic_monitoring_automated_annotation_todo.md) | **TODO references** | Monitoring scale, automated annotation |
| [Strong labelling](strong_labelling_time_frequency_localisation_todo.md) | **TODO references** | Event boxes, time-frequency localisation |
| [PCEN](pcen_audio_preprocessing_todo.md) | **TODO citation** | Audio preprocessing, representation ablation |
| [Human-in-the-loop](human_in_loop_active_learning_todo.md) | **TODO references** | Review triage, active learning |
| [Vision-language models](vision_language_scientific_annotation_todo.md) | **TODO references** | Scientific images, structured multimodal output |
| [Agentic validation](agentic_tool_use_validation_critic_todo.md) | **TODO references** | Tool use, validators, critic/referee systems |
| [Metrics and uncertainty](evaluation_metrics_calibration_uncertainty_todo.md) | **TODO references** | Precision/recall/F1, IoU, calibration |

## Introduction

- **Bioacoustic monitoring:** motivate scalable annotation, after a real review source is added.
- **BatDetect2:** introduce automated candidate detection and its role in bat monitoring.
- **Walters et al.:** motivate objective, standardized acoustic identification and uncertainty-aware decisions.
- **Human-in-the-loop:** frame reliable automation as selective assistance rather than complete replacement.

## Related Work

- **Walters et al.:** reference libraries, call parameters, and hierarchical identification.
- **Strong labelling:** distinguish clip classification from event-level time-frequency localisation.
- **Vision-language models:** position spectrogram annotation as structured scientific-image interpretation.
- **Agentic validation:** position tool-using single-agent and future critic/referee workflows.
- **PCEN:** position alternate audio representations once the original citation is verified.

## Methods

- **BatDetect2:** version, proposal schema, threshold, taxonomy caveat, and provenance.
- **Strong labelling:** one call per box and explicit time-frequency geometry.
- **PCEN:** reproducible parameters and controlled representation comparison.
- **Agentic validation:** overview/tile/proposal tools and deterministic shadow validation.
- **Metrics:** one-to-one matching, precision/recall/F1, temporal/frequency/box IoU, and review fields.

## Results Interpretation

- **PCEN:** a visually altered representation does not necessarily improve model localisation.
- **BatDetect2:** specialist proposals can outperform unconstrained VLM refinement.
- **Metrics:** recall gains can coexist with weaker box localisation.
- **Agentic validation:** validators repair some shifts but can suppress useful held-out expansions.

## Discussion and Future Work

- **Human-in-the-loop:** prioritize boundary, disagreement, new-event, and source-extent cases.
- **Agentic validation:** duration-normalized rules and a carefully scoped critic/referee prototype.
- **Metrics and uncertainty:** confidence calibration and threshold sensitivity remain open.
- **Bioacoustic monitoring:** evaluate generalisation to other species and recording conditions.

## Submission Checklist

1. Obtain the actual papers or authoritative records for every TODO card.
2. Replace placeholders with complete citations in the required UCL style.
3. Verify each methodological claim against the source text.
4. Remove any TODO note that is not ultimately cited.
5. Keep project findings clearly separated from claims attributed to literature.
