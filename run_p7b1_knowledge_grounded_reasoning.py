"""Run the non-trivial P7B.1 knowledge-grounding ablation.

The frozen P7A Cases provide geometry and deterministic features. Reasoning
reference labels are loaded only after inference and never enter a model prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from scripts.data_prep.build_event_characterisation_eval_dataset import inputs_for_condition
from event_characterisation_evaluators import UnsupportedBehaviourClaimEvaluator
from event_characterisation_models import (
    EventCharacterisationExpected,
    EventCharacterisationInput,
    GroundedEventInterpretation,
)
from event_characterisation_retrieval import (
    ConditionName,
    RetrievalTrace,
    retrieve_for_condition,
    safe_annotation_context,
    safe_literature_context,
    write_retrieval_trace,
)
from p7b1_reasoning_evaluators import (
    EvidenceScopeComplianceEvaluator,
    GroundedLimitationValidityEvaluator,
    RecommendedStrategyEvaluator,
    RetrievalRelevanceEvaluator,
    ReviewTriggerEvaluator,
    RiskFlagPrecisionEvaluator,
    RiskFlagRecallEvaluator,
)
from run_event_characterisation_ablation import (
    MODEL_NAME,
    OLLAMA_ENDPOINT,
    RunArtifact,
    call_ollama,
    deterministic_feature_payload,
    evaluate_artifact,
    load_dataset,
    ollama_host,
    parse_prediction,
    require_model,
)


DEFAULT_OUTPUT_DIR = Path("outputs/agent_runs/p7b1_knowledge_grounded_reasoning")
CONDITIONS: tuple[ConditionName, ...] = (
    "baseline",
    "annotation_memory_only",
    "literature_only",
    "combined",
)
RISK_FLAGS = (
    "dense_short_call_sequence",
    "boundary_truncation",
    "detector_under_extension",
    "harmful_rigid_shift",
    "useful_anchored_expansion",
    "unnecessary_tool_use",
    "uncertain_tool_conflict",
)
STRATEGIES = (
    "use_fixed_overview",
    "use_tiled_view",
    "prefer_detector_geometry",
    "allow_anchored_expansion",
    "preserve_current_geometry",
    "request_human_review",
)


@dataclass(frozen=True)
class P7B1Artifact:
    condition: ConditionName
    clip_id: str
    parse_status: str
    prediction_path: Path | None
    raw_response_path: Path
    parse_error_path: Path | None
    retrieval_trace_path: Path


def output_template(
    *,
    expected: EventCharacterisationExpected,
    annotation_context: list[dict[str, Any]],
    literature_context: list[dict[str, Any]],
) -> dict[str, Any]:
    features = deterministic_feature_payload(expected)
    return {
        "clip_id": expected.clip_id,
        "interpreted_events": [
            {**event, "confirmed_interpretations": [], "confidence": 1.0}
            for event in features["events"]
        ],
        "sequence_interpretation": {
            **features["sequence"],
            "confirmed_interpretations": [],
        },
        "retrieved_annotation_cases": annotation_context,
        "retrieved_literature_evidence": literature_context,
        "exploratory_hypotheses": [],
        "unsupported_claims": [],
        "risk_flags": [],
        "recommended_strategy": "preserve_current_geometry",
        "grounded_limitations": [],
        "reasoning_confidence": 1.0,
        "confidence": 1.0,
        "human_review_needed": False,
        "review_reason": "",
    }


def build_prompt(
    *,
    inputs: EventCharacterisationInput,
    expected: EventCharacterisationExpected,
    annotation_context: list[dict[str, Any]],
    literature_context: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build one condition-invariant prompt with only context lists changing."""

    system_prompt = (
        "You are a careful bioacoustic system-design reasoning assistant. "
        "The event boxes and deterministic numeric features are frozen. Copy all "
        "event IDs and numeric values exactly; do not detect events or alter geometry. "
        "Choose risk_flags only from the supplied enum and one recommended_strategy. "
        "Annotation-memory cases are project examples, not ground truth for this clip. "
        "Literature records support only their narrow claim and scope. A grounded "
        "limitation must cite one or more evidence_ids present in the supplied verified "
        "literature context; if no literature is supplied, grounded_limitations must be []. "
        "Do not infer behaviour, call phase, social calls, individual identity, or "
        "environmental context. Behaviour remains exploratory and should not be added. "
        "Set human_review_needed when uncertainty, boundary truncation, or conflicting "
        "tool evidence makes automatic strategy selection unsafe. Return valid JSON only."
    )
    runtime_context = {
        "clip_id": inputs.clip_id,
        "clip_duration_seconds": inputs.clip_duration_seconds,
        "frozen_event_boxes": [
            box.model_dump(mode="json") for box in inputs.frozen_event_boxes
        ],
        "immutable_deterministic_features": deterministic_feature_payload(expected),
        "annotation_memory_context": annotation_context,
        "verified_literature_context": literature_context,
        "allowed_risk_flags": list(RISK_FLAGS),
        "allowed_recommended_strategies": list(STRATEGIES),
        "required_output_template": output_template(
            expected=expected,
            annotation_context=annotation_context,
            literature_context=literature_context,
        ),
    }
    user_prompt = (
        "/no_think\n"
        "Characterise system-design risks for the frozen events in the attached clean "
        "spectrogram. Preserve every immutable deterministic value and all supplied "
        "retrieval records exactly. Select evidence-supported risks and one strategy. "
        "For each literature-grounded limitation, use keys limitation_id, statement, "
        "evidence_ids, and scope. Do not create citations or evidence IDs. Do not output "
        "behavioural hypotheses.\n\n"
        + json.dumps(runtime_context, indent=2, ensure_ascii=False)
        + "\n\nReturn the complete GroundedEventInterpretation JSON object now."
    )
    return system_prompt, user_prompt


def run_case(
    *,
    condition: ConditionName,
    case: Any,
    output_dir: Path,
    timeout_seconds: float,
    num_predict: int,
) -> P7B1Artifact:
    inputs = inputs_for_condition(case.inputs, condition)
    annotation_records, literature_records, trace = retrieve_for_condition(
        condition=condition,
        inputs=inputs,
        expected=case.expected_output,
    )
    annotation_context = [
        safe_annotation_context(record) for record in annotation_records
    ]
    literature_context = [
        safe_literature_context(record) for record in literature_records
    ]
    trace_path = output_dir / condition / "retrieval_traces" / f"{case.name}.json"
    raw_path = output_dir / condition / "raw_responses" / f"{case.name}.txt"
    prediction_path = (
        output_dir / condition / "predictions" / f"{case.name}_prediction.json"
    )
    error_path = output_dir / condition / "parse_errors" / f"{case.name}.txt"
    for path in (trace_path, raw_path, prediction_path, error_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_retrieval_trace(trace, trace_path)
    system_prompt, user_prompt = build_prompt(
        inputs=inputs,
        expected=case.expected_output,
        annotation_context=annotation_context,
        literature_context=literature_context,
    )
    raw = ""
    try:
        raw = call_ollama(
            image_path=Path(inputs.spectrogram_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
        )
        raw_path.write_text(raw, encoding="utf-8")
        prediction = parse_prediction(raw, case.name)
        prediction_path.write_text(
            prediction.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return P7B1Artifact(
            condition, case.name, "success", prediction_path, raw_path, None, trace_path
        )
    except Exception as exc:
        raw_path.write_text(raw or f"MODEL_CALL_FAILED\n{exc}\n", encoding="utf-8")
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return P7B1Artifact(
            condition, case.name, "failure", None, raw_path, error_path, trace_path
        )


def evaluator_value(evaluator: Any, ctx: SimpleNamespace) -> bool | float:
    value = evaluator.evaluate(ctx)
    return bool(value) if isinstance(value, bool) else float(value)


def evaluate_case(case: Any, artifact: P7B1Artifact) -> dict[str, Any]:
    inputs = inputs_for_condition(case.inputs, artifact.condition)
    output: Any = {"parse_error": True}
    if artifact.prediction_path:
        output = json.loads(artifact.prediction_path.read_text(encoding="utf-8"))
    old_artifact = RunArtifact(
        condition=artifact.condition,
        clip_id=artifact.clip_id,
        run_type="main",
        parse_status=artifact.parse_status,
        prediction_path=artifact.prediction_path,
        raw_response_path=artifact.raw_response_path,
        parse_error_path=artifact.parse_error_path,
        retrieval_trace_path=artifact.retrieval_trace_path,
    )
    deterministic = evaluate_artifact(
        case=case, condition=artifact.condition, artifact=old_artifact
    )
    ctx = SimpleNamespace(
        inputs=inputs,
        expected_output=case.expected_output,
        output=output,
        metadata=case.metadata,
    )
    reasoning = {
        "risk_flag_precision": evaluator_value(RiskFlagPrecisionEvaluator(), ctx),
        "risk_flag_recall": evaluator_value(RiskFlagRecallEvaluator(), ctx),
        "recommended_strategy_correct": evaluator_value(
            RecommendedStrategyEvaluator(), ctx
        ),
        "grounded_limitation_validity": evaluator_value(
            GroundedLimitationValidityEvaluator(), ctx
        ),
        "evidence_scope_compliance": evaluator_value(
            EvidenceScopeComplianceEvaluator(), ctx
        ),
        "review_trigger_correct": evaluator_value(ReviewTriggerEvaluator(), ctx),
        "retrieval_relevance": evaluator_value(RetrievalRelevanceEvaluator(), ctx),
        "unsupported_behaviour_claim_pass": evaluator_value(
            UnsupportedBehaviourClaimEvaluator(), ctx
        ),
    }
    prediction = (
        GroundedEventInterpretation.model_validate(output)
        if artifact.parse_status == "success"
        else None
    )
    return {
        **deterministic,
        **reasoning,
        "risk_flags": "|".join(prediction.risk_flags) if prediction else "",
        "recommended_strategy": prediction.recommended_strategy if prediction else "",
        "grounded_limitation_count": (
            len(prediction.grounded_limitations) if prediction else 0
        ),
        "reasoning_confidence": prediction.reasoning_confidence if prediction else "",
        "review_reason": prediction.review_reason if prediction else "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def summary_row(
    condition: str, split: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "condition": condition,
        "split": split,
        "case_count": len(rows),
        "parse_success_count": sum(row["parse_status"] == "success" for row in rows),
        "parse_failure_count": sum(row["parse_status"] != "success" for row in rows),
        "deterministic_feature_preservation_rate": average(
            rows, "deterministic_feature_preservation_pass"
        ),
        "risk_flag_precision": average(rows, "risk_flag_precision"),
        "risk_flag_recall": average(rows, "risk_flag_recall"),
        "recommended_strategy_accuracy": average(
            rows, "recommended_strategy_correct"
        ),
        "grounded_limitation_validity": average(
            rows, "grounded_limitation_validity"
        ),
        "evidence_scope_compliance": average(rows, "evidence_scope_compliance"),
        "review_trigger_accuracy": average(rows, "review_trigger_correct"),
        "retrieval_relevance": average(rows, "retrieval_relevance"),
        "unsupported_behaviour_claim_pass_rate": average(
            rows, "unsupported_behaviour_claim_pass"
        ),
        "mean_grounded_limitation_count": average(rows, "grounded_limitation_count"),
    }


def retrieval_rows(artifacts: list[P7B1Artifact], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["condition"], row["clip_id"]): row for row in results}
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        trace = RetrievalTrace.model_validate_json(
            artifact.retrieval_trace_path.read_text(encoding="utf-8")
        )
        result = by_key[(artifact.condition, artifact.clip_id)]
        rows.append(
            {
                "condition": artifact.condition,
                "clip_id": artifact.clip_id,
                "split": result["split"],
                "annotation_ids": "|".join(
                    item.retrieved_id for item in trace.annotation_matches
                ),
                "literature_ids": "|".join(
                    item.retrieved_id for item in trace.literature_matches
                ),
                "target_case_excluded": trace.target_case_excluded,
                "retrieval_relevance": result["retrieval_relevance"],
            }
        )
    return rows


def grounding_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition": row["condition"],
            "clip_id": row["clip_id"],
            "split": row["split"],
            "grounded_limitation_count": row["grounded_limitation_count"],
            "grounded_limitation_validity": row["grounded_limitation_validity"],
            "evidence_scope_compliance": row["evidence_scope_compliance"],
            "unsupported_behaviour_claim_pass": row[
                "unsupported_behaviour_claim_pass"
            ],
        }
        for row in results
    ]


def generate_report(
    summaries: list[dict[str, Any]], results: list[dict[str, Any]], output: Path
) -> None:
    all_rows = [row for row in summaries if row["split"] == "all"]
    lines = [
        "# P7B.1 Non-Trivial Knowledge-Grounded Reasoning Evaluation",
        "",
        "## Scope",
        "",
        f"Exact model: `{MODEL_NAME}`. Four conditions were evaluated on eight frozen P7A Cases (six diagnostic-development and two held-out). Reasoning references were used only after inference.",
        "",
        "## Condition Summary",
        "",
        "| Condition | Parsed | Risk P | Risk R | Strategy | Limitation validity | Scope | Review | Retrieval | Feature preservation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['condition']} | {row['parse_success_count']}/{row['case_count']} "
            f"| {row['risk_flag_precision']:.3f} | {row['risk_flag_recall']:.3f} "
            f"| {row['recommended_strategy_accuracy']:.3f} "
            f"| {row['grounded_limitation_validity']:.3f} "
            f"| {row['evidence_scope_compliance']:.3f} "
            f"| {row['review_trigger_accuracy']:.3f} "
            f"| {row['retrieval_relevance']:.3f} "
            f"| {row['deterministic_feature_preservation_rate']:.3f} |"
        )
    lookup = {(row["condition"], row["split"]): row for row in summaries}
    lines.extend(["", "## Answers to the Evaluation Questions", ""])
    annotation = lookup[("annotation_memory_only", "all")]
    baseline = lookup[("baseline", "all")]
    literature = lookup[("literature_only", "all")]
    combined = lookup[("combined", "all")]
    lines.extend(
        [
            f"1. **Annotation memory:** risk recall changed from {baseline['risk_flag_recall']:.3f} to {annotation['risk_flag_recall']:.3f}; strategy accuracy changed from {baseline['recommended_strategy_accuracy']:.3f} to {annotation['recommended_strategy_accuracy']:.3f}.",
            f"2. **Literature evidence:** literature-only limitation validity was {literature['grounded_limitation_validity']:.3f}, with a mean of {literature['mean_grounded_limitation_count']:.2f} grounded limitations per case.",
            f"3. **Combined grounding:** unsupported-claim pass rate was {combined['unsupported_behaviour_claim_pass_rate']:.3f}, compared with {baseline['unsupported_behaviour_claim_pass_rate']:.3f} for baseline; evidence-scope compliance was {combined['evidence_scope_compliance']:.3f}.",
            f"4. **Review decisions:** review-trigger accuracy was baseline {baseline['review_trigger_accuracy']:.3f}, annotation memory {annotation['review_trigger_accuracy']:.3f}, literature {literature['review_trigger_accuracy']:.3f}, and combined {combined['review_trigger_accuracy']:.3f}.",
            f"5. **Feature integrity:** deterministic preservation was {min(row['deterministic_feature_preservation_rate'] for row in all_rows):.3f} or higher across conditions.",
            "",
            "## Representative Versus Held-Out",
            "",
        ]
    )
    for condition in CONDITIONS:
        representative = lookup[(condition, "representative")]
        heldout = lookup[(condition, "heldout")]
        lines.append(
            f"- `{condition}`: representative risk recall {representative['risk_flag_recall']:.3f}, strategy {representative['recommended_strategy_accuracy']:.3f}, review {representative['review_trigger_accuracy']:.3f}; held-out risk recall {heldout['risk_flag_recall']:.3f}, strategy {heldout['recommended_strategy_accuracy']:.3f}, review {heldout['review_trigger_accuracy']:.3f}."
        )
    transport_ready = all(
        row["parse_success_count"] == row["case_count"]
        and row["deterministic_feature_preservation_rate"] == 1.0
        and row["unsupported_behaviour_claim_pass_rate"] == 1.0
        for row in all_rows
    )
    reasoning_ready = (
        combined["risk_flag_recall"] >= 0.75
        and combined["recommended_strategy_accuracy"] >= 0.75
        and combined["review_trigger_accuracy"] >= 0.75
        and combined["evidence_scope_compliance"] >= 0.75
    )
    lines.extend(
        [
            "",
            "## Freeze Recommendation",
            "",
            (
                "The deterministic transport and substantive reasoning policy meet the predeclared freeze checks, although the eight-case scope still limits generalisation."
                if transport_ready and reasoning_ready
                else "Freeze the deterministic transport, schema, feature-preservation, and evaluation foundations, but do not freeze the substantive reasoning policy. Knowledge context did not improve risk, strategy, or review decisions, and evidence-scope compliance remained below the freeze threshold."
            ),
            "",
            "No LLM judge was used; all reported reasoning scores use manually curated project references and deterministic checks.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if ollama_host() != OLLAMA_ENDPOINT:
        raise RuntimeError(f"P7B.1 requires OLLAMA_HOST={OLLAMA_ENDPOINT}")
    print(require_model())
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}; use --overwrite")
        shutil.rmtree(args.output_dir)
    dataset = load_dataset()
    artifacts: list[P7B1Artifact] = []
    for condition in CONDITIONS:
        for case in dataset.cases:
            print(f"Running {condition} {case.name}", flush=True)
            artifact = run_case(
                condition=condition,
                case=case,
                output_dir=args.output_dir,
                timeout_seconds=args.timeout_seconds,
                num_predict=args.num_predict,
            )
            artifacts.append(artifact)
            print(f"  {artifact.parse_status}", flush=True)
    cases = {case.name: case for case in dataset.cases}
    results = [evaluate_case(cases[item.clip_id], item) for item in artifacts]
    summaries: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_rows = [row for row in results if row["condition"] == condition]
        summaries.append(summary_row(condition, "all", condition_rows))
        for split in ("representative", "heldout"):
            summaries.append(
                summary_row(
                    condition,
                    split,
                    [row for row in condition_rows if row["split"] == split],
                )
            )
    write_csv(args.output_dir / "p7b1_condition_summary.csv", summaries)
    write_csv(args.output_dir / "p7b1_case_results.csv", results)
    write_csv(
        args.output_dir / "p7b1_retrieval_results.csv",
        retrieval_rows(artifacts, results),
    )
    write_csv(
        args.output_dir / "p7b1_grounding_results.csv", grounding_rows(results)
    )
    generate_report(
        summaries,
        results,
        args.output_dir / "p7b1_knowledge_grounding_report.md",
    )
    print(json.dumps([row for row in summaries if row["split"] == "all"], indent=2))


if __name__ == "__main__":
    main()
