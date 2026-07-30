"""Run qwen3.6 multi-agent Stage 2C selected-proposal classification pilot.

This pilot uses existing Stage 2C label-safe centred crops and deterministic
nearest-centre BatDetect2 proposals. Ground-truth species labels are used only
after prediction for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_prompt_v2_small_pilot import extract_json_text  # noqa: E402
from scripts.inference.run_stage1a_multispecies_classification import (  # noqa: E402
    ALLOWED_LABELS,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_NAME,
    call_ollama_generate,
    load_manifest,
    resolve_repo_path,
)
from scripts.inference.run_stage2c_selected_proposal_classification import (  # noqa: E402
    DEFAULT_SELECTED_PROPOSALS,
    selected_proposal_index,
    write_csv,
)


CONDITION_NAME = "qwen3_6_stage2c_pilot24"
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/agent_runs/multi_agent" / CONDITION_NAME
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs/analysis_reports/multi_agent" / CONDITION_NAME
QWEN_STAGE2C_FULL240 = (
    REPO_ROOT
    / "outputs/agent_runs/multispecies_classification/"
    "qwen3_6_stage2c_nearest_centre_proposal_classification_full240/parsed_predictions.csv"
)
GPT_STAGE2C_FULL240 = (
    REPO_ROOT
    / "outputs/analysis_reports/openrouter_model_comparison/"
    "gpt_5_6_sol_uk_node_stage2c_classification_full240/sample_level_results.csv"
)

AGENT1_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_species": {"type": "string", "enum": list(ALLOWED_LABELS)},
        "confidence": {"type": "number"},
        "reasoning_brief": {"type": "string"},
        "visual_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["predicted_species", "confidence", "reasoning_brief", "visual_evidence"],
}
AGENT2_SCHEMA = {
    "type": "object",
    "properties": {
        "review_decision": {"type": "string", "enum": ["accept", "revise", "uncertain"]},
        "revised_species": {"type": "string", "enum": list(ALLOWED_LABELS)},
        "confidence": {"type": "number"},
        "reasoning_brief": {"type": "string"},
        "human_review_recommended": {"type": "boolean"},
    },
    "required": [
        "review_decision",
        "revised_species",
        "confidence",
        "reasoning_brief",
        "human_review_recommended",
    ],
}
AGENT3_SCHEMA = {
    "type": "object",
    "properties": {
        "final_species": {"type": "string", "enum": list(ALLOWED_LABELS)},
        "confidence": {"type": "number"},
        "review_status": {"type": "string", "enum": ["accepted", "revised", "uncertain"]},
        "human_review_recommended": {"type": "boolean"},
        "reasoning_brief": {"type": "string"},
    },
    "required": [
        "final_species",
        "confidence",
        "review_status",
        "human_review_recommended",
        "reasoning_brief",
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ollama_tags(model_host: str, timeout: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(f"{model_host.rstrip('/')}/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def assert_model_available(model_name: str, model_host: str) -> dict[str, Any]:
    payload = ollama_tags(model_host)
    names = [item.get("name", "") for item in payload.get("models", []) if isinstance(item, dict)]
    if model_name not in names:
        raise RuntimeError(f"{model_name} is not available through {model_host}; available={names}")
    return {"model_name": model_name, "ollama_host": model_host, "available": True}


def comparison_prediction_index(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    return {row["anonymous_sample_id"]: row for row in rows if row.get("anonymous_sample_id")}


def select_pilot_samples(
    manifest_rows: list[dict[str, str]],
    selected_by_anon: dict[str, dict[str, str]],
    qwen_predictions: dict[str, dict[str, str]],
    *,
    per_species: int = 3,
) -> list[dict[str, str]]:
    selected_rows: list[dict[str, str]] = []
    for species in ALLOWED_LABELS:
        candidates = [
            row
            for row in manifest_rows
            if row.get("species") == species and row.get("anonymous_sample_id") in selected_by_anon
        ]

        def sort_key(row: dict[str, str]) -> tuple[int, str]:
            pred = qwen_predictions.get(row["anonymous_sample_id"], {})
            qwen_wrong = pred.get("parse_status") != "success" or pred.get("predicted_species") != species
            return (0 if qwen_wrong else 1, row["anonymous_sample_id"])

        selected_rows.extend(sorted(candidates, key=sort_key)[:per_species])
    return selected_rows


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    payload = json.loads(extract_json_text(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON must be an object")
    return payload


def _validate_species(value: Any, field_name: str) -> str:
    if value not in ALLOWED_LABELS:
        raise ValueError(f"invalid {field_name}: {value!r}")
    return str(value)


def _validate_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be numeric")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


def parse_agent1(raw_text: str) -> dict[str, Any]:
    payload = _parse_json_object(raw_text)
    return {
        "predicted_species": _validate_species(payload.get("predicted_species"), "predicted_species"),
        "confidence": _validate_confidence(payload.get("confidence")),
        "reasoning_brief": str(payload.get("reasoning_brief", "")),
        "visual_evidence": payload.get("visual_evidence") if isinstance(payload.get("visual_evidence"), list) else [],
    }


def parse_agent2(raw_text: str) -> dict[str, Any]:
    payload = _parse_json_object(raw_text)
    decision = payload.get("review_decision")
    if decision not in {"accept", "revise", "uncertain"}:
        raise ValueError(f"invalid review_decision: {decision!r}")
    human_review = payload.get("human_review_recommended")
    if not isinstance(human_review, bool):
        raise ValueError("human_review_recommended must be boolean")
    return {
        "review_decision": decision,
        "revised_species": _validate_species(payload.get("revised_species"), "revised_species"),
        "confidence": _validate_confidence(payload.get("confidence")),
        "reasoning_brief": str(payload.get("reasoning_brief", "")),
        "human_review_recommended": human_review,
    }


def parse_agent3(raw_text: str) -> dict[str, Any]:
    payload = _parse_json_object(raw_text)
    status = payload.get("review_status")
    if status not in {"accepted", "revised", "uncertain"}:
        raise ValueError(f"invalid review_status: {status!r}")
    human_review = payload.get("human_review_recommended")
    if not isinstance(human_review, bool):
        raise ValueError("human_review_recommended must be boolean")
    return {
        "final_species": _validate_species(payload.get("final_species"), "final_species"),
        "confidence": _validate_confidence(payload.get("confidence")),
        "review_status": status,
        "human_review_recommended": human_review,
        "reasoning_brief": str(payload.get("reasoning_brief", "")),
    }


def proposal_payload(proposal: dict[str, str]) -> dict[str, Any]:
    return {
        "proposal_id": proposal["proposal_id"],
        "coordinate_frame": "local_window_seconds_0.000_to_0.300",
        "start_time": float(proposal["start_time"]),
        "end_time": float(proposal["end_time"]),
        "low_freq": float(proposal["low_freq"]),
        "high_freq": float(proposal["high_freq"]),
        "det_prob": float(proposal["det_prob"]),
    }


def labels_block() -> str:
    return "\n".join(f"- {label}" for label in ALLOWED_LABELS)


def build_agent1_system_prompt() -> str:
    return (
        "You are Agent 1, a bioacoustic species classifier. A detector proposal "
        "has already been selected as the target candidate. Classify only that "
        "selected proposal in the centred crop image. Do not change coordinates, "
        "do not output extra detections, and do not use ground-truth labels.\n\n"
        "Allowed species:\n"
        f"{labels_block()}\n\n"
        "Return valid JSON only."
    )


def build_agent2_system_prompt() -> str:
    return (
        "You are Agent 2, an acoustic reviewer and critic. Review Agent 1's "
        "species prediction against the same centred crop image and selected "
        "proposal coordinates. Check difficult taxa, especially Myotis, "
        "Plecotus, and Ozimops confusions. Decide accept, revise, or uncertain. "
        "Do not change coordinates and do not output extra detections.\n\n"
        "Allowed species:\n"
        f"{labels_block()}\n\n"
        "Return valid JSON only."
    )


def build_agent3_prompt(agent1: dict[str, Any], agent2: dict[str, Any]) -> str:
    payload = {
        "agent1_classifier_output": agent1,
        "agent2_reviewer_output": agent2,
        "allowed_species": list(ALLOWED_LABELS),
        "instructions": [
            "Produce one final forced-choice species label from the allowed list.",
            "If Agent 2 accepts, normally keep Agent 1's species.",
            "If Agent 2 revises, use the revised species if it is supported.",
            "If uncertain, still choose one label but set human_review_recommended=true.",
            "Return valid JSON only.",
        ],
    }
    return "/no_think\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def build_image_user_message(row: dict[str, str], proposal: dict[str, str], extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "anonymous_sample_id": row["anonymous_sample_id"],
        "image_variant": "centred_crop_no_box",
        "selected_detector_proposal": proposal_payload(proposal),
        "instructions": [
            "Classify only this selected detector proposal.",
            "Preserve selected proposal coordinates exactly.",
            "Do not output additional detections.",
            "Choose labels only from the allowed species list.",
            "Ignore other calls, noise, axes, gridlines, and padding.",
            "Return valid JSON only.",
        ],
    }
    if extra:
        payload.update(extra)
    return "/no_think\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def call_ollama_text(
    *,
    system_prompt: str,
    user_message: str,
    model_name: str,
    timeout: float,
    num_predict: int,
    response_schema: dict[str, Any],
) -> str:
    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "format": response_schema,
        "prompt": f"{system_prompt}\n\n{user_message}",
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    from scripts.inference.run_stage1a_multispecies_classification import ollama_host

    request = urllib.request.Request(
        f"{ollama_host()}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    content = response_payload.get("response")
    return str(content) if content else json.dumps(response_payload, indent=2, ensure_ascii=False)


def safe_agent_record(
    *,
    row: dict[str, str],
    agent_name: str,
    parse_status: str,
    parsed: dict[str, Any],
    raw_response_path: Path,
    parse_error: str,
) -> dict[str, Any]:
    return {
        "agent": agent_name,
        "sample_id": row["sample_id"],
        "anonymous_sample_id": row["anonymous_sample_id"],
        "parse_status": parse_status,
        "parsed": parsed,
        "raw_response_path": raw_response_path.as_posix(),
        "parse_error": parse_error,
    }


def run_one_sample(
    *,
    row: dict[str, str],
    proposal: dict[str, str],
    run_dir: Path,
    model_name: str,
    timeout: float,
    num_predict: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    anon_id = row["anonymous_sample_id"]
    image_path = resolve_repo_path(row["centred_crop_image_path"])
    raw_dir = run_dir / "raw_responses" / anon_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    agent1_raw = raw_dir / "agent1_classifier_raw.txt"
    agent2_raw = raw_dir / "agent2_reviewer_raw.txt"
    agent3_raw = raw_dir / "agent3_adjudicator_raw.txt"

    try:
        raw1 = call_ollama_generate(
            image_path=image_path,
            system_prompt=build_agent1_system_prompt(),
            user_message=build_image_user_message(row, proposal),
            model_name=model_name,
            timeout=timeout,
            num_predict=num_predict,
            response_schema=AGENT1_SCHEMA,
        )
        agent1_raw.write_text(raw1, encoding="utf-8")
        parsed1 = parse_agent1(raw1)
        agent1_status, agent1_error = "success", ""
    except Exception as exc:
        if not agent1_raw.exists():
            agent1_raw.write_text("", encoding="utf-8")
        parsed1 = {}
        agent1_status, agent1_error = "failed", f"{type(exc).__name__}: {exc}"

    try:
        raw2 = call_ollama_generate(
            image_path=image_path,
            system_prompt=build_agent2_system_prompt(),
            user_message=build_image_user_message(row, proposal, {"agent1_classifier_output": parsed1}),
            model_name=model_name,
            timeout=timeout,
            num_predict=num_predict,
            response_schema=AGENT2_SCHEMA,
        )
        agent2_raw.write_text(raw2, encoding="utf-8")
        parsed2 = parse_agent2(raw2)
        agent2_status, agent2_error = "success", ""
    except Exception as exc:
        if not agent2_raw.exists():
            agent2_raw.write_text("", encoding="utf-8")
        parsed2 = {}
        agent2_status, agent2_error = "failed", f"{type(exc).__name__}: {exc}"

    try:
        raw3 = call_ollama_text(
            system_prompt=(
                "You are Agent 3, the final adjudicator. Use only Agent 1 and "
                "Agent 2 outputs. Choose one final species from the allowed list."
            ),
            user_message=build_agent3_prompt(parsed1, parsed2),
            model_name=model_name,
            timeout=timeout,
            num_predict=num_predict,
            response_schema=AGENT3_SCHEMA,
        )
        agent3_raw.write_text(raw3, encoding="utf-8")
        parsed3 = parse_agent3(raw3)
        agent3_status, agent3_error = "success", ""
    except Exception as exc:
        if not agent3_raw.exists():
            agent3_raw.write_text("", encoding="utf-8")
        parsed3 = {}
        agent3_status, agent3_error = "failed", f"{type(exc).__name__}: {exc}"

    prediction = {
        "sample_id": row["sample_id"],
        "anonymous_sample_id": anon_id,
        "true_species": row["species"],
        "selected_proposal_id": proposal["proposal_id"],
        "selected_start_time": proposal["start_time"],
        "selected_end_time": proposal["end_time"],
        "selected_low_freq": proposal["low_freq"],
        "selected_high_freq": proposal["high_freq"],
        "agent1_parse_status": agent1_status,
        "agent1_predicted_species": parsed1.get("predicted_species", ""),
        "agent1_confidence": parsed1.get("confidence", ""),
        "agent2_parse_status": agent2_status,
        "agent2_review_decision": parsed2.get("review_decision", ""),
        "agent2_revised_species": parsed2.get("revised_species", ""),
        "agent2_human_review_recommended": str(parsed2.get("human_review_recommended", "")).lower(),
        "agent3_parse_status": agent3_status,
        "final_species": parsed3.get("final_species", ""),
        "final_confidence": parsed3.get("confidence", ""),
        "review_status": parsed3.get("review_status", ""),
        "human_review_recommended": str(parsed3.get("human_review_recommended", "")).lower(),
        "final_correct": str(parsed3.get("final_species", "") == row["species"]).lower(),
        "parse_error": " | ".join(error for error in (agent1_error, agent2_error, agent3_error) if error),
    }
    return (
        safe_agent_record(row=row, agent_name="agent1_classifier", parse_status=agent1_status, parsed=parsed1, raw_response_path=agent1_raw, parse_error=agent1_error),
        safe_agent_record(row=row, agent_name="agent2_reviewer", parse_status=agent2_status, parsed=parsed2, raw_response_path=agent2_raw, parse_error=agent2_error),
        safe_agent_record(row=row, agent_name="agent3_adjudicator", parse_status=agent3_status, parsed=parsed3, raw_response_path=agent3_raw, parse_error=agent3_error),
        prediction,
    )


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def per_species_metrics(rows: list[dict[str, Any]], prediction_field: str) -> list[dict[str, Any]]:
    metrics = []
    for species in ALLOWED_LABELS:
        tp = sum(row["true_species"] == species and row.get(prediction_field) == species for row in rows)
        fp = sum(row["true_species"] != species and row.get(prediction_field) == species for row in rows)
        fn = sum(row["true_species"] == species and row.get(prediction_field) != species for row in rows)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        metrics.append(
            {
                "species": species,
                "support": sum(row["true_species"] == species for row in rows),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "F1": f1_score(precision, recall),
            }
        )
    return metrics


def confusion_matrix(rows: list[dict[str, Any]], prediction_field: str) -> list[dict[str, Any]]:
    matrix = []
    for true_species in ALLOWED_LABELS:
        out: dict[str, Any] = {"true_species": true_species}
        for predicted_species in ALLOWED_LABELS:
            out[predicted_species] = sum(
                row["true_species"] == true_species and row.get(prediction_field) == predicted_species
                for row in rows
            )
        matrix.append(out)
    return matrix


def aggregate_metrics(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get(prediction_field) in ALLOWED_LABELS]
    correct = sum(row["true_species"] == row.get(prediction_field) for row in rows)
    species_rows = per_species_metrics(rows, prediction_field)
    return {
        "sample_count": len(rows),
        "valid_prediction_count": len(valid_rows),
        "accuracy": safe_div(correct, len(rows)),
        "macro_F1": mean(row["F1"] for row in species_rows) if species_rows else 0.0,
        "balanced_accuracy": mean(row["recall"] for row in species_rows) if species_rows else 0.0,
    }


def subset_accuracy(rows: list[dict[str, Any]], predicate: str, value: str) -> dict[str, Any]:
    subset = [row for row in rows if str(row.get(predicate, "")).lower() == value]
    return {
        "subset": f"{predicate}={value}",
        "count": len(subset),
        "accuracy": safe_div(sum(row.get("final_correct") == "true" for row in subset), len(subset)),
    }


def comparison_rows(
    selected_rows: list[dict[str, str]],
    qwen_predictions: dict[str, dict[str, str]],
    gpt_predictions: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for row in selected_rows:
        anon_id = row["anonymous_sample_id"]
        qwen = qwen_predictions.get(anon_id, {})
        gpt = gpt_predictions.get(anon_id, {})
        gpt_species = gpt.get("predicted_species", "")
        rows.append(
            {
                "sample_id": row["sample_id"],
                "anonymous_sample_id": anon_id,
                "true_species": row["species"],
                "qwen_single_agent_species": qwen.get("predicted_species", ""),
                "qwen_single_agent_correct": str(qwen.get("predicted_species", "") == row["species"]).lower(),
                "gpt_single_agent_species": gpt_species,
                "gpt_single_agent_correct": str(gpt_species == row["species"]).lower() if gpt_species else "",
            }
        )
    return rows


def write_report(
    path: Path,
    *,
    selected_rows: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    agent_records: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    final_metrics: dict[str, Any],
    qwen_metrics: dict[str, Any],
    gpt_metrics: dict[str, Any] | None,
    review_metrics: list[dict[str, Any]],
) -> None:
    a1, a2, a3 = agent_records
    revised = sum(row.get("agent2_review_decision") == "revise" for row in predictions)
    uncertain = sum(
        row.get("review_status") == "uncertain" or row.get("human_review_recommended") == "true"
        for row in predictions
    )
    lines = [
        "# qwen3.6 Multi-Agent Stage 2C Pilot24",
        "",
        "## Scope",
        "",
        "This pilot used label-safe centred crops and deterministic nearest-centre BatDetect2 proposals. The selected proposal coordinates were preserved exactly; the agents were not asked to redraw boxes or output extra detections. Ground-truth species labels were used only after prediction for evaluation.",
        "",
        "## Pilot Samples",
        "",
        f"- Samples: {len(selected_rows)}",
        "- Selection: 3 samples per species where selected proposals were available, preferring qwen3.6 single-agent Stage 2C errors.",
        "",
        "## Parse Status",
        "",
        f"- Agent 1 parse success: {sum(r['parse_status'] == 'success' for r in a1)}/{len(a1)}",
        f"- Agent 2 parse success: {sum(r['parse_status'] == 'success' for r in a2)}/{len(a2)}",
        f"- Agent 3 parse success: {sum(r['parse_status'] == 'success' for r in a3)}/{len(a3)}",
        "",
        "## Final Classification Metrics",
        "",
        f"- Accuracy: {final_metrics['accuracy']:.3f}",
        f"- Macro-F1: {final_metrics['macro_F1']:.3f}",
        f"- Balanced accuracy: {final_metrics['balanced_accuracy']:.3f}",
        "",
        "## Review Metrics",
        "",
        f"- Revised cases: {revised}",
        f"- Uncertain or human-review cases: {uncertain}",
    ]
    for row in review_metrics:
        lines.append(f"- {row['subset']}: count {row['count']}, accuracy {row['accuracy']:.3f}")
    lines.extend(
        [
            "",
            "## Same-Sample Comparisons",
            "",
            f"- qwen3.6 single-agent Stage 2C accuracy on these samples: {qwen_metrics['accuracy']:.3f}",
        ]
    )
    if gpt_metrics is not None:
        lines.append(f"- GPT-5.6 Sol single-agent Stage 2C accuracy on these samples: {gpt_metrics['accuracy']:.3f}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This pilot tests whether a critic/adjudicator structure improves qwen3.6 species decisions once localisation is fixed by selected proposals. The strongest signal to inspect is whether review flags concentrate on incorrect or difficult samples, rather than only whether forced-choice accuracy improves.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected-proposals", type=Path, default=DEFAULT_SELECTED_PROPOSALS)
    parser.add_argument("--qwen-predictions", type=Path, default=QWEN_STAGE2C_FULL240)
    parser.add_argument("--gpt-predictions", type=Path, default=GPT_STAGE2C_FULL240)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir.exists() and (args.run_dir / "multi_agent_pilot_predictions.csv").exists():
        raise FileExistsError(f"Output already exists: {args.run_dir}")
    if args.ollama_host:
        import os

        os.environ["OLLAMA_HOST"] = args.ollama_host
    from scripts.inference.run_stage1a_multispecies_classification import ollama_host

    availability = assert_model_available(args.model_name, ollama_host())
    manifest_rows = load_manifest(args.manifest)
    selected_by_anon = selected_proposal_index(args.selected_proposals, "full240")
    qwen_by_anon = comparison_prediction_index(args.qwen_predictions)
    gpt_by_anon = comparison_prediction_index(args.gpt_predictions)
    selected_rows = select_pilot_samples(manifest_rows, selected_by_anon, qwen_by_anon, per_species=3)
    if len(selected_rows) != 24:
        raise RuntimeError(f"Expected 24 selected samples, got {len(selected_rows)}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.run_dir / "selected_pilot_samples.csv", selected_rows)
    write_csv(args.analysis_dir / "selected_pilot_samples.csv", selected_rows)
    (args.run_dir / "availability_check.json").write_text(
        json.dumps(availability, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    agent1_records: list[dict[str, Any]] = []
    agent2_records: list[dict[str, Any]] = []
    agent3_records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        anon_id = row["anonymous_sample_id"]
        print(f"[{index}/{len(selected_rows)}] {anon_id}", flush=True)
        proposal = selected_by_anon[anon_id]
        a1, a2, a3, prediction = run_one_sample(
            row=row,
            proposal=proposal,
            run_dir=args.run_dir,
            model_name=args.model_name,
            timeout=args.timeout,
            num_predict=args.num_predict,
        )
        agent1_records.append(a1)
        agent2_records.append(a2)
        agent3_records.append(a3)
        predictions.append(prediction)

    write_jsonl(args.run_dir / "agent1_classifier_outputs.jsonl", agent1_records)
    write_jsonl(args.run_dir / "agent2_reviewer_outputs.jsonl", agent2_records)
    write_jsonl(args.run_dir / "agent3_adjudicator_outputs.jsonl", agent3_records)
    write_csv(args.run_dir / "multi_agent_pilot_predictions.csv", predictions)
    write_csv(args.analysis_dir / "multi_agent_pilot_predictions.csv", predictions)

    final_metrics = aggregate_metrics(predictions, "final_species")
    species_rows = per_species_metrics(predictions, "final_species")
    confusion_rows = confusion_matrix(predictions, "final_species")
    write_csv(args.analysis_dir / "per_species_metrics.csv", species_rows)
    write_csv(args.analysis_dir / "confusion_matrix.csv", confusion_rows)

    qwen_comparison = comparison_rows(selected_rows, qwen_by_anon, gpt_by_anon)
    qwen_metrics = aggregate_metrics(
        [
            {"true_species": row["true_species"], "predicted": row["qwen_single_agent_species"]}
            for row in qwen_comparison
        ],
        "predicted",
    )
    gpt_metrics = None
    if any(row["gpt_single_agent_species"] for row in qwen_comparison):
        gpt_metrics = aggregate_metrics(
            [
                {"true_species": row["true_species"], "predicted": row["gpt_single_agent_species"]}
                for row in qwen_comparison
            ],
            "predicted",
        )
    comparison_out = []
    for pred in predictions:
        comp = next(row for row in qwen_comparison if row["anonymous_sample_id"] == pred["anonymous_sample_id"])
        comparison_out.append({**comp, "multi_agent_final_species": pred["final_species"], "multi_agent_correct": pred["final_correct"]})
    write_csv(args.analysis_dir / "same_sample_comparison.csv", comparison_out)

    review_metrics = [
        subset_accuracy(predictions, "review_status", "accepted"),
        subset_accuracy(predictions, "review_status", "revised"),
        subset_accuracy(predictions, "review_status", "uncertain"),
        subset_accuracy(predictions, "human_review_recommended", "true"),
        subset_accuracy(predictions, "human_review_recommended", "false"),
    ]
    summary = {
        "condition": CONDITION_NAME,
        "model_name": args.model_name,
        "sample_count": len(predictions),
        "agent1_parse_success": sum(r["parse_status"] == "success" for r in agent1_records),
        "agent2_parse_success": sum(r["parse_status"] == "success" for r in agent2_records),
        "agent3_parse_success": sum(r["parse_status"] == "success" for r in agent3_records),
        "final_metrics": final_metrics,
        "qwen_single_agent_same_sample_metrics": qwen_metrics,
        "gpt_single_agent_same_sample_metrics": gpt_metrics,
        "revised_cases": sum(row.get("agent2_review_decision") == "revise" for row in predictions),
        "uncertain_or_human_review_cases": sum(
            row.get("review_status") == "uncertain" or row.get("human_review_recommended") == "true"
            for row in predictions
        ),
        "review_metrics": review_metrics,
    }
    (args.analysis_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.analysis_dir / "multi_agent_pilot_report.md",
        selected_rows=selected_rows,
        predictions=predictions,
        agent_records=(agent1_records, agent2_records, agent3_records),
        final_metrics=final_metrics,
        qwen_metrics=qwen_metrics,
        gpt_metrics=gpt_metrics,
        review_metrics=review_metrics,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
