"""Run selected full45 single-agent localisation conditions.

The runner uses clean grid_v2 spectrogram images only. It never reads ground
truth files, overlays, raw PDFs, or full extracted source text. Condition
context is supplied as compact metadata:

- proposal_constrained: BatDetect2 proposal JSON;
- walters_acoustic: compact generic Walters card and evidence chunks;
- annotation_exemplars: source-recording-safe abstract annotation memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_prompt_v2_small_pilot import (  # noqa: E402
    PROMPT_VERSION,
    call_ollama_generate,
    load_prompt,
    parse_prediction,
    read_clip_duration,
    resolve_all_clip_ids,
    resolve_input_image,
)


ConditionName = Literal[
    "proposal_constrained",
    "p14_best_stack_qwen3_6_proposal_constrained_conservative",
    "walters_acoustic",
    "annotation_exemplars",
]

DEFAULT_MODEL_NAME = "qwen3.6:latest"
DEFAULT_EVAL_DIR = Path("outputs/evaluation_sets/ozimops_petersi_v1")
DEFAULT_IMAGE_DIR = Path("outputs/agent_inputs/prompt_v2_full_grid_v2")
DEFAULT_PROMPT = Path("prompts/prompt_v2_bat_strong_label.md")
DEFAULT_PROPOSAL_DIR = Path("outputs/tool_outputs/batdetect2_proposals/full45")
DEFAULT_ANNOTATION_MEMORY = Path(
    "docs/annotation_example_library/annotation_memory.jsonl"
)
DEFAULT_WALTERS_CARD = Path(
    "docs/acoustic_reference_library/species_cards/generic_walters_2012_acoustic_guidance.md"
)
DEFAULT_WALTERS_CHUNK_DIR = Path("docs/acoustic_reference_library/evidence_chunks")
OUTPUT_DIRS: dict[ConditionName, Path] = {
    "proposal_constrained": Path(
        "outputs/agent_runs/p7c_full45_proposal_constrained_qwen3_6"
    ),
    "p14_best_stack_qwen3_6_proposal_constrained_conservative": Path(
        "outputs/agent_runs/p14_best_stack_qwen3_6_proposal_constrained_conservative"
    ),
    "walters_acoustic": Path(
        "outputs/agent_runs/p7c_full45_walters_acoustic_qwen3_6"
    ),
    "annotation_exemplars": Path(
        "outputs/agent_runs/p7c_full45_annotation_memory_qwen3_6"
    ),
}


@dataclass(frozen=True)
class Full45RunResult:
    clip_id: str
    parse_status: str
    predicted_event_count: int | None
    prediction_path: Path
    raw_response_path: Path
    parse_error_path: Path | None


def parse_clip_ids(value: str) -> list[str]:
    clip_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not clip_ids:
        raise ValueError("--clip-list must contain at least one clip id")
    return list(dict.fromkeys(clip_ids))


def read_manifest_source_recordings(eval_dir: Path) -> dict[str, str]:
    """Return clip_id -> source recording mapping from the manifest."""
    manifest = eval_dir / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        return {row["clip_id"]: row["source_recording"] for row in csv.DictReader(handle)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def safe_annotation_record(record: dict[str, Any]) -> dict[str, Any]:
    """Strip figure paths and provenance paths from model-facing exemplars."""
    return {
        "case_id": record.get("case_id"),
        "case_type": record.get("case_type", []),
        "observable_features": record.get("observable_features", []),
        "known_failure_modes": record.get("known_failure_modes", []),
        "recommended_actions": record.get("recommended_actions", []),
        "anti_patterns": record.get("anti_patterns", []),
        "provenance": {
            "evidence_basis": record.get("provenance", {}).get("evidence_basis", ""),
            "curation_status": record.get("provenance", {}).get("curation_status", ""),
        },
    }


def retrieve_source_safe_annotation_examples(
    *,
    clip_id: str,
    source_recordings: dict[str, str],
    memory_records: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve abstract examples while excluding same-source-recording cases."""
    target_source = source_recordings.get(clip_id, "")
    selected: list[dict[str, Any]] = []
    excluded: list[str] = []
    for record in memory_records:
        case_id = str(record.get("case_id") or "")
        if not case_id or case_id == clip_id:
            excluded.append(case_id)
            continue
        if source_recordings.get(case_id) == target_source:
            excluded.append(case_id)
            continue
        selected.append(safe_annotation_record(record))
        if len(selected) >= top_k:
            break
    trace = {
        "clip_id": clip_id,
        "target_source_recording": target_source,
        "retrieved_case_ids": [str(row["case_id"]) for row in selected],
        "excluded_case_ids": excluded,
        "source_recording_safe": all(
            source_recordings.get(str(row["case_id"])) != target_source
            for row in selected
        ),
        "target_case_excluded": all(str(row["case_id"]) != clip_id for row in selected),
    }
    return selected, trace


def load_walters_context(card_path: Path, chunk_dir: Path) -> dict[str, Any]:
    """Load compact Walters card and chunk summaries without reading full text."""
    card_text = card_path.read_text(encoding="utf-8")
    chunks = []
    for path in sorted(chunk_dir.glob("walters_2012_chunk_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks.append(
            {
                "evidence_id": payload["evidence_id"],
                "page_number": payload.get("page_number"),
                "chunk_text": payload.get("chunk_text"),
                "detected_acoustic_concepts": payload.get(
                    "detected_acoustic_concepts", []
                ),
                "caution_note": payload.get("caution_note"),
            }
        )
    if not chunks:
        raise FileNotFoundError("No Walters evidence chunks found")
    return {
        "card_path": card_path.as_posix(),
        "card_text": card_text,
        "evidence_chunks": chunks,
        "safety": [
            "Generic European-bat acoustic guidance only.",
            "Do not infer Ozimops petersi species-specific numeric priors.",
            "Do not force duration, frequency range, bandwidth, or shape.",
        ],
    }


def load_proposal_context(proposal_dir: Path, clip_id: str) -> list[dict[str, Any]]:
    """Load compact BatDetect2 proposals for one clip."""
    path = proposal_dir / f"{clip_id}_batdetect2_proposals.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for event in payload.get("events", []):
        start = float(event["start_time_seconds"])
        end = float(event["end_time_seconds"])
        rows.append(
            {
                "proposal_id": event.get("proposal_id"),
                "start_time_seconds": start,
                "end_time_seconds": end,
                "duration_ms": round((end - start) * 1000.0, 3),
                "low_frequency_hz": event.get("low_frequency_hz"),
                "high_frequency_hz": event.get("high_frequency_hz"),
                "det_prob": event.get("det_prob"),
                "class_prob": event.get("class_prob"),
                "original_label": event.get("label"),
            }
        )
    return rows


def build_condition_user_message(
    *,
    condition: ConditionName,
    clip_id: str,
    clip_duration_seconds: float,
    condition_context: Any,
) -> str:
    """Build the condition-specific user message for clean grid_v2 input."""
    runtime = {
        "clip_id": clip_id,
        "clip_duration_seconds": clip_duration_seconds,
        "frequency_axis_unit": "kHz",
        "return_frequency_unit": "Hz",
        "condition": condition,
    }
    if condition == "proposal_constrained":
        instruction = (
            "Use the BatDetect2 proposals as candidate hints, not ground truth. "
            "Verify each proposal against the clean spectrogram. Preserve proposal "
            "geometry when it fits visible evidence; refine only when the image "
            "clearly supports a better onset, offset, or frequency bound. You may "
            "remove false-positive proposals and add clearly visible missing calls."
        )
        context_key = "batdetect2_proposals"
    elif condition == "p14_best_stack_qwen3_6_proposal_constrained_conservative":
        instruction = (
            "Use the BatDetect2 proposals as candidate events, not ground truth. "
            "This is the conservative final best-stack condition: preserve proposal "
            "start_time_seconds and end_time_seconds unless there is clear visual "
            "evidence that the onset or offset is wrong. Prefer retain, reject, or "
            "small evidence-based refinement over free-form redrawing. Avoid "
            "unsupported onset/offset shifts, duration expansion, or duration "
            "shrinkage. Refine frequency bounds conservatively only when the visible "
            "main harmonic clearly supports the adjustment. Do not invent new events "
            "unless there is clear spectrogram evidence outside the proposal list."
        )
        context_key = "batdetect2_proposals"
    elif condition == "walters_acoustic":
        instruction = (
            "Use the compact Walters acoustic guidance only as a generic checklist "
            "for onset, offset, frequency bounds, duration, bandwidth, ridge shape, "
            "and noise rejection. Do not use it as an Ozimops petersi prior and do "
            "not force European species-specific ranges."
        )
        context_key = "generic_acoustic_guidance"
    elif condition == "annotation_exemplars":
        instruction = (
            "Use the source-recording-safe annotation examples only as abstract "
            "lessons about failure modes and review strategy. They are not answers "
            "for this clip. Do not copy coordinates or infer ground truth from them."
        )
        context_key = "source_recording_safe_annotation_examples"
    else:
        raise ValueError(f"Unsupported condition: {condition}")
    payload = {
        "runtime_context": runtime,
        "condition_instruction": instruction,
        context_key: condition_context,
        "output_requirement": "Return valid JSON only with clip_id and events.",
    }
    return (
        "/no_think\n"
        "Annotate the attached clean grid_v2 spectrogram according to the system prompt.\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
        "Do not explain your reasoning. Return the valid JSON object immediately."
    )


def condition_context_for_clip(
    *,
    condition: ConditionName,
    clip_id: str,
    proposal_dir: Path,
    walters_context: dict[str, Any] | None,
    source_recordings: dict[str, str],
    memory_records: list[dict[str, Any]],
    retrieval_dir: Path,
    top_k: int,
) -> Any:
    if condition in {
        "proposal_constrained",
        "p14_best_stack_qwen3_6_proposal_constrained_conservative",
    }:
        return load_proposal_context(proposal_dir, clip_id)
    if condition == "walters_acoustic":
        if walters_context is None:
            raise ValueError("Walters context was not loaded")
        return walters_context
    if condition == "annotation_exemplars":
        examples, trace = retrieve_source_safe_annotation_examples(
            clip_id=clip_id,
            source_recordings=source_recordings,
            memory_records=memory_records,
            top_k=top_k,
        )
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        (retrieval_dir / f"{clip_id}_retrieval_trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not trace["source_recording_safe"]:
            raise ValueError(f"Unsafe annotation retrieval for {clip_id}")
        return examples
    raise ValueError(f"Unsupported condition: {condition}")


def write_prediction(
    *,
    path: Path,
    prediction: dict[str, Any],
    condition: ConditionName,
    model_name: str,
    image_path: Path,
    clip_duration_seconds: float,
) -> None:
    payload = {
        "clip_id": prediction["clip_id"],
        "prompt_version": PROMPT_VERSION,
        "condition": condition,
        "model_name": model_name,
        "backend": "ollama_generate",
        "input_image_path": image_path.as_posix(),
        "clip_duration_seconds": clip_duration_seconds,
        "events": prediction["events"],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_failed_prediction(
    *,
    path: Path,
    clip_id: str,
    condition: ConditionName,
    model_name: str,
    image_path: Path | None,
    clip_duration_seconds: float | None,
    error: str,
) -> None:
    payload = {
        "clip_id": clip_id,
        "prompt_version": PROMPT_VERSION,
        "condition": condition,
        "model_name": model_name,
        "backend": "ollama_generate",
        "input_image_path": "" if image_path is None else image_path.as_posix(),
        "clip_duration_seconds": clip_duration_seconds,
        "parse_status": "failed",
        "error": error,
        "events": [],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_clip(
    *,
    condition: ConditionName,
    clip_id: str,
    prompt_text: str,
    eval_dir: Path,
    image_dir: Path,
    output_dir: Path,
    model_name: str,
    timeout: float,
    num_predict: int,
    proposal_dir: Path,
    walters_context: dict[str, Any] | None,
    source_recordings: dict[str, str],
    memory_records: list[dict[str, Any]],
    top_k: int,
) -> Full45RunResult:
    predictions_dir = output_dir / "predictions"
    raw_dir = output_dir / "raw_responses"
    retrieval_dir = output_dir / "retrieval_traces"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{clip_id}_predictions.json"
    raw_path = raw_dir / f"{clip_id}_raw_response.txt"
    error_path = predictions_dir / f"{clip_id}_parse_error.txt"
    if prediction_path.exists():
        raise FileExistsError(f"Prediction already exists: {prediction_path}")
    image_path: Path | None = None
    clip_duration: float | None = None
    try:
        image_path = resolve_input_image(image_dir, clip_id)
        clip_duration = read_clip_duration(eval_dir, clip_id)
        context = condition_context_for_clip(
            condition=condition,
            clip_id=clip_id,
            proposal_dir=proposal_dir,
            walters_context=walters_context,
            source_recordings=source_recordings,
            memory_records=memory_records,
            retrieval_dir=retrieval_dir,
            top_k=top_k,
        )
        raw_text = call_ollama_generate(
            image_path=image_path,
            system_prompt=prompt_text,
            user_message=build_condition_user_message(
                condition=condition,
                clip_id=clip_id,
                clip_duration_seconds=clip_duration,
                condition_context=context,
            ),
            model_name=model_name,
            timeout=timeout,
            num_predict=num_predict,
        )
        raw_path.write_text(raw_text, encoding="utf-8")
        prediction = parse_prediction(raw_text, expected_clip_id=clip_id)
        write_prediction(
            path=prediction_path,
            prediction=prediction,
            condition=condition,
            model_name=model_name,
            image_path=image_path,
            clip_duration_seconds=clip_duration,
        )
        error_path.unlink(missing_ok=True)
        return Full45RunResult(
            clip_id=clip_id,
            parse_status="success",
            predicted_event_count=len(prediction["events"]),
            prediction_path=prediction_path,
            raw_response_path=raw_path,
            parse_error_path=None,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if not raw_path.exists():
            raw_path.write_text("", encoding="utf-8")
        error_path.write_text(message + "\n", encoding="utf-8")
        write_failed_prediction(
            path=prediction_path,
            clip_id=clip_id,
            condition=condition,
            model_name=model_name,
            image_path=image_path,
            clip_duration_seconds=clip_duration,
            error=message,
        )
        return Full45RunResult(
            clip_id=clip_id,
            parse_status="failed",
            predicted_event_count=None,
            prediction_path=prediction_path,
            raw_response_path=raw_path,
            parse_error_path=error_path,
        )


def write_run_summary(output_dir: Path, results: list[Full45RunResult]) -> None:
    rows = [
        {
            "clip_id": result.clip_id,
            "parse_status": result.parse_status,
            "predicted_event_count": "" if result.predicted_event_count is None else result.predicted_event_count,
            "prediction_path": result.prediction_path.as_posix(),
            "raw_response_path": result.raw_response_path.as_posix(),
            "parse_error_path": "" if result.parse_error_path is None else result.parse_error_path.as_posix(),
        }
        for result in results
    ]
    path = output_dir / "parse_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "proposal_constrained",
            "p14_best_stack_qwen3_6_proposal_constrained_conservative",
            "walters_acoustic",
            "annotation_exemplars",
        ],
    )
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--proposal-dir", type=Path, default=DEFAULT_PROPOSAL_DIR)
    parser.add_argument("--annotation-memory", type=Path, default=DEFAULT_ANNOTATION_MEMORY)
    parser.add_argument("--walters-card", type=Path, default=DEFAULT_WALTERS_CARD)
    parser.add_argument("--walters-chunk-dir", type=Path, default=DEFAULT_WALTERS_CHUNK_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clip-list", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--num-predict", type=int, default=8000)
    parser.add_argument("--top-k-exemplars", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    condition: ConditionName = args.condition
    output_dir = args.output_dir or OUTPUT_DIRS[condition]
    if output_dir.exists() and any((output_dir / "predictions").glob("OP_*_predictions.json")):
        raise FileExistsError(
            f"Output predictions already exist under {output_dir}. Choose a new --output-dir."
        )
    clip_ids = (
        resolve_all_clip_ids(args.eval_dir)
        if args.all or args.clip_list is None
        else parse_clip_ids(args.clip_list)
    )
    prompt_text = load_prompt(args.prompt)
    source_recordings = read_manifest_source_recordings(args.eval_dir)
    memory_records = (
        load_jsonl(args.annotation_memory)
        if condition == "annotation_exemplars"
        else []
    )
    walters_context = (
        load_walters_context(args.walters_card, args.walters_chunk_dir)
        if condition == "walters_acoustic"
        else None
    )
    results: list[Full45RunResult] = []
    for index, clip_id in enumerate(clip_ids, start=1):
        print(f"[{index}/{len(clip_ids)}] {condition} {clip_id}", flush=True)
        results.append(
            run_clip(
                condition=condition,
                clip_id=clip_id,
                prompt_text=prompt_text,
                eval_dir=args.eval_dir,
                image_dir=args.image_dir,
                output_dir=output_dir,
                model_name=args.model_name,
                timeout=args.timeout,
                num_predict=args.num_predict,
                proposal_dir=args.proposal_dir,
                walters_context=walters_context,
                source_recordings=source_recordings,
                memory_records=memory_records,
                top_k=args.top_k_exemplars,
            )
        )
    write_run_summary(output_dir, results)
    successes = sum(result.parse_status == "success" for result in results)
    print(
        f"Condition={condition} clips={len(results)} parse_success={successes} "
        f"parse_failure={len(results) - successes} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
