"""Evaluate P9-light predictions with the P8 multi-protocol evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_audio_agent.experiments.p9_light import (  # noqa: E402
    OPTIONAL_AGENT_CONDITION,
    PROPOSAL_ONLY_CONDITION,
    REQUIRED_AGENT_CONDITIONS,
    default_paths,
    evaluate_condition,
    summarise_parse,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional-condition", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = default_paths(REPO_ROOT)
    conditions = [PROPOSAL_ONLY_CONDITION, *REQUIRED_AGENT_CONDITIONS]
    if args.include_optional_condition:
        conditions.append(OPTIONAL_AGENT_CONDITION)
    summary_rows = []
    case_rows = []
    pair_rows = []
    for condition in conditions:
        condition_summary, condition_cases, condition_pairs = evaluate_condition(paths, condition)
        summary_rows.extend(condition_summary)
        case_rows.extend(condition_cases)
        pair_rows.extend(condition_pairs)
    paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(paths.analysis_dir / "p9_light_condition_summary.csv", summary_rows)
    write_csv(paths.analysis_dir / "p9_light_case_level_results.csv", case_rows)
    write_csv(paths.analysis_dir / "p9_light_protocol_comparison.csv", summary_rows)
    write_csv(paths.analysis_dir / "p9_light_matched_pair_box_quality.csv", pair_rows)
    write_csv(paths.analysis_dir / "p9_light_parse_summary.csv", summarise_parse(paths, conditions))
    print(f"Wrote P9-light evaluation outputs to {paths.analysis_dir}")


if __name__ == "__main__":
    main()
