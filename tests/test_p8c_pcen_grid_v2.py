from __future__ import annotations

import importlib.util
from pathlib import Path

from scripts.data_prep.prepare_pcen_spectrogram_inputs import GRID_STYLE, pcen_image_path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pcen_generator_uses_grid_v2_filename() -> None:
    assert GRID_STYLE == "grid_v2"
    assert pcen_image_path(Path("out"), "OP_016").as_posix() == "out/OP_016_pcen_grid_v2.png"


def test_p8c_runner_uses_separate_output_paths() -> None:
    module = load_module(REPO_ROOT / "scripts/inference/run_p8c_pcen_grid_v2_full45.py")
    assert module.INPUT_DIR.name == "p8c_pcen_grid_v2_full45"
    assert module.RUN_DIR.name == "p8c_pcen_grid_v2_qwen3_6_full45"
    assert module.IMAGE_SUFFIX == "_pcen_grid_v2.png"
    assert module.REQUIRED_OLLAMA_HOST == "http://127.0.0.1:11436"


def test_p8c_evaluator_includes_three_comparison_runs() -> None:
    module = load_module(REPO_ROOT / "scripts/evaluation/evaluate_p8c_pcen_grid_v2.py")
    runs = module.p8c_runs(module.DEFAULT_EVAL_DIR)
    assert [run.experiment_id for run in runs] == [
        "p5_qwen_grid_v2_full",
        "p8b_pcen_full45",
        "p8c_pcen_grid_v2_full45",
    ]
