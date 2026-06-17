import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import _resolve_grid_steps, choose_readable_grid_steps


def test_choose_readable_grid_steps_for_zoom_window() -> None:
    steps = choose_readable_grid_steps(time_span=4.0, frequency_span=80000.0)

    assert steps.time_major_step == 0.5
    assert steps.time_minor_step == 0.1
    assert steps.frequency_major_step == 20000
    assert steps.frequency_minor_step == 10000


def test_fixed_grid_steps_preserve_existing_defaults() -> None:
    steps = _resolve_grid_steps(
        grid_step_mode="fixed",
        time_span=10.0,
        frequency_span=100000.0,
        time_major_step=None,
        time_minor_step=None,
        frequency_major_step=None,
        frequency_minor_step=None,
    )

    assert steps.time_major_step == 0.5
    assert steps.time_minor_step == 0.1
    assert steps.frequency_major_step == 10000
    assert steps.frequency_minor_step == 5000


def test_auto_grid_steps_preserve_explicit_overrides() -> None:
    steps = _resolve_grid_steps(
        grid_step_mode="auto",
        time_span=4.0,
        frequency_span=80000.0,
        time_major_step=0.25,
        time_minor_step=None,
        frequency_major_step=None,
        frequency_minor_step=2500,
    )

    assert steps.time_major_step == 0.25
    assert steps.time_minor_step == 0.1
    assert steps.frequency_major_step == 20000
    assert steps.frequency_minor_step == 2500


def test_grid_step_validation_rejects_zero_explicit_step() -> None:
    with pytest.raises(ValueError, match="time_major_step"):
        _resolve_grid_steps(
            grid_step_mode="auto",
            time_span=4.0,
            frequency_span=80000.0,
            time_major_step=0,
            time_minor_step=None,
            frequency_major_step=None,
            frequency_minor_step=None,
        )
