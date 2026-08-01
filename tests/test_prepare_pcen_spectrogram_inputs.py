import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_prep.prepare_pcen_spectrogram_inputs import (
    DEFAULT_PCEN_PARAMETERS,
    PCENManifestRow,
    comparison_contact_sheet_path,
    linear_frequency_pcen,
    pcen_image_path,
    save_side_by_side_contact_sheet,
    write_pcen_manifest,
)


def test_pcen_image_and_contact_sheet_paths() -> None:
    output_dir = Path("outputs/agent_inputs/p6_pcen_spectrograms")

    assert pcen_image_path(output_dir, "OP_016") == (
        output_dir / "OP_016_pcen_grid_v2.png"
    )
    assert comparison_contact_sheet_path(output_dir, "OP_016") == (
        output_dir / "contact_sheets/OP_016_db_vs_pcen_contact_sheet.png"
    )


def test_linear_frequency_pcen_preserves_shape_and_is_finite() -> None:
    power = np.array(
        [[1.0, 1.0, 9.0, 1.0], [2.0, 2.0, 2.0, 8.0]], dtype=np.float64
    )

    enhanced = linear_frequency_pcen(power, hop_seconds=0.01)

    assert enhanced.shape == power.shape
    assert np.all(np.isfinite(enhanced))
    assert np.all(enhanced >= 0)


def test_manifest_records_parameters_as_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pcen_manifest.csv"
    parameter_json = json.dumps(
        DEFAULT_PCEN_PARAMETERS.__dict__, sort_keys=True, separators=(",", ":")
    )
    row = PCENManifestRow(
        clip_id="OP_016",
        image_path="outputs/pcen/OP_016_pcen_grid_v2.png",
        original_audio_path="outputs/eval/audio/OP_016.wav",
        representation="linear_frequency_pcen",
        grid_style="grid_v2",
        pcen_parameters=parameter_json,
        duration_seconds=1.0,
    )

    write_pcen_manifest([row], manifest_path)

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))[0]
    parameters = json.loads(saved["pcen_parameters"])
    assert saved["clip_id"] == "OP_016"
    assert saved["representation"] == "linear_frequency_pcen"
    assert parameters["alpha"] == 0.98
    assert parameters["time_constant_seconds"] == 0.4


def test_side_by_side_contact_sheet_is_created(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.png"
    pcen_path = tmp_path / "pcen.png"
    output_path = tmp_path / "contact/sheet.png"
    Image.new("RGB", (40, 20), "black").save(baseline_path)
    Image.new("RGB", (40, 20), "white").save(pcen_path)

    save_side_by_side_contact_sheet(
        baseline_path=baseline_path,
        pcen_path=pcen_path,
        clip_id="OP_TEST",
        output_path=output_path,
    )

    assert output_path.is_file()
    with Image.open(output_path) as contact_sheet:
        assert contact_sheet.width > 80
        assert contact_sheet.height > 20

