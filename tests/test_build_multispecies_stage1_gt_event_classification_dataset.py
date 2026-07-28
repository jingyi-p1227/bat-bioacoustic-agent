import csv
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from scripts.analysis.build_multispecies_stage1_gt_event_classification_dataset import (
    anonymous_sample_id,
    contains_species_token,
    human_overlay_path,
    model_image_path,
)


def test_anonymous_sample_id_has_no_species_text() -> None:
    anon = anonymous_sample_id(42)

    assert anon == "sample_000042"
    assert not contains_species_token(anon)


def test_model_image_paths_are_anonymous() -> None:
    path = model_image_path(Path("out"), "centred_crop_no_box", "sample_000001")

    assert path.as_posix() == "out/centred_crop_no_box/sample_000001.png"
    assert not contains_species_token(path.as_posix())


def test_human_overlay_path_is_separate_from_model_variants() -> None:
    overlay = human_overlay_path(Path("out"), "sample_000001")

    assert "human_review_overlays" in overlay.as_posix()
    assert overlay != model_image_path(Path("out"), "gt_box_marker", "sample_000001")


def test_stage1_generated_dataset_integrity_if_present() -> None:
    manifest_path = Path(
        "outputs/analysis_reports/"
        "multispecies_stage1_gt_event_classification_dataset/stage1_manifest.csv"
    )
    if not manifest_path.exists():
        pytest.skip("Stage 1 generated dataset is not present")

    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))

    assert rows
    assert all(row["target_centered_pass"] == "true" for row in rows)
    assert all(row["label_safe_pass"] == "true" for row in rows)
    assert all(row["embedded_label_text_detected"] == "false" for row in rows)
    assert all(row["split_group"] and row["source_recording_id"] for row in rows)
    assert all(
        "human_review_overlays" not in row["centred_crop_image_path"]
        and "human_review_overlays" not in row["gt_box_marker_image_path"]
        for row in rows
    )
    assert all(
        not contains_species_token(row["centred_crop_image_path"])
        and not contains_species_token(row["gt_box_marker_image_path"])
        for row in rows
    )
    counts = Counter(row["species"] for row in rows)
    assert set(counts.values()) == {30}
    for row in rows:
        with Image.open(row["centred_crop_image_path"]) as image:
            assert image.size == (800, 600)
        with Image.open(row["gt_box_marker_image_path"]) as image:
            assert image.size == (800, 600)
