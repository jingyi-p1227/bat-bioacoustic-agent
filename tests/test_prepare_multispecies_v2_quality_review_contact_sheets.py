from scripts.analysis.prepare_multispecies_v2_quality_review_contact_sheets import (
    markdown_table,
    padding_category,
    select_species_review_rows,
)


def make_row(species: str, sample_id: str, left: float, right: float) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "species": species,
        "source_recording": f"{sample_id}.wav",
        "left_padding_seconds": str(left),
        "right_padding_seconds": str(right),
        "target_center_x_fraction": "0.5",
        "image_path": f"clean/{sample_id}.png",
        "human_review_overlay_path": f"overlay/{sample_id}.png",
    }


def test_padding_category_prefers_left_then_right_then_none() -> None:
    assert padding_category(make_row("A", "left", 0.1, 0.0)) == "left_padding"
    assert padding_category(make_row("A", "right", 0.0, 0.1)) == "right_padding"
    assert padding_category(make_row("A", "none", 0.0, 0.0)) == "no_padding"


def test_select_species_review_rows_chooses_padding_mix() -> None:
    rows = [
        make_row("Species A", "a_left", 0.1, 0.0),
        make_row("Species A", "a_none", 0.0, 0.0),
        make_row("Species A", "a_right", 0.0, 0.1),
        make_row("Species A", "a_extra", 0.0, 0.0),
        make_row("Species B", "b_none", 0.0, 0.0),
    ]

    selected = select_species_review_rows(rows)

    assert [row["sample_id"] for row in selected[:3]] == [
        "a_none",
        "a_left",
        "a_right",
    ]
    assert [row["sample_id"] for row in selected[3:]] == ["b_none"]


def test_markdown_table_includes_required_paths() -> None:
    table = markdown_table([make_row("Species A", "a_none", 0.0, 0.0)])

    assert "sample_id" in table
    assert "image_path" in table
    assert "overlay_path" in table
    assert "clean/a_none.png" in table
