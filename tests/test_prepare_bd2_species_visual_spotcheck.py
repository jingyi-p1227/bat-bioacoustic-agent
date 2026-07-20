from pathlib import Path

from scripts.analysis.prepare_bd2_species_visual_spotcheck import (
    CandidateRow,
    clean_preview_path,
    diagnostic_overlay_path,
    resolve_audio_path,
    slugify,
)


def test_slugify_species_name() -> None:
    assert slugify("Rhinolophus hipposideros") == "rhinolophus_hipposideros"
    assert slugify("Scotorepens sp. (Parnaby)") == "scotorepens_sp_parnaby"


def test_output_paths_are_separate_for_clean_and_diagnostic() -> None:
    candidate = CandidateRow(
        species="Plecotus auritus",
        example_index=3,
        dataset="uk",
        source="echobank",
        recording_path="example.wav",
        event_count=12,
        duration_seconds=1.0,
        quality_hint="unmarked",
        difficulty_prior="challenging",
    )
    clean = clean_preview_path(Path("out"), candidate)
    diagnostic = diagnostic_overlay_path(Path("out"), candidate)
    assert clean.as_posix().endswith("plecotus_auritus/03_example_clean_preview.png")
    assert "diagnostic_overlays_human_review_only" in diagnostic.as_posix()
    assert clean != diagnostic


def test_resolve_audio_path_for_australia_and_uk() -> None:
    au = CandidateRow(
        species="Ozimops petersi",
        example_index=1,
        dataset="australia",
        source="australia",
        recording_path="pseudo_petersi_001.wav",
        event_count=19,
        duration_seconds=4.0,
        quality_hint="unmarked",
        difficulty_prior="moderate_or_unknown",
    )
    uk = CandidateRow(
        species="Rhinolophus hipposideros",
        example_index=1,
        dataset="uk",
        source="rhinolophus",
        recording_path="sample.wav",
        event_count=10,
        duration_seconds=1.0,
        quality_hint="unmarked",
        difficulty_prior="likely_easier",
    )
    assert resolve_audio_path(au).as_posix().endswith("datasets/australia/audio/pseudo_petersi_001.wav")
    assert resolve_audio_path(uk).as_posix().endswith("datasets/uk/sources/rhinolophus/audio/sample.wav")
