from scripts.analysis.audit_bd2_species_candidates import (
    difficulty_label,
    quality_hint_from_name,
)


def test_difficulty_labels_requested_species_groups() -> None:
    assert difficulty_label("Rhinolophus hipposideros") == "likely_easier"
    assert difficulty_label("Myotis daubentonii") == "likely_harder"
    assert difficulty_label("Plecotus auritus") == "challenging"
    assert difficulty_label("Ozimops petersi") == "moderate_or_unknown"


def test_quality_hint_from_filename() -> None:
    assert quality_hint_from_name("OP_045_clean.wav") == "clean_hint"
    assert quality_hint_from_name("example_overlap_low.wav") == "hard_hint"
    assert quality_hint_from_name("call_return_a_rc.wav") == "context_or_quality_hint"
    assert quality_hint_from_name("plain.wav") == "unmarked"
