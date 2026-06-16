import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import EventResult, SpectrogramEvent, TemporalEvent, TemporalEventResult


def test_temporal_event_result_validates_minimal_payload() -> None:
    payload = {
        "audio_path": "audio/example.wav",
        "events": [
            {
                "event_id": "event_001",
                "start_time_seconds": 1.2,
                "end_time_seconds": 1.8,
                "label": "possible bat call",
                "confidence": 0.72,
                "evidence": "Clear pulse-like energy in the spectrogram time window.",
                "tools_used": ["generate_spectrogram"],
                "human_review_needed": True,
                "review_reason": "Temporal boundary looks plausible but species is uncertain.",
            }
        ],
    }

    result = TemporalEventResult.model_validate(payload)

    assert result.audio_path == "audio/example.wav"
    assert result.notes == ""
    assert result.events[0].start_time_seconds == 1.2
    assert result.events[0].end_time_seconds == 1.8


def test_temporal_event_allows_zero_duration_boundary_case() -> None:
    event = TemporalEvent(
        event_id="event_001",
        start_time_seconds=2.0,
        end_time_seconds=2.0,
        label="instantaneous marker",
        confidence=0.5,
        evidence="Boundary marker only.",
        tools_used=[],
        human_review_needed=True,
        review_reason="Zero-duration event should be reviewed.",
    )

    assert event.end_time_seconds == event.start_time_seconds


def test_temporal_event_rejects_negative_time_range() -> None:
    with pytest.raises(ValidationError):
        TemporalEvent(
            event_id="event_001",
            start_time_seconds=2.0,
            end_time_seconds=1.0,
            label="invalid",
            confidence=0.5,
            evidence="Invalid range.",
            tools_used=[],
            human_review_needed=True,
            review_reason="End precedes start.",
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_temporal_event_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    with pytest.raises(ValidationError):
        TemporalEvent(
            event_id="event_001",
            start_time_seconds=0.0,
            end_time_seconds=1.0,
            label="invalid confidence",
            confidence=confidence,
            evidence="Confidence outside [0, 1].",
            tools_used=[],
            human_review_needed=True,
            review_reason="Invalid confidence.",
        )


def test_existing_spectrogram_event_result_still_validates_2d_boxes() -> None:
    event = SpectrogramEvent(
        event_id="box_001",
        start_time_seconds=0.5,
        end_time_seconds=0.8,
        low_frequency_hz=30000.0,
        high_frequency_hz=60000.0,
        label="possible bat call",
        confidence=0.6,
        evidence="Time-frequency box contains visible pulse energy.",
        tools_used=["generate_spectrogram", "zoom_spectrogram"],
        human_review_needed=True,
        review_reason="Species uncertain.",
    )
    result = EventResult(audio_path="audio/example.wav", events=[event])

    assert result.events[0].low_frequency_hz == 30000.0
    assert result.events[0].high_frequency_hz == 60000.0
