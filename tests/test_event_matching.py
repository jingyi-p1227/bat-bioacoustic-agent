from toy_audio_agent.evaluation.event_matching import (
    EventBox,
    MatchingProtocol,
    box_iou,
    evaluate_clip,
    frequency_iou,
    interval_iou,
    match_events,
)


def event(event_id: str, start: float, end: float, confidence: float | None = None) -> EventBox:
    return EventBox(
        event_id=event_id,
        start_time=start,
        end_time=end,
        low_frequency=30_000.0,
        high_frequency=40_000.0,
        confidence=confidence,
    )


def test_temporal_iou_threshold_boundary_is_inclusive() -> None:
    pred = event("p1", 0.0, 0.1)
    gt = event("g1", 0.05, 0.15)

    pairs, unmatched, missed = match_events([pred], [gt], MatchingProtocol.TEMPORAL_IOU_0_3)

    assert len(pairs) == 1
    assert unmatched == []
    assert missed == []
    assert round(pairs[0].temporal_iou, 6) == round(1 / 3, 6)


def test_temporal_iou_0p1_is_more_permissive_than_0p3() -> None:
    pred = event("p1", 0.0, 0.1)
    gt = event("g1", 0.08, 0.18)

    pairs_loose, _, _ = match_events([pred], [gt], MatchingProtocol.TEMPORAL_IOU_0_1)
    pairs_strict, unmatched, missed = match_events([pred], [gt], MatchingProtocol.TEMPORAL_IOU_0_3)

    assert len(pairs_loose) == 1
    assert pairs_strict == []
    assert unmatched == [pred]
    assert missed == [gt]


def test_start_time_proximity_10ms_uses_onset_difference() -> None:
    pred = event("p1", 0.110, 0.200)
    gt = event("g1", 0.100, 0.120)

    pairs, unmatched, missed = match_events([pred], [gt], MatchingProtocol.START_TIME_PROXIMITY_10MS)

    assert len(pairs) == 1
    assert pairs[0].match_score == -0.009999999999999995
    assert unmatched == []
    assert missed == []


def test_start_time_proximity_rejects_outside_10ms() -> None:
    pred = event("p1", 0.111, 0.200)
    gt = event("g1", 0.100, 0.120)

    pairs, unmatched, missed = match_events([pred], [gt], MatchingProtocol.START_TIME_PROXIMITY_10MS)

    assert pairs == []
    assert unmatched == [pred]
    assert missed == [gt]


def test_confidence_ordered_one_to_one_matching() -> None:
    low_conf = event("p_low", 0.10, 0.20, confidence=0.2)
    high_conf = event("p_high", 0.11, 0.21, confidence=0.9)
    gt = event("g1", 0.10, 0.20)

    pairs, unmatched, missed = match_events([low_conf, high_conf], [gt], MatchingProtocol.TEMPORAL_IOU_0_3)

    assert pairs[0].prediction.event_id == "p_high"
    assert unmatched == [low_conf]
    assert missed == []


def test_prediction_overlapping_multiple_gt_chooses_best_iou() -> None:
    pred = event("p1", 0.10, 0.20)
    gt_weaker = event("g1", 0.08, 0.18)
    gt_better = event("g2", 0.10, 0.20)

    pairs, _, missed = match_events([pred], [gt_weaker, gt_better], MatchingProtocol.TEMPORAL_IOU_0_3)

    assert pairs[0].ground_truth.event_id == "g2"
    assert [item.event_id for item in missed] == ["g1"]


def test_box_quality_calculations() -> None:
    pred = event("p1", 0.0, 0.1)
    gt = event("g1", 0.05, 0.15)

    assert round(interval_iou(0.0, 0.1, 0.05, 0.15), 6) == round(1 / 3, 6)
    assert frequency_iou(pred, gt) == 1.0
    assert round(box_iou(pred, gt), 6) == round(1 / 3, 6)


def test_clip_metrics_from_pooled_counts() -> None:
    result = evaluate_clip(
        "OP_TEST",
        [event("p1", 0.0, 0.1), event("p2", 0.5, 0.6)],
        [event("g1", 0.0, 0.1), event("g2", 0.8, 0.9)],
        MatchingProtocol.TEMPORAL_IOU_0_3,
    )

    assert result.tp == 1
    assert result.fp == 1
    assert result.fn == 1
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5
