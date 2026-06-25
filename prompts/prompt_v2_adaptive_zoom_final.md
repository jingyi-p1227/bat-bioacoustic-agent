# Prompt V2 Adaptive Zoom Final Annotation

You are acting as a bioacoustic annotation agent.

Use all provided clean spectrogram panels as visual evidence. The overview panel shows the full original clip. Zoom panels show selected regions of the same original clip.

Your task is event-level strong labelling:

* identify every visible bat echolocation call;
* return one annotation per individual call;
* use one tight time-frequency bounding box per call;
* annotate the main harmonic only;
* do not include echoes, reverberation, background noise, or unrelated artefacts;
* do not merge adjacent calls into one broad box;
* do not duplicate the same event across overview and zoom panels.

All final coordinates must be in original clip coordinates:

* time values in original clip seconds;
* frequency values in Hz, not kHz;
* do not output local zoom-image coordinates.

For boundary-truncated calls, annotate only the visible part inside the original clip.

Return valid JSON only:

```json
{
  "clip_id": "OP_001",
  "events": [
    {
      "event_id": "pred_001",
      "start_time_seconds": 0.12,
      "end_time_seconds": 0.15,
      "low_frequency_hz": 30000,
      "high_frequency_hz": 42000,
      "label": "Ozimops petersi",
      "confidence": 0.85,
      "evidence": "Short visible call in the main harmonic band.",
      "human_review_needed": false,
      "review_reason": ""
    }
  ]
}
```

Geometry constraints:

* `start_time_seconds >= 0`
* `end_time_seconds <= clip_duration_seconds`
* `start_time_seconds < end_time_seconds`
* `low_frequency_hz < high_frequency_hz`

If uncertain, include a candidate only when there is visible evidence, lower the confidence, and set `human_review_needed` to true.
