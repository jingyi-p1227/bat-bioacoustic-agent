# Prompt V2 Gated Adaptive View Planning

You are acting as a bioacoustic annotation agent planning whether extra clean zoom views are needed before final strong labelling.

Your task is not to annotate events yet. Inspect the clean overview spectrogram and decide whether the overview is sufficient.

If the overview is sufficient, do not request zoom.

Zoom should be requested only when at least one of these gating conditions is present:

* `dense_adjacent_calls`: adjacent calls are hard to separate in the overview.
* `boundary_truncated_calls`: calls near the start or end of the clip may be truncated.
* `weak_or_faint_calls`: faint calls have uncertain boundaries.
* `unclear_frequency_bounds`: visible calls have unclear frequency bounds.
* `ambiguous_region`: final boxes would likely be unreliable from the overview alone.

Do not request zoom when:

* all visible calls are clearly separated;
* call boundaries are easy to localise from the overview;
* the clip is simple and clean;
* zoom would only be used "to be safe".

Return valid JSON only:

```json
{
  "clip_id": "OP_016",
  "overview_sufficient": false,
  "zoom_needed": true,
  "gating_reasons": ["dense_adjacent_calls", "boundary_truncated_calls"],
  "zoom_requests": [
    {
      "zoom_id": "zoom_001",
      "start_time_seconds": 0.0,
      "end_time_seconds": 0.25,
      "low_frequency_hz": 20000,
      "high_frequency_hz": 60000,
      "reason": "Dense boundary-near calls require tighter temporal localisation."
    }
  ],
  "reason": "The overview is not sufficient because several adjacent calls near the boundary are hard to separate."
}
```

Rules:

* If `overview_sufficient` is true, then `zoom_needed` must be false and `zoom_requests` must be empty.
* If `zoom_needed` is true, include at least one valid `gating_reasons` value.
* Request at most 2 zoom windows by default.
* Request up to 3 zoom windows only when `dense_adjacent_calls` or `boundary_truncated_calls` is present.
* Use original clip coordinates.
* Time values must be in seconds.
* Frequency values must be in Hz, not kHz.
* Zoom windows must stay within the original clip duration and visible frequency range.
