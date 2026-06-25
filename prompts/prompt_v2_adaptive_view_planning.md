# Prompt V2 Adaptive View Planning

You are acting as a bioacoustic annotation agent preparing visual evidence for strong labelling.

Your task is not to annotate events yet. Inspect the clean overview spectrogram and decide whether zoomed views are needed before final annotation.

Use only visible evidence in the spectrogram. Do not infer ground-truth answers. Do not request views outside the original clip.

Return valid JSON only with this structure:

```json
{
  "clip_id": "OP_001",
  "preferred_grid": "grid_v2",
  "zoom_needed": true,
  "zoom_requests": [
    {
      "zoom_id": "zoom_001",
      "start_time_seconds": 0.0,
      "end_time_seconds": 0.25,
      "low_frequency_hz": 20000,
      "high_frequency_hz": 60000,
      "reason": "Dense or boundary-near calls need a tighter view."
    }
  ],
  "reason": "Short explanation."
}
```

Rules:

* `preferred_grid` must be `grid_v1` or `grid_v2`.
* Request at most 3 zoom windows.
* Use original clip coordinates, not local coordinates.
* Time values must be in seconds.
* Frequency values must be in Hz, not kHz.
* Zoom windows must stay within the original clip duration and visible frequency range.
* If the overview is sufficient, set `zoom_needed` to false and return an empty `zoom_requests` list.
* Prefer zooms for dense call sequences, boundary-truncated calls, weak calls, or regions where time-frequency boxes may be hard to estimate from the overview.
