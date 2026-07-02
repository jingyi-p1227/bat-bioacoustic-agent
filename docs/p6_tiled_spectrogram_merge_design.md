# P6A.1 Tiled Spectrogram Merge Design

## Purpose

P6A.1 prepares overlapping clean spectrogram tiles for the six representative clips. It does not run a model or alter the frozen evaluation protocol. The next stage will test whether larger on-image call geometry improves strong labelling while preserving original clip coordinates.

## Tile Conditions

| Setting | Nominal tile duration | Nominal overlap | Step |
| --- | ---: | ---: | ---: |
| `tile_0p5_overlap_0p1` | 0.50 s | 0.10 s | 0.40 s |
| `tile_0p25_overlap_0p05` | 0.25 s | 0.05 s | 0.20 s |

The final tile is clipped to the true WAV duration. No empty padding is added. Every tile uses `grid_v2`, and its x-axis is displayed in original clip coordinates rather than restarting at zero.

## Coordinate Conversion

The preferred prediction contract is for the model to return original clip time coordinates directly because those coordinates are visible on each tile axis. Each model request should include:

- `clip_id`;
- `tile_id`;
- `tile_start_seconds`;
- `tile_end_seconds`;
- an instruction that output times must use original clip coordinates.

Before merging, validate each prediction as follows:

1. Require `start_time_seconds < end_time_seconds`.
2. Require the predicted interval to lie within the tile bounds, allowing only a small numeric tolerance.
3. Clamp tolerance-level floating-point excursions to the tile boundary.
4. Reject larger out-of-range coordinates and record a parse or coordinate warning.
5. Preserve frequency coordinates in Hz without conversion beyond the existing kHz-axis-to-Hz output instruction.

If a future runner deliberately requests tile-local coordinates, it must record `coordinate_frame: tile_local` and convert with:

```text
original_start = tile_start_seconds + local_start
original_end = tile_start_seconds + local_end
```

The merge code must not guess the coordinate frame from numeric values.

## Duplicate Merge Rule

Overlapping tiles can produce multiple predictions for the same call. The proposed first implementation is confidence-ordered, class-aware two-dimensional non-maximum suppression:

1. Combine predictions from all tiles belonging to one clip.
2. Validate and clamp coordinates before comparison.
3. Sort predictions by confidence descending. For absent confidence, use a documented neutral fallback and record a warning.
4. Prefer a prediction that does not touch a tile boundary over an otherwise similar boundary-clipped prediction.
5. Keep the highest-ranked prediction.
6. Suppress a lower-ranked prediction as a duplicate when labels are compatible and either:
   - `box_iou >= 0.30`; or
   - `temporal_iou >= 0.60` and `frequency_iou >= 0.50`.
7. Continue until every prediction is kept or suppressed.

These are merge thresholds, not evaluation thresholds. They must be stored in run metadata and should be tested on the representative-six pilot before being frozen.

The initial version should keep the selected box unchanged rather than averaging coordinates. This makes the merge decision traceable and avoids creating a synthetic broad box from two imperfect predictions. Retain suppressed predictions and their parent tile IDs in a merge audit file.

## Boundary-Spanning Calls

Overlap is intended to ensure that most calls near an internal tile boundary are fully visible in at least one neighbouring tile. Merge ranking should therefore prefer the candidate whose box lies away from its tile edge.

If no tile contains a complete view and two compatible predictions represent opposite visible portions of the same call, mark the case for review rather than automatically joining them in the first implementation. A later, explicitly tested stitching rule may join partial boxes only when:

- they come from adjacent overlapping tiles;
- their temporal intervals overlap or meet within a small tolerance;
- their frequency IoU is sufficiently high;
- both predictions touch the relevant tile boundaries;
- the resulting union remains call-sized rather than spanning adjacent calls.

Calls at the original clip boundary remain clipped to `[0, clip_duration]`. Tiling must not infer sound outside the source clip.

## Proposed Merge Output

For each final event, retain:

```text
event_id
start_time_seconds
end_time_seconds
low_frequency_hz
high_frequency_hz
label
confidence
source_tile_ids
merge_status
suppressed_duplicate_count
coordinate_warnings
```

The final merged prediction JSON should use the existing prompt-v2 event schema plus merge provenance. Only final merged events should be passed to the frozen evaluator; raw tile predictions must also be retained.

## Risks and Assumptions

- **Duplicate calls:** overlap can inflate predictions unless NMS is deterministic.
- **Merged adjacent calls:** thresholds that are too permissive may suppress distinct dense calls.
- **Boundary truncation:** thresholds that are too strict may fail to recognize two partial views of one call.
- **Coordinate-frame errors:** local and original times can look numerically plausible; the frame must be explicit.
- **Loss of context:** 0.25 s tiles may improve geometry while making call sequences or artefacts harder to interpret.
- **Tool cost:** more tiles increase image generation, inference time, and opportunities for parsing failures.
- **Confidence comparability:** model confidence may vary by crop scale and may not be calibrated across tile settings.
- **Frequency consistency:** all tile plots must use the same frequency range and Hz output convention.
- **Partial final clips:** the final tile may be shorter than the nominal duration and must use its true end time.

## P6A.2 Validation Before Inference

Before any model call:

1. Spot-check `OP_016` tiles for readable axes and complete frequency coverage.
2. Confirm every manifest interval is inside its WAV duration.
3. Confirm adjacent windows have the configured overlap except for a clipped final window.
4. Confirm image paths and audio paths are portable and relative.
5. Freeze the tile-level prompt context and merge thresholds for the pilot.

