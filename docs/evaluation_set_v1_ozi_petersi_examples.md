# Evaluation Set V1: Representative Examples

## Purpose

These examples provide a small, stable set of clips for P1.5/P3 inspection and
prompt iteration. They cover common multi-event clips, dense clips, partial
final clips, and boundary-truncated annotations. The goal is to make it easy to
compare model outputs against known ground truth without reviewing all 45 clips
each time.

## Selected Examples

| clip_id | role | event_count | truncation_summary | figure_path | why_this_clip_is_useful |
| --- | --- | ---: | --- | --- | --- |
| `OP_001` | canonical multi-event example | 5 | `none: 5` | `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_001_gt_overlay.png` | Good first visual example: multiple clearly separated target calls from the first source recording. |
| `OP_010` | dense multi-event example | 7 | `none: 7` | `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_010_gt_overlay.png` | Highest event count without boundary truncation, useful for testing recall under dense activity. |
| `OP_045` | simple clean / partial-final-clip example | 3 | `none: 3` | `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_045_gt_overlay.png` | Final partial clip from `pseudo_petersi_010.wav`; useful for checking that short clips are handled correctly. |
| `OP_003` | right-truncated boundary example | 5 | `none: 4`, `right: 1` | `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_003_gt_overlay.png` | Contains a call that continues past the right edge of the clip, testing boundary-aware evaluation. |
| `OP_004` | left-truncated boundary example | 6 | `left: 1`, `none: 5` | `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_004_gt_overlay.png` | Contains the same source event as `OP_003`, clipped at the left edge after crossing the one-second boundary. |
| `OP_016` | dense boundary-stress example | 7 | `left: 1`, `none: 5`, `right: 1` | `outputs/evaluation_sets/ozimops_petersi_v1/figures/OP_016_gt_overlay.png` | Combines high event density with both left- and right-truncated events; useful as a harder stress case. |

## Boundary-Truncated Events

The table below lists all clip-level events where
`is_truncated_by_clip_boundary` is `true` or `truncation_side` is not `none`.

| clip_id | event_index | truncation_side | start_time | end_time | low_frequency | high_frequency | source_recording | source_event_uuid |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `OP_003` | 5 | `right` | 0.99786 | 1.0 | 28808.6 | 34179.7 | `pseudo_petersi_001.wav` | `77fb461a-60fa-57ca-8848-281a82ab2b09` |
| `OP_004` | 1 | `left` | 0.0 | 0.03695 | 28808.6 | 34179.7 | `pseudo_petersi_001.wav` | `77fb461a-60fa-57ca-8848-281a82ab2b09` |
| `OP_015` | 5 | `right` | 0.994318 | 1.0 | 31066.528875 | 35644.53125 | `pseudo_petersi_004.wav` | `e241f814-3243-5aa4-b062-901319653f2a` |
| `OP_016` | 1 | `left` | 0.0 | 0.004016 | 31066.528875 | 35644.53125 | `pseudo_petersi_004.wav` | `e241f814-3243-5aa4-b062-901319653f2a` |
| `OP_016` | 7 | `right` | 0.994125 | 1.0 | 31310.6695 | 37279.2015 | `pseudo_petersi_004.wav` | `b9a195cc-dc0d-5c21-8923-49b9c7e2a962` |
| `OP_017` | 1 | `left` | 0.0 | 0.003139 | 31310.6695 | 37279.2015 | `pseudo_petersi_004.wav` | `b9a195cc-dc0d-5c21-8923-49b9c7e2a962` |
| `OP_019` | 3 | `right` | 0.975958 | 1.0 | 30029.3 | 33935.5 | `pseudo_petersi_005.wav` | `2d5339e6-ad96-5e6c-8fd1-d286a38a0ee2` |
| `OP_020` | 1 | `left` | 0.0 | 0.01299 | 30029.3 | 33935.5 | `pseudo_petersi_005.wav` | `2d5339e6-ad96-5e6c-8fd1-d286a38a0ee2` |
| `OP_020` | 5 | `right` | 0.95235 | 1.0 | 29541.0 | 34179.7 | `pseudo_petersi_005.wav` | `ecffa093-35b2-5b6f-9f14-3d7f4b3d049d` |
| `OP_021` | 1 | `left` | 0.0 | 0.00419 | 29541.0 | 34179.7 | `pseudo_petersi_005.wav` | `ecffa093-35b2-5b6f-9f14-3d7f4b3d049d` |
| `OP_027` | 4 | `right` | 0.96356 | 1.0 | 27832.0 | 33935.5 | `pseudo_petersi_006.wav` | `97542394-ba63-5169-bbb3-bac2e9aaaf08` |
| `OP_028` | 1 | `left` | 0.0 | 0.01952 | 27832.0 | 33935.5 | `pseudo_petersi_006.wav` | `97542394-ba63-5169-bbb3-bac2e9aaaf08` |
| `OP_031` | 3 | `right` | 0.96454 | 1.0 | 29296.9 | 35156.2 | `pseudo_petersi_007.wav` | `d3efca38-f870-5106-b96d-4094d185015e` |
| `OP_032` | 1 | `left` | 0.0 | 0.01103 | 29296.9 | 35156.2 | `pseudo_petersi_007.wav` | `d3efca38-f870-5106-b96d-4094d185015e` |

## Notes

- `OP_003` and `OP_004` are a paired boundary example: the same source event is
  right-truncated in `OP_003` and left-truncated in `OP_004`.
- `OP_016` is the best compact stress case because it combines 7 events with
  both left and right truncation.
- A weak/noisy or ambiguous example is not assigned yet. The current metadata
  contains event counts and truncation flags, but no manual quality labels such
  as `weak_call`, `noisy`, or `borderline`. This should be filled after
  spectrogram spot-checking.
