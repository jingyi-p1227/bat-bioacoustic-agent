# BatDetect2 Australia Annotation Schema Audit

## Scope and Method

This audit examines the local BatDetect2 Australia dataset without modifying it. It describes fields that are actually present in the AOEF annotation export and associated study configuration. It does not infer labels from recording filenames, project papers, or acoustic appearance.

Dataset root:

```text
<DATASET_ROOT>
```

The dataset contains 274 WAV recordings and an AOEF 1.1.0 `annotations.json`. The annotation project contains 274 recording records, 274 full-recording clips, 274 clip annotations, 274 completed tasks, 5,723 sound events, and 5,723 sound-event annotations. Every recording has at least one sound event. The 5,723 event annotations refer one-to-one to 5,723 unique sound events.

Associated study files define a 216-recording train split, a 58-recording test split, and detection/classification target mappings. These files configure dataset use; they do not add event-level behavioural annotations.

## Files Inspected

| File | Role |
|---|---|
| `annotations.json` | AOEF 1.1.0 annotation project containing recordings, event geometry, tag assignments, clips, notes, and task state. |
| `study/split.json` | Recording UUIDs assigned to non-overlapping train and test splits. |
| `study/targets.yaml` | BatDetect2 detection and species-classification target definitions derived from annotation tags. |
| `study/train_set.yaml` | Train-set source and recording-filter configuration. |
| `study/test_set.yaml` | Test-set source and recording-filter configuration. |
| `build_evaluation_set.py` | Project parser used to build the Ozimops petersi evaluation set. |

No additional JSON, YAML, CSV, or text metadata files were found under the dataset root to a depth of three directories. Audio content was not interpreted as annotation metadata.

## AOEF Project Structure

The JSON document has top-level fields `version`, `created_on`, and `data`. The `data` object contains:

```text
clip_annotations
clips
collection_type
created_on
description
instructions
name
project_tags
recordings
sound_event_annotations
sound_events
tags
tasks
uuid
```

Project-level fields identify the collection as an annotation project and provide a narrative description. That description mentions the broad collection region, a Nanobat Systems recorder, field recordings from released and positively identified bats, and selection of recordings with calls close to search-phase echolocation behaviour. These statements describe the collection or selection process. They are not structured per-recording or per-event labels.

## Recording-Level Fields

### `recordings`

| Field | Coverage | Meaning observed in the file |
|---|---:|---|
| `uuid` | 274/274 | Recording identifier used by clips and sound events. |
| `path` | 274/274 | Relative WAV path or filename. It is an identifier/path, not a ground-truth label. |
| `duration` | 274/274 | Recording duration in seconds. Observed range: 0.230878 to 10.00225 seconds. |
| `channels` | 274/274 | Channel count; all recordings are mono (`1`). |
| `samplerate` | 274/274 | Sample rate in Hz: 65 recordings at 384,000 Hz and 209 at 500,000 Hz. |
| `hash` | 216/274 | Optional audio hash; absent from 58 records. |
| `owners` | 274/274 | List field, empty for every recording. It does not provide annotator or individual identity here. |

### Related Full-Recording Objects

Each recording is represented by one AOEF clip spanning the complete recording:

| Object | Fields | Coverage and interpretation |
|---|---|---|
| `clips` | `uuid`, `recording`, `start_time`, `end_time` | 274 records; links a recording to its full time interval. |
| `clip_annotations` | `uuid`, `clip`, `sound_events`, `created_on`, optional `notes` | 274 records; `sound_events` contains sound-event-annotation UUIDs. `notes` occurs on 166 records. |
| `tasks` | `uuid`, `clip`, `status_badges`, `created_on` | 274 records; all contain a `completed` state. This is workflow status, not ecological behaviour. |

Each item in `clip_annotations.notes` contains `uuid`, `message`, `is_issue`, and `created_on`. Each item in `tasks.status_badges` contains `state` and `created_on`. No additional nested recording-level annotation fields were observed.

Clip notes are provenance/review comments such as `Automatically generated. Adjusted by Callan`, `Checked by Tanya 30Jan2020`, and `Done by Tanya 31Jan2020`. One note asks whether a second species may occur at the end of a recording. All note objects have `is_issue: false`. Notes are free text and cannot be treated as a consistent event-level label schema.

## Event-Level Annotation Fields

AOEF separates event geometry from annotation metadata.

### `sound_events`

| Field | Coverage | Meaning |
|---|---:|---|
| `uuid` | 5,723/5,723 | Stable sound-event identifier. |
| `recording` | 5,723/5,723 | Recording UUID containing the event. |
| `geometry` | 5,723/5,723 | Event geometry object. Every geometry is a `BoundingBox`. |
| `geometry.type` | 5,723/5,723 | Always `BoundingBox`. |
| `geometry.coordinates` | 5,723/5,723 | Four values ordered as `[start_time, low_frequency, end_time, high_frequency]`. Time is in seconds and frequency is in Hz. |
| `features` | 121/5,723 | Optional object containing duration and frequency-extent values. |

All bounding boxes contain four coordinates. No event has `start_time >= end_time` or `low_frequency >= high_frequency`.

The optional `features` object contains:

| Feature | Coverage | Audit result |
|---|---:|---|
| `ac:mediaDuration` | 121/5,723 | Exactly equals `end_time - start_time` for all 121 events. |
| `ac:freqLow` | 121/5,723 | Exactly equals the bounding-box low-frequency coordinate. |
| `ac:freqHigh` | 121/5,723 | Exactly equals the bounding-box high-frequency coordinate. |
| `soundevent:bandwidth` | 121/5,723 | Exactly equals `ac:freqHigh - ac:freqLow`. |

These optional features duplicate or deterministically derive from bounding-box geometry; they are not independent call-shape measurements or behavioural labels.

### `sound_event_annotations`

| Field | Coverage | Meaning |
|---|---:|---|
| `uuid` | 5,723/5,723 | Annotation identifier. |
| `sound_event` | 5,723/5,723 | UUID of the annotated `sound_events` record. |
| `tags` | 5,723/5,723 | List of shared tag IDs resolved through `data.tags`. |
| `created_on` | 5,723/5,723 | Annotation creation timestamp. |

There is exactly one sound-event annotation per sound event, and no annotation has an empty tag list.

## Label and Category Fields

Only three tag keys occur in event annotations.

| Tag key | Representative or complete values | Event coverage | Interpretation |
|---|---|---:|---|
| `soundevent:call_type` | `Echolocation` | 5,723/5,723 | Direct event-level call-category label. It does not distinguish search, approach, or feeding buzz. |
| `dwc:order` | `Chiroptera` | 5,723/5,723 | Direct event-level taxonomic order label. |
| `dwc:scientificName` | 15 values listed below | 5,441/5,723 | Direct event-level scientific-name tag when present. 282 events have no scientific-name tag. |

Scientific-name values and event counts:

| Value | Count |
|---|---:|
| `Austronomus australis` | 160 |
| `Chalinolobus gouldii` | 706 |
| `Chalinolobus morio` | 546 |
| `Chalinolobus picatus` | 419 |
| `Nyctophilus corbeni` | 632 |
| `Nyctophilus geoffroyi` | 198 |
| `Nyctophilus gouldi` | 468 |
| `Ozimops petersi` | 191 |
| `Ozimops planiceps` | 192 |
| `Ozimops ridei` | 170 |
| `Saccolaimus flaviventris` | 154 |
| `Scotorepens balstoni` | 314 |
| `Scotorepens greyii` | 349 |
| `Scotorepens sp. (Parnaby)` | 9 |
| `Vespadelus vulturnus` | 933 |
| No `dwc:scientificName` tag | 282 |

`Scotorepens sp. (Parnaby)` is retained verbatim. The audit does not reinterpret its taxonomic precision. Events without a scientific-name tag are not assigned species from recording paths.

## Availability of Requested Annotation Concepts

| Concept | Status | Evidence and limitation |
|---|---|---|
| Event presence | **Directly represented, but not as a Boolean label** | A `sound_events` record represents an annotated positive event. No explicit event-absence or negative-event label exists. All 274 recordings contain at least one event. |
| Start/end time | **Directly annotated** | First and third bounding-box coordinates. |
| Low/high frequency | **Directly annotated** | Second and fourth bounding-box coordinates. |
| Species label | **Directly annotated when present** | `dwc:scientificName` exists for 5,441 events; absent for 282. |
| Call-type label | **Directly annotated at a coarse level** | Every event is tagged `Echolocation`. No finer call-type taxonomy is present. |
| Search / approach / feeding-buzz label | **Absent at event level** | No such event tag exists. The project description says selected calls were close to search-phase behaviour, but this is dataset-level context and cannot serve as per-event ground truth. |
| Social-call label | **Absent** | No social-call tag or field exists. |
| Behaviour label | **Absent** | No structured behaviour field exists. Task status and review notes are not behaviour labels. |
| Individual identity | **Absent** | Recording `owners` lists are empty, and no bat-individual identifier is present. The project description's reference to identified released individuals is not a structured identity field. |
| Recording context or environmental metadata | **Limited project-level narrative only** | No per-recording location, date/time, weather, habitat, temperature, microphone position, or individual context fields are present. Technical audio metadata are available. Region and recorder information occur only in the project description. |

## Direct Ground Truth, Derived Values, and Unsupported Labels

### Directly Annotated Ground Truth

- Event UUID and recording linkage.
- Event start and end time.
- Event low and high frequency.
- Coarse call type: `Echolocation`.
- Taxonomic order: `Chiroptera`.
- Scientific-name tag for the 5,441 events where it is present.
- Annotation creation time and clip-level review/provenance notes.

### Deterministically Derivable Values

The following can be calculated without additional expert judgement:

- Event duration: `end_time - start_time`.
- Frequency bandwidth: `high_frequency - low_frequency`.
- Temporal and frequency centres.
- Event ordering and inter-event intervals within a recording.
- Event count or density within a defined recording or generated window.
- Whether an event overlaps a defined window.
- Clip-relative times and left/right boundary truncation after deterministic clipping.
- Event-presence Boolean for a specifically defined window, assuming the source annotations are complete for that window.
- Species frequency distributions for events carrying `dwc:scientificName`.

Derived quantities are not new annotated labels. In particular, the evaluation-set builder's `event_density`, `auto_scenario`, `has_target_event`, and boundary-truncation fields are operational derivatives of event count and geometry, not expert behavioural annotations.

### Absent Labels Requiring Expert Annotation

- Search-phase, approach-phase, and feeding-buzz category per event or sequence.
- Social-call category.
- Behavioural state or behavioural context.
- Individual bat identity.
- Call-sequence functional role.
- Echo, reverberation, artefact, or signal-quality labels as explicit annotation fields.
- Per-event confidence, uncertainty, annotator agreement, or adjudication outcome.
- Per-recording environmental context such as habitat, weather, temperature, and precise collection location.

### Exploratory Hypotheses Only

The following may be generated as model hypotheses for qualitative review, but cannot be scored as ground-truth reasoning targets from this dataset:

- Whether an event is a search, approach, or feeding-buzz call.
- Whether a call is social or behavioural rather than echolocation.
- Behavioural interpretation of a sequence.
- Individual identity or individual-level consistency.
- Acoustic quality, ambiguity, echo status, or artefact type unless a new expert-reviewed annotation layer is created.
- Species assignment for the 282 events lacking a scientific-name tag.

The phrase `close to search-phase echolocation behaviour` in the project description must not be converted into per-event search-phase labels. At most, it is a dataset-level selection statement whose scope and consistency are not encoded in the event schema.

## Existing Ozimops petersi Evaluation-Set Parser

`build_evaluation_set.py` uses the schema as follows:

1. Loads `annotations.json` and requires a top-level `data` object.
2. Builds lookup tables for shared tags, sound-event annotations, and recording-to-event links.
3. Selects recordings by a configured filename prefix, but does **not** infer species from that prefix.
4. Resolves each event's tags and retains it only when the direct scientific-name tag equals the requested species.
5. Requires `BoundingBox` geometry with four valid coordinates.
6. Splits source WAVs using sample indices and includes every source event overlapping each output clip.
7. Converts source times to clip-relative times and records deterministic boundary truncation.
8. Preserves source event UUIDs and resolved tags.
9. Derives event density and scenario names from event count for dataset organisation.

For `Ozimops petersi`, the parser uses 191 directly species-tagged source events. Overlap with generated one-second clip boundaries can create more clip-level event instances than unique source events; this does not add new source annotations.

## Quantitatively Evaluable Event-Interpretation Targets

The existing ground truth supports quantitative evaluation of:

1. **Event detection/presence within a defined interval:** TP, FP, FN, precision, recall, and F1, using annotated sound events as positives.
2. **Temporal localisation:** start/end error, temporal IoU, event duration error, and centre-time error.
3. **Frequency localisation:** low/high-frequency error, bandwidth error, frequency IoU, and centre-frequency error.
4. **Joint time-frequency localisation:** two-dimensional box IoU and thresholded box-match counts.
5. **Coarse call-type classification:** `Echolocation` versus an explicit alternative only if an evaluation set containing alternative annotated classes is later added. In the current dataset all events share the same call-type value, so this field cannot test discrimination among call types.
6. **Species classification:** accuracy or class-wise metrics for the 5,441 events with direct scientific-name tags, while excluding or explicitly treating the 282 unlabelled events. Taxonomic ambiguity such as `Scotorepens sp. (Parnaby)` must remain explicit.
7. **Deterministic geometric reasoning:** duration, bandwidth, event order, inter-event interval, overlap with a specified window, and boundary truncation.

The dataset does **not** support quantitative ground-truth evaluation of search/approach/feeding-buzz interpretation, social-call interpretation, behaviour, individual identity, or environmental reasoning. Such outputs would require a new expert annotation protocol and should otherwise be reported only as exploratory hypotheses.

## Recommendation for Event-Reasoning Task Design

The first event-reasoning task should remain grounded in directly annotated geometry and deterministic transformations. Suitable outputs are event presence, start/end time, low/high frequency, duration, bandwidth, boundary status, event order, and scientific name where a direct tag exists. A model can also provide a rationale or uncertainty estimate, but these fields cannot be scored against existing ground truth unless an expert-reviewed reference layer is added.

Behavioural interpretation should not be introduced as a supervised evaluation target for this dataset. If search, approach, feeding buzz, social calls, signal quality, or behavioural context are important research questions, they require a separate expert annotation exercise with explicit definitions, uncertainty handling, and inter-annotator review.
