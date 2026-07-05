# Strong Labelling and Time-Frequency Localisation

## Citation

**TODO:** add verified literature defining strong labels for sound events and literature evaluating time-frequency event localisation. No citation is currently available locally.

## Summary

The dissertation uses strong labelling to mean one event annotation per individual echolocation call with explicit start time, end time, low frequency, and high frequency. This is stricter than clip-level presence or species classification because the output must localise each call in two dimensions. A literature source is needed to place this operational definition within established sound-event detection or bioacoustic annotation terminology.

## Key Methodological Idea

Represent each event as a time-frequency bounding box and evaluate detection, timing, frequency extent, and joint box quality separately.

## Relevance to This Dissertation

The frozen protocol reports temporal IoU, frequency IoU, box IoU, precision, recall, and F1. Cases such as OP_045 show why finding the call centre is not enough when the annotation standard expects a wider event extent.

## How It Supports the Project Argument

Strong localisation exposes failure modes hidden by clip-level classification. It makes clear why specialist proposals, VLM visual judgement, and validators can disagree even when all identify roughly the same call location.

## Dissertation Use

- **Related work:** distinguish clip classification, event detection, and strong labelling.
- **Methods:** define the annotation target and geometry.
- **Results:** interpret timing-versus-frequency trade-offs.

## TODO

- Verify a canonical strong-labelling definition.
- Add a source for time-frequency event-box evaluation if one is used in the final text.
