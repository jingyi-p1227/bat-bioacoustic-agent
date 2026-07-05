# Agentic Tool Use, Validation, and Critic/Referee Systems

## Citation

**TODO:** add verified literature on tool-using language-model agents, external verification, and critic/referee or self-correction architectures. No relevant paper is stored locally.

## Summary

The project's agentic contribution is narrow and testable: a single annotation agent can receive alternate visual views or detector metadata, while deterministic validators audit and constrain its output. The experiments show that tool access alone is insufficient. Unconstrained use can damage correct proposals, and a validator tuned on representative cases can fail held out. A future critic/referee should therefore be compared with deterministic baselines rather than assumed to improve reliability.

## Key Methodological Idea

Separate proposal generation, evidence inspection, final prediction, and validation. Preserve provenance at every stage so a critic or human can identify which tool supplied each event and what geometry changed.

## Relevance to This Dissertation

P5 tests view planning; P6 tests tiled/PCEN tools, BatDetect2 metadata, proposal-deviation auditing, and deterministic preservation. P6E.5 supplies held-out evidence that a simple rule does not generalise reliably.

## How It Supports the Project Argument

Reliable agentic annotation requires controlled tool interfaces and independent validation. The project's results support this empirically, while literature is needed to position the architecture among established agent and critic systems.

## Dissertation Use

- **Related work:** tool-using and critic/referee agents.
- **Methods:** single-agent tool interface and shadow validators.
- **Results:** unconstrained versus validated outputs.
- **Discussion/Future work:** scoped critic prototype and limits of self-correction.

## TODO

- Add verified tool-use and critic/referee citations.
- Keep claims about multi-agent improvement prospective until a real experiment is run.
