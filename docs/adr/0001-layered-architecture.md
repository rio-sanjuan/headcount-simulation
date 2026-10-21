# ADR 0001: Layered architecture with a one-way dependency rule

**Status:** Accepted, 2026-10-21

## Context

The system combines statistical modelling, simulation orchestration, and financial accounting. When these are interleaved, improving one model means editing code that owns every other concern, and checking one accounting rule means running a whole simulation.

## Decision

Split the package into layers with one dependency direction, enforced by `import-linter` contracts in CI:

- **Model Layer** (`hcsim.model`): protocols, schema, models, and samplers. It's the leaf and imports nothing else in `hcsim`.
- **Financial Layer** (`hcsim.financial`): award sizing, vesting, and expense. It reads only the Model Layer's contract surface and runs without a simulation.
- **Guidance** (`hcsim.guidance`): depends only on the Model Layer.
- **Engine** (`hcsim.engine`): depends on the Model and Financial layers.
- **Studies** (`hcsim.studies`): depends on everything except the synthetic world.
- **Synthetic world** (`hcsim.synth`): imports nothing from `hcsim`.

## Alternatives considered

- **One package with no enforced boundaries.** Simpler at first, but boundaries erode silently.
- **Separate repositories per layer.** Strong boundaries, but cross-layer changes need coordinated releases, which is too heavy for one maintainer.

## Consequences

- A model can be trained, and an expense calculation checked, without the engine.
- A boundary violation fails CI with the contract's name.
- Shared types must live in the Model Layer's contract surface, which takes deliberate design.
