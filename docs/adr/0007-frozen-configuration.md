# ADR 0007: One frozen configuration

**Status:** Accepted, 2026-10-21

## Context

When parameters are spread across nested sub-configurations and module-level constants, no one can list the tunable surface or reproduce a past run.

## Decision

One frozen dataclass, `SimulationConfig`, holds every parameter of a run. Scalars load in layers (defaults, then YAML, then environment, then overrides), and object-valued fields are passed in code. Validation collects every problem before raising. The configuration serializes to plain data and back, re-attaching models by their recorded versions.

## Alternatives considered

- **Nested per-subsystem configurations.** They're locally convenient, but can't be enumerated or frozen as one record.

## Consequences

- A run is reproducible from its configuration, seed, and artifact versions.
- Any change during a run raises an error.
- Eligibility rules are declared with raw values and resolved per schema, so one declaration means the same thing under any vocabulary.
