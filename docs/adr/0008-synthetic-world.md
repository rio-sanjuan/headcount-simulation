# ADR 0008: A synthetic world as the only data source

**Status:** Accepted, 2026-10-21

## Context

The repository is public, and the method needs realistic data: workforce transactions, requisition snapshots, plan versions, and an equity ledger. Real data can't be published, and hand-written fixtures are too small to train or validate models.

## Decision

A seeded generator, `hcsim.synth`, simulates a company with its own ground-truth processes and writes the raw source tables to DuckDB. Its processes differ in functional form from the engine's models, so fitted models must learn the world rather than mirror it, and it writes a manifest of its true parameters. It shares no code with the rest of the package.

## Alternatives considered

- **Small hand-built fixtures only.** Fine for unit tests, but too small for training and studies.
- **Public datasets.** None covers this domain with the needed joint structure.

## Consequences

- Every model and study can report how well it recovers the true parameters, which real data can't offer.
- Data-quality traps are reproduced as toggleable quirks, so defensive code has something to defend against.
- Results describe the synthetic world, not any real one.
