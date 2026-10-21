# ADR 0011: Local-only execution, with cloud deferred

**Status:** Accepted, 2026-10-21

## Context

The project is developed and run by one maintainer, and its data is generated locally.

## Decision

Everything runs locally: world generation, training, guidance fits, simulations, studies, and the explorer. GitHub Actions is used only as CI to check code, and it deploys nothing.

## Alternatives considered

- **Managed training and scheduled pipelines in the cloud.** Useful at scale, but premature now, and each service adds operational surface.

## Consequences

- Runs are bounded by one machine, so streaming trajectories and chunked expense matter.
- Cloud work is tracked as `future` issues outside every milestone.
