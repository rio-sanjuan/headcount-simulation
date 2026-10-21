# ADR 0006: Known events as a capability, not a run mode

**Status:** Accepted, 2026-10-21

## Context

Some events are known in advance: replayed history, accepted offers, declared reorganizations, and termination programs. A separate replay mode would duplicate the execution path for them.

## Decision

Known events are available in every run mode. Within a domain, a month's known events fire first, and sampling runs on the remaining eligible rows. An employee with a future known event is excluded from earlier draws, and a requisition reserved for a future known hire is held back from fills.

## Alternatives considered

- **A dedicated replay mode.** It isolates replay, but forks the code path that every domain must maintain.

## Consequences

- Replay is just the rung of the ladder that pins every movement.
- Under quota sampling, a known event counts against its group's quota, so a termination program changes who leaves rather than how many, unless it exceeds the quota.
- Known terminations create no backfill by default, so a planned reduction doesn't regenerate the headcount it removes.
