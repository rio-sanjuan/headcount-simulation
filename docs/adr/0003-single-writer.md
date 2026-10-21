# ADR 0003: Pure handlers and a single writer

**Status:** Accepted, 2026-10-21

## Context

Handlers that mutate state directly make it impossible to say which domain changed what, and they hide missed updates, such as a moved employee being resampled because no state update followed the move.

## Decision

Domain handlers are pure: they read state and return a declarative result that describes appends, drops, writes, grants, and departures. One apply step writes every change, in a fixed order, and keeps the per-cost-center counters in step with the buffers.

## Alternatives considered

- **Handlers that mutate buffers.** No allocation for results, but no audit trail, and every handler must keep every counter consistent.

## Consequences

- Each result allocates a small description of its change, which is cheap next to the buffers.
- Later domains in a step read settled state, and the accounting identity holds by construction.
- Handlers can be unit-tested on toy states, and trajectories can stream one at a time.
