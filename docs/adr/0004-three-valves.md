# ADR 0004: Three separate valves

**Status:** Accepted, 2026-10-21

## Context

A plan target, an event-count forecast, and a hiring ceiling are different things. When one artifact drives both requisition creation and fills, one number is forced to mean two things, and every gap between supply and flow looks like an inconsistency.

## Decision

Three valves, each with one job and one owner:

- **Plan targets** size the growth gap and the backfill ceiling (supply).
- **Guidance** governs event counts (flow).
- **Capacity caps** veto requisition creation (veto). The effective ceiling is the tighter of the plan target and the cap, so a target never switches the veto off.

## Alternatives considered

- **One target artifact for everything.** Simpler to supply, but contradictory in use.

## Consequences

- An open but unfilled backlog is the intended gap between supply and flow, not an error.
- An alignment method must say which valve a plan drives, so a plan never silently drives two.
