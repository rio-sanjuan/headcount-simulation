# ADR 0005: Amortize expense after the simulation

**Status:** Accepted, 2026-10-21

## Context

Accruing expense every step carries unbounded per-tranche state through the loop and couples accounting to orchestration.

## Decision

Grants are emitted and sized during the simulation and recorded on the RSU tracker. After the last month, a stateless calculator amortizes every tranche over its award-to-vest window and computes the attrition, rate-change, and vesting true-ups. The vesting true-up is the residual, so the per-tranche accounting identity holds by construction.

## Alternatives considered

- **Accrual inside the loop.** Expense would be available mid-run, but at the cost of state and coupling.

## Consequences

- Expense runs once per trajectory over a sparse list of tranches, in bounded chunks.
- The calculator can be tested on hand-built schedules.
- A failed identity check signals a defect in the expense code, never an input problem.
