# ADR 0002: Exact-count quota sampling for aggregate alignment

**Status:** Accepted, 2026-10-21

## Context

Drawing every entity independently compounds variance: totals drift from the aggregate forecast and from run to run, and the usual patch is a post-hoc multiplier per domain that can't remove the drift and documents nothing.

## Decision

Where a guidance forecast carries a group, the domain realizes exactly `min(K, eligible)` events per group and month. Quotas are apportioned to integers by largest remainder, and selection is a segmented Gumbel-top-k on the model's log-probabilities, so the count is fixed and the model decides who. Where no forecast exists, rows draw independently and the count is an output.

## Alternatives considered

- **Better-calibrated multipliers.** They still correct after the fact, and can't remove run-to-run variance.
- **Exact conditional-Bernoulli sampling.** It's the model conditioned on its count, but it's costly. Successive sampling is close enough that the gap is measured rather than engineered away.

## Consequences

- Totals match guidance by construction, so a bad forecast gives a confidently wrong answer: trust moves upstream to the guidance pipeline.
- Finer guidance groups give tighter control but leak more shortfall, because each group clamps at its own eligible count.
- Guided and unguided runs separate uncertainty in volume from uncertainty in identity.
