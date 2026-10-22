"""Alignment: a guidance forecast moved onto plan targets, `H*(g, t)`.

Alignment takes a group's reconciled forecasts and an alignment target, and moves the anchor
to the target where the plan sets one, re-deriving the flows so the stock-flow balance still
holds:

    Ĥ(t) - Ĥ(t-1) = Σ sign_c · flow_c(t) + ε_H(t)

where `ε_H` is the change declared movements make to the group, such as people moved in or
out by a reorganization, which no flow forecasts.

**Which months.** Only months the plan actively plans (up to its evaluation month `t*`) are
aligned. By default only its target dates are, each to within its band; with `all_months`,
every planned month is aligned to the interpolated target, exactly.

**Fixed flows.** A flow the configuration fixes keeps its forecast; only the rest move.

**Greedy re-derivation** (`greedy`): each aligned month's anchor is clipped into the target's
band, and the flows are re-derived from it as greedy reconciliation does, the smallest free
flow first, never below zero. Where the free flows can't reach the band, as when only inflows
are free and the target falls faster than they can, the anchor is what the flows reach, and the
gap to the band is the month's **shortfall**. The anchor is then rebuilt from the flows, so the
balance holds exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

import numpy as np
from numpy.typing import NDArray

from hcsim.guidance.reconcile import reconcile_greedy
from hcsim.guidance.registry import Entry, Registry

Floats = NDArray[np.float64]
#: How close counts as on target, in heads.
TOLERANCE: Final = 1e-6


@dataclass(frozen=True)
class AlignmentProblem:
    """One group's forecasts, and the targets and constraints it's aligned to."""

    anchor: Floats
    last_actual: float
    flows: Mapping[str, Floats]
    signs: Mapping[str, int]
    #: Per month, the target, NaN where the month isn't aligned.
    target: Floats
    #: Per month, how far from the target still counts as on it.
    band: Floats
    #: Flows that keep their forecast.
    fixed: frozenset[str] = frozenset()
    #: Per month, `ε_H`: the change declared movements make.
    movements: Floats | None = None
    #: The QP's weight on each flow's change; a flow it doesn't name weighs 1.
    weights: Mapping[str, float] = field(default_factory=dict)
    #: The QP: measure a flow's change as a share of the anchor, its exposure.
    rate_aware: bool = True
    #: The QP: the cost of a head of shortfall, against a flow's change.
    shortfall_penalty: float = 1e4

    @property
    def epsilon(self) -> Floats:
        months = len(self.anchor)
        return np.zeros(months) if self.movements is None else np.asarray(self.movements, float)


@dataclass(frozen=True)
class Aligned:
    anchor: Floats
    flows: dict[str, Floats]
    #: Per month, the achieved anchor less the band's nearest edge; zero on target.
    shortfall: Floats
    method: str


def path(
    last_actual: float, flows: Mapping[str, Floats], signs: Mapping[str, int], epsilon: Floats
) -> Floats:
    """The anchor the flows and movements make from `last_actual`, month by month."""
    change = epsilon + sum(
        (signs[c] * np.asarray(v, float) for c, v in flows.items()), start=np.zeros(len(epsilon))
    )
    result: Floats = float(last_actual) + np.cumsum(change)
    return result


def shortfall_of(anchor: Floats, target: Floats, band: Floats) -> Floats:
    """How far `anchor` is outside each aligned month's band: signed, zero within it."""
    low, high = target - band, target + band
    result = np.where(anchor > high, anchor - high, np.where(anchor < low, anchor - low, 0.0))
    result = np.where(np.isnan(target), 0.0, result)
    result[np.abs(result) < TOLERANCE] = 0.0
    return result


def greedy(problem: AlignmentProblem) -> Aligned:
    aligned = ~np.isnan(problem.target)
    wanted = np.where(
        aligned,
        np.clip(problem.anchor, problem.target - problem.band, problem.target + problem.band),
        problem.anchor,
    )
    free = {c: v for c, v in problem.flows.items() if c not in problem.fixed}
    fixed = {c: np.asarray(v, float) for c, v in problem.flows.items() if c in problem.fixed}
    # The free flows carry what the wanted anchor asks, less the fixed flows and movements.
    others = path(0.0, fixed, problem.signs, problem.epsilon)
    flows: dict[str, Floats] = dict(fixed)
    if free:
        moved = reconcile_greedy(
            wanted - others,
            problem.last_actual,
            dict(free),
            {c: problem.signs[c] for c in free},
            0.0,
        )
        flows |= moved.flows
    anchor = path(problem.last_actual, flows, problem.signs, problem.epsilon)
    flows = {c: flows[c] for c in problem.flows}
    return Aligned(anchor, flows, shortfall_of(anchor, problem.target, problem.band), "greedy")


class Aligner(Protocol):
    def __call__(self, problem: AlignmentProblem) -> Aligned: ...


#: The alignment methods a configuration can name.
ALIGNERS: Registry[Aligner] = Registry("alignment method", (Entry("greedy", lambda: greedy),))
