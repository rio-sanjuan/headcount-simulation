"""The effective ceiling, `Cap_eff(c, t)`: the tighter of the plan target and the capacity cap.

Two inputs can bound a cost center's committed strength, `S = H + O`:

- the **capacity cap** `Cap(c)`, a fixed ceiling per cost center;
- the **plan target** `H*(c, t)`, read from an alignment target's cost-center marginal at the
  step's month, interpolated as the target says. Outside its window, or for a cost center it
  has no target for, there's none.

`Cap_eff(c, t)` is the smaller of the two where either exists, and infinite where neither does.
It's the capacity veto, one of the three valves: no requisition opens in a cost center whose
committed strength has reached its ceiling, and `room` says how many more can.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.model.core.alignment import AlignmentTarget
from hcsim.model.core.schema import COST_CENTER, SimulationSchema

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]


class CeilingError(ValueError):
    """Caps or a plan that don't fit the run, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("bad ceiling: " + "; ".join(problems))


@dataclass(frozen=True)
class Ceiling:
    #: `Cap(c)` per cost-center code, infinite where there's none.
    caps: Floats
    plan: AlignmentTarget | None = None
    #: Each cost-center code's group in the plan's cost-center marginal, or -1.
    _plan_group: Ints = field(default_factory=lambda: np.zeros(0, dtype=np.int64), repr=False)

    @classmethod
    def build(
        cls,
        schema: SimulationSchema,
        caps: Mapping[str, int] | None = None,
        plan: AlignmentTarget | None = None,
    ) -> Ceiling:
        vocab = schema.vocabulary(COST_CENTER)
        problems = []
        values = np.full(len(vocab), np.inf)
        for label, cap in sorted((caps or {}).items()):
            if label not in vocab:
                problems.append(f"no cost center {label!r} to cap")
            elif not isinstance(cap, int | np.integer) or cap < 0:
                problems.append(f"the cap on {label} must be a non-negative integer")
            else:
                values[vocab.code(label)] = float(cap)
        group = np.full(len(vocab), -1, dtype=np.int64)
        if plan is not None:
            try:
                marginal = plan.marginal(COST_CENTER)
            except ValueError as error:
                problems.append(f"the plan has no cost-center targets: {error}")
            else:
                unknown = sorted(g for g in marginal.groups if g not in vocab)
                if unknown:
                    problems.append(f"the plan targets cost centers the run lacks: {unknown[:5]}")
                for k, label in enumerate(marginal.groups):
                    if label in vocab:
                        group[vocab.code(label)] = k
        if problems:
            raise CeilingError(problems)
        return cls(values, plan, group)

    @property
    def binds(self) -> bool:
        """Whether anything bounds any cost center."""
        return self.plan is not None or bool(np.isfinite(self.caps).any())

    def at(self, run_date: dt.date, step: int) -> Floats:
        """`Cap_eff(c, t)` for every cost center at `step`."""
        ceiling = self.caps.copy()
        if self.plan is None:
            return ceiling
        month = pd.Period(run_date, freq="M") + step
        first, last = self.plan.window
        if not first <= month <= last:
            return ceiling
        targets = self.plan.target(COST_CENTER, month)
        has = self._plan_group >= 0
        planned = np.full(len(ceiling), np.inf)
        planned[has] = targets[self._plan_group[has]]
        planned = np.where(np.isnan(planned), np.inf, planned)
        result: Floats = np.minimum(ceiling, planned)
        return result

    def room(self, run_date: dt.date, step: int, strength: Ints) -> Floats:
        """How many more openings each cost center can take before its committed strength
        reaches its ceiling: never negative, and infinite where nothing binds."""
        result: Floats = np.maximum(0.0, np.floor(self.at(run_date, step) - strength))
        return result
