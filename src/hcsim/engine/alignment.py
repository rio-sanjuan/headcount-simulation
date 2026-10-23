"""Alignment methods: which valve a run's plan target, `H*(c, t)`, drives.

There are three valves (ADR 0004): supply, which plan targets size; flow, which guidance governs;
and the capacity veto. A plan must say which it drives, so it never silently drives two. The
method says:

- `none`: nothing. The plan is read for scoring alone: growth runs at its rate or by its
  guidance, and the effective ceiling is the capacity cap.
- `supply`: requisition supply. Wherever the plan targets a cost center, growth seeks the target
  by the supply rule, whatever guidance says of requisitions opened, and the target bounds
  committed strength in the effective ceiling.
- `demand`: event counts, through guidance aligned to the plan. Growth follows its guidance or
  its rate, and the effective ceiling is the capacity cap.
- `hybrid`: demand where guidance covers a cost center's requisitions opened, and supply where
  it doesn't. Each cost center's target drives one valve, and coverage says which.
- `rate_calibration`: the domain rates. The attrition and fill handlers shift their hazards so
  the expected path meets the target. Growth runs at its rate, and the effective ceiling is the
  capacity cap.

**Overrides per arm.** A study's arms can each run under their own method: `MethodChoice`
holds the run's method and any arm's override, and `resolve` gives the one an arm runs under.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

Mask = NDArray[np.bool_]


class AlignmentMethod(StrEnum):
    NONE = "none"
    SUPPLY = "supply"
    DEMAND = "demand"
    HYBRID = "hybrid"
    RATE_CALIBRATION = "rate_calibration"


class Valve(StrEnum):
    """What a plan can drive: requisition supply, event counts, or the domain rates."""

    SUPPLY = "supply"
    FLOW = "flow"
    RATE = "rate"


#: The valves each method can drive. Hybrid drives supply in some cost centers and flow in the
#: rest, never both in one.
VALVES: Final[Mapping[AlignmentMethod, frozenset[Valve]]] = MappingProxyType(
    {
        AlignmentMethod.NONE: frozenset(),
        AlignmentMethod.SUPPLY: frozenset({Valve.SUPPLY}),
        AlignmentMethod.DEMAND: frozenset({Valve.FLOW}),
        AlignmentMethod.HYBRID: frozenset({Valve.SUPPLY, Valve.FLOW}),
        AlignmentMethod.RATE_CALIBRATION: frozenset({Valve.RATE}),
    }
)


class MethodError(ValueError):
    """Methods that can't be resolved, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("bad alignment methods: " + "; ".join(problems))


def supplied(method: AlignmentMethod, covered: Mask) -> Mask:
    """Which cost centers the plan drives supply in, given which ones guidance covers for
    requisitions opened: all of them under `supply`, the uncovered ones under `hybrid`, and
    none otherwise."""
    covered = np.asarray(covered, dtype=bool)
    if method is AlignmentMethod.SUPPLY:
        return np.ones(len(covered), dtype=bool)
    if method is AlignmentMethod.HYBRID:
        return ~covered
    return np.zeros(len(covered), dtype=bool)


@dataclass(frozen=True)
class MethodChoice:
    """The run's method, and the arms that override it."""

    method: AlignmentMethod = AlignmentMethod.SUPPLY
    overrides: Mapping[str, AlignmentMethod] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MethodChoice:
        """From `{"method": ..., "overrides": {arm: method}}`, refusing, with every problem
        named, a method that isn't one, or a key it doesn't know."""
        known = {m.value for m in AlignmentMethod}
        problems = [f"no setting {k!r}" for k in sorted(set(data) - {"method", "overrides"})]
        method = str(data.get("method", AlignmentMethod.SUPPLY.value))
        if method not in known:
            problems.append(f"no method {method!r}; the methods are {', '.join(sorted(known))}")
        overrides = data.get("overrides", {})
        if not isinstance(overrides, Mapping):
            problems.append("overrides map each arm to its method")
            overrides = {}
        for arm, value in sorted(overrides.items()):
            if str(value) not in known:
                problems.append(f"arm {arm!r} overrides with no method {value!r}")
        if problems:
            raise MethodError(problems)
        return cls(
            AlignmentMethod(method),
            {str(a): AlignmentMethod(str(v)) for a, v in overrides.items()},
        )

    def resolve(self, arm: str | None = None) -> AlignmentMethod:
        """The method `arm` runs under: its override, or the run's."""
        return self.overrides.get(arm, self.method) if arm is not None else self.method

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "overrides": {a: m.value for a, m in sorted(self.overrides.items())},
        }
