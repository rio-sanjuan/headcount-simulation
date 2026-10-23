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

**Preflight** (`refusals`) refuses, every problem at once, a method its run can't honor:
`demand`, `hybrid`, or `rate_calibration` with no plan; `demand` or `hybrid` without guidance
aligned to the plan; guidance aligned to the plan under any other method, since it would drive
the plan's flow beside whatever the method drives; and `rate_calibration` where guidance fixes
the counts a rate shift would move, attrition's or a fill channel's.

**Declared movements.** Guidance aligned to a plan counts the change declared movements make to
each group, `ε_H`, which no flow forecasts, and records their digest (`movements_digest`). A run
under `demand` or `hybrid` computes its own (`declared_movements`): each guidance group's net
heads moved per month by its declared reorganizations and conversions out, where one moved
between cost centers of one group nets to nothing. It refuses guidance aligned under other
movements than it declares (`movement_refusals`); guidance that records none counts as aligned
with none declared.

**Resolved metadata** (`ResolvedAlignment`) records what a run resolved: its method, the plan
and the plan the guidance was aligned to, by digest, and the cost centers whose target each
valve drives, with any target no valve reaches, as one under `demand` that guidance doesn't
cover.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.known_events import DECLARABLE, Source
from hcsim.engine.known_resolution import ResolvedPlan
from hcsim.model.core.aligner import GroupAligner
from hcsim.model.core.alignment import AlignmentTarget
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast, movements_digest
from hcsim.model.core.layouts import Buffer, EmployeeColumn
from hcsim.model.core.schema import COST_CENTER, SimulationSchema

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


#: The methods that align to a plan, and the ones that drive flow through aligned guidance.
NEEDS_PLAN: Final = frozenset(
    {AlignmentMethod.DEMAND, AlignmentMethod.HYBRID, AlignmentMethod.RATE_CALIBRATION}
)
NEEDS_ALIGNED_GUIDANCE: Final = frozenset({AlignmentMethod.DEMAND, AlignmentMethod.HYBRID})
#: The counts rate calibration moves, by shifting attrition's and the fill channels' hazards.
CALIBRATED: Final = frozenset(
    {
        GuidanceCategory.ATTRITION,
        GuidanceCategory.EXTERNAL_HIRES,
        GuidanceCategory.CONVERSION_HIRES,
        GuidanceCategory.INTERNAL_STARTS,
        GuidanceCategory.HIRES,
        GuidanceCategory.FILLS,
    }
)
#: What each method makes the plan drive, in words, for a refusal.
_DRIVES: Final = {
    AlignmentMethod.NONE: "nothing",
    AlignmentMethod.SUPPLY: "supply",
    AlignmentMethod.RATE_CALIBRATION: "the domain rates",
}


def refusals(
    method: AlignmentMethod, plan: AlignmentTarget | None, guidance: GuidanceForecast | None
) -> list[str]:
    """Every reason a run can't align under `method`, given its plan and guidance."""
    problems = []
    if method in NEEDS_PLAN and plan is None:
        problems.append(f"alignment_method {method} aligns to a plan, and the run has none")
    if method in NEEDS_ALIGNED_GUIDANCE:
        if guidance is None:
            problems.append(
                f"alignment_method {method} drives the plan's flow through guidance aligned to "
                "it, and the run has no guidance"
            )
        elif guidance.aligned_to is None:
            problems.append(
                f"alignment_method {method} needs guidance aligned to the plan, and the run's "
                "guidance isn't aligned to any"
            )
    elif guidance is not None and guidance.aligned_to is not None and plan is not None:
        problems.append(
            f"the guidance is aligned to the plan, so it drives the plan's flow, while "
            f"alignment_method {method} makes the plan drive {_DRIVES[method]}: run demand or "
            "hybrid, or guidance that isn't aligned to it"
        )
    if method is AlignmentMethod.RATE_CALIBRATION and guidance is not None:
        fixed = sorted(c.value for c in CALIBRATED if guidance.covers(c))
        if fixed:
            problems.append(
                "rate calibration shifts the attrition and fill hazards, but guidance fixes "
                f"the counts of {fixed}, which no shift moves"
            )
    return problems


def declared_movements(
    known: ResolvedPlan | None,
    employees: Buffer,
    guidance: GuidanceForecast,
    aligner: GroupAligner,
) -> dict[str, list[float]]:
    """Each guidance group's net change in heads from the run's declared movements, per
    forecast month: `ε_H`, as guidance alignment counts it. A person moves from the cost center
    the seed holds them in."""
    found: dict[str, list[float]] = {}
    if known is None:
        return found
    where = dict(
        zip(
            employees[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64).tolist(),
            employees[:, EmployeeColumn.COST_CENTER].astype(np.int64).tolist(),
            strict=True,
        )
    )

    def group_of(code: int | None) -> str | None:
        if code is None:
            return None
        index = int(aligner.groups_of_codes(np.array([code], dtype=np.int64))[0])
        return guidance.groups[index] if index >= 0 else None

    for resolved in known.events:
        event = resolved.event
        if event.source is not Source.DECLARED or event.kind not in DECLARABLE:
            continue
        if not 0 <= event.step < guidance.horizon:
            continue
        source = None if resolved.boundary == "in" else where.get(event.employee_id)
        destination = (
            None if resolved.boundary == "out" else resolved.codes.get(EmployeeColumn.COST_CENTER)
        )
        left, joined = group_of(source), group_of(destination)
        if left == joined:
            continue
        for group, change in ((left, -1.0), (joined, 1.0)):
            if group is not None:
                found.setdefault(group, [0.0] * guidance.horizon)[event.step] += change
    return found


def movement_refusals(
    method: AlignmentMethod,
    guidance: GuidanceForecast | None,
    aligner: GroupAligner | None,
    known: ResolvedPlan | None,
    employees: Buffer,
) -> list[str]:
    """Why a run can't consume its aligned guidance: under `demand` or `hybrid`, guidance
    aligned under other declared movements than the run declares."""
    if method not in NEEDS_ALIGNED_GUIDANCE or guidance is None or aligner is None:
        return []
    if guidance.aligned_to is None:
        return []
    ours = movements_digest(declared_movements(known, employees, guidance, aligner))
    theirs = guidance.movements or movements_digest(None)
    if ours == theirs:
        return []
    return [
        f"the guidance was aligned under declared movements {theirs}, but the run declares "
        f"{ours}: realign the guidance with the run's declared movements"
    ]


@dataclass(frozen=True)
class ResolvedAlignment:
    """What a run resolved: its method, the plan and the guidance's alignment by digest, and
    the cost centers, by label, whose target each valve drives, or none does."""

    method: AlignmentMethod
    plan: str | None = None
    aligned_to: str | None = None
    supply: tuple[str, ...] = ()
    flow: tuple[str, ...] = ()
    rate: tuple[str, ...] = ()
    undriven: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        method: AlignmentMethod,
        schema: SimulationSchema,
        plan: AlignmentTarget | None,
        guidance: GuidanceForecast | None,
        opened: Mask,
        grouped: Mask,
    ) -> ResolvedAlignment:
        """`opened` and `grouped` say, per cost-center code, whether guidance covers its
        requisitions opened, and whether it's in any guidance group at all."""
        labels = schema.vocabulary(COST_CENTER).labels
        targeted = np.zeros(len(labels), dtype=bool)
        if plan is not None and any(m.grouping == COST_CENTER for m in plan.marginals):
            groups = set(plan.marginal(COST_CENTER).groups)
            targeted = np.array([label in groups for label in labels], dtype=bool)
        supply = targeted & supplied(method, opened)
        flow = np.zeros(len(labels), dtype=bool)
        if method is AlignmentMethod.DEMAND:
            flow = targeted & np.asarray(grouped, dtype=bool)
        elif method is AlignmentMethod.HYBRID:
            flow = targeted & ~supply & np.asarray(grouped, dtype=bool)
        rate = targeted & (method is AlignmentMethod.RATE_CALIBRATION)
        undriven = targeted & ~(supply | flow | rate) & (method is not AlignmentMethod.NONE)

        def named(mask: Mask) -> tuple[str, ...]:
            return tuple(labels[int(c)] for c in np.flatnonzero(mask))

        return cls(
            method,
            None if plan is None else plan.digest(),
            None if guidance is None else guidance.aligned_to,
            named(supply),
            named(flow),
            named(rate),
            named(undriven),
        )

    @property
    def valves(self) -> tuple[Valve, ...]:
        """The valves the plan drives somewhere."""
        driven = (
            (Valve.SUPPLY, self.supply),
            (Valve.FLOW, self.flow),
            (Valve.RATE, self.rate),
        )
        return tuple(v for v, where in driven if where)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "plan": self.plan,
            "aligned_to": self.aligned_to,
            "valves": [v.value for v in self.valves],
            "supply": list(self.supply),
            "flow": list(self.flow),
            "rate": list(self.rate),
            "undriven": list(self.undriven),
        }


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
