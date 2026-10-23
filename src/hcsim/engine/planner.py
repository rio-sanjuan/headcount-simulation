"""Planner rules: the controls a planner sets beside the plan, each over a scope.

**Scopes** (`PlannerScope`). A rule applies to the rows its scope selects: a subset's rules over
the buffer attributes and org rollups, in raw labels (see `model.data.subset`). So a scope of
`{"org_level1": ["ORG-A"], "job_level": [5]}` selects level-5 rows anywhere under ORG-A, and an
empty scope selects every row. A scope is declared once and resolved per schema, as eligibility
rules are, with every problem in every scope collected.

**Specificity.** Where several rules of one kind select a row, the most specific one decides
it: the scope that constrains the most attributes, an org level counting as one, and among
those, the narrowest, the one admitting the fewest codes in all. Two scopes that tie on both
and can select the same row are ambiguous, since neither is more specific, and resolution
refuses them, naming both.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import Buffer, BufferLayout
from hcsim.model.core.schema import SimulationSchema
from hcsim.model.data.subset import LoweredSubset, SubsetError, SubsetSpec

Ints = NDArray[np.int64]


class PlannerError(ValueError):
    """Planner rules that can't be declared or resolved, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("bad planner rules: " + "; ".join(problems))


@dataclass(frozen=True)
class PlannerScope:
    """The rows a planner rule applies to, in raw labels: every row, with no rules."""

    where: SubsetSpec = field(default_factory=lambda: SubsetSpec({}))

    @classmethod
    def of(cls, **where: Iterable[object]) -> PlannerScope:
        return cls(
            SubsetSpec({c: frozenset(str(v) for v in values) for c, values in where.items()})
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Iterable[object]]) -> PlannerScope:
        try:
            return cls.of(**{str(c): v for c, v in data.items()})
        except (TypeError, SubsetError) as error:
            raise PlannerError([f"can't read scope {dict(data)!r}: {error}"]) from None

    def to_dict(self) -> dict[str, Any]:
        return {c: sorted(v) for c, v in self.where.rules.items()}

    def resolve(self, schema: SimulationSchema) -> ResolvedScope:
        try:
            return ResolvedScope(self, self.where.lower(schema))
        except SubsetError as error:
            raise PlannerError([f"scope {self.to_dict()}: {error}"]) from None

    def __str__(self) -> str:
        if not self.where.rules:
            return "everyone"
        return ", ".join(f"{c} in {sorted(v)}" for c, v in self.where.rules.items())


@dataclass(frozen=True)
class ResolvedScope:
    """A scope lowered to codes through one schema."""

    scope: PlannerScope
    lowered: LoweredSubset

    @property
    def specificity(self) -> tuple[int, int]:
        """How specific the scope is, greater first: the attributes it constrains, then how
        few codes it admits."""
        admitted = sum(len(codes) for codes in self.lowered.codes.values())
        return len(self.scope.where.rules), -admitted

    def mask(self, rows: Buffer, layout: BufferLayout) -> NDArray[np.bool_]:
        return self.lowered.mask(rows, layout)

    def overlaps(self, other: ResolvedScope) -> bool:
        """Whether some row could fall in both: no attribute both constrain is disjoint."""
        mine, theirs = self.lowered.codes, other.lowered.codes
        return all(mine[c] & theirs[c] for c in set(mine) & set(theirs))


def resolve_scopes(scopes: Sequence[PlannerScope], schema: SimulationSchema) -> list[ResolvedScope]:
    """Every scope lowered through the schema, refusing, with every problem named, one that
    doesn't lower and any pair that's ambiguous."""
    problems, resolved = [], []
    for k, scope in enumerate(scopes):
        try:
            resolved.append(scope.resolve(schema))
        except PlannerError as error:
            problems += [f"rule {k + 1}: {p}" for p in error.problems]
    for i, a in enumerate(resolved):
        for j in range(i + 1, len(resolved)):
            b = resolved[j]
            if a.specificity == b.specificity and a.overlaps(b):
                problems.append(
                    f"rules {i + 1} ({a.scope}) and {j + 1} ({b.scope}) are ambiguous: equally "
                    "specific, they can select the same rows"
                )
    if problems:
        raise PlannerError(problems)
    return resolved


def most_specific(scopes: Sequence[ResolvedScope], rows: Buffer, layout: BufferLayout) -> Ints:
    """Each row's deciding scope: the index of the most specific scope that selects it, or -1
    where none does."""
    winner = np.full(len(rows), -1, dtype=np.int64)
    order = sorted(range(len(scopes)), key=lambda k: scopes[k].specificity, reverse=True)
    for k in order:
        open_rows = winner < 0
        if not open_rows.any():
            break
        winner[open_rows & scopes[k].mask(rows, layout)] = k
    return winner
