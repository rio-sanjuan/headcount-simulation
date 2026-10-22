"""Subsets: a declared part of a population that a run simulates on its own.

A `SubsetSpec` holds rules over the 8 employee-buffer attributes and the org rollups. Each rule
is a column and the values it admits, and a row is inside when every rule admits it.

- `cost_center`, `job_family`, `job_level`, `location`, `company`, and `tech_indicator` admit
  labels in the schema's vocabularies;
- `is_manager` admits `true` or `false`;
- `perf_rating` admits rating ranks, `0` for the unrated, and applies to employees only;
- `org_level<k>` admits org nodes at level k, and lowers to `cost_center`.

**Lowering to leaves.** An org rule becomes the set of leaf cost centers under its nodes, through
the schema's rollups. Several org rules, and a cost-center rule, intersect.

**Pushdown versus in-memory filtering.** Every rule but the rating can be pushed down: merged
into the population, so the warehouse never extracts rows outside the subset. The rating isn't
in the extraction join, so it's filtered in memory, on the employee buffer. A requisition has
no rating, so a rating rule never excludes an opening.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import EMPLOYEE, OPENING, Buffer, BufferLayout, EmployeeColumn
from hcsim.model.core.schema import COST_CENTER, SimulationSchema
from hcsim.model.data.population import PopulationSpec
from hcsim.model.sql.catalog import is_org_level

#: The 8 buffer attributes a subset may constrain, and their employee-buffer columns.
SUBSET_COLUMNS: Final[Mapping[str, EmployeeColumn]] = MappingProxyType(
    {
        "cost_center": EmployeeColumn.COST_CENTER,
        "job_family": EmployeeColumn.JOB_FAMILY,
        "job_level": EmployeeColumn.LEVEL,
        "location": EmployeeColumn.LOCATION,
        "company": EmployeeColumn.COMPANY,
        "tech_indicator": EmployeeColumn.TECH_INDICATOR,
        "is_manager": EmployeeColumn.IS_MANAGER,
        "perf_rating": EmployeeColumn.PERF_RATING,
    }
)
#: Columns the warehouse can't filter on, so they're filtered in memory.
IN_MEMORY: Final = frozenset({"perf_rating"})
_FLAGS: Final = {"true": 1, "false": 0}


class SubsetError(ValueError):
    """A subset that can't be declared or lowered."""


@dataclass(frozen=True)
class SubsetSpec:
    rules: Mapping[str, frozenset[str]]
    name: str = ""

    def __post_init__(self) -> None:
        problems = []
        frozen = {}
        for column, values in self.rules.items():
            if column not in SUBSET_COLUMNS and not is_org_level(column):
                known = ", ".join([*SUBSET_COLUMNS, "org_level<k>"])
                problems.append(f"a subset can't constrain {column!r}; it can constrain {known}")
                continue
            admitted = frozenset(
                str(v).lower() if column == "is_manager" else str(v) for v in values
            )
            if not admitted:
                problems.append(f"the rule on {column!r} admits nothing")
            if column == "is_manager" and not admitted <= set(_FLAGS):
                problems.append(f"is_manager admits true or false, not {sorted(admitted)}")
            if column == "perf_rating" and not all(v.isdigit() for v in admitted):
                problems.append(f"perf_rating admits rating ranks, not {sorted(admitted)}")
            frozen[column] = admitted
        if problems:
            raise SubsetError("bad subset: " + "; ".join(problems))
        object.__setattr__(self, "rules", MappingProxyType(dict(sorted(frozen.items()))))

    @classmethod
    def of(cls, name: str = "", **rules: Iterable[object]) -> SubsetSpec:
        return cls({c: frozenset(str(v) for v in values) for c, values in rules.items()}, name)

    @property
    def org_rules(self) -> dict[str, frozenset[str]]:
        return {c: v for c, v in self.rules.items() if is_org_level(c)}

    def lower(self, schema: SimulationSchema) -> LoweredSubset:
        """Codes per buffer column, with org rules lowered to leaf cost centers."""
        problems = []
        codes: dict[str, frozenset[int]] = {}
        leaves: set[int] | None = None
        for level, nodes in self.org_rules.items():
            if level not in schema.rollups:
                problems.append(f"the schema has no rollup {level!r}")
                continue
            vocab = schema.vocabulary(level)
            unknown = sorted(n for n in nodes if n not in vocab)
            if unknown:
                problems.append(f"{level} has no nodes {unknown}")
            wanted = {vocab.code(n) for n in nodes if n in vocab}
            under = {int(i) for i in np.flatnonzero(np.isin(schema.rollups[level], list(wanted)))}
            leaves = under if leaves is None else leaves & under
        for column, values in self.rules.items():
            if is_org_level(column):
                continue
            if column == "is_manager":
                codes[column] = frozenset(_FLAGS[v] for v in values)
            elif column == "perf_rating":
                codes[column] = frozenset(int(v) for v in values)
            else:
                vocab = schema.vocabulary(column)
                unknown = sorted(v for v in values if v not in vocab)
                if unknown:
                    problems.append(f"{column} has no labels {unknown}")
                codes[column] = frozenset(vocab.code(v) for v in values if v in vocab)
        if leaves is not None:
            own = codes.get(COST_CENTER)
            codes[COST_CENTER] = frozenset(leaves if own is None else leaves & own)
        if problems:
            raise SubsetError(f"subset {self.name!r} can't be lowered: " + "; ".join(problems))
        return LoweredSubset(self, schema, MappingProxyType(codes))


@dataclass(frozen=True)
class LoweredSubset:
    spec: SubsetSpec
    schema: SimulationSchema = field(repr=False)
    #: Admitted codes per subset column, org rules folded into `cost_center`.
    codes: Mapping[str, frozenset[int]]

    @property
    def empty(self) -> bool:
        """Whether no row could be inside: some rule admits no code at all."""
        return any(not admitted for admitted in self.codes.values())

    def mask(self, buffer: Buffer, layout: BufferLayout) -> NDArray[np.bool_]:
        """The buffer rows inside the subset. The rating constrains employees only."""
        inside = np.ones(len(buffer), dtype=bool)
        for column, admitted in self.codes.items():
            if layout is OPENING and column == "perf_rating":
                continue
            index = _column_index(column, layout)
            inside &= np.isin(buffer[:, index].astype(np.int64), np.fromiter(admitted, np.int64))
        return inside

    def pushdown(self, population: PopulationSpec) -> PopulationSpec:
        """The population narrowed by every rule the warehouse can apply. Org rules go down
        as their leaf cost centers. A column both constrain admits what both admit."""
        rules = {c: set(v) for c, v in population.rules.items()}
        for column, admitted in self.codes.items():
            if column in IN_MEMORY:
                continue
            if column == "is_manager":
                labels = {name for name, flag in _FLAGS.items() if flag in admitted}
            else:
                labels = set(self.schema.decode(column, sorted(admitted)).tolist())
            rules[column] = labels if column not in rules else rules[column] & labels
        merged = {c: frozenset(v) for c, v in rules.items()}
        if any(not v for v in merged.values()):
            raise SubsetError(f"subset {self.spec.name!r} leaves nothing of {population.name!r}")
        return PopulationSpec(merged)

    @property
    def in_memory(self) -> frozenset[str]:
        """The subset's columns that only an in-memory filter can apply."""
        return frozenset(self.codes) & IN_MEMORY


def _column_index(column: str, layout: BufferLayout) -> int:
    employee = SUBSET_COLUMNS[column]
    if layout is EMPLOYEE:
        return int(employee)
    return int(layout.columns[employee.name])  # every subset column but the rating is shared
