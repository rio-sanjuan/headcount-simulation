"""Eligibility rules: which rows a domain may select, declared in raw labels.

A rule names a domain and a selection of rows, written as a subset's rules over the buffer
attributes and org rollups, with raw labels such as `job_level: [8]` or `org_level1: [ORG-A]`.
It either **excludes** the rows it selects, or admits **only** them. A domain's rows are
eligible when every one of its rules allows them.

Rules live in the configuration, never in a model (the overview's design rule), and they're
declared once and resolved per schema: `resolve` lowers each selection to codes through the
schema, so the same declaration selects the same people under any vocabulary. Every problem
in every rule is collected: an unknown domain, a column a subset can't name, or a label the
schema lacks.

A domain scores rows with its model, then sets the probability of every ineligible row to
zero. Quota sampling never selects a row with probability zero, so eligibility always wins.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import EMPLOYEE, OPENING, Buffer, BufferLayout
from hcsim.model.core.schema import SimulationSchema
from hcsim.model.data.subset import LoweredSubset, SubsetError, SubsetSpec

#: The domains a rule may name, and the buffer whose rows each one selects from.
DOMAIN_LAYOUTS: Final[Mapping[str, BufferLayout]] = MappingProxyType(
    {
        "attrition": EMPLOYEE,
        "performance": EMPLOYEE,
        "promotion": EMPLOYEE,
        "transfers": EMPLOYEE,
        "backfill": EMPLOYEE,  # the departures, as employee rows
        "external_fill": OPENING,
        "cancellation": OPENING,
    }
)


class EligibilityError(ValueError):
    """Rules that can't be declared or resolved, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("bad eligibility rules: " + "; ".join(problems))


def _selection(where: Mapping[str, Iterable[object]]) -> SubsetSpec:
    return SubsetSpec({c: frozenset(str(v) for v in values) for c, values in where.items()})


@dataclass(frozen=True)
class EligibilityRule:
    domain: str
    where: SubsetSpec
    #: True: the selected rows are ineligible. False: only they are eligible.
    exclude: bool = True
    reason: str = ""

    @classmethod
    def of(
        cls, domain: str, *, exclude: bool = True, reason: str = "", **where: Iterable[object]
    ) -> EligibilityRule:
        return cls(domain, _selection(where), exclude, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "where": {c: sorted(v) for c, v in self.where.rules.items()},
            "exclude": self.exclude,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EligibilityRule:
        try:
            where = _selection(data["where"])
            return cls(str(data["domain"]), where, bool(data["exclude"]), str(data["reason"]))
        except (KeyError, TypeError, SubsetError) as error:
            raise EligibilityError([f"can't read rule {dict(data)!r}: {error}"]) from None


@dataclass(frozen=True)
class Eligibility:
    """Rules resolved against one schema."""

    rules: tuple[tuple[EligibilityRule, LoweredSubset], ...] = ()

    def eligible(self, domain: str, rows: Buffer) -> NDArray[np.bool_]:
        """Which of a domain's rows its rules allow."""
        allowed = np.ones(len(rows), dtype=bool)
        layout = DOMAIN_LAYOUTS.get(domain)
        for rule, lowered in self.rules:
            if rule.domain != domain or layout is None:
                continue
            selected = lowered.mask(rows, layout)
            allowed &= ~selected if rule.exclude else selected
        return allowed

    def restrict(self, domain: str, rows: Buffer, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """`p` with every ineligible row's probability set to zero."""
        if not any(rule.domain == domain for rule, _ in self.rules):
            return p
        result: NDArray[np.float64] = np.where(self.eligible(domain, rows), p, 0.0)
        return result

    def for_domain(self, domain: str) -> list[EligibilityRule]:
        return [rule for rule, _ in self.rules if rule.domain == domain]


def resolve(rules: Sequence[EligibilityRule], schema: SimulationSchema) -> Eligibility:
    """Lower every rule through the schema, or report every problem."""
    problems = []
    resolved = []
    for k, rule in enumerate(rules):
        where = f"rule {k + 1} ({rule.domain})"
        if rule.domain not in DOMAIN_LAYOUTS:
            known = ", ".join(DOMAIN_LAYOUTS)
            problems.append(f"{where}: no domain {rule.domain!r}; known: {known}")
            continue
        if DOMAIN_LAYOUTS[rule.domain] is OPENING and "perf_rating" in rule.where.rules:
            problems.append(f"{where}: requisitions have no rating to select by")
            continue
        try:
            resolved.append((rule, rule.where.lower(schema)))
        except SubsetError as error:
            problems.append(f"{where}: {error}")
    if problems:
        raise EligibilityError(problems)
    return Eligibility(tuple(resolved))
