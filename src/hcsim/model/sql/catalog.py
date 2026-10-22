"""The column catalog: the allowlist of columns a reader may name, and what each may do.

Every grouping, population rule, and filter a profile or artifact names resolves through this
catalog, never through free SQL. An entry says where the column lives in the standard
extraction join and which uses it permits:

- `group`: a series or quota may be grouped by it;
- `population`: a population may be defined by it;
- `filter`: a query may restrict rows by it.

The standard join puts the fact row `t` (transactions or requisitions, with its own cost
center, job code, location, company, and account) beside its dimensions: `o` for `dim_org`,
`j` for `dim_job`, `l` for `dim_location`, `c` for `dim_company`, and `a` for `dim_account`.

Org levels are open-ended (`org_level1` to the deepest), so one entry serves them all.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Use(StrEnum):
    GROUP = "group"
    POPULATION = "population"
    FILTER = "filter"


_VERB: Final = {
    Use.GROUP: "group by",
    Use.POPULATION: "define a population",
    Use.FILTER: "filter on",
}


class Kind(StrEnum):
    LABEL = "label"
    NUMBER = "number"
    FLAG = "flag"
    DATE = "date"


class CatalogError(ValueError):
    """A column the catalog doesn't list, or a use it doesn't permit."""


@dataclass(frozen=True)
class CatalogColumn:
    name: str
    #: Its SQL in the standard extraction join.
    expression: str
    kind: Kind
    uses: frozenset[Use]
    description: str


_ALL = frozenset(Use)
_SELECT = frozenset({Use.POPULATION, Use.FILTER})
_FILTER = frozenset({Use.FILTER})
_ORG_LEVEL: Final = re.compile(r"org_level([1-9][0-9]*)")


def _column(
    name: str, expression: str, kind: Kind, uses: frozenset[Use], description: str
) -> CatalogColumn:
    return CatalogColumn(name, expression, kind, uses, description)


COLUMNS: Final[Mapping[str, CatalogColumn]] = MappingProxyType(
    {
        c.name: c
        for c in (
            _column("cost_center", "t.cost_center", Kind.LABEL, _ALL, "the leaf cost center"),
            _column("job_family", "j.job_family", Kind.LABEL, _ALL, "the job family"),
            _column("job_level", "j.job_level", Kind.NUMBER, _ALL, "the job level"),
            _column("tech_indicator", "j.tech_indicator", Kind.LABEL, _ALL, "the technical group"),
            _column("location", "t.location", Kind.LABEL, _ALL, "the work location"),
            _column("country", "l.country", Kind.LABEL, _ALL, "the location's country"),
            _column("region", "l.region", Kind.LABEL, _ALL, "the country's region"),
            _column("company", "t.company", Kind.LABEL, _ALL, "the legal company"),
            _column(
                "planning_company", "c.planning_company", Kind.LABEL, _ALL, "the planning rollup"
            ),
            _column("cost_type", "o.cost_type", Kind.LABEL, _SELECT, "fixed or variable"),
            _column("field_type", "o.field_type", Kind.LABEL, _SELECT, "corporate or field"),
            _column("account", "t.account", Kind.LABEL, _SELECT, "the account"),
            _column(
                "account_group", "a.account_group", Kind.LABEL, _SELECT, "regular, and the rest"
            ),
            _column("is_manager", "t.is_manager", Kind.FLAG, _SELECT, "whether it manages"),
            _column("is_executive", "j.is_executive", Kind.FLAG, _SELECT, "the executive band"),
            _column("is_offline", "c.is_offline", Kind.FLAG, _SELECT, "an offline subsidiary"),
            _column("trans_type", "t.trans_type", Kind.LABEL, _FILTER, "the transaction code"),
            _column("event_date", "t.event_date", Kind.DATE, _FILTER, "when the event fell"),
        )
    }
)


def entry(name: str) -> CatalogColumn:
    """The catalog's entry for `name`. Every org level shares one shape."""
    if name in COLUMNS:
        return COLUMNS[name]
    if (match := _ORG_LEVEL.fullmatch(name)) is not None:
        return CatalogColumn(
            name, f"o.{name}", Kind.LABEL, _ALL, f"the org node at level {match.group(1)}"
        )
    known = ", ".join([*sorted(COLUMNS), "org_level<k>"])
    raise CatalogError(f"{name!r} isn't in the column catalog; known: {known}")


def require(name: str, use: Use) -> CatalogColumn:
    """The entry for `name`, refusing a use it doesn't permit."""
    column = entry(name)
    if use not in column.uses:
        permitted = ", ".join(sorted(u.value for u in column.uses))
        raise CatalogError(f"{name!r} can't {_VERB[use]}; it permits {permitted}")
    return column


def check(requests: Iterable[tuple[str, Use]]) -> None:
    """Refuse, all at once, every (column, use) the catalog doesn't allow."""
    problems = []
    for name, use in requests:
        try:
            require(name, use)
        except CatalogError as error:
            problems.append(str(error))
    if problems:
        raise CatalogError("; ".join(problems))


def is_org_level(name: str) -> bool:
    return _ORG_LEVEL.fullmatch(name) is not None
