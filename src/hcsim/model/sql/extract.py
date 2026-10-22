"""The population compiler and the extraction queries built on it.

Every extract reads a fact table through a standard join, so the column catalog's expressions
(`t.cost_center`, `j.job_family`, `a.account_group`, ...) mean the same thing in every query:

- transactions: `workforce_txn t` beside `dim_org o`, `dim_job j`, `dim_location l`,
  `dim_company c`, and `dim_account a`;
- requisitions: `req_interval t` beside the same dimensions but `a`, since a requisition books
  to no account. Its cost-center column comes from the data contract.

The compiler turns a population into one SQL predicate over the join: each rule becomes
`expression IN (values)`, and the rules are ANDed. A rule on a dimension the join lacks is
refused, or skipped where the caller says so, as requisition seeds do with account rules.

Data never enters the SQL: dates and other values are bound parameters (`$snapshot`,
`$start`, `$end`), and labels go through `literal`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from hcsim.model.core.measures import TransTypeMeasure
from hcsim.model.data.population import PopulationError, PopulationSpec
from hcsim.model.sql.catalog import Use, entry, require
from hcsim.model.sql.render import identifier, literal, load

_DIMENSIONS: Final = """
JOIN dim_org o ON o.cost_center = t.cost_center
JOIN dim_job j ON j.job_code = t.job_code
JOIN dim_location l ON l.location = t.location
JOIN dim_company c ON c.company = t.company
"""


@dataclass(frozen=True)
class Join:
    name: str
    sql: str
    aliases: frozenset[str]


TRANSACTIONS: Final = Join(
    "transactions",
    "workforce_txn t" + _DIMENSIONS + "JOIN dim_account a ON a.account = t.account",
    frozenset({"t", "o", "j", "l", "c", "a"}),
)


def requisitions(cost_center: str = '"cost_center"') -> Join:
    """The requisition join, with `cost_center` naming the table's cost-center column, as
    `role_column` gives it."""
    source = (
        "req_interval t"
        if cost_center == '"cost_center"'
        else f"(SELECT *, {cost_center} AS cost_center FROM req_interval) t"
    )
    return Join("requisitions", source + _DIMENSIONS, frozenset({"t", "o", "j", "l", "c"}))


def predicate(
    spec: PopulationSpec, join: Join, *, unavailable: Literal["raise", "skip"] = "raise"
) -> str:
    """The population as one predicate over `join`. With no rules left, it's `TRUE`."""
    clauses, missing = [], []
    for column, values in spec.rules.items():
        expression = require(column, Use.POPULATION).expression
        if expression.split(".")[0] not in join.aliases:
            missing.append(column)
            continue
        listed = ", ".join(literal(v) for v in sorted(values))
        clauses.append(f"{expression} IN ({listed})")
    if missing and unavailable == "raise":
        raise PopulationError(
            f"population {spec.name!r} has rules on {missing}, which {join.name} don't carry"
        )
    return "(" + " AND ".join(clauses) + ")" if clauses else "TRUE"


def seed_employees(spec: PopulationSpec) -> str:
    """The population's month-end stock; binds `$snapshot`."""
    return load("seed_employees").render(
        source=TRANSACTIONS.sql, population=predicate(spec, TRANSACTIONS)
    )


def seed_openings(spec: PopulationSpec, cost_center: str = '"cost_center"') -> str:
    """The requisitions open at a month-end; binds `$snapshot`. Account rules don't constrain
    requisitions, which book to no account."""
    join = requisitions(cost_center)
    return load("seed_openings").render(
        source=join.sql, population=predicate(spec, join, unavailable="skip")
    )


def person_months(
    spec: PopulationSpec, codes: Sequence[str], attributes: Sequence[str] = ()
) -> str:
    """Person-months at risk, labelled by an event among `codes` the next month; binds
    `$start` and `$end`. `attributes` are catalog columns carried as features."""
    if not codes:
        raise PopulationError("a training extract needs at least one event code")
    carried = "".join(f", {entry(a).expression} AS {identifier(a)}" for a in attributes)
    return load("person_months").render(
        source=TRANSACTIONS.sql,
        population=predicate(spec, TRANSACTIONS),
        codes=", ".join(literal(c) for c in sorted(set(codes))),
        attributes=carried,
    )


def realized_flows(measure: TransTypeMeasure, spec: PopulationSpec, grouping: str) -> str:
    """A measure's realized count per month and group; binds `$start` and `$end`."""
    group = require(grouping, Use.GROUP).expression
    return load("realized_flows").render(
        source=TRANSACTIONS.sql,
        group=group,
        expression=measure.expression(code="t.trans_type", weight="1"),
        codes=", ".join(literal(c.value) for c in sorted(measure.codes)),
        population=predicate(spec, TRANSACTIONS),
    )
