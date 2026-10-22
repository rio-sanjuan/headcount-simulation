"""Measures: how a guidance series is counted from the warehouse.

Each measure renders one SQL query that yields `(month, cost_center, job_family, job_level,
location, company, value)`: the month as the first of the month, the attributes a count can be
grouped by (`KEYS`), and the count.

A `TransTypeMeasure` sums the aggregate fact's rows for a set of codes, each with a sign, over
regular accounts. Net hires, for instance, are `HIRE` less `HIRE_REV`, so a hire recorded in
error and reversed counts for nothing. A measure checks its codes against the warehouse's
catalog: every code must be defined and measured, never excluded or ignored.

A `TemplateMeasure` counts something the transaction codes can't express, from its own SQL:
requisitions, which belong to the opening identity rather than the headcount one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from hcsim.model.core.categories import CATEGORIES, Grain, GuidanceCategory
from hcsim.model.core.layouts import MAX_OPENINGS_PER_REQUISITION
from hcsim.model.core.transactions import Disposition, TransType, TransTypeCatalog
from hcsim.model.sql.render import Template, literal

#: Regular employees only: contingent and temporary workers book to their own accounts.
REGULAR_ACCOUNTS: Final = "SELECT account FROM dim_account WHERE account_group = 'regular'"
#: The attributes every measure's rows carry, beside the month: the leaf cost center, and the
#: job and place a series can be grouped by.
KEYS: Final = ("cost_center", "job_family", "job_level", "location", "company")
#: The same attributes as each measure selects them, past the cost center.
_ATTRIBUTES: Final = "job_family, job_level, location, company"


class MeasureError(ValueError):
    """A measure that can't be counted as declared."""


_FLOW: Final = Template(
    """
    SELECT date_trunc('month', CAST(period AS DATE)) AS month, {{cost_center}} AS cost_center,
           {{attributes}}, sum({{expression}}) AS value
    FROM workforce_agg
    WHERE trans_type IN ({{codes}})
      AND account IN ({{regular}})
    GROUP BY ALL
    """,
    "transaction measure",
)


@dataclass(frozen=True)
class TransTypeMeasure:
    """Signed sums of transaction codes. A code's sign is +1 or -1: it counts its rows, or
    takes them away, whatever its headcount amount."""

    name: str
    terms: Mapping[TransType, int]
    grain: Grain = Grain.HEADCOUNT
    #: The table the measure reads, whose cost-center column `sql` takes.
    table: str = "workforce_agg"

    def __post_init__(self) -> None:
        problems = []
        if not self.terms:
            problems.append("no codes")
        bad = sorted(code.value for code, sign in self.terms.items() if sign not in (-1, 1))
        if bad:
            problems.append(f"signs must be +1 or -1: {bad}")
        if self.grain is Grain.REQUISITION:
            problems.append("transaction codes count heads, not requisitions")
        if problems:
            raise MeasureError(f"{self.name}: " + "; ".join(problems))
        object.__setattr__(self, "terms", MappingProxyType(dict(self.terms)))

    @property
    def codes(self) -> frozenset[TransType]:
        return frozenset(self.terms)

    def expression(self, code: str = "trans_type", weight: str = "headcount") -> str:
        """`weight` times each row's sign, as a SQL expression."""
        cases = " ".join(
            f"WHEN {literal(c.value)} THEN {sign}" for c, sign in sorted(self.terms.items())
        )
        return f"(CASE {code} {cases} ELSE 0 END) * {weight}"

    def check(self, catalog: TransTypeCatalog) -> None:
        """Refuse codes the catalog doesn't define, or doesn't measure."""
        catalog.check(c.value for c in self.codes)
        unmeasured = sorted(
            c.value for c in self.codes if catalog.disposition(c) is not Disposition.MEASURED
        )
        if unmeasured:
            raise MeasureError(f"{self.name}: codes that aren't measured: {unmeasured}")

    def sql(self, cost_center: str = "cost_center") -> str:
        codes = ", ".join(literal(c.value) for c in sorted(self.codes))
        return _FLOW.render(
            expression=self.expression(),
            codes=codes,
            regular=REGULAR_ACCOUNTS,
            cost_center=cost_center,
            attributes=_ATTRIBUTES,
        )


@dataclass(frozen=True)
class TemplateMeasure:
    """A measure with its own SQL. `{{cost_center}}` names its table's cost-center column,
    which the data contract resolves per warehouse."""

    name: str
    grain: Grain
    template: Template = field(repr=False)
    table: str = "req_interval"

    def check(self, catalog: TransTypeCatalog) -> None:
        """A template reads no transaction codes, so any catalog will do."""

    def sql(self, cost_center: str = "cost_center") -> str:
        return self.template.render(cost_center=cost_center)


Measure = TransTypeMeasure | TemplateMeasure


def _identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def consolidated_sql(measures: Sequence[Measure], columns: Mapping[str, str] | None = None) -> str:
    """One query for every measure, yielding its `KEYS` with the month, measure, and value.

    The transaction measures share one pass over the aggregate fact, a column each, unpivoted
    to rows; each template measure runs its own SQL beside them. `columns` names each table's
    cost-center column, as the data contract resolves it; a table it leaves out uses
    `cost_center`.
    """
    resolved = dict(columns or {})
    names = [m.name for m in measures]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise MeasureError(f"each measure is counted once; {repeated} repeat")
    if not measures:
        raise MeasureError("no measures to count")
    flows = [m for m in measures if isinstance(m, TransTypeMeasure)]
    keys = ", ".join(KEYS)
    parts = []
    if flows:
        codes = ", ".join(literal(c.value) for c in sorted(set().union(*(m.codes for m in flows))))
        sums = ",\n".join(f"sum({m.expression()}) AS {_identifier(m.name)}" for m in flows)
        cost_center = resolved.get("workforce_agg", "cost_center")
        parts.append(
            f"""
            SELECT month, {keys}, measure, CAST(value AS DOUBLE) AS value
            FROM (
                UNPIVOT (
                    SELECT date_trunc('month', CAST(period AS DATE)) AS month,
                           {cost_center} AS cost_center, {_ATTRIBUTES}, {sums}
                    FROM workforce_agg
                    WHERE trans_type IN ({codes}) AND account IN ({REGULAR_ACCOUNTS})
                    GROUP BY ALL
                )
                ON {", ".join(_identifier(m.name) for m in flows)}
                INTO NAME measure VALUE value
            )
            """
        )
    for m in measures:
        if isinstance(m, TemplateMeasure):
            parts.append(
                f"SELECT month, {keys}, {literal(m.name)} AS measure, "
                f"CAST(value AS DOUBLE) AS value "
                f"FROM ({m.sql(resolved.get(m.table, 'cost_center'))})"
            )
    return "\nUNION ALL\n".join(parts)


_HIRE = {TransType.HIRE: 1, TransType.HIRE_REV: -1}

HEADCOUNT_STOCK: Final = TransTypeMeasure("headcount", {TransType.ACTIVE: 1}, grain=Grain.STOCK)
#: Departures net of reversals. A termination program's departures (`RIF`) are known events,
#: not sampled attrition, so they don't count.
NET_DEPARTURES: Final = TransTypeMeasure("attrition", {TransType.TERM: 1, TransType.TERM_REV: -1})
NET_EXTERNAL_HIRES: Final = TransTypeMeasure("external_hires", _HIRE)
CONVERSION_HIRES: Final = TransTypeMeasure("conversion_hires", {TransType.CONV_IN: 1})
TRANSFERS_IN: Final = TransTypeMeasure("internal_starts", {TransType.XFER_IN: 1})
CONVERSIONS_OUT: Final = TransTypeMeasure("conversions_out", {TransType.CONV_OUT: 1})
NET_HIRES: Final = TransTypeMeasure("hires", {**_HIRE, TransType.CONV_IN: 1})
NET_FILLS: Final = TransTypeMeasure("fills", {**_HIRE, TransType.CONV_IN: 1, TransType.XFER_IN: 1})
#: Promotions change a level in place: they move no one.
PROMOTIONS_IN_PLACE: Final = TransTypeMeasure("promotions", {TransType.PROMO: 1})

#: A requisition counts once, in the month of its first sighting, with the openings it opened
#: with: the most any of its intervals carries, capped as the opening buffer caps them.
REQUISITIONS_OPENED: Final = TemplateMeasure(
    "requisitions_opened",
    Grain.REQUISITION,
    Template(
        f"""
        SELECT date_trunc('month', CAST(opened_date AS DATE)) AS month, cost_center,
               {_ATTRIBUTES}, sum(least(openings, {MAX_OPENINGS_PER_REQUISITION})) AS value
        FROM (
            SELECT req_id, min(opened_date) AS opened_date,
                   any_value({{{{cost_center}}}}) AS cost_center,
                   any_value(job_family) AS job_family, any_value(job_level) AS job_level,
                   any_value(location) AS location, any_value(company) AS company,
                   max(openings) AS openings
            FROM req_interval GROUP BY req_id
        )
        GROUP BY ALL
        """,
        "requisitions opened",
    ),
)

#: A requisition that closes unfilled counts once, in the month of its last sighting: the end
#: of its last interval.
REQUISITIONS_CLOSED_UNFILLED: Final = TemplateMeasure(
    "requisitions_closed_unfilled",
    Grain.REQUISITION,
    Template(
        """
        SELECT date_trunc('month', CAST(closed_date AS DATE)) AS month, cost_center,
               job_family, job_level, location, company, count(*) AS value
        FROM (
            SELECT req_id, arg_max(close_reason, interval_number) AS close_reason,
                   max(closed_date) AS closed_date,
                   any_value({{cost_center}}) AS cost_center, any_value(job_family) AS job_family,
                   any_value(job_level) AS job_level, any_value(location) AS location,
                   any_value(company) AS company
            FROM req_interval GROUP BY req_id
        )
        WHERE close_reason = 'unfilled'
        GROUP BY ALL
        """,
        "requisitions closed unfilled",
    ),
)

#: One measure per guidance category, named for it.
CATEGORY_MEASURES: Final[Mapping[GuidanceCategory, Measure]] = MappingProxyType(
    {
        GuidanceCategory(m.name): m
        for m in (
            HEADCOUNT_STOCK,
            NET_DEPARTURES,
            NET_EXTERNAL_HIRES,
            CONVERSION_HIRES,
            TRANSFERS_IN,
            CONVERSIONS_OUT,
            NET_HIRES,
            NET_FILLS,
            PROMOTIONS_IN_PLACE,
            REQUISITIONS_OPENED,
            REQUISITIONS_CLOSED_UNFILLED,
        )
    }
)
if set(CATEGORY_MEASURES) != set(CATEGORIES):  # pragma: no cover - an import-time check
    raise MeasureError("every guidance category needs exactly one measure")
