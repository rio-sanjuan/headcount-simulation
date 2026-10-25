"""Ingestion: read the warehouse, compose guidance categories, and aggregate to series.

Categories are composed here, before any model is fitted, so a forecaster's output is the
category's forecast. Adding forecasts after fitting would be invalid for anything but point
values. Composition works at two levels:

- each counted category is its measure: signed sums of transaction codes, or a template's own
  count (see `model.core.measures`);
- an inner node of the fill tree, `hires` or `fills`, isn't counted on its own. Its series is
  the sum of its leaves' series, so the series a forecast starts from are coherent by
  construction.

The result is long-form: one row per month, group, and category. The grid is complete: every
group has every month, with zeros where nothing happened, and trailing months with no data at
all are trimmed, because they usually mark months not yet loaded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.guidance.sources import TABLES, missing_tables
from hcsim.model.core.aligner import TOPLINE, TOPLINE_LABEL
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.contract import role_column
from hcsim.model.core.fill_tree import leaves
from hcsim.model.core.measures import CATEGORY_MEASURES, Measure, MeasureError, consolidated_sql
from hcsim.model.core.transactions import TransTypeCatalog, TransTypeError
from hcsim.model.sql.catalog import CatalogError, Use, is_org_level, require

MONTH: Final = "month"
GROUP: Final = "group"
CATEGORY: Final = "category"
VALUE: Final = "value"

#: Groupings the series support: every column the column catalog lets a series group by, and
#: the whole population. Each resolves on the measures' rows: the cost center and its org
#: levels (`org_level1` to the deepest), the job family and level, the tech indicator, the
#: location and its country and region, and the company and its planning rollup.
_GROUP_SQL: Final = {
    "cost_center": "m.cost_center",
    "job_family": "m.job_family",
    "job_level": "CAST(m.job_level AS VARCHAR)",
    "tech_indicator": "j.tech_indicator",
    "location": "m.location",
    "country": "l.country",
    "region": "l.region",
    "company": "m.company",
    "planning_company": "c.planning_company",
}
GROUPINGS: Final = (*_GROUP_SQL, "org_level<k>", TOPLINE)
#: The dimensions a grouping reads, joined to the measures' rows on their keys. The tech
#: indicator belongs to a job, which the aggregate fact names by family and level.
_JOINS: Final = """
    LEFT JOIN dim_org o USING (cost_center)
    LEFT JOIN dim_location l USING (location)
    LEFT JOIN dim_company c USING (company)
    LEFT JOIN (SELECT DISTINCT job_family, job_level, tech_indicator FROM dim_job) j
        USING (job_family, job_level)
"""

# Each counted category is one measure, a SQL query yielding (month, cost_center, value).
# Workforce measures read the pre-summed fact, `workforce_agg`, as the data contract intends
# for the guidance pipeline, and count regular employees only.
#: The categories a configuration forecasts by default: the walking skeleton's four.
DEFAULT_CATEGORIES: Final = (
    GuidanceCategory.ATTRITION,
    GuidanceCategory.EXTERNAL_HIRES,
    GuidanceCategory.HEADCOUNT,
    GuidanceCategory.REQUISITIONS_OPENED,
)
MEASURES: Final[dict[GuidanceCategory, Measure]] = {
    c: CATEGORY_MEASURES[c] for c in DEFAULT_CATEGORIES
}


def counted(categories: Sequence[GuidanceCategory]) -> list[GuidanceCategory]:
    """The categories counted by their own measure, for `categories`: each one, but an inner
    fill-tree node is its leaves."""
    found: list[GuidanceCategory] = []
    for category in categories:
        found += [leaf for leaf in leaves(category) if leaf not in found]
    return found


class IngestionError(ValueError):
    """Inputs the pipeline can't turn into series."""


@dataclass(frozen=True)
class Series:
    """Long-form monthly series, with the snapshot: the last month with any data.

    The frame is read, never changed, so what's read from it (the groups, the categories, and
    each series' rows) is found once and kept: a run reads every series several times."""

    frame: pd.DataFrame
    grouping: str
    snapshot: pd.Period
    #: The months in which any measure had any row, whatever its count.
    observed: tuple[pd.Period, ...] = ()
    #: The absolute count on rows no group took, left out of every series.
    ungrouped: float = 0.0

    @cached_property
    def groups(self) -> list[str]:
        return sorted(self.frame[GROUP].unique())

    @cached_property
    def categories(self) -> list[GuidanceCategory]:
        """The categories the series carry, in the categories' declared order."""
        carried = set(self.frame[CATEGORY])
        return [c for c in GuidanceCategory if c.value in carried]

    @cached_property
    def keys(self) -> list[tuple[str, GuidanceCategory]]:
        """Each series' (group, category), by group and then the categories' declared order."""
        present = self._rows
        return [(g, c) for g in self.groups for c in self.categories if (g, c.value) in present]

    @cached_property
    def _rows(self) -> dict[tuple[str, str], NDArray[np.intp]]:
        """Each series' row positions in the frame, in month order, by group and category value."""
        by_month = self.frame[MONTH].argsort(kind="stable").to_numpy()
        found: dict[Any, Any] = (
            self.frame.iloc[by_month].groupby([GROUP, CATEGORY], sort=False).indices
        )
        return {key: by_month[positions] for key, positions in found.items()}

    @cached_property
    def _values(self) -> NDArray[Any]:
        values: NDArray[Any] = self.frame[VALUE].to_numpy()
        return values

    def carries(self, group: str, category: GuidanceCategory) -> bool:
        """Whether there's a series for `group` and `category`."""
        return (group, category.value) in self._rows

    def values(self, group: str, category: GuidanceCategory) -> pd.Series:
        rows = self.frame.iloc[self._rows.get((group, category.value), [])]
        return rows.set_index(MONTH)[VALUE].sort_index()

    def history(self, group: str, category: GuidanceCategory) -> NDArray[Any]:
        """The values of a series it carries, month by month, as a new array: `values`, without
        the months."""
        history: NDArray[Any] = self._values[self._rows[group, category.value]]
        return history


def grouping_problems(grouping: str) -> list[str]:
    """Why the series can't be grouped by `grouping`, as far as the catalog alone can tell."""
    if grouping == TOPLINE:
        return []
    try:
        column = require(grouping, Use.GROUP)
    except CatalogError as error:
        return [str(error)]
    if column.name not in _GROUP_SQL and not is_org_level(column.name):  # pragma: no cover
        return [f"series can't be grouped by {grouping!r}; supported: {', '.join(GROUPINGS)}"]
    return []


def _group_expression(con: duckdb.DuckDBPyConnection, grouping: str) -> str:
    if grouping == TOPLINE:
        return f"'{TOPLINE_LABEL}'"
    found = grouping_problems(grouping)
    if found:
        raise IngestionError(found[0])
    if grouping in _GROUP_SQL:
        if grouping == "tech_indicator":
            (ambiguous,) = con.execute(
                "SELECT count(*) FROM (SELECT job_family, job_level FROM dim_job "
                "GROUP BY ALL HAVING count(DISTINCT tech_indicator) > 1)"
            ).fetchone() or (0,)
            if ambiguous:
                raise IngestionError(
                    f"dim_job gives {ambiguous} job family and level pair(s) more than one tech "
                    "indicator, so the aggregate fact's rows can't be grouped by it"
                )
        return _GROUP_SQL[grouping]
    levels = {c[0] for c in con.execute("DESCRIBE dim_org").fetchall()}
    if grouping not in levels:
        raise IngestionError(f"dim_org has no {grouping!r}")
    return f"o.{grouping}"


def problems(
    con: duckdb.DuckDBPyConnection,
    grouping: str = "cost_center",
    categories: Sequence[GuidanceCategory] = DEFAULT_CATEGORIES,
) -> list[str]:
    """Everything that keeps the warehouse from yielding `categories` grouped by `grouping`."""
    measures = {c: CATEGORY_MEASURES[c] for c in counted(categories)}
    missing = missing_tables(con)
    if missing:
        return [f"the source has no {', '.join(missing)}: ingestion reads {', '.join(TABLES)}"]
    found = []
    (count, example) = con.execute(
        "SELECT count(*), min(period) FROM workforce_agg "
        "WHERE CAST(period AS DATE) <> last_day(CAST(period AS DATE))"
    ).fetchone() or (0, None)
    if count:
        found.append(
            f"{count} rows of workforce_agg have a period that isn't a month end, such as "
            f"{example}: the aggregate fact is counted at month ends"
        )
    try:
        catalog = TransTypeCatalog.from_warehouse(con)
        used = con.execute("SELECT DISTINCT trans_type FROM workforce_agg").fetchall()
        catalog.check(row[0] for row in used)
        for measure in measures.values():
            measure.check(catalog)
    except (TransTypeError, MeasureError) as error:
        found.append(str(error))
    try:
        _group_expression(con, grouping)
    except IngestionError as error:
        found.append(str(error))
    return found


def ingest(
    con: duckdb.DuckDBPyConnection,
    grouping: str = "cost_center",
    categories: Sequence[GuidanceCategory] = DEFAULT_CATEGORIES,
) -> Series:
    """Each of `categories`' monthly series per group. The counted categories come from one
    query: the transaction measures in one pass over the aggregate fact, and the template
    measures beside them. An inner fill-tree node is then the sum of its leaves."""
    measures = {c: CATEGORY_MEASURES[c] for c in counted(categories)}
    found = problems(con, grouping, categories)
    if found:
        raise IngestionError("; ".join(found))
    tables = {m.table for m in measures.values()}
    sql = consolidated_sql(list(measures.values()), {t: role_column(con, t) for t in tables})
    raw = grouped(con, sql, grouping)
    if raw.empty:
        raise IngestionError("the warehouse holds no workforce or requisition data")
    # A row whose attribute names no group, such as a location missing from its dimension, is
    # left out of every series; the count it carried is reported.
    ungrouped = float(raw.loc[raw["group"].isna(), VALUE].abs().sum())
    names = {m.name: c.value for c, m in measures.items()}
    raw[CATEGORY] = raw[CATEGORY].map(names)
    raw[MONTH] = pd.to_datetime(raw[MONTH]).dt.to_period("M")

    # The snapshot is the last month any measure counted anything. Later months are trimmed.
    snapshot = raw.loc[raw[VALUE] > 0, MONTH].max()
    observed = tuple(sorted(m for m in raw[MONTH].unique() if m <= snapshot))
    months = pd.period_range(raw[MONTH].min(), snapshot, freq="M")
    groups = sorted(raw["group"].dropna().unique())
    if not groups:
        raise IngestionError(
            f"no row falls in any {grouping!r} group: its dimension names none of their keys"
        )
    grid = pd.MultiIndex.from_product(
        [months, groups, [c.value for c in measures]], names=[MONTH, GROUP, CATEGORY]
    )
    frame = (
        raw.dropna(subset=["group"])
        .groupby([MONTH, GROUP, CATEGORY])[VALUE]
        .sum()
        .reindex(grid, fill_value=0.0)
        .reset_index()
    )
    frame = compose(frame, categories)
    return Series(frame, grouping, snapshot, observed, ungrouped)


def grouped(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    grouping: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """The rows of `sql`, which yields the month, `KEYS`, measure, and value, summed per month,
    group of `grouping`, and measure, as `category`. A row whose attribute names no group keeps
    a missing group, for the caller to count."""
    group = _group_expression(con, grouping)
    return con.execute(
        f"""
        SELECT m.month, {group} AS "group", m.measure AS category, sum(m.value)::DOUBLE AS value
        FROM ({sql}) m
        {_JOINS}
        GROUP BY ALL
        """,
        dict(parameters or {}),
    ).df()


def compose(frame: pd.DataFrame, categories: Sequence[GuidanceCategory]) -> pd.DataFrame:
    """The requested categories from their counted ones, long-form, each inner fill-tree node
    the sum of its leaves."""
    wide = frame.pivot(index=[MONTH, GROUP], columns=CATEGORY, values=VALUE)
    parts = []
    for category in categories:
        values = wide[[leaf.value for leaf in leaves(category)]].sum(axis=1)
        parts.append(values.rename(VALUE).reset_index().assign(**{CATEGORY: category.value}))
    composed = pd.concat(parts, ignore_index=True)[[MONTH, GROUP, CATEGORY, VALUE]]
    return composed.sort_values([MONTH, GROUP, CATEGORY], ignore_index=True)


def basis_counts(
    con: duckdb.DuckDBPyConnection,
    grouping: str,
    leaf: str,
    categories: Sequence[GuidanceCategory],
    first: pd.Period,
    last: pd.Period,
    groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """`categories`' monthly counts per group of `grouping` and leaf of `leaf`, in months
    `first` through `last`, for `groups` when given: one row per month, group, leaf, and
    category, an inner fill-tree node the sum of its leaves. The rows a disaggregation basis is
    built from."""
    measures = {c: CATEGORY_MEASURES[c] for c in counted(categories)}
    found = problems(con, grouping, categories) + problems(con, leaf, categories)
    if found:
        raise IngestionError("; ".join(sorted(set(found))))
    group, child = _group_expression(con, grouping), _group_expression(con, leaf)
    tables = {m.table for m in measures.values()}
    sql = consolidated_sql(list(measures.values()), {t: role_column(con, t) for t in tables})
    within = ""
    parameters: list[object] = [first.start_time.date(), last.start_time.date()]
    if groups is not None:
        within = f"AND {group} IN (SELECT unnest(?::VARCHAR[]))"
        parameters.append(list(groups))
    raw = con.execute(
        f"""
        SELECT m.month, {group} AS "group", {child} AS leaf, m.measure AS category,
               sum(m.value)::DOUBLE AS value
        FROM ({sql}) m
        {_JOINS}
        WHERE m.month BETWEEN ? AND ? {within}
        GROUP BY ALL
        """,
        parameters,
    ).df()
    raw = raw.dropna(subset=["group", "leaf"])
    raw["group"], raw["leaf"] = raw["group"].astype(str), raw["leaf"].astype(str)
    raw[MONTH] = pd.to_datetime(raw[MONTH]).dt.to_period("M")
    names = {m.name: c.value for c, m in measures.items()}
    raw[CATEGORY] = raw[CATEGORY].map(names)
    parts = []
    for category in categories:
        wanted = [leaf_.value for leaf_ in leaves(category)]
        rows = raw[raw[CATEGORY].isin(wanted)]
        summed = rows.groupby([MONTH, "group", "leaf"], as_index=False)[VALUE].sum()
        parts.append(summed.assign(**{CATEGORY: category.value}))
    columns = [MONTH, "group", "leaf", CATEGORY, VALUE]
    return pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)
