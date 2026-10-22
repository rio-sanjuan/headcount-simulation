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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import duckdb
import pandas as pd

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

#: Groupings the series support: the leaf cost center, any org level from the column catalog
#: (`org_level1` to the deepest), and the whole population. Series are counted per cost
#: center, so nothing finer or unrelated to the org can group them.
GROUPINGS: Final = ("cost_center", "org_level<k>", TOPLINE)

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
    """Long-form monthly series, with the snapshot: the last month with any data."""

    frame: pd.DataFrame
    grouping: str
    snapshot: pd.Period

    @property
    def groups(self) -> list[str]:
        return sorted(self.frame[GROUP].unique())

    @property
    def categories(self) -> list[GuidanceCategory]:
        """The categories the series carry, in the categories' declared order."""
        carried = set(self.frame[CATEGORY])
        return [c for c in GuidanceCategory if c.value in carried]

    def values(self, group: str, category: GuidanceCategory) -> pd.Series:
        rows = self.frame[(self.frame[GROUP] == group) & (self.frame[CATEGORY] == category.value)]
        return rows.set_index(MONTH)[VALUE].sort_index()


def grouping_problems(grouping: str) -> list[str]:
    """Why the series can't be grouped by `grouping`, as far as the catalog alone can tell."""
    if grouping == TOPLINE:
        return []
    try:
        column = require(grouping, Use.GROUP)
    except CatalogError as error:
        return [str(error)]
    if column.name != "cost_center" and not is_org_level(column.name):
        return [
            f"series are counted per cost center, so they can't be grouped by {grouping!r}; "
            f"supported: {', '.join(GROUPINGS)}"
        ]
    return []


def _group_expression(con: duckdb.DuckDBPyConnection, grouping: str) -> str:
    if grouping == TOPLINE:
        return f"'{TOPLINE_LABEL}'"
    found = grouping_problems(grouping)
    if found:
        raise IngestionError(found[0])
    column = require(grouping, Use.GROUP)
    if column.name == "cost_center":
        return "m.cost_center"
    levels = {c[0] for c in con.execute("DESCRIBE dim_org").fetchall()}
    if column.name not in levels:
        raise IngestionError(f"dim_org has no {column.name!r}")
    return column.expression


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
    group = _group_expression(con, grouping)
    tables = {m.table for m in measures.values()}
    sql = consolidated_sql(list(measures.values()), {t: role_column(con, t) for t in tables})
    raw = con.execute(
        f"""
        SELECT m.month, {group} AS "group", m.measure AS category, sum(m.value)::DOUBLE AS value
        FROM ({sql}) m
        LEFT JOIN dim_org o USING (cost_center)
        GROUP BY ALL
        """
    ).df()
    if raw.empty:
        raise IngestionError("the warehouse holds no workforce or requisition data")
    names = {m.name: c.value for c, m in measures.items()}
    raw[CATEGORY] = raw[CATEGORY].map(names)
    raw[MONTH] = pd.to_datetime(raw[MONTH]).dt.to_period("M")

    # The snapshot is the last month any measure observed. Later months are trimmed.
    snapshot = raw.loc[raw[VALUE] > 0, MONTH].max()
    months = pd.period_range(raw[MONTH].min(), snapshot, freq="M")
    groups = sorted(raw["group"].dropna().unique())
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
    frame = _compose(frame, categories)
    return Series(frame, grouping, snapshot)


def _compose(frame: pd.DataFrame, categories: Sequence[GuidanceCategory]) -> pd.DataFrame:
    """The requested categories, each inner fill-tree node the sum of its leaves."""
    wide = frame.pivot(index=[MONTH, GROUP], columns=CATEGORY, values=VALUE)
    parts = []
    for category in categories:
        values = wide[[leaf.value for leaf in leaves(category)]].sum(axis=1)
        parts.append(values.rename(VALUE).reset_index().assign(**{CATEGORY: category.value}))
    composed = pd.concat(parts, ignore_index=True)[[MONTH, GROUP, CATEGORY, VALUE]]
    return composed.sort_values([MONTH, GROUP, CATEGORY], ignore_index=True)
