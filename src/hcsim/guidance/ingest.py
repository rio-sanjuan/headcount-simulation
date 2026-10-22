"""Ingestion: read the warehouse, compose guidance categories, and aggregate to series.

Measures are composed into categories here, before any model is fitted, so a forecaster's
output is the category's forecast. Adding forecasts after fitting would be invalid for anything
but point values.

The result is long-form: one row per month, group, and category. The grid is complete: every
group has every month, with zeros where nothing happened, and trailing months with no data at
all are trimmed, because they usually mark months not yet loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import duckdb
import pandas as pd

from hcsim.model.core.aligner import TOPLINE, TOPLINE_LABEL
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.contract import role_column
from hcsim.model.core.measures import (
    HEADCOUNT_STOCK,
    NET_DEPARTURES,
    NET_HIRES,
    REQUISITIONS_OPENED,
    Measure,
    MeasureError,
)
from hcsim.model.core.transactions import TransTypeCatalog, TransTypeError

MONTH: Final = "month"
GROUP: Final = "group"
CATEGORY: Final = "category"
VALUE: Final = "value"

#: Supported grouping columns. Org levels resolve through dim_org.
GROUPINGS: Final = ("cost_center", "org_level1", TOPLINE)

# Each category is counted by one measure, a SQL query yielding (month, cost_center, value).
# Workforce measures read the pre-summed fact, `workforce_agg`, as the data contract intends
# for the guidance pipeline, and count regular employees only.
MEASURES: Final[dict[GuidanceCategory, Measure]] = {
    GuidanceCategory.ATTRITION: NET_DEPARTURES,
    GuidanceCategory.HIRES: NET_HIRES,
    GuidanceCategory.HEADCOUNT: HEADCOUNT_STOCK,
    GuidanceCategory.REQUISITIONS_OPENED: REQUISITIONS_OPENED,
}


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

    def values(self, group: str, category: GuidanceCategory) -> pd.Series:
        rows = self.frame[(self.frame[GROUP] == group) & (self.frame[CATEGORY] == category.value)]
        return rows.set_index(MONTH)[VALUE].sort_index()


def _group_expression(grouping: str) -> str:
    if grouping == "cost_center":
        return "m.cost_center"
    if grouping == "org_level1":
        return "o.org_level1"
    if grouping == TOPLINE:
        return f"'{TOPLINE_LABEL}'"
    raise IngestionError(f"unsupported grouping {grouping!r}; known: {', '.join(GROUPINGS)}")


def ingest(con: duckdb.DuckDBPyConnection, grouping: str = "cost_center") -> Series:
    try:
        catalog = TransTypeCatalog.from_warehouse(con)
        used = con.execute("SELECT DISTINCT trans_type FROM workforce_agg").fetchall()
        catalog.check(row[0] for row in used)
        for measure in MEASURES.values():
            measure.check(catalog)
    except (TransTypeError, MeasureError) as error:
        raise IngestionError(str(error)) from None
    group = _group_expression(grouping)
    parts = []
    for category, measure in MEASURES.items():
        sql = measure.sql(role_column(con, measure.table))
        frame = con.execute(
            f"""
            SELECT m.month, {group} AS "group", sum(m.value)::DOUBLE AS value
            FROM ({sql}) m
            LEFT JOIN dim_org o USING (cost_center)
            GROUP BY ALL
            """
        ).df()
        if not frame.empty:
            parts.append(frame.assign(category=category.value))
    if not parts:
        raise IngestionError("the warehouse holds no workforce or requisition data")
    raw = pd.concat(parts, ignore_index=True)
    raw[MONTH] = pd.to_datetime(raw[MONTH]).dt.to_period("M")

    # The snapshot is the last month any measure observed. Later months are trimmed.
    snapshot = raw.loc[raw[VALUE] > 0, MONTH].max()
    months = pd.period_range(raw[MONTH].min(), snapshot, freq="M")
    groups = sorted(raw["group"].dropna().unique())
    grid = pd.MultiIndex.from_product(
        [months, groups, [c.value for c in MEASURES]], names=[MONTH, GROUP, CATEGORY]
    )
    frame = (
        raw.dropna(subset=["group"])
        .groupby([MONTH, GROUP, CATEGORY])[VALUE]
        .sum()
        .reindex(grid, fill_value=0.0)
        .reset_index()
    )
    return Series(frame, grouping, snapshot)
