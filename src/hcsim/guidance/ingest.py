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
from hcsim.model.core.layouts import MAX_OPENINGS_PER_REQUISITION

MONTH: Final = "month"
GROUP: Final = "group"
CATEGORY: Final = "category"
VALUE: Final = "value"

#: Supported grouping columns. Org levels resolve through dim_org.
GROUPINGS: Final = ("cost_center", "org_level1", TOPLINE)

# Each measure is a SQL query yielding (month, cost_center, value). Workforce measures read
# the pre-summed fact, `workforce_agg`, as the data contract intends for the guidance pipeline.
# The trans_type codes are the contract's. Headcount measures count regular employees only:
# contingent and temporary workers book to their own account groups.
_FLOW = """
    SELECT date_trunc('month', CAST(period AS DATE)) AS month, cost_center,
           sum(headcount) AS value
    FROM workforce_agg
    WHERE trans_type = '{code}'
      AND account IN (SELECT account FROM dim_account WHERE account_group = 'regular')
    GROUP BY ALL
"""
MEASURES: Final[dict[GuidanceCategory, str]] = {
    GuidanceCategory.ATTRITION: _FLOW.format(code="TERM"),
    GuidanceCategory.HIRES: _FLOW.format(code="HIRE"),
    GuidanceCategory.HEADCOUNT: _FLOW.format(code="ACTIVE"),
    # A requisition counts once, in the month it opened, with the openings it opened with: the
    # most any of its state intervals carries. Openings are capped per requisition, as they
    # are in the opening buffer. The table's cost-center column comes from the contract.
    GuidanceCategory.REQUISITIONS_OPENED: f"""
        SELECT date_trunc('month', CAST(opened_date AS DATE)) AS month, cost_center,
               sum(least(openings, {MAX_OPENINGS_PER_REQUISITION})) AS value
        FROM (
            SELECT req_id, min(opened_date) AS opened_date,
                   any_value({{req_cost_center}}) AS cost_center, max(openings) AS openings
            FROM req_interval GROUP BY req_id
        )
        GROUP BY ALL
    """,
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
    unknown = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT trans_type FROM workforce_agg "
            "WHERE trans_type NOT IN (SELECT trans_type FROM dim_trans_type) ORDER BY 1"
        ).fetchall()
    ]
    if unknown:
        raise IngestionError(f"transaction codes missing from dim_trans_type: {unknown}")
    group = _group_expression(grouping)
    parts = []
    req_cost_center = role_column(con, "req_interval")
    for category, sql in MEASURES.items():
        frame = con.execute(
            f"""
            SELECT m.month, {group} AS "group", sum(m.value)::DOUBLE AS value
            FROM ({sql.format(req_cost_center=req_cost_center)}) m
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
