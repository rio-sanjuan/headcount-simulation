"""A data profile of a written world: what's in it, at a glance, as Markdown.

The profile reads a warehouse the way a new analyst would. It covers:

- the tables and their sizes;
- each column's type, missing share, distinct values, and range;
- the workforce's flows by year;
- the transaction codes and their dispositions;
- the requisitions and how they closed;
- the plan vintages;
- the equity awards and their expense;
- the coherence checks.

It reads only the source tables and the truth schema's sizes, so it profiles a world with
any quirks on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import duckdb
import pandas as pd

from hcsim.synth.validate import validate

_REGULAR: Final = "SELECT account FROM dim_account WHERE account_group = 'regular'"

_WORKFORCE = f"""
    WITH regular AS (
        SELECT * FROM workforce_txn WHERE account IN ({_REGULAR})
    ),
    stock AS (
        SELECT event_date, count(*) AS n FROM regular WHERE trans_type = 'ACTIVE' GROUP BY 1
    ),
    yearly AS (
        SELECT year(event_date) AS year, avg(n) AS average, arg_max(n, event_date) AS year_end
        FROM stock GROUP BY 1
    ),
    flows AS (
        SELECT year(event_date) AS year,
               count(*) FILTER (trans_type = 'HIRE')
                 - count(*) FILTER (trans_type = 'HIRE_REV') AS hires,
               count(*) FILTER (trans_type IN ('TERM', 'RIF'))
                 - count(*) FILTER (trans_type = 'TERM_REV') AS departures,
               count(*) FILTER (trans_type = 'XFER_IN') AS transfers,
               count(*) FILTER (trans_type = 'PROMO') AS promotions,
               count(*) FILTER (trans_type = 'CONV_IN') AS conversions_in
        FROM regular GROUP BY 1
    )
    SELECT y.year, y.year_end AS year_end_headcount,
           coalesce(f.hires, 0) AS hires, coalesce(f.departures, 0) AS departures,
           round(coalesce(f.departures, 0) / y.average, 4) AS attrition_rate,
           coalesce(f.transfers, 0) AS transfers, coalesce(f.promotions, 0) AS promotions,
           coalesce(f.conversions_in, 0) AS conversions_in
    FROM yearly y LEFT JOIN flows f USING (year)
    ORDER BY y.year
"""

_CODES = """
    SELECT c.trans_type, c.disposition, c.amount,
           coalesce(t.n, 0) AS transaction_rows, coalesce(a.n, 0) AS aggregate_rows
    FROM dim_trans_type c
    LEFT JOIN (SELECT trans_type, count(*) AS n FROM workforce_txn GROUP BY 1) t USING (trans_type)
    LEFT JOIN (SELECT trans_type, count(*) AS n FROM workforce_agg GROUP BY 1) a USING (trans_type)
    ORDER BY c.disposition, c.trans_type
"""

_REQUISITIONS = """
    WITH reqs AS (
        SELECT req_id, any_value(req_type) AS req_type, min(opened_date) AS opened_date,
               max(openings) AS openings
        FROM req_interval GROUP BY req_id
    )
    SELECT year(opened_date) AS year,
           count(*) FILTER (req_type = 'growth') AS growth,
           count(*) FILTER (req_type = 'backfill') AS backfill,
           count(*) FILTER (req_type = 'campus') AS campus,
           sum(openings) AS openings,
           max(openings) AS most_openings
    FROM reqs GROUP BY 1 ORDER BY 1
"""

_CLOSURES = """
    SELECT coalesce(close_reason, '(still open)') AS close_reason, count(*) AS intervals,
           median(closed_date - opened_date) AS median_days_since_opened
    FROM req_interval GROUP BY 1 ORDER BY 2 DESC
"""

_VINTAGES = """
    SELECT vintage, min(published_date) AS published_date, count(*) AS rows,
           count(*) FILTER (is_hedge) AS hedge_rows,
           count(*) FILTER (target_headcount < 0) AS negative_rows,
           min(month_end) AS first_month, max(month_end) AS last_month,
           round(sum(target_headcount) FILTER (month_end = (
               SELECT max(month_end) FROM plan_headcount p WHERE p.vintage = plan_headcount.vintage
           )), 1) AS final_target
    FROM plan_headcount GROUP BY vintage ORDER BY vintage
"""

_AWARDS = """
    SELECT award_type, count(*) AS awards, sum(shares) AS shares,
           round(sum(shares * fair_value), 2) AS grant_value,
           count(DISTINCT employee_id) AS holders
    FROM equity_award GROUP BY 1 ORDER BY 1
"""

_EXPENSE = """
    SELECT year(period) AS year, round(sum(gross), 2) AS gross, round(sum(net), 2) AS net,
           round(sum(trueup_attrition), 2) AS trueup_attrition,
           round(sum(trueup_rate), 2) AS trueup_rate,
           round(sum(trueup_vesting), 2) AS trueup_vesting,
           round(sum(recognized), 2) AS recognized
    FROM equity_expense GROUP BY 1 ORDER BY 1
"""


@dataclass
class Profile:
    #: Sections in order, each a table.
    sections: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Each table's column profile.
    columns: dict[str, pd.DataFrame] = field(default_factory=dict)


def _tables(con: duckdb.DuckDBPyConnection, schema: str) -> pd.DataFrame:
    listed = con.execute(
        "SELECT table_name, table_type FROM information_schema.tables "
        "WHERE table_schema = ? ORDER BY table_name",
        [schema],
    ).fetchall()
    rows = []
    for name, kind in listed:
        (count,) = con.execute(f'SELECT count(*) FROM {schema}."{name}"').fetchone() or (0,)
        (width,) = con.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = ?",
            [schema, name],
        ).fetchone() or (0,)
        rows.append(
            {
                "table": name if schema == "main" else f"{schema}.{name}",
                "kind": "view" if kind == "VIEW" else "table",
                "rows": int(count),
                "columns": int(width),
            }
        )
    return pd.DataFrame(rows, columns=["table", "kind", "rows", "columns"])


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    summary = con.execute(f'SUMMARIZE "{table}"').df()
    return pd.DataFrame(
        {
            "column": summary["column_name"],
            "type": summary["column_type"],
            "missing": summary["null_percentage"].astype(float).div(100).round(4),
            "distinct": summary["approx_unique"],
            "min": summary["min"],
            "max": summary["max"],
        }
    )


def profile(con: duckdb.DuckDBPyConnection) -> Profile:
    tables = _tables(con, "main")
    result = Profile()
    result.sections["Tables"] = tables
    result.sections["Truth"] = _tables(con, "truth")
    for title, sql in (
        ("Regular workforce by year", _WORKFORCE),
        ("Transaction codes", _CODES),
        ("Requisitions by year opened", _REQUISITIONS),
        ("Requisition intervals by how they closed", _CLOSURES),
        ("Plan vintages", _VINTAGES),
        ("Equity awards by type", _AWARDS),
        ("Equity expense by year", _EXPENSE),
    ):
        result.sections[title] = con.execute(sql).df()
    result.sections["Coherence checks"] = pd.DataFrame(
        [
            {
                "check": r.name,
                "status": "ok" if r.passed else "FAIL",
                "violations": r.violations,
                "description": r.description,
            }
            for r in validate(con, examples=0)
        ]
    )
    for name in tables.loc[tables["kind"] == "table", "table"]:
        result.columns[str(name)] = _columns(con, str(name))
    return result


#: Columns whose numbers are labels, printed without thousands separators.
_LABELS: Final = frozenset({"year"})


def _cell(value: object, label: bool = False) -> str:
    if value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return str(int(value)) if label else f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0")
    if isinstance(value, pd.Timestamp) and value == value.normalize():
        return str(value.date())
    return str(value).replace("|", "\\|")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_none_"
    labels = [str(c) in _LABELS for c in frame.columns]
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(_cell(v, k) for v, k in zip(row, labels, strict=True)) + " |"
        for row in frame.astype(object).itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def render(result: Profile, title: str = "World profile") -> str:
    parts = [f"# {title}", ""]
    for heading, frame in result.sections.items():
        parts += [f"## {heading}", "", markdown_table(frame), ""]
    parts += ["## Columns", ""]
    for name, frame in result.columns.items():
        parts += [f"### {name}", "", markdown_table(frame), ""]
    return "\n".join(parts)
