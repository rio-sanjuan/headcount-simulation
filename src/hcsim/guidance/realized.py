"""Realized totals: what happened in a window of months, as a guidance artifact.

A `guided_actuals` run pins volumes and quota: it draws each month's quota from what actually
happened, so it isolates quota sampling given correct volumes. `realized(con, start, months)`
counts those volumes from the warehouse's own record of people and requisitions, rather than
from the aggregate fact the guidance pipeline fits:

- **Head flows and the headcount stock** come from the employee-grain fact, `workforce_txn`:
  each category's codes with their signs, over regular accounts, by the month of the event.
  A reversal cancels what it reverses, as in the guidance measures, so corrections and
  no-show pairs net out within their month, and a backdated row counts in the month it took
  effect.
- **Requisition flows** come from the month-end snapshots of open requisitions,
  `req_snapshot`, and of accepted offers waiting to start, `pending_start`:
  - a requisition opens in the month of its `opened_date`, with the openings it had at its
    first sighting plus the offers already accepted by then. A requisition that never reached
    a month-end open, but whose accepted offer did, opens in the month of its first offer;
  - a requisition closes in the month of the first month-end it's missing from. The closure
    was a fill when the offers accepted since its last sighting cover the openings it had,
    or, failing that, when its cost center's starters do: the fills, by the employee-grain
    fact, whose start no offer waited at a month-end for, less the openings they filled on
    requisitions that stayed open. Those starters fill each closing requisition once, the
    ones needing fewest first. Otherwise it closed unfilled. A requisition still open at the
    last month-end hasn't closed.

Snapshot spellings of the open state may change (a feed rebuilt partway through), so the state
is read without regard to case, and any state that doesn't mean open is refused. A month-end
missing from the snapshots (a missed feed run) can't date the requisitions that closed across
it, so a window that needs one, from the month-end before it to its last, is refused for the
requisition categories unless the gaps are allowed. Allowed, a requisition stays open across a
gap, and a closure in the missing month counts in the month after.

The snapshots can't see everything the requisition intervals record: a requisition opened and
closed between two month-ends, and an offer accepted and started between them, leave no trace.
So the requisition counts can differ from the guidance pipeline's measures, which read the
intervals, while the head flows match the aggregate fact's exactly.

The artifact is written at the window's step offsets (step 0 is `start`), with a snapshot of
the day before `start`, on the whole grid of months and groups, zeros included, so every group
the window touches is guided. The groups are those holding headcount at any month-end from the
one before `start` to the window's last, those any count names, and any the caller adds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import duckdb
import pandas as pd

from hcsim.guidance.ingest import (
    CATEGORY,
    GROUP,
    MONTH,
    VALUE,
    compose,
    counted,
    grouped,
    grouping_problems,
)
from hcsim.model.core.categories import CATEGORIES, Grain, GuidanceCategory
from hcsim.model.core.contract import role_column
from hcsim.model.core.guidance import FORECAST_MONTH, GuidanceForecast
from hcsim.model.core.layouts import MAX_OPENINGS_PER_REQUISITION
from hcsim.model.core.measures import (
    CATEGORY_MEASURES,
    HEADCOUNT_STOCK,
    KEYS,
    REGULAR_ACCOUNTS,
    MeasureError,
    TransTypeMeasure,
)
from hcsim.model.core.transactions import TransTypeCatalog, TransTypeError
from hcsim.model.sql.render import identifier, literal

_C = GuidanceCategory
#: The tables realized totals read.
TABLES: Final = (
    "workforce_txn",
    "req_snapshot",
    "pending_start",
    "dim_trans_type",
    "dim_account",
    "dim_org",
    "dim_job",
    "dim_location",
    "dim_company",
)
#: Every category some domain draws a quota for: what a `guided_actuals` run pins.
QUOTA_CATEGORIES: Final = tuple(c for c, spec in CATEGORIES.items() if spec.consumers)
#: The one state a requisition snapshot records, in any case.
OPEN_STATE: Final = "open"


class RealizedError(ValueError):
    """A window the warehouse can't count realized totals for."""


@dataclass(frozen=True)
class Realized:
    """A window's realized totals: the artifact, the same counts by calendar month, and what
    counting them found."""

    artifact: GuidanceForecast
    #: Long-form: one row per month, group, and category, zeros included.
    frame: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


def _months(start: pd.Period, months: int) -> pd.PeriodIndex:
    return pd.period_range(start, periods=months, freq="M")


def heads_sql(measures: Sequence[TransTypeMeasure]) -> str:
    """The transaction measures counted from the employee-grain fact, a row per transaction,
    by the event's month, yielding the month, `KEYS`, measure, and value. Binds `$first` and
    `$last`, the window's first and last days."""
    codes = ", ".join(literal(c.value) for c in sorted(set().union(*(m.codes for m in measures))))
    sums = ",\n".join(
        f"sum({m.expression(code='t.trans_type', weight='1')}) AS {identifier(m.name)}"
        for m in measures
    )
    names = ", ".join(identifier(m.name) for m in measures)
    return f"""
        SELECT month, {", ".join(KEYS)}, measure, CAST(value AS DOUBLE) AS value
        FROM (
            UNPIVOT (
                SELECT date_trunc('month', t.event_date) AS month, t.cost_center,
                       j.job_family, j.job_level, t.location, t.company, {sums}
                FROM workforce_txn t
                LEFT JOIN dim_job j ON j.job_code = t.job_code
                WHERE t.trans_type IN ({codes}) AND t.account IN ({REGULAR_ACCOUNTS})
                  AND t.event_date BETWEEN $first AND $last
                GROUP BY ALL
            )
            ON {names}
            INTO NAME measure VALUE value
        )
    """


def _offers(cost_center: str) -> str:
    """Each accepted offer once: its requisition, person, offer date, and role."""
    return f"""
        SELECT req_id, employee_id, min(offer_date) AS offer_date,
               any_value({cost_center}) AS cost_center, any_value(job_code) AS job_code,
               any_value(location) AS location, any_value(company) AS company
        FROM pending_start
        GROUP BY req_id, employee_id
    """


def opened_sql(cost_center: str = '"cost_center"') -> str:
    """Requisitions opened in the window, with their openings, from the snapshots and the
    accepted offers. `cost_center` names `pending_start`'s cost-center column. Binds `$first`
    and `$last`."""
    cap = MAX_OPENINGS_PER_REQUISITION
    return f"""
        WITH first_seen AS (
            SELECT req_id, min(snapshot_date) AS seen, arg_min(openings, snapshot_date) AS openings,
                   min(opened_date) AS opened_date,
                   arg_min(cost_center, snapshot_date) AS cost_center,
                   arg_min(job_code, snapshot_date) AS job_code,
                   arg_min(location, snapshot_date) AS location,
                   arg_min(company, snapshot_date) AS company
            FROM req_snapshot
            GROUP BY req_id
        ),
        offers AS ({_offers(cost_center)}),
        seen AS (
            SELECT f.opened_date, f.cost_center, f.job_code, f.location, f.company,
                   least(f.openings + count(o.employee_id), {cap}) AS openings
            FROM first_seen f
            LEFT JOIN offers o ON o.req_id = f.req_id AND o.offer_date <= f.seen
            GROUP BY f.req_id, f.opened_date, f.cost_center, f.job_code, f.location, f.company,
                     f.openings
        ),
        unseen AS (
            SELECT min(offer_date) AS opened_date, any_value(cost_center) AS cost_center,
                   any_value(job_code) AS job_code, any_value(location) AS location,
                   any_value(company) AS company, least(count(*), {cap}) AS openings
            FROM offers
            WHERE req_id NOT IN (SELECT req_id FROM first_seen)
            GROUP BY req_id
        ),
        opened AS (SELECT * FROM seen UNION ALL BY NAME SELECT * FROM unseen)
        SELECT date_trunc('month', o.opened_date) AS month, o.cost_center, j.job_family,
               j.job_level, o.location, o.company, 'requisitions_opened' AS measure,
               CAST(sum(o.openings) AS DOUBLE) AS value
        FROM opened o
        LEFT JOIN dim_job j ON j.job_code = o.job_code
        WHERE o.opened_date BETWEEN $first AND $last
        GROUP BY ALL
    """


def _closures(cost_center: str) -> str:
    """Each requisition that left the snapshots in the window, with what it needed filled,
    the starters its cost center had to fill it, and whether they did."""
    fills = CATEGORY_MEASURES[_C.FILLS]
    assert isinstance(fills, TransTypeMeasure)
    codes = ", ".join(literal(c.value) for c in sorted(fills.codes))
    return f"""
        WITH dates AS (
            SELECT d, lead(d) OVER (ORDER BY d) AS next
            FROM (SELECT DISTINCT snapshot_date AS d FROM req_snapshot)
        ),
        last_seen AS (
            SELECT req_id, max(snapshot_date) AS seen, arg_max(openings, snapshot_date) AS openings,
                   arg_max(cost_center, snapshot_date) AS cost_center,
                   arg_max(job_code, snapshot_date) AS job_code,
                   arg_max(location, snapshot_date) AS location,
                   arg_max(company, snapshot_date) AS company
            FROM req_snapshot
            GROUP BY req_id
        ),
        offers AS ({_offers(cost_center)}),
        closed AS (
            SELECT l.req_id, l.cost_center, l.job_code, l.location, l.company, l.seen,
                   d.next AS gone, l.openings - count(o.employee_id) AS need
            FROM last_seen l
            JOIN dates d ON d.d = l.seen
            LEFT JOIN offers o
                ON o.req_id = l.req_id AND o.offer_date > l.seen AND o.offer_date <= d.next
            WHERE d.next BETWEEN $first AND $last
            GROUP BY l.req_id, l.cost_center, l.job_code, l.location, l.company, l.seen, d.next,
                     l.openings
        ),
        starters AS (
            SELECT t.cost_center, t.event_date,
                   {fills.expression(code="t.trans_type", weight="1")} AS value
            FROM workforce_txn t
            WHERE t.trans_type IN ({codes}) AND t.account IN ({REGULAR_ACCOUNTS})
              AND NOT EXISTS (
                  SELECT 1 FROM pending_start p
                  WHERE p.employee_id = t.employee_id AND p.start_date = t.event_date
              )
        ),
        pairs AS (
            SELECT a.req_id, a.cost_center, a.snapshot_date AS seen, d.next AS gone,
                   a.openings - b.openings AS dropped
            FROM req_snapshot a
            JOIN dates d ON d.d = a.snapshot_date
            JOIN req_snapshot b ON b.req_id = a.req_id AND b.snapshot_date = d.next
            WHERE d.next BETWEEN $first AND $last
        ),
        kept AS (
            SELECT cost_center, gone, sum(greatest(dropped - offered, 0)) AS taken
            FROM (
                SELECT p.req_id, p.cost_center, p.gone, p.dropped,
                       count(o.employee_id) AS offered
                FROM pairs p
                LEFT JOIN offers o
                    ON o.req_id = p.req_id AND o.offer_date > p.seen AND o.offer_date <= p.gone
                GROUP BY p.req_id, p.cost_center, p.gone, p.dropped
            )
            GROUP BY cost_center, gone
        ),
        budget AS (
            SELECT b.cost_center, b.gone, greatest(b.starters - coalesce(k.taken, 0), 0) AS starters
            FROM (
                SELECT c.cost_center, c.gone, coalesce(sum(s.value), 0) AS starters
                FROM (SELECT DISTINCT cost_center, seen, gone FROM closed) c
                LEFT JOIN starters s
                    ON s.cost_center = c.cost_center AND s.event_date > c.seen
                   AND s.event_date <= c.gone
                GROUP BY c.cost_center, c.gone
            ) b
            LEFT JOIN kept k USING (cost_center, gone)
        )
        SELECT *, need > 0 AND covered > starters AS unfilled
        FROM (
            SELECT c.*, b.starters,
                   sum(greatest(c.need, 0)) OVER (
                       PARTITION BY c.cost_center, c.gone ORDER BY c.need, c.req_id
                   ) AS covered
            FROM closed c
            JOIN budget b USING (cost_center, gone)
        )
    """


def closed_sql(cost_center: str = '"cost_center"') -> str:
    """Requisitions closed unfilled in the window, from the snapshots, the accepted offers, and
    the employee-grain fact's starters. Binds `$first` and `$last`."""
    return f"""
        SELECT date_trunc('month', c.gone) AS month, c.cost_center, j.job_family, j.job_level,
               c.location, c.company, 'requisitions_closed_unfilled' AS measure,
               CAST(count(*) AS DOUBLE) AS value
        FROM ({_closures(cost_center)}) c
        LEFT JOIN dim_job j ON j.job_code = c.job_code
        WHERE c.unfilled
        GROUP BY ALL
    """


def _held(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    return {str(r[0]) for r in rows}


def _month_ends(con: duckdb.DuckDBPyConnection) -> tuple[pd.Period, pd.Period] | None:
    (first, last) = con.execute(
        "SELECT min(event_date), max(event_date) FROM workforce_txn WHERE trans_type = 'ACTIVE'"
    ).fetchone() or (None, None)
    if first is None or last is None:
        return None
    return pd.Period(first, "M"), pd.Period(last, "M")


def problems(
    con: duckdb.DuckDBPyConnection,
    start: pd.Period,
    months: int,
    grouping: str = "cost_center",
    categories: Sequence[GuidanceCategory] = QUOTA_CATEGORIES,
    allow_gaps: bool = False,
) -> list[str]:
    """Everything that keeps the warehouse from yielding the window's realized totals."""
    missing = [t for t in TABLES if t not in _held(con)]
    if missing:
        return [f"the source has no {', '.join(missing)}: realized totals read {', '.join(TABLES)}"]
    found = list(grouping_problems(grouping))
    if months < 1:
        found.append(f"a window needs at least one month, not {months}")
    try:
        catalog = TransTypeCatalog.from_warehouse(con)
        used = con.execute("SELECT DISTINCT trans_type FROM workforce_txn").fetchall()
        catalog.check(row[0] for row in used)
        for category in counted(categories):
            CATEGORY_MEASURES[category].check(catalog)
    except (TransTypeError, MeasureError) as error:
        found.append(str(error))
    states = [
        str(row[0])
        for row in con.execute("SELECT DISTINCT state FROM req_snapshot ORDER BY 1").fetchall()
    ]
    other = [s for s in states if s.lower() != OPEN_STATE]
    if other:
        found.append(f"req_snapshot records states that don't mean open: {other}")
    held = _month_ends(con)
    if held is None:
        found.append("workforce_txn holds no month-end stock")
    elif months >= 1:
        first, last = held
        end = start + months - 1
        if start <= first:
            found.append(
                f"the window starts in {start}, but realized flows need the month-end before "
                f"it, and the data's first is in {first}"
            )
        (snapshot,) = con.execute("SELECT max(snapshot_date) FROM req_snapshot").fetchone() or (
            None,
        )
        latest = min(last, pd.Period(snapshot, "M")) if snapshot is not None else last
        if end > latest:
            found.append(
                f"the window runs to {end}, past the last month-end the data holds, {latest}"
            )
        requisitions = [c for c in counted(categories) if CATEGORIES[c].grain is Grain.REQUISITION]
        gaps = [m for m in _missing_snapshots(con, start, months) if pd.Period(m, "M") <= latest]
        if requisitions and gaps and not allow_gaps:
            found.append(
                f"req_snapshot is missing the month-ends of {', '.join(gaps)}, so requisitions "
                "that closed across them can't be dated: allow the gaps to count those closures "
                "in the month after, or choose a window without them"
            )
    return found


def _measures(categories: Sequence[GuidanceCategory]) -> list[TransTypeMeasure]:
    heads = [HEADCOUNT_STOCK]
    for category in counted(categories):
        measure = CATEGORY_MEASURES[category]
        if CATEGORIES[category].grain is not Grain.REQUISITION and measure not in heads:
            assert isinstance(measure, TransTypeMeasure)
            heads.append(measure)
    return heads


def _missing_snapshots(con: duckdb.DuckDBPyConnection, start: pd.Period, months: int) -> list[str]:
    dates = {
        pd.Period(row[0], "M")
        for row in con.execute("SELECT DISTINCT snapshot_date FROM req_snapshot").fetchall()
    }
    wanted = pd.period_range(start - 1, start + months - 1, freq="M")
    return [str(m) for m in wanted if m not in dates]


def realized(
    con: duckdb.DuckDBPyConnection,
    start: pd.Period,
    months: int,
    grouping: str = "cost_center",
    categories: Sequence[GuidanceCategory] = QUOTA_CATEGORIES,
    groups: Sequence[str] = (),
    allow_gaps: bool = False,
) -> Realized:
    """The realized totals of `categories` per group of `grouping` in the `months` months from
    `start`, as a guidance artifact for a run from `start`. Refuses, naming every problem, a
    window or warehouse it can't count."""
    start = pd.Period(start, "M")
    found = problems(con, start, months, grouping, categories, allow_gaps)
    if found:
        raise RealizedError("; ".join(found))
    end = start + months - 1
    window = {"first": start.start_time.date(), "last": end.end_time.date()}
    heads = grouped(
        con,
        heads_sql(_measures(categories)),
        grouping,
        {"first": (start - 1).start_time.date(), "last": window["last"]},
    )
    parts = [heads]
    wanted = set(counted(categories))
    cost_center = role_column(con, "pending_start")
    if _C.REQUISITIONS_OPENED in wanted:
        parts.append(grouped(con, opened_sql(cost_center), grouping, window))
    closures: dict[str, int] = {}
    if _C.REQUISITIONS_CLOSED_UNFILLED in wanted:
        parts.append(grouped(con, closed_sql(cost_center), grouping, window))
        closures = _closure_counts(con, cost_center, window)
    raw = pd.concat([p for p in parts if not p.empty] or parts[:1], ignore_index=True)
    raw[MONTH] = pd.to_datetime(raw[MONTH]).dt.to_period("M")
    ungrouped = float(raw.loc[raw[GROUP].isna() & (raw[MONTH] >= start), VALUE].abs().sum())
    raw = raw.dropna(subset=[GROUP])
    raw[GROUP] = raw[GROUP].astype(str)
    # Measure names are the categories' own, so a row's measure is its category.
    stock = raw[(raw[CATEGORY] == HEADCOUNT_STOCK.name) & (raw[VALUE] > 0)]
    named = raw[(raw[CATEGORY] != HEADCOUNT_STOCK.name) & (raw[VALUE] != 0) & (raw[MONTH] >= start)]
    universe = sorted(set(stock[GROUP]) | set(named[GROUP]) | {str(g) for g in groups})
    if not universe:
        raise RealizedError(f"no {grouping!r} group holds headcount or any count in the window")
    leaves = [c.value for c in counted(categories)]
    grid = pd.MultiIndex.from_product(
        [_months(start, months), universe, leaves], names=[MONTH, GROUP, CATEGORY]
    )
    frame = (
        raw[raw[CATEGORY].isin(leaves)]
        .groupby([MONTH, GROUP, CATEGORY])[VALUE]
        .sum()
        .reindex(grid, fill_value=0.0)
        .reset_index()
    )
    frame = compose(frame, categories)
    artifact = GuidanceForecast(
        pd.DataFrame(
            {
                FORECAST_MONTH: (frame[MONTH] - start).map(lambda d: d.n).astype("int64"),
                CATEGORY: frame[CATEGORY],
                VALUE: frame[VALUE],
                grouping: frame[GROUP],
            }
        ),
        grouping,
        (start - 1).end_time.date().isoformat(),
        months,
    )
    report: dict[str, Any] = {
        "window": f"{start}..{end}",
        "grouping": grouping,
        "groups": len(universe),
        "totals": {str(c): float(v) for c, v in frame.groupby(CATEGORY)[VALUE].sum().items()},
        "ungrouped": ungrouped,
        "missing_snapshots": _missing_snapshots(con, start, months),
    }
    if closures:
        report["closures"] = closures
    return Realized(artifact, frame, report)


def _closure_counts(
    con: duckdb.DuckDBPyConnection, cost_center: str, window: Mapping[str, object]
) -> dict[str, int]:
    """How the window's closures were told apart: filled by offers, by starters, or not."""
    row = con.execute(
        f"""
        SELECT count(*), count(*) FILTER (WHERE need <= 0),
               count(*) FILTER (WHERE need > 0 AND NOT unfilled), count(*) FILTER (WHERE unfilled)
        FROM ({_closures(cost_center)})
        """,
        dict(window),
    ).fetchone() or (0, 0, 0, 0)
    keys = ("requisitions", "filled_by_offers", "filled_by_starters", "unfilled")
    return dict(zip(keys, (int(v) for v in row), strict=True))
