"""Coherence validators: the identities a written world must satisfy.

Each check is a query over the warehouse that returns the rows breaking an identity; a world
passes when every query returns none. The checks read only the source tables, so they hold
with every quirk on: quirks change how the data looks, never what it adds up to.

- Stock and flow: each month-end stock is the last one plus the month's summed flows.
- The aggregate fact is the transactions summed, apart from ignored codes found only there.
- Each requisition's state intervals follow from one another.
- Fills tie to the workforce moves that filled them, and pending starts to their offers.
- Vest tranches add up to their awards, and the reference ledger's rows to their totals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import duckdb

#: The relative tolerance for money.
TOLERANCE: Final = 1e-6


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    #: Selects the rows that break the identity.
    sql: str


@dataclass(frozen=True)
class Result:
    name: str
    description: str
    violations: int
    #: A few of the offending rows, for a report.
    examples: list[dict[str, object]]

    @property
    def passed(self) -> bool:
        return self.violations == 0


_STOCK_FLOW = """
    WITH stock AS (
        SELECT event_date AS month_end, account, cost_center, count(*) AS n
        FROM workforce_txn WHERE trans_type = 'ACTIVE' GROUP BY ALL
    ),
    flow AS (
        SELECT last_day(event_date) AS month_end, account, cost_center, sum(amount) AS net
        FROM workforce_txn WHERE trans_type <> 'ACTIVE' GROUP BY ALL
    ),
    ends AS (
        SELECT month_end, lag(month_end) OVER (ORDER BY month_end) AS previous FROM calendar
    ),
    keys AS (
        SELECT account, cost_center FROM stock UNION SELECT account, cost_center FROM flow
    )
    SELECT e.month_end, k.account, k.cost_center,
           coalesce(s.n, 0) - coalesce(p.n, 0) AS stock_change, coalesce(f.net, 0) AS net_flow
    FROM ends e CROSS JOIN keys k
    LEFT JOIN stock s
      ON s.month_end = e.month_end AND s.account = k.account AND s.cost_center = k.cost_center
    LEFT JOIN stock p
      ON p.month_end = e.previous AND p.account = k.account AND p.cost_center = k.cost_center
    LEFT JOIN flow f
      ON f.month_end = e.month_end AND f.account = k.account AND f.cost_center = k.cost_center
    WHERE e.previous IS NOT NULL AND coalesce(s.n, 0) - coalesce(p.n, 0) <> coalesce(f.net, 0)
"""

_ONE_STOCK_ROW = """
    SELECT employee_id, event_date, count(*) AS rows
    FROM workforce_txn WHERE trans_type = 'ACTIVE' GROUP BY ALL HAVING count(*) > 1
"""

_KNOWN_CODES = """
    SELECT DISTINCT trans_type FROM (
        SELECT trans_type FROM workforce_txn UNION ALL SELECT trans_type FROM workforce_agg
    )
    WHERE trans_type NOT IN (SELECT trans_type FROM dim_trans_type)
"""

_AGGREGATE = """
    WITH summed AS (
        SELECT last_day(t.event_date) AS period, t.cost_center, j.job_family, j.job_level,
               t.location, t.company, t.account, t.trans_type,
               count(*) AS headcount, sum(t.amount) AS amount
        FROM workforce_txn t LEFT JOIN dim_job j USING (job_code)
        GROUP BY ALL
    )
    SELECT period, cost_center, job_family, job_level, location, company, account, trans_type,
           s.headcount AS summed_headcount, a.headcount AS aggregate_headcount,
           s.amount AS summed_amount, a.amount AS aggregate_amount
    FROM summed s
    FULL OUTER JOIN workforce_agg a
      USING (period, cost_center, job_family, job_level, location, company, account, trans_type)
    WHERE (s.headcount IS DISTINCT FROM a.headcount OR s.amount IS DISTINCT FROM a.amount)
      AND NOT (
          s.headcount IS NULL
          AND trans_type IN (SELECT trans_type FROM dim_trans_type WHERE disposition = 'ignored')
      )
"""

# A fill takes one opening and starts the next interval the same day; a no-show gives one
# back, ending an open interval, or reopening a filled requisition with one opening, on or
# after the day it filled. Only the last interval may be open or exited unfilled.
_REQUISITION_CHAIN = """
    WITH r AS (
        SELECT req_id, interval_number, openings, opened_date, interval_start, closed_date,
               close_reason, filled_by,
               lead(openings) OVER w AS next_openings,
               lead(interval_start) OVER w AS next_start,
               max(openings) OVER (PARTITION BY req_id) AS opened_with,
               coalesce(starts_with(close_reason, 'filled'), false) AS filled
        FROM req_interval
        WINDOW w AS (PARTITION BY req_id ORDER BY interval_number)
    )
    SELECT req_id, interval_number, openings, interval_start, closed_date, close_reason,
           next_openings, next_start
    FROM r
    WHERE NOT coalesce(
        (closed_date IS NULL OR closed_date >= interval_start)
        AND interval_start >= opened_date
        AND (interval_number > 1 OR (interval_start = opened_date AND openings = opened_with))
        AND filled = (filled_by IS NOT NULL)
        AND CASE
            WHEN next_openings IS NULL THEN
                close_reason IS NULL OR close_reason = 'unfilled' OR (filled AND openings = 1)
            WHEN close_reason = 'reopened' THEN
                next_openings = openings + 1 AND next_start = closed_date
            WHEN filled AND openings > 1 THEN
                next_openings = openings - 1 AND next_start = closed_date
            WHEN filled THEN next_openings = 1 AND next_start >= closed_date
            ELSE false
        END,
        false
    )
"""

# Transfers and conversions arrive the day they fill; campus hires start that day. Every
# transfer or conversion arrival fills a requisition.
_FILLS = """
    WITH fills AS (
        SELECT filled_by AS employee_id, closed_date AS event_date,
               CASE close_reason
                   WHEN 'filled_transfer' THEN 'XFER_IN'
                   WHEN 'filled_conversion' THEN 'CONV_IN'
                   ELSE 'HIRE'
               END AS trans_type
        FROM req_interval
        WHERE close_reason IN ('filled_transfer', 'filled_conversion', 'filled_campus')
    ),
    arrivals AS (
        SELECT employee_id, event_date, trans_type
        FROM workforce_txn WHERE trans_type IN ('XFER_IN', 'CONV_IN')
    ),
    rows AS (
        SELECT DISTINCT employee_id, event_date, trans_type
        FROM workforce_txn WHERE trans_type IN ('XFER_IN', 'CONV_IN', 'HIRE')
    )
    SELECT 'fill without its move' AS problem, * FROM (
        SELECT * FROM fills EXCEPT ALL SELECT * FROM rows
    )
    UNION ALL
    SELECT 'move filling nothing' AS problem, * FROM (
        SELECT * FROM arrivals EXCEPT ALL SELECT * FROM fills
    )
"""

# Each pending start waits on an accepted offer: an external fill by that person, made on
# the offer date, starting after the snapshot.
_PENDING = """
    SELECT p.snapshot_date, p.employee_id, p.req_id, p.offer_date, p.start_date
    FROM pending_start p
    WHERE NOT (p.offer_date <= p.snapshot_date AND p.snapshot_date < p.start_date)
       OR NOT EXISTS (
           SELECT 1 FROM req_interval r
           WHERE r.req_id = p.req_id AND r.filled_by = p.employee_id
             AND r.close_reason = 'filled_external' AND r.closed_date = p.offer_date
       )
"""

_VESTS = """
    SELECT a.award_id, a.shares, v.shares AS tranche_shares
    FROM equity_award a
    FULL OUTER JOIN (
        SELECT award_id, sum(shares) AS shares FROM equity_vest GROUP BY award_id
    ) v USING (award_id)
    WHERE a.shares IS DISTINCT FROM v.shares
"""

_VEST_STATUS = """
    SELECT award_id, vest_number, vest_date, status, forfeit_date
    FROM equity_vest
    WHERE NOT coalesce(
        CASE status
            WHEN 'forfeited' THEN forfeit_date IS NOT NULL AND forfeit_date <= vest_date
            WHEN 'vested' THEN
                forfeit_date IS NULL AND vest_date <= (SELECT max(month_end) FROM calendar)
            WHEN 'unvested' THEN
                forfeit_date IS NULL AND vest_date > (SELECT max(month_end) FROM calendar)
            ELSE false
        END,
        false
    )
"""

_LEDGER_ROWS = f"""
    SELECT award_id, period, recognized,
           net + trueup_attrition + trueup_rate + trueup_vesting AS parts
    FROM equity_expense
    WHERE abs(recognized - (net + trueup_attrition + trueup_rate + trueup_vesting))
          > {TOLERANCE} * (1 + abs(recognized))
       OR award_id NOT IN (SELECT award_id FROM equity_award)
       OR period < (SELECT last_day(award_date) FROM equity_award a
                    WHERE a.award_id = equity_expense.award_id)
"""

# An award granted and settled inside the window recognizes exactly its vested value: the
# forfeited tranches reverse to nothing.
_LEDGER_TOTALS = f"""
    WITH window_ AS (
        SELECT date_trunc('month', min(month_end)) AS first, max(month_end) AS last
        FROM calendar
    ),
    settled AS (
        SELECT v.award_id,
               sum(CASE WHEN v.status = 'vested' THEN v.shares ELSE 0 END) AS vested_shares
        FROM equity_vest v
        GROUP BY v.award_id
        HAVING bool_and(v.status <> 'unvested'
                        AND coalesce(v.forfeit_date, v.vest_date) <= (SELECT last FROM window_))
    )
    SELECT a.award_id, s.vested_shares * a.fair_value AS vested_value, e.recognized
    FROM equity_award a
    JOIN settled s USING (award_id)
    LEFT JOIN (
        SELECT award_id, sum(recognized) AS recognized FROM equity_expense GROUP BY award_id
    ) e USING (award_id)
    WHERE a.award_date >= (SELECT first FROM window_)
      AND abs(coalesce(e.recognized, 0) - s.vested_shares * a.fair_value)
          > {TOLERANCE} * (1 + s.vested_shares * a.fair_value)
"""

CHECKS: Final[tuple[Check, ...]] = (
    Check(
        "stock_flow",
        "each month-end stock is the last one plus the month's summed flows",
        _STOCK_FLOW,
    ),
    Check("one_stock_row", "nobody is in a month-end stock twice", _ONE_STOCK_ROW),
    Check("known_codes", "every transaction code is in the catalog", _KNOWN_CODES),
    Check(
        "aggregate_ties",
        "the aggregate fact is the transactions summed, apart from ignored codes",
        _AGGREGATE,
    ),
    Check(
        "requisition_chain",
        "each requisition's state intervals follow from one another",
        _REQUISITION_CHAIN,
    ),
    Check("fills_tie_to_moves", "fills tie to the moves that filled them", _FILLS),
    Check("pending_starts", "pending starts wait on their accepted offers", _PENDING),
    Check("vests_sum_to_awards", "vest tranches add up to their awards", _VESTS),
    Check("vest_status", "each tranche's status agrees with its dates", _VEST_STATUS),
    Check("ledger_rows", "each ledger row's parts add up to what it recognized", _LEDGER_ROWS),
    Check(
        "ledger_totals",
        "an award settled inside the window recognizes its vested value",
        _LEDGER_TOTALS,
    ),
)


def validate(con: duckdb.DuckDBPyConnection, examples: int = 5) -> list[Result]:
    results = []
    for check in CHECKS:
        (violations,) = con.execute(f"SELECT count(*) FROM ({check.sql})").fetchone() or (0,)
        sample = (
            con.execute(f"SELECT * FROM ({check.sql}) LIMIT {examples}").df()
            if violations
            else None
        )
        results.append(
            Result(
                check.name,
                check.description,
                int(violations),
                []
                if sample is None
                else [
                    {str(k): v for k, v in row.items()}
                    for row in sample.astype(str).to_dict("records")
                ],
            )
        )
    return results
