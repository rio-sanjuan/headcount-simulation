"""Seed equity: the warehouse's unvested awards, read into the run's holder ledger.

`seed_equity` reads every primary-plan tranche granted before the run date that's still to vest
at its start and wasn't forfeited before it, from `equity_award` and `equity_vest`, with:

- its award and vest month as steps from the run's first month, its shares, and its grant's
  fair value per share;
- its holder at the origin and their cost center then, from the `equity_holder` interval that
  covers the run date, or the cost center at grant where none does;
- the month history records its holder leaving, for a replay to read;
- its holder's status (see `financial.seed`): active when in the seeded workforce; outside when
  in post at the snapshot but left out of the run's subset; a future entrant when an accepted
  offer starts on or after the run date; departed otherwise.

A tranche whose cost center, level, or award type the schema can't encode is left out, and
counted by cause.

`record_seed` puts the seed into the run's ledger: an active holder's tranches under their vest
slot, so a simulated departure forfeits them and sizing counts their vests; an outside holder's
and a future entrant's under new slots, in scratch from the run's first month, a future entrant
joining at their start. A departed holder's tranches stay out.

`offline_equity` reads the offline subsidiaries' plan the same way, for the offline stream (see
`financial.streams`): its tranches outstanding at the run's start, each at its latest appraisal
before the run, its grant's value when it has none, with no attrition, so a tranche whose holder
history says leaves during the run stays in. A replay's stream is history's: it adds the awards
granted during the replayed months, each at its grant's appraisal and recorded for routing, and
each tranche forfeits in the month history records.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from hcsim.engine.state import SimulationState
from hcsim.financial.expense import SBC_KEYS, SBC_VALUES, TRUEUPS
from hcsim.financial.output import PART_KEYS
from hcsim.financial.routing import RecordedAwards
from hcsim.financial.seed import HolderStatus, SeedTranches, classify
from hcsim.financial.streams import OFFLINE_COLUMNS, OfflineStream
from hcsim.model.core.contract import role_column
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.core.measures import REGULAR_ACCOUNTS
from hcsim.model.core.schema import AWARD_TYPE, COST_CENTER, JOB_LEVEL, SimulationSchema

_TRANCHES = """
WITH tranches AS (
    SELECT a.award_id, a.employee_id, a.award_type, a.award_date, a.fair_value, a.job_level,
           a.{award_cc} AS granted_in, v.vest_date, v.shares, v.forfeit_date
    FROM equity_award a JOIN equity_vest v USING (award_id)
    WHERE a.plan = 'primary' AND a.award_date < $run AND v.vest_date >= $run
      AND (v.forfeit_date IS NULL OR v.forfeit_date >= $run) AND v.shares > 0
),
holders AS (
    SELECT award_id, any_value(employee_id) AS holder, any_value({holder_cc}) AS cost_center
    FROM equity_holder
    WHERE valid_from <= $run AND valid_to > $run
    GROUP BY award_id
)
SELECT t.award_id, coalesce(h.holder, t.employee_id) AS holder, t.award_type,
       datediff('month', date_trunc('month', $run::DATE), date_trunc('month', t.award_date))
           AS award_month,
       datediff('month', date_trunc('month', $run::DATE), date_trunc('month', t.vest_date))
           AS vest_month,
       t.shares, t.fair_value AS value_per_share, t.job_level AS level,
       coalesce(h.cost_center, t.granted_in) AS cost_center,
       CASE WHEN t.forfeit_date IS NULL THEN -1
            ELSE datediff('month', date_trunc('month', $run::DATE),
                          date_trunc('month', t.forfeit_date)) END AS forfeit_month
FROM tranches t LEFT JOIN holders h USING (award_id)
ORDER BY t.award_id, t.vest_date
"""

_IN_POST = f"""
SELECT DISTINCT employee_id FROM workforce_txn
WHERE trans_type = 'ACTIVE' AND event_date = $snapshot AND account IN ({REGULAR_ACCOUNTS})
"""

_FUTURE = """
SELECT employee_id, min(start_date) AS start_date FROM pending_start
WHERE snapshot_date = $snapshot AND start_date >= $run
GROUP BY employee_id
"""


@dataclass(frozen=True)
class SeedEquity:
    """The seed's tranches, and those left out because the schema couldn't encode them."""

    seed: SeedTranches
    pruned: Mapping[str, int] = field(default_factory=dict)


def seed_equity(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run_date: dt.date,
    snapshot: dt.date,
    state: SimulationState,
) -> SeedEquity:
    """The unvested primary-plan equity at `run_date`, its holders classified against `state`,
    the run's seeded workforce, as of `snapshot`, the month-end before the run."""
    sql = _TRANCHES.format(
        award_cc=role_column(con, "equity_award"), holder_cc=role_column(con, "equity_holder")
    )
    frame = con.execute(sql, {"run": run_date}).df()
    active = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
    in_post = con.execute(_IN_POST, {"snapshot": snapshot}).df()["employee_id"]
    outside = np.setdiff1d(in_post.to_numpy(dtype=np.int64), active)
    future = con.execute(_FUTURE, {"snapshot": snapshot, "run": run_date}).df()
    holders = frame["holder"].to_numpy(dtype=np.int64)
    frame["status"] = classify(holders, active, outside, future["employee_id"].to_numpy())
    starts = dict(zip(future["employee_id"], pd.to_datetime(future["start_date"]), strict=True))
    run = pd.Period(run_date, "M")
    frame["start_month"] = [
        (pd.Period(starts[h], "M") - run).n if s == HolderStatus.FUTURE.value else -1
        for h, s in zip(holders, frame["status"], strict=True)
    ]
    frame, pruned = _encoded(frame, schema)
    columns = [c for c in frame.columns if c != "employee_id"]
    return SeedEquity(SeedTranches(frame[columns]), pruned)


def record_seed(state: SimulationState, seed: SeedTranches) -> dict[str, int]:
    """Record the seed into `state`'s ledger, returning the tranches recorded per status."""
    if state.ledger is None:
        raise ValueError("recording seed equity needs a ledger in the run's state")
    live = state.live_employees
    slot_of = dict(
        zip(
            live[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64).tolist(),
            live[:, EmployeeColumn.VEST_INDEX].astype(np.int64).tolist(),
            strict=True,
        )
    )
    recorded = dict.fromkeys((s.value for s in HolderStatus), 0)
    waiting = seed.of(HolderStatus.OUTSIDE, HolderStatus.FUTURE)
    for holder in sorted(set(waiting["holder"].astype(int))):
        slot_of[holder] = state.next_vest_slot
        state.next_vest_slot += 1
    for status in (HolderStatus.ACTIVE, HolderStatus.OUTSIDE, HolderStatus.FUTURE):
        rows = seed.of(status)
        if rows.empty:
            continue
        slots = np.array([slot_of[int(h)] for h in rows["holder"]], dtype=np.int64)
        recorded[status.value] = state.ledger.record_tranches(
            slots,
            rows["award_month"].to_numpy(),
            rows["vest_month"].to_numpy(),
            rows["shares"].to_numpy(),
            rows["award_type"].to_numpy(),
            rows["level"].to_numpy(),
            rows["cost_center"].to_numpy(),
            rows["value_per_share"].to_numpy(dtype=np.float64),
        )
        if status is not HolderStatus.ACTIVE:
            distinct = np.unique(slots)
            state.ledger.leave(distinct, 0)
        if status is HolderStatus.FUTURE:
            for slot, start in sorted(set(zip(slots.tolist(), rows["start_month"], strict=True))):
                state.ledger.join(np.array([slot]), int(start))
    return recorded


_OFFLINE = """
WITH tranches AS (
    SELECT a.award_id, a.employee_id, a.award_type, a.award_date, a.fair_value, a.job_level,
           a.{award_cc} AS granted_in, v.vest_date, v.shares, v.forfeit_date
    FROM equity_award a JOIN equity_vest v USING (award_id)
    WHERE a.plan = 'offline' AND a.award_date < $end AND v.vest_date >= $run AND v.shares > 0
      AND (v.forfeit_date IS NULL OR v.forfeit_date >= $run)
),
appraised AS (
    SELECT award_id, arg_max(value_per_share, valuation_date) AS appraisal
    FROM equity_valuation
    WHERE basis = 'appraisal' AND valuation_date < $run
    GROUP BY award_id
),
holders AS (
    SELECT award_id, any_value(employee_id) AS holder, any_value({holder_cc}) AS cost_center
    FROM equity_holder
    WHERE valid_from <= $run AND valid_to > $run
    GROUP BY award_id
)
SELECT t.award_id, coalesce(h.holder, t.employee_id) AS holder, t.award_type,
       datediff('month', date_trunc('month', $run::DATE), date_trunc('month', t.award_date))
           AS award_month,
       datediff('month', date_trunc('month', $run::DATE), date_trunc('month', t.vest_date))
           AS vest_month,
       t.shares, coalesce(p.appraisal, t.fair_value) AS value_per_share, t.job_level AS level,
       coalesce(h.cost_center, t.granted_in) AS cost_center,
       CASE WHEN t.forfeit_date IS NULL THEN -1
            ELSE datediff('month', date_trunc('month', $run::DATE),
                          date_trunc('month', t.forfeit_date)) END AS forfeit_month
FROM tranches t LEFT JOIN appraised p USING (award_id) LEFT JOIN holders h USING (award_id)
ORDER BY t.award_id, t.vest_date
"""


@dataclass(frozen=True)
class OfflineEquity:
    """The offline stream's tranches, and those left out because the schema couldn't encode
    them."""

    stream: OfflineStream
    pruned: Mapping[str, int] = field(default_factory=dict)


def offline_equity(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run_date: dt.date,
    replayed_months: int = 0,
) -> OfflineEquity:
    """The offline subsidiaries' equity outstanding at `run_date`, at its latest appraisals. A
    replay of `replayed_months` months adds the awards granted during them, and its tranches
    forfeit as history records."""
    sql = _OFFLINE.format(
        award_cc=role_column(con, "equity_award"), holder_cc=role_column(con, "equity_holder")
    )
    end = (pd.Period(run_date, "M") + replayed_months).start_time.date()
    frame, pruned = _encoded(con.execute(sql, {"run": run_date, "end": end}).df(), schema)
    if not replayed_months:
        return OfflineEquity(OfflineStream(frame[list(OFFLINE_COLUMNS)]), pruned)
    window = frame[frame["award_month"] >= 0]
    recorded = RecordedAwards.from_frame(
        pd.DataFrame(
            {
                "employee_id": window["holder"],
                "award_type": schema.decode(AWARD_TYPE, window["award_type"]),
                "step": window["award_month"],
                "after": window["vest_month"] - window["award_month"],
                "shares": window["shares"],
            }
        )
    )
    stream = OfflineStream(frame[[*OFFLINE_COLUMNS, "forfeit_month"]], recorded)
    return OfflineEquity(stream, pruned)


def _encoded(frame: pd.DataFrame, schema: SimulationSchema) -> tuple[pd.DataFrame, dict[str, int]]:
    """`frame` with its cost centers, levels, and award types as schema codes, the rows whose
    label the schema can't encode left out and counted by cause."""
    pruned = {}
    for name, vocabulary, column in (
        ("unknown_cost_center", COST_CENTER, "cost_center"),
        ("unknown_level", JOB_LEVEL, "level"),
        ("unknown_award_type", AWARD_TYPE, "award_type"),
    ):
        codes = schema.encode(vocabulary, frame[column].astype(str), unknown="unattributed")
        keep = codes >= 0
        if (~keep).any():
            pruned[name] = int((~keep).sum())
        frame = frame[keep].copy()
        frame[column] = codes[keep]
    return frame.reset_index(drop=True), pruned


_ACTUAL = """
SELECT CASE WHEN a.plan = 'offline' THEN 'offline'
            WHEN a.award_date < $run THEN 'seed' ELSE 'sim' END AS part,
       datediff('month', date_trunc('month', $run::DATE), date_trunc('month', e.period))
           AS month,
       coalesce(h.cost_center, a.{award_cc}) AS cost_center, a.award_type,
       a.job_level AS level, e.gross, e.net, e.trueup_attrition, e.trueup_rate,
       e.trueup_vesting, e.recognized
FROM equity_expense e
JOIN equity_award a USING (award_id)
LEFT JOIN equity_holder h
    ON h.award_id = e.award_id AND h.valid_from <= e.period AND h.valid_to > e.period
WHERE e.period >= $run AND e.period < $end
"""


def actual_sbc(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema, run_date: dt.date, months: int
) -> pd.DataFrame:
    """The SBC history records for the run's `months` months, from the reference ledger
    (`equity_expense`), by part and the SBC keys, each award's expense in its holder's cost
    center that month: `offline` for an offline plan's, `seed` for one granted before the run,
    `sim` for the rest, all replayed. Rows the schema can't encode are left out."""
    end = (pd.Period(run_date, "M") + months).start_time.date()
    sql = _ACTUAL.format(award_cc=role_column(con, "equity_award"))
    frame = con.execute(sql, {"run": run_date, "end": end}).df()
    frame, _ = _encoded(frame, schema)
    keys = ["part", *SBC_KEYS]
    out = frame.groupby(keys, as_index=False)[["gross", "net", *TRUEUPS, "recognized"]].sum()
    out["trueup"] = out[list(TRUEUPS)].sum(axis=1)
    out["basis"] = "replayed"
    return out[[*PART_KEYS, *SBC_VALUES]].sort_values(PART_KEYS, ignore_index=True)
