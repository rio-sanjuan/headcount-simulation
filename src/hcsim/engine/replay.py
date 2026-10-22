"""Replay: the run-mode rung that pins every movement to realized history.

In replay, nothing is sampled. Every realized movement of the run's window reaches its domain
as a known event (`known_events.from_replay`), so the domains that fire known events replay
history and every sampled draw is zero:

- terminations and reductions leave through attrition; conversions out, reorganizations, and
  reclassifications (as other movements) through their known-event domains; promotions through
  promotion;
- a realized transfer moves its employee through the transfers domain, filling the requisition
  it filled in history;
- a realized hire starts through pending starts, filling its requisition. A hire with no
  requisition the run holds is a **boundary fill** (`F_bdy`), when `replay_boundary_fill` allows
  one, and is left out otherwise.

**Requisitions.** Requisitions are replayed too, so the opening stock follows history as the
headcount does. `ReplayedRequisitions` holds, per step, every requisition history holds during
that month, with the openings it shows at the month's end. The replay-requisitions domain runs
first in each month and brings each requisition to those openings, plus the ones the month's
known starts and transfers will take from it, so they find it open: it appends a requisition
history opens or reopens, writes the openings of one that changes, and drops one history no
longer holds. The openings it adds count by the requisition's type, growth as `G`, backfill as
`B_att`, and campus as `G_camp`, whether they're new, reopened, or come into view; the ones it
removes count as `U`. Every change to the stock is a flow, so `OPENING_IDENTITY` holds exactly.

**Accepted offers.** History closes an opening on the day its offer is accepted, often a month
or more before the person starts. The run closes it when the person starts, as every run does:
a pending start fills its opening then (`F_pend`), and until then the opening stays open,
reserved for them. So at a month's end, the run's open openings are history's plus one per
offer accepted during the run whose person hasn't started. The domain records those per cost
center as the `reserved_openings` diagnostic, and history's stock is `open_openings -
reserved_openings`. Counting them explicitly keeps the flows as they are: closing the opening on
the offer day would need a flow that takes an opening while no one starts and nothing exits,
and the notation has none. The run's first month is the boundary, as it is for the seed, which
reads history at the month-end before it:

- an offer accepted before the run had already closed its opening in the seed, so the run never
  holds it, and its start is a boundary fill;
- a no-show on such an offer gives back an opening the run never held: history's reopening is
  replayed, counted in its requisition's flow. A no-show on an offer accepted during the run
  gives back nothing, since the run never closed its opening: its reservation lapses in the
  month it was due, the month history reopens it;
- a requisition that exits unfilled keeps an opening reserved for an accepted offer until its
  person starts, or is due.

**Clipped requisitions.** A requisition shows at most `max_openings`, as the seed clips it: its
open openings first, then its reserved ones. While it holds more, a shown opening that fills
brings a held-back one into view.

**Replayed awards.** A run that replays awards passes a `ReplayedAwards` hook in its award
settings: for a grant it covers, the hook returns the award as recorded, and the apply step
records it instead of sizing one.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol

import duckdb
import numpy as np
from numpy.typing import NDArray

from hcsim.engine.known_events import EventKind, step_of
from hcsim.engine.results import DomainResult, RowChange
from hcsim.engine.state import SimulationState
from hcsim.financial.prices import StockPriceModel
from hcsim.financial.sizing import AwardBatch
from hcsim.model.core.contract import role_column
from hcsim.model.core.layouts import (
    MAX_OPENINGS_PER_REQUISITION,
    OPENING,
    Buffer,
    EmployeeColumn,
    OpeningColumn,
)
from hcsim.model.core.schema import (
    COMPANY,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    REQ_TYPE,
    TECH_INDICATOR,
    SimulationSchema,
)

if TYPE_CHECKING:
    from hcsim.engine.domains import StepContext

Ints = NDArray[np.int64]
REPLAY_REQUISITIONS: Final = "replay_requisitions"
#: The diagnostic counting, per cost center, the openings a replay holds at each month's end
#: for offers accepted during the run and still to start, which history has already closed.
RESERVED_OPENINGS: Final = "reserved_openings"
#: The opening flow each requisition type replays as.
TYPE_FLOWS: Final = {"growth": "G", "backfill": "B_att", "campus": "G_camp"}

#: Every requisition history holds during each month of the window, with the openings it has
#: open at the month's end (`NULL` when it has none), and the offers accepted during the run
#: that wait on it then.
_MONTHLY: Final = """
WITH steps AS (
    SELECT step, first, (first + INTERVAL 1 MONTH - INTERVAL 1 DAY)::DATE AS month_end
    FROM (
        SELECT range AS step, ($run::DATE + to_months(range::INTEGER))::DATE AS first
        FROM range($months)
    )
),
shown AS (
    SELECT s.step, r.req_id,
           max(r.openings) FILTER (WHERE r.closed_date IS NULL OR r.closed_date > s.month_end)
               AS openings
    FROM steps s JOIN req_interval r
      ON r.interval_start <= s.month_end AND (r.closed_date IS NULL OR r.closed_date >= s.first)
    GROUP BY ALL
),
awaiting AS (
    SELECT s.step, p.req_id, count(*) AS offers
    FROM steps s JOIN pending_start p ON p.snapshot_date = s.month_end
    WHERE p.offer_date >= $run
    GROUP BY ALL
)
SELECT step, req_id, openings, coalesce(offers, 0) AS offers
FROM shown FULL JOIN awaiting USING (step, req_id)
ORDER BY step, req_id
"""


class ReplayedAwards(Protocol):
    """Awards as recorded, for the grants a replay covers."""

    def batch(
        self, rows: Buffer, step: int, award_type: str, prices: StockPriceModel
    ) -> AwardBatch | None: ...


@dataclass(frozen=True)
class ReplayedRequisitions:
    #: One opening-buffer row per requisition the replay tracks, sorted by id, with no openings.
    rows: Buffer = field(default_factory=lambda: OPENING.empty(0))
    #: Step to the requisitions history holds during that month, open or with an offer still
    #: to start at its end: their places in `rows`, and the openings each shows at the end.
    held: Mapping[int, tuple[Ints, Ints]] = field(default_factory=dict)
    #: Step to how many of the openings shown at its end are reserved for accepted offers, per
    #: cost center: the `reserved_openings` diagnostic.
    reserved: Mapping[int, Ints] = field(default_factory=dict)
    #: Requisitions left out because a label didn't encode, by cause.
    pruned: Mapping[str, int] = field(default_factory=dict)

    @property
    def ids(self) -> Ints:
        ids: Ints = self.rows[:, OpeningColumn.REQ_ID].astype(np.int64)
        return ids


def replayed_requisitions(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run_date: dt.date,
    months: int,
    max_openings: int = MAX_OPENINGS_PER_REQUISITION,
) -> ReplayedRequisitions:
    """Every requisition history holds during the run's window, with the openings it shows at
    each month's end."""
    monthly = con.execute(_MONTHLY, {"run": run_date, "months": months}).df()
    cost_center = role_column(con, "req_interval")
    frame = con.execute(
        f"""
        SELECT t.req_id, t.req_type, t.{cost_center} AS cost_center, j.job_family, j.job_level,
               j.tech_indicator, t.location, t.company, t.is_manager, t.opened_date
        FROM req_interval t JOIN dim_job j USING (job_code)
        WHERE t.interval_number = 1 AND t.req_id IN (SELECT unnest($ids))
        ORDER BY t.req_id
        """,
        {"ids": sorted(set(monthly["req_id"].astype(int).tolist()))},
    ).df()
    rows = OPENING.empty(len(frame))
    columns = {
        COST_CENTER: OpeningColumn.COST_CENTER,
        JOB_FAMILY: OpeningColumn.JOB_FAMILY,
        JOB_LEVEL: OpeningColumn.LEVEL,
        TECH_INDICATOR: OpeningColumn.TECH_INDICATOR,
        LOCATION: OpeningColumn.LOCATION,
        COMPANY: OpeningColumn.COMPANY,
        REQ_TYPE: OpeningColumn.REQ_TYPE,
    }
    keep = np.ones(len(frame), dtype=bool)
    pruned: dict[str, int] = {}
    for vocabulary, column in columns.items():
        codes = schema.encode(vocabulary, frame[vocabulary], unknown="unattributed")
        rows[:, column] = codes
        unknown = keep & (codes < 0)
        if unknown.any():
            pruned[f"unknown_{vocabulary}"] = int(unknown.sum())
        keep &= ~unknown
    rows[:, OpeningColumn.REQ_ID] = frame["req_id"].to_numpy(dtype=np.float64)
    rows[:, OpeningColumn.IS_MANAGER] = frame["is_manager"].astype(bool).to_numpy(np.float64)
    rows[:, OpeningColumn.CREATION_STEP] = [step_of(run_date, d) for d in frame["opened_date"]]
    rows = rows[keep]

    ids = rows[:, OpeningColumn.REQ_ID].astype(np.int64)
    monthly = monthly[np.isin(monthly["req_id"].to_numpy(dtype=np.int64), ids)]
    place = np.searchsorted(ids, monthly["req_id"].to_numpy(dtype=np.int64))
    steps = monthly["step"].to_numpy(dtype=np.int64)
    raw = monthly["openings"].to_numpy(dtype=np.float64, na_value=np.nan)
    is_open = ~np.isnan(raw)
    open_now = np.where(is_open, np.maximum(np.nan_to_num(raw), 1), 0).astype(np.int64)
    shown = np.minimum(open_now, max_openings)
    target = np.minimum(open_now + monthly["offers"].to_numpy(dtype=np.int64), max_openings)
    cost_centers = rows[place, OpeningColumn.COST_CENTER].astype(np.int64)
    n = len(schema.vocabulary(COST_CENTER))
    held: dict[int, tuple[Ints, Ints]] = {}
    reserved: dict[int, Ints] = {}
    for step in range(months):
        here = steps == step
        held[step] = (place[here], target[here])
        counts = np.bincount(cost_centers[here], weights=(target - shown)[here], minlength=n)
        reserved[step] = counts.astype(np.int64)
    return ReplayedRequisitions(rows, held, reserved, pruned)


def replay_requisitions(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Bring every requisition the replay tracks to the openings history shows at the month's
    end, plus those the month's known starts and transfers will take from it."""
    replay = ctx.replay
    if replay is None:
        return DomainResult(REPLAY_REQUISITIONS)
    n = state.n_cost_centers
    tracked = replay.ids
    place, target = replay.held.get(ctx.step, (_none(), _none()))
    live = np.zeros(len(tracked), dtype=bool)
    live[place] = True
    want = np.zeros(len(tracked), dtype=np.int64)
    want[place] = target
    want += _taking(state, ctx, tracked)

    reqs = state.live_openings
    rows, at = _find(tracked, reqs[:, OpeningColumn.REQ_ID].astype(np.int64))
    held = np.zeros(len(tracked), dtype=bool)
    held[at] = True
    old = reqs[rows, OpeningColumn.NUM_OPENINGS].astype(np.int64)
    new = want[at]
    drop = rows[new == 0]
    changed = (new != old) & (new > 0)
    written = reqs[rows[changed]].copy()
    written[:, OpeningColumn.NUM_OPENINGS] = new[changed]
    opening = np.flatnonzero(live & ~held & (want > 0))
    appended = replay.rows[opening].copy()
    appended[:, OpeningColumn.NUM_OPENINGS] = want[opening]

    gained = np.concatenate([appended, written[new[changed] > old[changed]]])
    gains = np.concatenate([want[opening], (new - old)[changed & (new > old)]])
    lost = reqs[rows[new < old]]
    losses = (old - new)[new < old]
    schema = ctx.config.schema
    flows = {symbol: np.zeros(n, dtype=np.int64) for symbol in (*TYPE_FLOWS.values(), "U")}
    for label, symbol in TYPE_FLOWS.items():
        if label in schema.vocabulary(REQ_TYPE):
            mine = gained[:, OpeningColumn.REQ_TYPE] == schema.vocabulary(REQ_TYPE).code(label)
            flows[symbol] = _per_cost_center(gained[mine], gains[mine], n)
    flows["U"] = _per_cost_center(lost, losses, n)
    reserved = replay.reserved.get(ctx.step, np.zeros(n, dtype=np.int64))
    return DomainResult(
        REPLAY_REQUISITIONS,
        openings=RowChange(
            OPENING, drop=drop, write_index=rows[changed], write_rows=written, append=appended
        ),
        flows=flows,
        diagnostics={RESERVED_OPENINGS: reserved},
    )


def _none() -> Ints:
    return np.zeros(0, dtype=np.int64)


def _find(tracked: Ints, ids: Ints) -> tuple[Ints, Ints]:
    """Which of `ids` the replay tracks: their positions in `ids`, and their places in
    `tracked`, which is sorted."""
    if not len(tracked):
        return _none(), _none()
    at = np.minimum(np.searchsorted(tracked, ids), len(tracked) - 1)
    found = np.flatnonzero(tracked[at] == ids)
    return found, at[found]


def _taking(state: SimulationState, ctx: StepContext, tracked: Ints) -> Ints:
    """The openings this month's known starts and transfers will take from each tracked
    requisition. A transfer takes one only for someone the run holds, as the transfers domain
    moves no one else."""
    starts = [e.event.req_id for e in ctx.known_at(EventKind.PENDING_START)]
    transfers = ctx.known_at(EventKind.TRANSFER)
    people = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
    movers = [-1 if e.event.employee_id is None else e.event.employee_id for e in transfers]
    present = np.isin(np.array(movers, dtype=np.int64), people)
    moves = [e.event.req_id for e, here in zip(transfers, present, strict=True) if here]
    ids = np.array([r for r in starts + moves if r is not None], dtype=np.int64)
    _, at = _find(tracked, ids)
    taking: Ints = np.bincount(at, minlength=len(tracked)).astype(np.int64)
    return taking


def _per_cost_center(rows: Buffer, openings: Ints, n: int) -> Ints:
    cc = rows[:, OpeningColumn.COST_CENTER].astype(np.int64)
    result: Ints = np.bincount(cc, weights=openings, minlength=n).astype(np.int64)
    return result
