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

Requisitions are replayed too. `ReplayedRequisitions` holds, per step, the requisitions that
opened that month, under their realized ids, roles, and openings, and those that exited
unfilled. The replay-requisitions domain opens and closes them first in each month, so a
hire or transfer finds its requisition open. Its openings count by type: growth as `G`,
backfill as `B_att`, and campus as `G_camp`; its exits as `U`.

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

from hcsim.engine.known_events import step_of
from hcsim.engine.results import DomainResult, RowChange
from hcsim.engine.state import SimulationState
from hcsim.financial.prices import StockPriceModel
from hcsim.financial.sizing import AwardBatch
from hcsim.model.core.contract import role_column
from hcsim.model.core.layouts import OPENING, Buffer, OpeningColumn
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
#: The opening flow each requisition type replays as.
TYPE_FLOWS: Final = {"growth": "G", "backfill": "B_att", "campus": "G_camp"}


class ReplayedAwards(Protocol):
    """Awards as recorded, for the grants a replay covers."""

    def batch(
        self, rows: Buffer, step: int, award_type: str, prices: StockPriceModel
    ) -> AwardBatch | None: ...


@dataclass(frozen=True)
class ReplayedRequisitions:
    #: Step to the opening-buffer rows of the requisitions that opened that month.
    opened: Mapping[int, Buffer] = field(default_factory=dict)
    #: Step to the ids of the requisitions that exited unfilled that month.
    closed: Mapping[int, Ints] = field(default_factory=dict)
    #: Requisitions left out because a label didn't encode, by cause.
    pruned: Mapping[str, int] = field(default_factory=dict)


def replayed_requisitions(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run_date: dt.date,
    months: int,
    max_openings: int = 10,
) -> ReplayedRequisitions:
    """The requisitions that opened, and that exited unfilled, in the run's window."""
    cost_center = role_column(con, "req_interval")
    frame = con.execute(
        f"""
        SELECT t.req_id, t.req_type, t.{cost_center} AS cost_center, j.job_family, j.job_level,
               j.tech_indicator, t.location, t.company, t.is_manager, t.openings, t.opened_date
        FROM req_interval t JOIN dim_job j USING (job_code)
        WHERE t.interval_number = 1 AND t.opened_date >= $run
        ORDER BY t.opened_date, t.req_id
        """,
        {"run": run_date},
    ).df()
    exits = con.execute(
        "SELECT req_id, closed_date FROM req_interval "
        "WHERE close_reason = 'unfilled' AND closed_date >= $run ORDER BY closed_date, req_id",
        {"run": run_date},
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
    rows[:, OpeningColumn.NUM_OPENINGS] = np.clip(frame["openings"].to_numpy(), 1, max_openings)
    steps = np.array([step_of(run_date, d) for d in frame["opened_date"]], dtype=np.int64)
    rows[:, OpeningColumn.CREATION_STEP] = steps
    inside = keep & (steps < months)
    opened = {int(s): rows[inside & (steps == s)] for s in np.unique(steps[inside])}
    exit_steps = np.array([step_of(run_date, d) for d in exits["closed_date"]], dtype=np.int64)
    ids = exits["req_id"].to_numpy(dtype=np.int64)
    closed = {int(s): ids[exit_steps == s] for s in np.unique(exit_steps) if 0 <= s < months}
    return ReplayedRequisitions(opened, closed, pruned)


def replay_requisitions(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Open the month's realized requisitions and close its realized unfilled exits."""
    replay = ctx.replay
    if replay is None:
        return DomainResult(REPLAY_REQUISITIONS)
    reqs = state.live_openings
    n = state.n_cost_centers
    opened = replay.opened.get(ctx.step, OPENING.empty(0))
    held = set(reqs[:, OpeningColumn.REQ_ID].astype(np.int64).tolist())
    opened = opened[~np.isin(opened[:, OpeningColumn.REQ_ID].astype(np.int64), list(held))]
    closing = replay.closed.get(ctx.step, np.zeros(0, dtype=np.int64))
    drop = np.flatnonzero(np.isin(reqs[:, OpeningColumn.REQ_ID].astype(np.int64), closing))
    schema = ctx.config.schema
    flows = {symbol: np.zeros(n, dtype=np.int64) for symbol in (*TYPE_FLOWS.values(), "U")}
    for label, symbol in TYPE_FLOWS.items():
        if label in schema.vocabulary(REQ_TYPE):
            mine = opened[:, OpeningColumn.REQ_TYPE] == schema.vocabulary(REQ_TYPE).code(label)
            flows[symbol] = _per_cost_center(opened[mine], n)
    flows["U"] = _per_cost_center(reqs[drop], n)
    return DomainResult(
        REPLAY_REQUISITIONS, openings=RowChange(OPENING, drop=drop, append=opened), flows=flows
    )


def _per_cost_center(rows: Buffer, n: int) -> Ints:
    cc = rows[:, OpeningColumn.COST_CENTER].astype(np.int64)
    weights = rows[:, OpeningColumn.NUM_OPENINGS]
    result: Ints = np.bincount(cc, weights=weights, minlength=n).astype(np.int64)
    return result
