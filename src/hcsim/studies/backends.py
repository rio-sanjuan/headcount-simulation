"""Backends: how a study's arms produce their results.

**The in-simulation backend** (`simulate_arm`) simulates an arm, under any rung of the run-mode
ladder, from the inputs its mode reads (`mode_inputs`), resolved once per study:

- `replay` replays realized history: every realized movement as a known event, the realized
  requisitions, and, where it computes SBC, the recorded awards and the realized price;
- `guided_actuals` is guided by the window's realized totals, on every cost center, with the
  known events history and pending starts give;
- `guided_forecast` and `unguided` read the guidance artifact a study is given, or none.

A mode that reads the window's actuals has inputs only once the window is wholly closed; until
then, its arms are deferred (see `harness`).

**The offline backend** (`offline_metrics`) scores a study from its inputs alone, with the
offline evaluators, and never simulates: an arm's results are its metrics.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import duckdb
import pandas as pd

from hcsim.engine.config import AwardSettings, SimulationConfig
from hcsim.engine.known_events import KnownEvents, Source, collect, from_replay, known_events
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import ResolvedPlan, jobs_from_world, resolve_plan
from hcsim.engine.loop import SimulationResult, simulate
from hcsim.engine.modes import RunMode
from hcsim.engine.replay import ReplayedRequisitions, recorded_awards, replayed_requisitions
from hcsim.engine.seeding import realized_price
from hcsim.engine.state import SimulationState
from hcsim.financial.prices import AnnualGrowthPrice, StockPriceModel
from hcsim.guidance.realized import realized
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import COST_CENTER, SimulationSchema
from hcsim.studies.evaluators import OFFLINE_EVALUATORS, Metric
from hcsim.studies.inputs import FittedModels

_C = GuidanceCategory
#: The categories a guided-actuals run is guided on, at their realized counts.
ACTUALS: Final = (
    _C.ATTRITION,
    _C.PROMOTIONS,
    _C.EXTERNAL_HIRES,
    _C.CONVERSION_HIRES,
    _C.INTERNAL_STARTS,
    _C.REQUISITIONS_OPENED,
    _C.REQUISITIONS_CLOSED_UNFILLED,
)


@dataclass(frozen=True)
class ModeInputs:
    """What a mode reads beyond the study's shared inputs: guidance from the window's actuals,
    known events, realized requisitions, and its awards, where it reads them."""

    guidance: GuidanceForecast | None = None
    known: ResolvedPlan | None = None
    replay: ReplayedRequisitions | None = None
    awards: AwardSettings | None = None


def _plan(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    state: SimulationState,
    events: KnownEvents,
) -> ResolvedPlan:
    employees = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(int).tolist()
    reqs = state.live_openings[:, OpeningColumn.REQ_ID].astype(int).tolist()
    return resolve_plan(sanitize(events, employees, reqs), schema, jobs_from_world(con))


def _realized_price(
    con: duckdb.DuckDBPyConnection, run_date: dt.date, model: StockPriceModel
) -> StockPriceModel:
    """The realized price path from the run's start, continued past the world's last close by
    the study's model, reopened there."""
    (last,) = con.execute(
        "SELECT close FROM stock_price ORDER BY month_end DESC LIMIT 1"
    ).fetchone() or (None,)
    if last is None:
        return model
    if isinstance(model, AnnualGrowthPrice):
        extension = dataclasses.replace(model, opening_price=float(last))
    else:
        extension = AnnualGrowthPrice(float(last))
    return realized_price(con, run_date, extension)


def mode_inputs(
    con: duckdb.DuckDBPyConnection,
    mode: RunMode | str,
    schema: SimulationSchema,
    state: SimulationState,
    run_date: dt.date,
    months: int,
    awards: AwardSettings | None = None,
) -> ModeInputs:
    """The inputs `mode` reads for a run of `months` from `run_date`, from the seeded `state`,
    with the study's `awards` as the mode reads them."""
    mode = RunMode(mode)
    if mode is RunMode.REPLAY:
        events = collect(run_date, months, [(Source.REPLAY, from_replay(con, run_date))])
        if awards is not None:
            awards = dataclasses.replace(
                awards,
                price_model=_realized_price(con, run_date, awards.price_model),
                replayed=recorded_awards(con, run_date, months),
            )
        return ModeInputs(
            known=_plan(con, schema, state, events),
            replay=replayed_requisitions(con, schema, run_date, months),
            awards=awards,
        )
    if mode is RunMode.GUIDED_ACTUALS:
        labels = schema.vocabulary(COST_CENTER).labels
        start = pd.Period(run_date, freq="M")
        totals = realized(con, start, months, COST_CENTER, ACTUALS, groups=labels)
        events = known_events(con, run_date, months)
        return ModeInputs(
            guidance=totals.artifact, known=_plan(con, schema, state, events), awards=awards
        )
    return ModeInputs(awards=awards)


def simulate_arm(
    config: SimulationConfig, state: SimulationState, inputs: ModeInputs
) -> SimulationResult:
    """The in-simulation backend: the arm's run, from its mode's inputs."""
    return simulate(config, state, known=inputs.known, replay=inputs.replay)


def offline_metrics(names: Sequence[str], fitted: FittedModels) -> tuple[Metric, ...]:
    """The offline backend: the named offline evaluators' metrics, from the study's inputs."""
    return tuple(m for name in names for m in OFFLINE_EVALUATORS[name](fitted))
