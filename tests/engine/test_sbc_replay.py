"""An SBC replay study: a year of history replayed with SBC, scored against the reference ledger.

The run replays 2025 on a `small` world. Its seed is the workforce's unvested equity, and the
offline stream is history's: the offline plans' tranches outstanding at the start and the awards
they grant during the year, each forfeiting as history records. Every grant a domain makes
routes to the award history records for it, of the primary plan or the offline stream. A
recorded award may fall up to two months after the grant's month, as a new hire's lags their
start, and it's awarded in its own month. The annual award runs in the world's annual month,
and the retention and off-cycle awards history records are the run's anticipated awards.

History's record of awards is complete through the year, so a grant whose window lies within
it and finds no recorded award is granted nothing, as history granted none. Those are chiefly
contingent workers converted in, whom the world grants no new-hire award. Only a grant whose
window reaches past the year's end is sized, and a sized new hire's award falls one or two
months after their start, as the world's do. So a December hire's falls after the world ends,
and a hire who leaves first is never awarded.

Each score is held to a tolerance set from this run, with room to spare: the recognized SBC per
month against `equity_expense`, and each part's total. The run's tranches also hold the identity
within 0.01 under both forfeiture modes, and their expense is the same, bit for bit, however
it's chunked.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.config import AwardSettings, ForecastScope, RunMode, SimulationConfig
from hcsim.engine.known_events import Source, collect, from_replay
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import jobs_from_world, resolve_plan
from hcsim.engine.loop import TrajectoryResult, simulate, values_per_share
from hcsim.engine.replay import recorded_awards, replayed_requisitions
from hcsim.engine.seed_equity import actual_sbc, offline_equity
from hcsim.engine.seeding import (
    range_table_from_world,
    realized_price,
    schema_from_world,
    seed_for,
)
from hcsim.financial.expense import IDENTITY_TOLERANCE, ExpenseConfig, calculate, prepare
from hcsim.financial.output import align
from hcsim.financial.prices import AnnualGrowthPrice
from hcsim.financial.routing import AnticipatedAwards
from hcsim.financial.sizing import NewHireGapFill
from hcsim.financial.timing import AwardTiming, LagDistribution
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.studies.inputs import fit_models
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

pytestmark = pytest.mark.acceptance

RUN = dt.date(2025, 1, 1)
MONTHS = 12
WORLD = preset("small", seed=6)


@pytest.fixture(scope="module")
def replayed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(WORLD), path)
    with duckdb.connect(str(path), read_only=True) as con:
        schema = schema_from_world(con)
        recorded = recorded_awards(con, RUN, MONTHS)
        anticipated = AnticipatedAwards.from_frame(
            pd.DataFrame(
                [
                    (employee, step, award_type)
                    for (award_type, step), by_id in recorded.schedules.items()
                    if award_type in ("retention", "off_cycle")
                    for employee in by_id
                ],
                columns=["employee_id", "step", "award_type"],
            )
        )
        (last_close,) = con.execute(
            "SELECT close FROM stock_price ORDER BY month_end DESC LIMIT 1"
        ).fetchone() or (None,)
        awards = AwardSettings(
            NewHireGapFill(range_table_from_world(con, schema)),
            realized_price(con, RUN, AnnualGrowthPrice(float(last_close))),
            replayed=recorded,
            # In this world a new hire's award falls one or two months after their start's
            # month, about evenly, a promotion's in its month or the next, and a transfer's
            # within two.
            replay_lag={"new_hire": 2, "promotion": 2, "transfer": 2},
            timing={"new_hire": AwardTiming(LagDistribution((0.0, 0.5, 0.5)))},
            annual_month=WORLD.grants.annual_month,
            anticipated=anticipated,
            offline=offline_equity(con, schema, RUN, replayed_months=MONTHS).stream,
        )
        config = SimulationConfig(
            run_date=RUN,
            schema=schema,
            models=fit_models(con, schema, RUN).models,
            simulation_months=MONTHS,
            num_trajectories=1,
            seed=3,
            forecast_scope=ForecastScope.HEADCOUNT_AND_SBC,
            run_mode=RunMode.REPLAY,
            awards=awards,
        )
        state, report = seed_for(con, config)
        events = collect(RUN, MONTHS, [(Source.REPLAY, from_replay(con, RUN))])
        employees = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(int).tolist()
        reqs = state.live_openings[:, OpeningColumn.REQ_ID].astype(int).tolist()
        plan = resolve_plan(sanitize(events, employees, reqs), schema, jobs_from_world(con))
        replay = replayed_requisitions(con, schema, RUN, MONTHS)
        (trajectory,) = simulate(config, state, known=plan, replay=replay).trajectories
        yield {
            "trajectory": trajectory,
            "actual": actual_sbc(con, schema, RUN, MONTHS),
            "report": report,
        }


def test_the_replay_scores_against_the_reference_ledger(replayed: dict[str, Any]) -> None:
    t: TrajectoryResult = replayed["trajectory"]
    assert t.sbc is not None
    joined = align(t.sbc, replayed["actual"])
    monthly = joined.groupby("month")[["forecast", "actual"]].sum()
    assert monthly.index.tolist() == list(range(MONTHS))
    # 2.1% on the run that set it.
    wape = (monthly["forecast"] - monthly["actual"]).abs().sum() / monthly["actual"].sum()
    assert wape < 0.03
    parts = joined.groupby("part")[["forecast", "actual"]].sum()
    gap = parts["forecast"] / parts["actual"] - 1
    # The seed is history's own tranches and forfeits: 0.8% short on the run that set it.
    assert abs(gap["seed"]) < 0.02
    # The run's own grants: 0.4% short on the run that set it. Their replayed SBC alone is 0.6%
    # short, and the few sized grants, whose windows reach past the year's end, make up the rest.
    assert abs(gap["sim"]) < 0.015
    # The offline stream: 5.9% high on the run that set it, since a tranche outstanding at the
    # start is valued at its latest appraisal, above its grant's.
    assert abs(gap["offline"]) < 0.09
    # Nearly all of the run's grants replay history's awards: 80.0M replayed, and 0.14M sized.
    window = t.sbc[t.sbc["month"].between(0, MONTHS - 1) & (t.sbc["part"] == "sim")]
    by_basis = window.groupby("basis")["recognized"].sum()
    assert by_basis["replayed"] > 300 * by_basis["sized"]
    # Every promotion takes a recorded award, the offline stream's among them.
    assert t.unsized_grants["promotion"] == t.ungranted_grants["promotion"] == 0
    # The starts through October with no new-hire award in history are granted none: 140 on
    # the run that set it, chiefly contingent workers converted in.
    assert t.ungranted_grants["pending_starts"] > 100


def test_every_tranche_holds_the_identity_and_chunks_change_nothing(
    replayed: dict[str, Any],
) -> None:
    t: TrajectoryResult = replayed["trajectory"]
    assert t.tranches is not None
    assert t.expense is not None
    assert len(t.tranches) > 100_000
    assert np.abs(t.expense.residual).max() < IDENTITY_TOLERANCE
    prepared = prepare(t.tranches, values_per_share(t.tranches, t.prices), pd.Period(RUN, "M"))
    cohort = calculate(prepared, ExpenseConfig(forfeiture="cohort"), first_month=0)
    assert cohort.rows["trueup_rate"].abs().sum() > 0  # the cohorts' rates moved
    assert np.abs(cohort.residual).max() < IDENTITY_TOLERANCE
    flat = ExpenseConfig()
    whole = calculate(prepared, flat, first_month=0, chunk_size=len(t.tranches))
    for size in (997, 20_000):
        chunked = calculate(prepared, dataclasses.replace(flat, chunk_size=size), first_month=0)
        pd.testing.assert_frame_equal(chunked.rows, whole.rows, check_exact=True)
        assert np.array_equal(chunked.residual, whole.residual)
