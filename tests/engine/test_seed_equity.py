"""Tests for seed equity: the warehouse's unvested awards, classified and recorded in the
ledger."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.config import AwardSettings, ForecastScope, SimulationConfig
from hcsim.engine.loop import simulate
from hcsim.engine.seed_equity import actual_sbc, offline_equity, record_seed, seed_equity
from hcsim.engine.seeding import (
    opening_price,
    range_table_from_world,
    schema_from_world,
    seed_for,
    seed_state,
)
from hcsim.financial.output import align
from hcsim.financial.prices import AnnualGrowthPrice
from hcsim.financial.seed import HolderStatus, SeedTranches
from hcsim.financial.sizing import NewHireGapFill
from hcsim.financial.streams import ManualAdjustments, OfflineStream
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.data.subset import SubsetSpec
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world
from tests.engine.test_loop import models

RUN = dt.date(2025, 7, 1)


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def configured(con: duckdb.DuckDBPyConnection, months: int = 6) -> SimulationConfig:
    schema = schema_from_world(con)
    awards = AwardSettings(
        NewHireGapFill(range_table_from_world(con, schema)),
        AnnualGrowthPrice(opening_price(con, RUN)),
    )
    return SimulationConfig(
        run_date=RUN,
        schema=schema,
        models=models(attrition=0.02),
        simulation_months=months,
        num_trajectories=2,
        seed=3,
        forecast_scope=ForecastScope.HEADCOUNT_AND_SBC,
        awards=awards,
    )


def test_the_seed_is_every_unvested_primary_plan_tranche_granted_before_the_run(
    con: duckdb.DuckDBPyConnection,
) -> None:
    config = configured(con)
    state, report = seed_state(con, config.schema, RUN, with_ledger=True)
    equity = seed_equity(con, config.schema, RUN, report.snapshot_date, state)
    (expected,) = con.execute(
        "SELECT count(*) FROM equity_award a JOIN equity_vest v USING (award_id) "
        "WHERE a.plan = 'primary' AND a.award_date < ? AND v.vest_date >= ? AND v.shares > 0 "
        "AND (v.forfeit_date IS NULL OR v.forfeit_date >= ?)",
        [RUN, RUN, RUN],
    ).fetchone() or (0,)
    frame = equity.seed.frame
    assert len(frame) == expected > 0
    assert equity.pruned == {}
    assert (frame["award_month"] < 0).all()
    assert (frame["vest_month"] >= 0).all()
    # The whole workforce is seeded, so every holder still in post is active.
    assert equity.seed.counts()["active"] == expected
    # The cost center is the holder's at the origin, from the interval covering the run date.
    award = int(frame["award_id"].iloc[0])
    (origin,) = con.execute(
        "SELECT cost_center FROM equity_holder WHERE award_id = ? AND valid_from <= ? "
        "AND valid_to > ?",
        [award, RUN, RUN],
    ).fetchone() or (None,)
    code = config.schema.vocabulary("cost_center").code(str(origin))
    assert int(frame["cost_center"].iloc[0]) == code


def test_a_seeded_run_records_the_seed_and_holds_the_identity(
    con: duckdb.DuckDBPyConnection,
) -> None:
    config = configured(con)
    state, report = seed_for(con, config)
    assert report.seed_tranches["active"] > 0
    assert state.ledger is not None
    t = state.ledger.tranches()
    assert len(t) == sum(report.seed_tranches.values())
    assert (t.award_month < 0).all()
    assert not np.isnan(t.values).any()
    # Each tranche sits under its holder's vest slot.
    slots = set(state.live_employees[:, EmployeeColumn.VEST_INDEX].astype(int))
    assert set(t.holder.tolist()) <= slots
    result = simulate(config, state)
    for trajectory in result.trajectories:
        assert trajectory.expense is not None
        assert trajectory.expense.max_identity_residual < 0.01
        assert trajectory.sbc is not None
        assert trajectory.sbc["month"].min() >= 0
        opening = trajectory.expense.opening
        assert opening is not None
        assert opening.sum() > 0  # the seed accrued before the run


def test_a_subset_runs_holders_outside_it_wait_in_scratch(con: duckdb.DuckDBPyConnection) -> None:
    config = configured(con)
    subset = SubsetSpec({"cost_center": frozenset({"CC-01", "CC-02"})})
    state, report = seed_for(con, config, subset=subset)
    assert report.seed_tranches["outside"] > 0
    assert state.ledger is not None
    scratch = state.ledger.scratch(0)
    assert int(scratch.sum()) == report.seed_tranches["outside"]
    t = state.ledger.tranches()
    live_slots = state.live_employees[:, EmployeeColumn.VEST_INDEX].astype(int)
    assert not np.isin(t.holder[scratch], live_slots).any()  # their own slots, past the buffer's


def test_a_future_entrant_joins_at_their_start_and_the_departed_stay_out(
    con: duckdb.DuckDBPyConnection,
) -> None:
    config = configured(con)
    state, _ = seed_state(con, config.schema, RUN, with_ledger=True)
    first = int(state.next_vest_slot)
    holder = int(state.live_employees[0, EmployeeColumn.EMPLOYEE_ID])
    frame = pd.DataFrame(
        {
            "award_id": [1, 2, 3],
            "holder": [holder, 77, 78],
            "award_type": [0, 0, 0],
            "award_month": [-2, -2, -2],
            "vest_month": [4, 4, 4],
            "shares": [5, 6, 7],
            "value_per_share": [10.0, 10.0, 10.0],
            "level": [0, 0, 0],
            "cost_center": [0, 0, 0],
            "status": ["active", "future", "departed"],
            "start_month": [-1, 3, -1],
            "forfeit_month": [-1, -1, -1],
        }
    )
    recorded = record_seed(state, SeedTranches(frame))
    assert recorded == {"active": 1, "outside": 0, "future": 1, "departed": 0}
    assert state.ledger is not None
    assert state.next_vest_slot == first + 1
    assert state.ledger.inside(np.array([first, first]), np.array([2, 3])).tolist() == [False, True]
    assert HolderStatus.DEPARTED.value not in set(frame["status"].iloc[:2])


def test_attrition_forfeits_seed_equity(con: duckdb.DuckDBPyConnection) -> None:
    config = dataclasses.replace(
        configured(con, months=12), models=models(attrition=0.2), num_trajectories=1, seed=5
    )
    state, _ = seed_for(con, config)
    t = simulate(config, state).trajectories[0].tranches
    assert t is not None
    assert ((t.forfeit_month >= 0) & (t.award_month < 0)).any()


# --- the offline stream --------------------------------------------------------------------------


def test_a_world_without_offline_subsidiaries_has_an_empty_offline_stream(
    con: duckdb.DuckDBPyConnection,
) -> None:
    equity = offline_equity(con, schema_from_world(con), RUN)
    assert len(equity.stream) == 0
    assert equity.stream.sbc().empty


@pytest.mark.slow
def test_the_offline_stream_is_every_outstanding_offline_tranche_at_its_appraisal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "small.duckdb"
    write_world(generate_world(preset("small", seed=6)), path)
    with duckdb.connect(str(path), read_only=True) as small:
        equity = offline_equity(small, schema_from_world(small), RUN)
        frame = equity.stream.frame
        (expected,) = small.execute(
            "SELECT count(*) FROM equity_award a JOIN equity_vest v USING (award_id) "
            "WHERE a.plan = 'offline' AND a.award_date < ? AND v.vest_date >= ? "
            "AND v.shares > 0 AND (v.forfeit_date IS NULL OR v.forfeit_date >= ?)",
            [RUN, RUN, RUN],
        ).fetchone() or (0,)
        assert len(frame) == expected > 0
        award = int(frame["award_id"].iloc[0])
        (appraisal,) = small.execute(
            "SELECT arg_max(value_per_share, valuation_date) FROM equity_valuation "
            "WHERE award_id = ? AND basis = 'appraisal' AND valuation_date < ?",
            [award, RUN],
        ).fetchone() or (None,)
        assert float(frame["value_per_share"].iloc[0]) == pytest.approx(appraisal)
        # No attrition: tranches whose holder history says leaves during the run stay in.
        (leaving,) = small.execute(
            "SELECT count(*) FROM equity_award a JOIN equity_vest v USING (award_id) "
            "WHERE a.plan = 'offline' AND a.award_date < ? AND v.vest_date >= ? "
            "AND v.forfeit_date >= ? AND v.shares > 0",
            [RUN, RUN, RUN],
        ).fetchone() or (0,)
        assert leaving > 0
        result = equity.stream.expense()
        assert result.max_identity_residual < 0.01
        assert (result.rows["month"] >= 0).all()


# --- SBC output by part, and actuals -------------------------------------------------------------


def test_a_runs_sbc_carries_its_seed_and_sim_parts_and_the_streams(
    con: duckdb.DuckDBPyConnection,
) -> None:
    config = configured(con)
    offline = OfflineStream(
        pd.DataFrame(
            {
                "award_id": [1],
                "holder": [5],
                "award_type": [1],
                "award_month": [-2],
                "vest_month": [3],
                "shares": [12],
                "value_per_share": [5.0],
                "level": [0],
                "cost_center": [0],
            }
        )
    )
    manual = ManualAdjustments.from_labels(
        pd.DataFrame(
            {
                "month": [1],
                "cost_center": ["CC-01"],
                "award_type": ["annual"],
                "level": [3],
                "amount": [250.0],
                "note": ["a known grant"],
            }
        ),
        config.schema,
        config.simulation_months,
    )
    assert config.awards is not None
    config = dataclasses.replace(
        config, awards=dataclasses.replace(config.awards, offline=offline, manual=manual)
    )
    state, _ = seed_for(con, config)
    result = simulate(config, state)
    frames = [t.sbc for t in result.trajectories if t.sbc is not None]
    assert len(frames) == 2
    for frame in frames:
        totals = frame.groupby(["part", "basis"])["recognized"].sum()
        assert {"seed", "sim", "offline", "manual"} <= set(frame["part"])
        assert totals[("manual", "sized")] == pytest.approx(250.0)
        assert ("seed", "sized") not in totals
    streams = [f[f["part"].isin(["offline", "manual"])].reset_index(drop=True) for f in frames]
    pd.testing.assert_frame_equal(streams[0], streams[1])


def test_the_actual_sbc_is_the_reference_ledgers_for_the_window(
    con: duckdb.DuckDBPyConnection,
) -> None:
    config = configured(con)
    actual = actual_sbc(con, config.schema, RUN, 6)
    (total,) = con.execute(
        "SELECT sum(recognized) FROM equity_expense WHERE period >= ? AND period < ?",
        [RUN, dt.date(2026, 1, 1)],
    ).fetchone() or (0.0,)
    assert actual["recognized"].sum() == pytest.approx(total)
    assert set(actual["month"]) == set(range(6))
    assert set(actual["part"]) <= {"seed", "sim"}  # the tiny world has no offline plan
    assert (actual["basis"] == "replayed").all()
    state, _ = seed_for(con, config)
    forecast = simulate(config, state).trajectories[0].sbc
    assert forecast is not None
    joined = align(forecast, actual)
    assert joined["actual"].sum() == pytest.approx(total)
    assert set(joined["part"]) >= {"seed", "sim"}
