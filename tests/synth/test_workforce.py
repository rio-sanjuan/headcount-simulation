"""Tests for the world's monthly workforce process."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import AttritionTruth, preset
from hcsim.synth.process import Shocks
from hcsim.synth.world import World, generate_world
from tests.support.stat_check import stat_check

FLOW_SIGN = {
    "HIRE": 1,
    "TERM": -1,
    "XFER_IN": 1,
    "XFER_OUT": -1,
    "REORG_IN": 1,
    "REORG_OUT": -1,
    "CONV_IN": 1,
    "CONV_OUT": -1,
    "CTG_START": 1,
    "CTG_END": -1,
    "PROMO": 0,
    "RECLASS": 0,
    "RIF": -1,
    "TERM_PENDING": 0,
}


@pytest.fixture(scope="module")
def world() -> World:
    return generate_world(preset("tiny", seed=11))


def monthly(txn: pd.DataFrame) -> pd.DataFrame:
    txn = txn.assign(month=pd.to_datetime(txn["event_date"]).dt.to_period("M"))
    return txn


def test_every_month_has_an_active_stock(world: World) -> None:
    txn = monthly(world.tables["workforce_txn"])
    months = txn.loc[txn["trans_type"] == "ACTIVE", "month"].nunique()
    assert months == 60


def test_stock_and_flows_agree_for_every_cost_center_and_month(world: World) -> None:
    txn = monthly(world.tables["workforce_txn"])
    stock = (
        txn[txn["trans_type"] == "ACTIVE"]
        .groupby(["month", "cost_center"], observed=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    flows = txn[txn["trans_type"] != "ACTIVE"]
    assert set(flows["trans_type"]) <= set(FLOW_SIGN)
    net = (
        flows.assign(delta=flows["amount"])
        .groupby(["month", "cost_center"], observed=False)["delta"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=stock.index, columns=stock.columns, fill_value=0)
    )
    change = stock.diff().iloc[1:]
    pd.testing.assert_frame_equal(change.astype(int), net.iloc[1:].astype(int))


def test_amounts_carry_the_transaction_sign(world: World) -> None:
    txn = world.tables["workforce_txn"]
    for code, sign in FLOW_SIGN.items():
        assert (txn.loc[txn["trans_type"] == code, "amount"] == sign).all()


def test_moves_record_both_legs_and_the_previous_cost_center(world: World) -> None:
    txn = world.tables["workforce_txn"]
    outs = txn[txn["trans_type"] == "XFER_OUT"].set_index(["employee_id", "event_date"])
    ins = txn[txn["trans_type"] == "XFER_IN"].set_index(["employee_id", "event_date"])
    assert len(outs) == len(ins) > 0
    paired = outs.join(ins, lsuffix="_out", rsuffix="_in", how="inner")
    assert len(paired) == len(outs)
    assert (paired["cost_center_previous_in"] == paired["cost_center_out"]).all()
    assert (paired["cost_center_in"] != paired["cost_center_out"]).all()


def test_departed_employees_never_reappear(world: World) -> None:
    txn = world.tables["workforce_txn"]
    terms = txn[txn["trans_type"] == "TERM"]
    term_date = pd.to_datetime(terms.set_index("employee_id")["event_date"])
    active = txn[txn["trans_type"] == "ACTIVE"]
    left_on = active["employee_id"].map(term_date)  # NaT for people who never left
    assert not (left_on < pd.to_datetime(active["event_date"])).any()


def test_hires_get_fresh_ids_and_their_hire_date(world: World) -> None:
    txn = world.tables["workforce_txn"]
    hires = txn[txn["trans_type"] == "HIRE"]
    assert len(hires) > 0
    assert hires["employee_id"].is_unique
    assert (hires["employee_id"] > 100_200).all()
    assert (hires["hire_date"] == hires["event_date"]).all()


def test_the_truth_manifest_records_every_process(world: World) -> None:
    processes = world.truth["processes"]
    assert set(processes) == {"attrition", "backfill", "equity", "movements", "programs"}
    assert len(processes["attrition"]["cost_center_effect"]) == 10


def _shock_multiplier(world: World, months: list[pd.Period]) -> pd.DataFrame:
    """The regime and org shock multiplier per month and cost center, from the manifest."""
    truth, recorded = world.config.attrition, world.truth["processes"]["attrition"]
    turbulent = set(recorded["turbulent_months"])
    org = world.tables["dim_org"].set_index("cost_center")["org_level1"]
    active: dict[str, set[str]] = {}
    for shock in recorded["org_shocks"]:
        start = pd.Period(shock["start"], freq="M")
        for k in range(truth.org_shock_months):
            active.setdefault(str(start + k), set()).add(shock["org"])
    rows = [
        {
            "month": month,
            "cost_center": cc,
            "shock": (truth.turbulent_factor if str(month) in turbulent else 1.0)
            * (truth.org_shock_factor if org[cc] in active.get(str(month), set()) else 1.0),
        }
        for month in months
        for cc in org.index
    ]
    return pd.DataFrame(rows)


def test_departures_recover_the_true_hazard(small_world: World) -> None:
    """Departures match the full true hazard: level, tenure, rating, season, the cost center's
    random effect, and the regime and org shocks the manifest records."""
    world = small_world
    txn = monthly(world.tables["workforce_txn"]).astype({"cost_center": str, "job_code": str})
    jobs = world.tables["dim_job"].set_index("job_code")["job_level"]
    truth = world.config.attrition
    effect = world.truth["processes"]["attrition"]["cost_center_effect"]
    scale = list(world.config.ratings.scale)
    ratings = world.tables["rating"].assign(
        code=lambda f: f["rating"].astype(str).map(scale.index),
        cycle_date=lambda f: pd.to_datetime(f["cycle_date"]),
    )
    active = txn[(txn["trans_type"] == "ACTIVE") & (txn["account"].astype(str) == "ACCT-100")]
    months = sorted(active["month"].unique())
    shocks = _shock_multiplier(world, months[1:])
    # Someone who gave notice sits out the draw until they leave on schedule.
    pending = txn[txn["trans_type"] == "TERM_PENDING"].assign(
        reported=lambda f: pd.to_datetime(f["reporting_date"]).dt.to_period("M")
    )
    expected = variance = 0.0
    for previous, month in itertools.pairwise(months):
        at_risk = active[active["month"] == previous]
        waiting = pending[(pending["reported"] < month) & (pending["month"] >= month)]
        at_risk = at_risk[~at_risk["employee_id"].isin(waiting["employee_id"])]
        start, end = month.start_time, month.to_timestamp(how="end").normalize()
        latest = ratings[ratings["cycle_date"] <= start].drop_duplicates("employee_id", keep="last")
        code = at_risk["employee_id"].map(latest.set_index("employee_id")["code"]).fillna(-1)
        shock = at_risk["cost_center"].map(
            shocks[shocks["month"] == month].set_index("cost_center")["shock"]
        )
        h = truth.hazards(
            at_risk["job_code"].map(jobs).to_numpy(),
            ((end - pd.to_datetime(at_risk["hire_date"])).dt.days / 30.4).to_numpy(),
            month.month,
            at_risk["cost_center"].map(effect).to_numpy(),
            code.to_numpy(dtype=np.int64),
            shock.to_numpy(),
        )
        expected += float(h.sum())
        variance += float((h * (1 - h)).sum())
    # A hazard event is a departure at once or a notice; a scheduled departure isn't a new one.
    terms = txn[txn["trans_type"] == "TERM"]
    observed = int((~terms["employee_id"].isin(pending["employee_id"])).sum()) + len(pending)
    stat_check(observed, expected, float(np.sqrt(variance)), label="departures")


def test_the_regime_and_org_shocks_are_recorded(small_world: World) -> None:
    recorded = small_world.truth["processes"]["attrition"]
    months = {str(pd.Period(f"2021-{m:02d}", freq="M") + k) for m in (2,) for k in range(59)}
    assert set(recorded["turbulent_months"]) <= months
    orgs = set(small_world.tables["dim_org"]["org_level1"])
    assert all(
        shock["org"] in orgs and shock["start"] in months for shock in recorded["org_shocks"]
    )


def _shocks(**overrides: Any) -> Shocks:
    config = dataclasses.replace(
        preset("tiny"), attrition=dataclasses.replace(AttritionTruth(), **overrides)
    )
    return Shocks(config, np.array([0, 1, 0, 1]))


def test_the_regime_switches_with_its_transition_chances() -> None:
    always = _shocks(calm_to_turbulent=1.0, turbulent_to_calm=0.0, org_shock_probability=0.0)
    rng = np.random.default_rng(0)
    assert all((always.step(m, rng) == 1.5).all() for m in range(1, 5))
    never = _shocks(calm_to_turbulent=0.0, org_shock_probability=0.0)
    assert (never.step(1, rng) == 1.0).all()
    assert never.regime == [False]


def test_an_org_shock_hits_its_cost_centers_for_its_duration() -> None:
    shocks = _shocks(calm_to_turbulent=0.0, org_shock_probability=1.0, org_shock_months=3)
    rng = np.random.default_rng(1)
    first = shocks.step(1, rng)
    assert first.tolist() == [2.0, 2.0, 2.0, 2.0]  # both orgs start a shock
    assert [shocks.step(m, rng).tolist() for m in (2, 3)] == [[2.0] * 4, [2.0] * 4]
    shocks.step(4, rng)  # the first shocks end and new ones start the same month
    assert shocks.org_shocks == [(1, 0), (1, 1), (4, 0), (4, 1)]


def test_a_turbulent_world_loses_more_people() -> None:
    def departures(**overrides: Any) -> int:
        attrition = dataclasses.replace(AttritionTruth(), org_shock_probability=0.0, **overrides)
        world = generate_world(dataclasses.replace(preset("small", seed=8), attrition=attrition))
        return int((world.tables["workforce_txn"]["trans_type"] == "TERM").sum())

    calm = departures(calm_to_turbulent=0.0)
    turbulent = departures(calm_to_turbulent=1.0, turbulent_to_calm=0.0)
    # Turbulence multiplies every hazard by 1.5; headcount shrinks a little faster, hence 1.35.
    assert turbulent > 1.35 * calm
