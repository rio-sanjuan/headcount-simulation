"""Tests for the world's monthly workforce process."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import AttritionTruth, preset
from hcsim.synth.world import World, generate_world
from tests.support.stat_check import stat_check

FLOW_SIGN = {"HIRE": 1, "TERM": -1, "XFER_IN": 1, "XFER_OUT": -1}


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
    flows = txn[txn["trans_type"].isin(FLOW_SIGN)]
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
    assert set(processes) == {"attrition", "requisitions", "equity"}
    assert len(processes["attrition"]["cost_center_effect"]) == 10


def test_departures_recover_the_true_hazard(world: World) -> None:
    """Each month's departures, summed, match the hazard the truth manifest records."""
    txn = monthly(world.tables["workforce_txn"])
    jobs = world.tables["dim_job"].set_index("job_code")["job_level"]
    truth = AttritionTruth()
    effect = world.truth["processes"]["attrition"]["cost_center_effect"]
    active = txn[txn["trans_type"] == "ACTIVE"]
    months = sorted(active["month"].unique())
    expected = variance = 0.0
    for previous, month in itertools.pairwise(months):
        at_risk = active[active["month"] == previous]
        month_end = month.to_timestamp(how="end").normalize()
        for row in at_risk.itertuples():
            tenure = (month_end - row.hire_date).days / 30.4
            h = truth.hazard(int(jobs[row.job_code]), tenure, month.month, effect[row.cost_center])
            expected += h
            variance += h * (1 - h)
    observed = int((txn["trans_type"] == "TERM").sum())
    stat_check(observed, expected, float(np.sqrt(variance)), label="departures")
