"""Tests for the world's compensation and equity."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import PriceTruth, WorldConfig, preset
from hcsim.synth.equity import add_months
from hcsim.synth.world import World, generate_world

WORLD_END = dt.date(2025, 12, 31)


@pytest.fixture(scope="module")
def world() -> World:
    return generate_world(preset("tiny", seed=11))


def test_compensation_ranges_cover_every_family_level_and_location(world: World) -> None:
    ranges = world.tables["comp_range"]
    assert len(ranges) == 4 * 5 * 1
    assert not ranges.duplicated(["job_family", "job_level", "location"]).any()
    assert (ranges["min_base"] < ranges["max_base"]).all()
    assert (ranges["min_target"] < ranges["max_target"]).all()
    assert (ranges["min_target"] > ranges["min_base"]).all()  # equity sits on top of base


def test_prices_are_positive_month_end_closes(world: World) -> None:
    prices = world.tables["stock_price"]
    assert len(prices) == 60
    assert (prices["close"] > 0).all()


def test_daily_prices_close_each_month_and_set_fair_value(world: World) -> None:
    daily = world.tables["stock_price_daily"].assign(day=lambda f: pd.to_datetime(f["price_date"]))
    assert (daily["day"].dt.weekday < 5).all()  # business days only
    assert daily["close"].iloc[0] == 100.0
    last = daily.groupby(daily["day"].dt.to_period("M"))["close"].last().to_numpy()
    assert last.tolist() == world.tables["stock_price"]["close"].tolist()
    window = world.config.prices.averaging_days
    expected = daily["close"].rolling(window, min_periods=1).mean()
    assert np.allclose(daily["fair_value"], expected, atol=1e-3)
    assert (daily["fair_value"] != daily["close"]).mean() > 0.9  # it differs from the close
    # Monthly log returns carry the configured volatility.
    returns = np.diff(np.log(world.tables["stock_price"]["close"].to_numpy()))
    assert returns.std() == pytest.approx(world.config.prices.monthly_volatility, rel=0.35)


def test_awards_take_the_fair_value_of_their_business_day(world: World) -> None:
    awards = world.tables["equity_award"]
    daily = world.tables["stock_price_daily"].set_index("price_date")["fair_value"]
    assert (pd.to_datetime(awards["award_date"]).dt.weekday < 5).all()
    assert (awards["fair_value"].to_numpy() == daily.loc[awards["award_date"]].to_numpy()).all()


def test_awards_go_to_external_hires_still_there_on_the_award_date(world: World) -> None:
    txn, awards = world.tables["workforce_txn"], world.tables["equity_award"]
    hires = txn[txn["trans_type"] == "HIRE"].set_index("employee_id")["event_date"]
    left = txn[txn["trans_type"] == "TERM"].set_index("employee_id")["event_date"]
    assert len(awards) > 0
    assert awards["employee_id"].is_unique
    for award in awards.to_dict("records"):
        employee, award_date = award["employee_id"], award["award_date"]
        assert 30 <= (pd.Timestamp(award_date) - hires[employee]).days <= 60
        assert employee not in left or left[employee] > pd.Timestamp(award_date)
        assert award_date <= WORLD_END


def test_tranches_split_every_award_exactly(world: World) -> None:
    awards, vests = world.tables["equity_award"], world.tables["equity_vest"]
    per_award = vests.groupby("award_id")["shares"].agg(["sum", "count"])
    joined = awards.set_index("award_id").join(per_award)
    assert (joined["sum"] == joined["shares"]).all()
    assert (joined["count"] == 13).all()
    assert vests.groupby("award_id")["vest_date"].apply(lambda d: d.is_monotonic_increasing).all()


def test_a_tranche_forfeits_exactly_when_its_holder_leaves_before_it_vests(world: World) -> None:
    txn, awards = world.tables["workforce_txn"], world.tables["equity_award"]
    vests = world.tables["equity_vest"].merge(awards[["award_id", "employee_id"]], on="award_id")
    left = txn[txn["trans_type"] == "TERM"].set_index("employee_id")["event_date"]
    departed = vests["employee_id"].map(left)
    expected = departed.notna() & (pd.to_datetime(departed) < pd.to_datetime(vests["vest_date"]))
    assert ((vests["status"] == "forfeited") == expected).all()
    assert (vests["status"] == "forfeited").any()
    assert (
        (vests.loc[~expected, "status"] == "vested")
        .eq(pd.to_datetime(vests.loc[~expected, "vest_date"]) <= pd.Timestamp(WORLD_END))
        .all()
    )


def test_the_ledger_recognizes_exactly_the_vested_value(world: World) -> None:
    """For every award whose last tranche vests inside the window, total recognized expense
    equals vested shares times fair value: forfeited tranches net to zero."""
    awards, vests, ledger = (
        world.tables["equity_award"],
        world.tables["equity_vest"],
        world.tables["equity_expense"],
    )
    last_vest = pd.to_datetime(vests.groupby("award_id")["vest_date"].max())
    closed = last_vest[last_vest <= pd.Timestamp(WORLD_END)].index
    assert len(closed) > 0
    vested_value = (
        vests[vests["status"] == "vested"]
        .merge(awards[["award_id", "fair_value"]], on="award_id")
        .assign(value=lambda f: f["shares"] * f["fair_value"])
        .groupby("award_id")["value"]
        .sum()
    )
    recognized = ledger.groupby("award_id")["recognized"].sum()
    for award_id in closed:
        # A fully forfeited award nets to zero up to float residue, hence an absolute tolerance.
        expected = vested_value.get(award_id, 0.0)
        assert recognized.get(award_id, 0.0) == pytest.approx(expected, abs=1e-6)


def test_ledger_signs(world: World) -> None:
    ledger = world.tables["equity_expense"]
    assert (ledger["gross"] >= 0).all()
    assert (ledger["trueup_attrition"] <= 0).all()
    assert (ledger["net"] == ledger["gross"]).all()  # the world books no forfeiture estimate
    assert ledger["trueup_attrition"].lt(0).any()


@pytest.mark.parametrize(
    ("date", "months", "expected"),
    [
        (dt.date(2024, 1, 31), 1, dt.date(2024, 2, 29)),
        (dt.date(2023, 3, 15), 12, dt.date(2024, 3, 15)),
    ],
)
def test_add_months_clamps_to_short_months(date: dt.date, months: int, expected: dt.date) -> None:
    assert add_months(date, months) == expected


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        (PriceTruth(start_price=0.0), "positive start"),
        (PriceTruth(averaging_days=0), "positive trading and averaging days"),
    ],
)
def test_bad_price_settings_are_refused(prices: PriceTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(prices=prices)
