"""Tests for the world's compensation and equity."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.awards import add_months
from hcsim.synth.config import PriceTruth, WorldConfig, preset
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


def _exits(world: World) -> pd.DataFrame:
    txn = world.tables["workforce_txn"]
    rows = txn[txn["trans_type"].isin(["TERM", "RIF", "CONV_OUT"])]
    return rows[["employee_id", "event_date"]].rename(columns={"event_date": "exit"})


def _next_exit(world: World, frame: pd.DataFrame, when: str) -> pd.Series:
    """Each row's first exit on or after its date."""
    keyed = (
        frame[["employee_id", when]]
        .reset_index()
        .assign(**{when: lambda f: pd.to_datetime(f[when])})
    )
    exits = _exits(world).assign(exit=lambda f: pd.to_datetime(f["exit"])).sort_values("exit")
    found = pd.merge_asof(
        keyed.sort_values(when),
        exits,
        left_on=when,
        right_on="exit",
        by="employee_id",
        direction="forward",
    )
    return found.set_index("index")["exit"].reindex(frame.index)


def test_new_hires_are_awarded_after_their_lag_if_still_employed(world: World) -> None:
    txn, awards = world.tables["workforce_txn"], world.tables["equity_award"]
    hires = txn[txn["trans_type"] == "HIRE"].set_index("employee_id")["event_date"]
    new = awards[awards["award_type"] == "new_hire"]
    assert len(new) > 0
    assert new["employee_id"].is_unique
    assert awards["award_id"].is_unique
    lag = (pd.to_datetime(new["award_date"]) - new["employee_id"].map(hires)).dt.days
    assert lag.between(30, 62).all()  # 30-60 days, then on to a business day
    assert (pd.to_datetime(new["award_date"]) <= pd.Timestamp(WORLD_END)).all()
    left = _next_exit(world, new.assign(hired=new["employee_id"].map(hires)), "hired")
    assert (left.isna() | (left > pd.to_datetime(new["award_date"]))).all()


def test_tranches_follow_each_type_s_template_and_split_exactly(world: World) -> None:
    awards, vests = world.tables["equity_award"], world.tables["equity_vest"]
    per_award = vests.groupby("award_id")["shares"].agg(["sum", "count"])
    joined = awards.set_index("award_id").join(per_award)
    assert (joined["sum"] == joined["shares"]).all()
    grants = world.config.grants
    expected = joined["award_type"].map(lambda t: len(grants.award_type(t).vest_months))
    assert (joined["count"] == expected).all()
    assert vests.groupby("award_id")["vest_date"].apply(lambda d: d.is_monotonic_increasing).all()


def test_a_tranche_forfeits_exactly_when_its_holder_leaves_before_it_vests(world: World) -> None:
    awards = world.tables["equity_award"]
    vests = world.tables["equity_vest"].merge(
        awards[["award_id", "employee_id", "award_date"]], on="award_id"
    )
    left = _next_exit(world, vests, "award_date")
    expected = left.notna() & (left < pd.to_datetime(vests["vest_date"]))
    assert ((vests["status"] == "forfeited") == expected).all()
    assert (vests["status"] == "forfeited").any()
    vested = pd.to_datetime(vests.loc[~expected, "vest_date"]) <= pd.Timestamp(WORLD_END)
    assert ((vests.loc[~expected, "status"] == "vested") == vested).all()


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


def test_add_months_clamps_to_short_months() -> None:
    dates = np.array(["2024-01-31", "2023-03-15", "2024-12-31"], dtype="datetime64[D]")
    result = add_months(dates, np.array([1, 12, 2]))
    assert result.astype(str).tolist() == ["2024-02-29", "2024-03-15", "2025-02-28"]


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
