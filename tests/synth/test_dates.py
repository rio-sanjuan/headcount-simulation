"""Tests for event days within the month."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import DateTruth, WorldConfig
from hcsim.synth.dates import MonthDays
from hcsim.synth.world import World
from tests.support.stat_check import stat_check

MAY_2024 = pd.Period("2024-05", freq="M")  # starts on a Wednesday; Fridays the 3rd and 31st


def weekday(dates: np.ndarray | pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).weekday.to_numpy()


def test_starts_fall_on_the_start_weekday() -> None:
    days = MonthDays(MAY_2024, DateTruth()).sample("start", 400, np.random.default_rng(0))
    assert (weekday(days) == 0).all()
    assert set(pd.to_datetime(days).day) == {6, 13, 20, 27}


def test_exits_fall_on_business_days_and_lean_on_the_last_one() -> None:
    truth = DateTruth(exit_month_end_weight=8.0)
    days = MonthDays(MAY_2024, truth).sample("exit", 6_000, np.random.default_rng(1))
    assert (weekday(days) < 5).all()
    business = 23  # business days in May 2024
    p = 8.0 / (8.0 + business - 1)
    last = int((pd.to_datetime(days).day == 31).sum())
    stat_check(last, 6_000 * p, float(np.sqrt(6_000 * p * (1 - p))), label="month-end exits")


def test_an_earliest_day_is_respected_with_a_fallback() -> None:
    month = MonthDays(MAY_2024, DateTruth())
    rng = np.random.default_rng(2)
    earliest = np.full(300, np.datetime64("2024-05-22"))
    starts = month.sample("start", 300, rng, earliest)
    assert set(pd.to_datetime(starts).day) == {27}  # the one Monday on or after the 22nd
    late = np.full(300, np.datetime64("2024-05-28"))  # no Monday left in May
    fallback = month.sample("start", 300, rng, late)
    assert set(pd.to_datetime(fallback).day) <= {28, 29, 30, 31}
    business = month.sample("business", 300, rng, np.full(300, np.datetime64("2024-04-15")))
    assert (pd.to_datetime(business).month == 5).all()


def test_any_covers_every_calendar_day() -> None:
    days = MonthDays(MAY_2024, DateTruth()).sample("any", 5_000, np.random.default_rng(3))
    assert set(pd.to_datetime(days).day) == set(range(1, 32))


def test_the_world_follows_the_profiles(small_world: World) -> None:
    txn = small_world.tables["workforce_txn"].astype({"trans_type": str})
    hires = txn[txn["trans_type"] == "HIRE"]
    assert (weekday(hires["event_date"]) == 0).mean() > 0.95  # a few fall back after a late opening
    for code in ("TERM", "XFER_IN", "REORG_IN", "CONV_OUT", "CTG_END"):
        rows = txn[txn["trans_type"] == code]
        assert (weekday(rows["event_date"]) < 5).all(), code
    reqs = small_world.tables["req_interval"]
    closed = reqs.dropna(subset=["closed_date"])
    assert (pd.to_datetime(closed["closed_date"]) >= pd.to_datetime(closed["opened_date"])).all()


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (DateTruth(start_weekday=6), "dates.start_weekday"),
        (DateTruth(exit_month_end_weight=0.0), "dates.exit_month_end_weight"),
    ],
)
def test_a_bad_date_truth_is_refused(truth: DateTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(dates=truth)
