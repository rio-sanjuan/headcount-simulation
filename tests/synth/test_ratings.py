"""Tests for the world's performance ratings."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import RatingTruth, WorldConfig, preset
from hcsim.synth.ratings import last_cycle_before
from hcsim.synth.world import World, generate_world
from tests.support.stat_check import stat_check


@pytest.fixture
def world(small_world: World) -> World:
    return small_world


def ratings(world: World) -> pd.DataFrame:
    return world.tables["rating"].assign(rating=lambda f: f["rating"].astype(str))


def test_ratings_take_effect_on_the_first_day_of_each_cycle_month(world: World) -> None:
    dates = sorted(pd.to_datetime(ratings(world)["cycle_date"]).unique())
    assert [d.strftime("%Y-%m-%d") for d in dates] == [f"{y}-03-01" for y in range(2020, 2026)]
    assert ratings(world)["rating"].isin(RatingTruth().scale).all()
    assert (ratings(world)["rating_rank"] == ratings(world)["rating"].str[1].astype(int)).all()


def test_each_cycle_rates_everyone_active_with_enough_tenure(world: World) -> None:
    txn = world.tables["workforce_txn"]
    cycle = pd.Timestamp("2023-03-01")
    # Active at the start of the cycle month is active at the prior month-end.
    stock = txn[
        (txn["trans_type"] == "ACTIVE")
        & (txn["event_date"] == pd.Timestamp("2023-02-28"))
        & (txn["account"] == "ACCT-100")  # contingent workers aren't rated
    ]
    tenure = (cycle - pd.to_datetime(stock["hire_date"])).dt.days / 30.4
    eligible = set(stock.loc[tenure >= 6.0, "employee_id"])
    rated = ratings(world)
    assert set(rated.loc[pd.to_datetime(rated["cycle_date"]) == cycle, "employee_id"]) == eligible


def test_first_ratings_follow_the_entry_distribution(world: World) -> None:
    first = ratings(world).loc[lambda f: f["is_first"], "rating"]
    entry = RatingTruth().entry_distribution
    for label, p in zip(RatingTruth().scale, entry, strict=True):
        observed = int((first == label).sum())
        se = float(np.sqrt(len(first) * p * (1 - p)))
        stat_check(observed, len(first) * p, se, label=f"first {label}")
    # Only someone who converts out and back in is rated as new twice.
    txn = world.tables["workforce_txn"]
    returned = set(txn.loc[txn["trans_type"] == "CONV_IN", "employee_id"])
    firsts = ratings(world)["is_first"].groupby(ratings(world)["employee_id"]).sum()
    assert firsts.drop(list(returned & set(firsts.index))).max() == 1


def test_later_ratings_follow_the_transition_matrix(world: World) -> None:
    rated = ratings(world).sort_values(["employee_id", "cycle_date"])
    rated["previous"] = rated.groupby("employee_id")["rating"].shift()
    steps = rated.dropna(subset=["previous"])
    steps = steps[~steps["is_first"]]
    truth = RatingTruth()
    for i, before in enumerate(truth.scale):
        row = steps[steps["previous"] == before]
        if len(row) < 200:
            continue
        for j, after in enumerate(truth.scale):
            p = truth.transitions[i][j]
            observed = int((row["rating"] == after).sum())
            stat_check(
                observed,
                len(row) * p,
                float(np.sqrt(len(row) * p * (1 - p))),
                label=f"{before}->{after}",
            )


def test_incumbents_start_from_the_long_run_distribution(world: World) -> None:
    seeded = ratings(world).loc[
        lambda f: pd.to_datetime(f["cycle_date"]) == pd.Timestamp("2020-03-01")
    ]
    assert not seeded["is_first"].any()
    for label, p in zip(RatingTruth().scale, RatingTruth().stationary(), strict=True):
        n = len(seeded)
        stat_check(
            int((seeded["rating"] == label).sum()),
            n * p,
            float(np.sqrt(n * p * (1 - p))),
            label=label,
        )


def test_the_stationary_distribution_is_fixed_by_the_chain() -> None:
    truth = RatingTruth()
    pi = truth.stationary()
    assert pi.sum() == pytest.approx(1)
    assert pi @ np.asarray(truth.transitions) == pytest.approx(pi)


def test_the_last_cycle_before_the_start() -> None:
    truth = RatingTruth(cycle_months=(3, 9))
    assert str(last_cycle_before(truth, pd.Period("2021-01", "M"))) == "2020-09"
    assert str(last_cycle_before(truth, pd.Period("2021-10", "M"))) == "2021-09"
    assert str(last_cycle_before(truth, pd.Period("2021-09", "M"))) == "2021-03"


def test_ratings_draw_from_their_own_stream() -> None:
    """With every rating effect held flat, changing the rating truth leaves the workforce
    alone."""
    tiny = preset("tiny", seed=4)
    flat = dataclasses.replace(tiny.attrition, rating_factor=(1.0,) * 5, unrated_factor=1.0)
    even = dataclasses.replace(tiny.promotions, readiness_drift=(1.0,) * 5, unrated_drift=1.0)
    base = dataclasses.replace(tiny, attrition=flat, promotions=even)
    other = dataclasses.replace(base, ratings=RatingTruth(cycle_months=(6, 12)))
    a, b = generate_world(base), generate_world(other)
    pd.testing.assert_frame_equal(a.tables["workforce_txn"], b.tables["workforce_txn"])
    assert not a.tables["rating"].equals(b.tables["rating"])


def test_low_ratings_leave_more_often(world: World) -> None:
    txn = world.tables["workforce_txn"]
    left = set(txn.loc[txn["trans_type"] == "TERM", "employee_id"])
    latest = ratings(world).drop_duplicates("employee_id", keep="last")
    rate = latest.assign(left=latest["employee_id"].isin(left)).groupby("rating")["left"].mean()
    assert rate["R1"] > rate["R3"] > rate["R5"]


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (RatingTruth(scale=("A",)), "at least two distinct labels"),
        (RatingTruth(cycle_months=(13,)), "calendar months 1-12"),
        (
            RatingTruth(entry_distribution=(1.0, 0, 0, 0)),
            "entry_distribution must be a distribution",
        ),
        (RatingTruth(transitions=((1.0,),)), "square stochastic matrix"),
    ],
)
def test_a_bad_rating_truth_is_refused(truth: RatingTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(ratings=truth)
