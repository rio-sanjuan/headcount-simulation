"""Tests for the world's compensation policy: one check per award type's trigger and sizing."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import AwardTypeTruth, GrantTruth, WorldConfig
from hcsim.synth.world import World
from tests.support.stat_check import stat_check


def awards(world: World, award_type: str) -> pd.DataFrame:
    frame = world.tables["equity_award"]
    return frame[frame["award_type"] == award_type].assign(
        day=lambda f: pd.to_datetime(f["award_date"])
    )


def ratings_on(world: World, frame: pd.DataFrame, when: str) -> pd.Series:
    """Each row's latest rating on its date."""
    ratings = world.tables["rating"].assign(
        cycle_date=lambda f: pd.to_datetime(f["cycle_date"]).astype("datetime64[ns]")
    )
    keyed = frame[["employee_id", when]].reset_index()
    keyed[when] = pd.to_datetime(keyed[when]).astype("datetime64[ns]")
    found = pd.merge_asof(
        keyed.sort_values(when),
        ratings.sort_values("cycle_date"),
        left_on=when,
        right_on="cycle_date",
        by="employee_id",
    )
    return found.set_index("index")["rating"].astype(object).reindex(frame.index)


def test_every_type_is_awarded(small_world: World) -> None:
    assert set(small_world.tables["equity_award"]["award_type"]) == set(GrantTruth.TYPES)


def test_annual_refreshes_fall_on_april_s_first_business_day_and_skip_the_lowest_rating(
    small_world: World,
) -> None:
    annual = awards(small_world, "annual")
    days = sorted(annual["day"].unique())
    assert [d.strftime("%Y-%m-%d") for d in days] == [
        "2021-04-01",
        "2022-04-01",
        "2023-04-03",
        "2024-04-01",
        "2025-04-01",
    ]
    rating = ratings_on(small_world, annual, "day")
    assert "R1" not in set(rating)
    # Better ratings get bigger refreshes, at the same level.
    level3 = annual.assign(rating=rating)[annual["job_level"] == 3]
    median = level3.groupby("rating")["shares"].median()
    assert median["R5"] > median["R3"] > median["R2"]


def test_promotion_awards_follow_promotions_at_the_new_level(small_world: World) -> None:
    txn = small_world.tables["workforce_txn"]
    promos = txn[txn["trans_type"] == "PROMO"].assign(day=lambda f: pd.to_datetime(f["event_date"]))
    awarded = awards(small_world, "promotion")
    assert 0.8 * len(promos) < len(awarded) <= len(promos)  # less those who left first
    jobs = small_world.tables["dim_job"].set_index("job_code")["job_level"]
    merged = pd.merge_asof(
        awarded.sort_values("day"),
        promos[["employee_id", "day", "job_code"]].sort_values("day"),
        on="day",
        by="employee_id",
        direction="backward",
    )
    lag = (merged["day"] - merged["day"]).dt.days  # same column; check the level instead
    assert (lag == 0).all()
    assert (merged["job_level"] == merged["job_code"].astype(str).map(jobs)).all()


def test_a_share_of_transfers_is_awarded(small_world: World) -> None:
    transfers = int((small_world.tables["workforce_txn"]["trans_type"] == "XFER_IN").sum())
    p = small_world.config.grants.transfer_award_probability
    observed = len(awards(small_world, "transfer"))
    # Some transferred people leave before the award, so allow a little below the share.
    stat_check(
        observed, transfers * p * 0.97, float(np.sqrt(transfers * p)) * 1.5, label="transfers"
    )


def test_retention_awards_go_only_to_the_top_two_ratings(small_world: World) -> None:
    """The trigger is the first of the award's month or of the month before (the lag is at
    most 30 days), so the rating on one of those days is in the top two."""
    retention = awards(small_world, "retention")
    assert len(retention) > 0
    month = retention["day"].dt.to_period("M")
    this = ratings_on(small_world, retention.assign(start=month.dt.start_time), "start")
    last = ratings_on(small_world, retention.assign(start=(month - 1).dt.start_time), "start")
    assert (this.isin(["R4", "R5"]) | last.isin(["R4", "R5"])).all()


def test_award_values_follow_the_type_s_value_by_level(small_world: World) -> None:
    grants = small_world.config.grants
    for name in ("new_hire", "promotion", "off_cycle"):
        frame = awards(small_world, name)
        value = frame["shares"] * frame["fair_value"]
        for level, rows in value.groupby(frame["job_level"]):
            if len(rows) < 30:
                continue
            target = grants.award_type(name).value_by_level[int(level)]
            assert np.median(rows) == pytest.approx(target, rel=0.1), (name, level)


@pytest.mark.parametrize(
    ("grants", "message"),
    [
        (GrantTruth(annual_month=13), "grants.annual_month"),
        (GrantTruth(annual_rating_factor=(1.0,)), "annual_rating_factor needs one factor"),
        (GrantTruth(retention_monthly_rate=2.0), "grants.retention_monthly_rate"),
        (
            GrantTruth(promotion=AwardTypeTruth({1: 1.0}, (6,), (1.0,))),
            "grants.promotion.value_by_level has no value",
        ),
        (
            dataclasses.replace(
                GrantTruth(),
                off_cycle=dataclasses.replace(GrantTruth().off_cycle, vest_shares=(0.5,) * 4),
            ),
            "vest shares must sum to 1",
        ),
    ],
)
def test_a_bad_grant_truth_is_refused(grants: GrantTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(grants=grants)
