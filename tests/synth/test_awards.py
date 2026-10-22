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


# --- schedules, plans, valuations, and holder links ------------------------------------------


def test_every_award_vests_on_its_schedule(small_world: World) -> None:
    catalog = small_world.tables["dim_vest_schedule"]
    assert set(catalog["schedule"]) == {*GrantTruth.TYPES, "executive"}
    assert np.allclose(catalog.groupby("schedule")["share"].sum(), 1)
    award = small_world.tables["equity_award"]
    vests = small_world.tables["equity_vest"].merge(
        award[["award_id", "schedule", "award_date"]], on="award_id"
    )
    months = catalog.set_index(["schedule", "vest_number"])["months_after_award"]
    expected = vests.apply(lambda r: months[(r["schedule"], r["vest_number"])], axis=1)
    span = (
        pd.to_datetime(vests["vest_date"]).dt.to_period("M")
        - pd.to_datetime(vests["award_date"]).dt.to_period("M")
    ).map(lambda d: d.n)
    assert (span == expected).all()


def test_executives_new_hire_and_annual_awards_use_the_executive_schedule(
    small_world: World,
) -> None:
    award = small_world.tables["equity_award"]
    executive = (award["job_level"] >= small_world.config.executive_min_level) & award[
        "award_type"
    ].isin(["new_hire", "annual"])
    assert executive.any()
    assert (award.loc[executive, "schedule"] == "executive").all()
    others = award.loc[~executive, ["schedule", "award_type"]].astype(str)
    assert (others["schedule"] == others["award_type"]).all()


def _quarter_ends(world: World) -> pd.Series:
    monthly = world.tables["stock_price"].assign(month_end=lambda f: pd.to_datetime(f["month_end"]))
    quarters = monthly[monthly["month_end"].dt.month.isin([3, 6, 9, 12])]
    discount = world.config.grants.offline_appraisal_discount
    return pd.Series(quarters["close"].to_numpy() * discount, index=quarters["month_end"])


def test_offline_subsidiaries_grant_at_an_appraised_value(small_world: World) -> None:
    award = small_world.tables["equity_award"]
    offline_companies = set(
        small_world.config.companies[-small_world.config.offline_subsidiaries :]
    )
    assert ((award["plan"] == "offline") == award["company"].isin(offline_companies)).all()
    offline = award[award["plan"] == "offline"]
    appraisal = _quarter_ends(small_world)
    position = (
        np.searchsorted(
            appraisal.index.to_numpy(),
            pd.to_datetime(offline["award_date"]).to_numpy(),
            side="right",
        )
        - 1
    )
    opening = (
        small_world.tables["stock_price"]["close"].iloc[0]
        * small_world.config.grants.offline_appraisal_discount
    )
    expected = np.where(position >= 0, appraisal.to_numpy()[np.maximum(position, 0)], opening)
    assert np.allclose(offline["fair_value"], expected, atol=1e-3)


def test_valuations_record_grants_and_each_offline_quarter_end(small_world: World) -> None:
    award = small_world.tables["equity_award"].set_index("award_id")
    valuation = small_world.tables["equity_valuation"]
    first = valuation.groupby("award_id").first()
    assert len(first) == len(award)
    assert (
        pd.to_datetime(first["valuation_date"])
        == pd.to_datetime(award.loc[first.index, "award_date"])
    ).all()
    primary = award.index[award["plan"] == "primary"]
    assert (valuation[valuation["award_id"].isin(primary)].groupby("award_id").size() == 1).all()
    later = (
        valuation[valuation["award_id"].isin(award.index[award["plan"] == "offline"])]
        .groupby("award_id")
        .nth(slice(1, None))
    )
    assert len(later) > 0
    assert set(pd.to_datetime(later["valuation_date"]).dt.month) <= {3, 6, 9, 12}
    assert (later["basis"] == "appraisal").all()


def test_holder_links_follow_the_holder_from_grant_to_settlement(small_world: World) -> None:
    holders = small_world.tables["equity_holder"].assign(
        start=lambda f: pd.to_datetime(f["valid_from"]), end=lambda f: pd.to_datetime(f["valid_to"])
    )
    award = small_world.tables["equity_award"].set_index("award_id")
    first = holders.groupby("award_id").first()
    assert (first["start"] == pd.to_datetime(award.loc[first.index, "award_date"])).all()
    assert (first["cost_center"] == award.loc[first.index, "cost_center"].astype(str)).all()
    assert (holders["start"] <= holders["end"]).all()
    # Each interval starts where the one before it ended.
    previous_end = holders.groupby("award_id")["end"].shift()
    follows = previous_end.isna() | (holders["start"] == previous_end)
    assert follows.all()
    # A later interval starts on a move, into the cost center that move records.
    txn = small_world.tables["workforce_txn"]
    moves = txn[txn["trans_type"].isin(["XFER_IN", "REORG_IN", "CONV_IN"])]
    keys = set(
        zip(
            moves["employee_id"],
            pd.to_datetime(moves["event_date"]),
            moves["cost_center"].astype(str),
            strict=True,
        )
    )
    later = holders[holders.groupby("award_id").cumcount() > 0]
    assert len(later) > 0
    assert all(
        key in keys
        for key in zip(later["employee_id"], later["start"], later["cost_center"], strict=True)
    )


@pytest.mark.parametrize(
    ("grants", "message"),
    [
        (GrantTruth(executive_vest_shares=(0.5, 0.5, 0.5)), "executive vest shares"),
        (GrantTruth(offline_appraisal_discount=0.0), "offline_appraisal_discount"),
    ],
)
def test_bad_executive_and_offline_settings_are_refused(grants: GrantTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(grants=grants)


@pytest.mark.parametrize(
    ("grants", "message"),
    [
        (GrantTruth(forfeiture_prior=0.7), "forfeiture estimates need"),
        (GrantTruth(forfeiture_credibility_years=0.0), "forfeiture_credibility_years"),
    ],
)
def test_bad_forfeiture_settings_are_refused(grants: GrantTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(grants=grants)
