"""Tests for the world's promotions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import PromotionTruth, WorldConfig
from hcsim.synth.world import World
from tests.support.stat_check import stat_check


def promotions(world: World) -> pd.DataFrame:
    txn = world.tables["workforce_txn"].astype(
        {"trans_type": str, "job_code": str, "job_code_previous": str, "cost_center": str}
    )
    jobs = world.tables["dim_job"].set_index("job_code")
    promos = txn[txn["trans_type"] == "PROMO"]
    return promos.assign(
        level=promos["job_code"].map(jobs["job_level"]),
        level_before=promos["job_code_previous"].map(jobs["job_level"]),
        family=promos["job_code"].map(jobs["job_family"]),
        family_before=promos["job_code_previous"].map(jobs["job_family"]),
        manager_title=promos["job_code"].map(jobs["manager_title"]),
    )


def test_promotions_take_effect_on_the_first_day_of_each_cycle_month(small_world: World) -> None:
    promos = promotions(small_world)
    assert len(promos) > 0
    assert set(pd.to_datetime(promos["event_date"]).dt.strftime("%m-%d")) == {"03-01", "09-01"}


def test_a_promotion_moves_one_level_up_in_place(small_world: World) -> None:
    promos = promotions(small_world)
    levels = list(small_world.config.levels)
    assert (
        promos["level"] == promos["level_before"].map(lambda lv: levels[levels.index(lv) + 1])
    ).all()
    assert (promos["cost_center"] == promos["cost_center_previous"].astype(str)).all()
    assert (promos["amount"] == 0).all()
    assert (promos["level_before"] < max(levels)).all()  # the top level is never promoted
    assert (promos["manager_title"] == promos["is_manager"]).all()
    assert (
        not promos.loc[promos["is_manager"], "level"].lt(small_world.config.manager_min_level).any()
    )


def test_some_promotions_change_family(small_world: World) -> None:
    promos = promotions(small_world)
    changed = int((promos["family"] != promos["family_before"]).sum())
    p = small_world.config.promotions.family_change_probability
    stat_check(changed, len(promos) * p, float(np.sqrt(len(promos) * p * (1 - p))), label="family")


def test_readiness_resets_so_no_one_is_promoted_two_cycles_running(small_world: World) -> None:
    promos = promotions(small_world)
    day = pd.to_datetime(promos["event_date"]).to_numpy().astype("datetime64[D]").astype(np.int64)
    gaps = promos.assign(day=day).sort_values("day").groupby("employee_id")["day"].diff().dropna()
    assert gaps.min() > 190  # days: more than one six-month cycle apart


def test_better_ratings_are_promoted_sooner(small_world: World) -> None:
    ratings = small_world.tables["rating"].assign(rating=lambda f: f["rating"].astype(str))
    cycles = ratings[pd.to_datetime(ratings["cycle_date"]).dt.year.between(2023, 2025)]
    promos = promotions(small_world)
    promoted = set(zip(promos["employee_id"], pd.to_datetime(promos["event_date"]), strict=True))
    keys = zip(cycles["employee_id"], pd.to_datetime(cycles["cycle_date"]), strict=True)
    rate = cycles.assign(promoted=[key in promoted for key in keys]).groupby("rating")["promoted"]
    assert rate.mean()["R5"] > rate.mean()["R3"] > rate.mean()["R1"]


def test_the_promotion_rate_is_plausible(small_world: World) -> None:
    txn = small_world.tables["workforce_txn"]
    person_years = (txn["trans_type"] == "ACTIVE").sum() / 12
    assert 0.05 < len(promotions(small_world)) / person_years < 0.25


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (PromotionTruth(cycle_months=()), "promotions.cycle_months"),
        (PromotionTruth(readiness_drift=(1.0,)), "readiness_drift needs one value per rating"),
        (PromotionTruth(promotion_probability={1: 0.1}), "has no value for levels [2, 3, 4]"),
        (PromotionTruth(family_change_probability=-0.1), "family_change_probability"),
        (PromotionTruth(readiness_threshold=0), "readiness_threshold must be positive"),
    ],
)
def test_a_bad_promotion_truth_is_refused(truth: PromotionTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        WorldConfig(promotions=truth)
