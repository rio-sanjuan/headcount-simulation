"""Tests for backfill requisitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import BackfillTruth, WorldConfig
from hcsim.synth.world import World
from tests.support.stat_check import stat_check


def vacancies(world: World) -> pd.DataFrame:
    """Every vacancy, each with the day it arose and the role it left: an immediate departure,
    a role left by a transfer fill, or a notice of a later departure."""
    txn = world.tables["workforce_txn"].astype(
        {"trans_type": str, "cost_center": str, "job_code": str}
    )
    jobs = world.tables["dim_job"].set_index("job_code")["job_level"]
    pending = txn[txn["trans_type"] == "TERM_PENDING"]
    scheduled = set(zip(pending["employee_id"], pending["event_date"], strict=True))
    terms = txn[txn["trans_type"] == "TERM"]
    immediate = terms[
        [
            key not in scheduled
            for key in zip(terms["employee_id"], terms["event_date"], strict=True)
        ]
    ]
    rows = pd.concat([immediate, txn[txn["trans_type"] == "XFER_OUT"], pending])
    return rows.assign(level=rows["job_code"].map(jobs), day=pd.to_datetime(rows["reporting_date"]))


def linked(world: World) -> pd.DataFrame:
    """Each backfill joined to the vacancy it replaces, on the person and the day."""
    reqs = world.tables["req_interval"].astype({"req_type": str, "cost_center": str})
    first = reqs.drop_duplicates("req_id")  # a requisition's first interval, from its opening
    bf = first[first["req_type"] == "backfill"].assign(
        employee_id=lambda f: f["backfill_of"].astype(int),
        day=lambda f: pd.to_datetime(f["opened_date"]),
    )
    vac = vacancies(world)[["employee_id", "day", "trans_type", "cost_center", "event_date"]]
    return bf.merge(
        vac.rename(columns={"cost_center": "vacated"}),
        on=["employee_id", "day"],
        validate="one_to_one",
    )


def test_every_backfill_names_the_vacancy_it_replaces(small_world: World) -> None:
    reqs = small_world.tables["req_interval"]
    assert reqs.loc[reqs["req_type"] == "backfill", "backfill_of"].notna().all()
    assert reqs.loc[reqs["req_type"] != "backfill", "backfill_of"].isna().all()
    assert len(linked(small_world)) == reqs.loc[reqs["req_type"] == "backfill", "req_id"].nunique()


def test_backfills_open_with_the_probability_of_their_vacancy(small_world: World) -> None:
    truth = small_world.config.backfill
    effect = small_world.truth["processes"]["backfill"]["cost_center_effect"]
    vac = vacancies(small_world)
    logit = np.log(truth.base_probability / (1 - truth.base_probability))
    z = logit + vac["cost_center"].map(effect) + vac["level"].map(truth.level_shift)
    p = 1 / (1 + np.exp(-z.to_numpy()))
    reqs = small_world.tables["req_interval"]
    observed = int(reqs.loc[reqs["req_type"] == "backfill", "req_id"].nunique())
    stat_check(observed, float(p.sum()), float(np.sqrt((p * (1 - p)).sum())), label="backfills")


def test_a_share_opens_in_a_sibling_cost_center(small_world: World) -> None:
    bf = linked(small_world)
    org = small_world.tables["dim_org"].set_index("cost_center")["org_level1"]
    moved = bf["cost_center"] != bf["vacated"]
    assert (bf.loc[moved, "cost_center"].map(org) == bf.loc[moved, "vacated"].map(org)).all()
    p = small_world.config.backfill.other_cost_center_share
    # Not every org has a sibling to move to, so allow a little below the share.
    stat_check(
        int(moved.sum()),
        len(bf) * p,
        float(np.sqrt(len(bf) * p * (1 - p))),
        label="elsewhere",
        alpha=1e-3,
    )


def test_notice_comes_one_to_two_months_ahead_and_opens_the_backfill_then(
    small_world: World,
) -> None:
    txn = small_world.tables["workforce_txn"]
    pending = txn[txn["trans_type"] == "TERM_PENDING"]
    reported = pd.to_datetime(pending["reporting_date"]).dt.to_period("M")
    leaves = pd.to_datetime(pending["event_date"]).dt.to_period("M")
    assert set((leaves - reported).map(lambda d: d.n)) == {1, 2}
    # Each notice is followed by the departure on its day, unless the world ends first.
    terms = set(
        zip(
            txn.loc[txn["trans_type"] == "TERM", "employee_id"],
            txn.loc[txn["trans_type"] == "TERM", "event_date"],
            strict=True,
        )
    )
    inside = pd.to_datetime(pending["event_date"]) <= pd.Timestamp("2025-12-31")
    keys = list(zip(pending["employee_id"], pending["event_date"], strict=True))
    followed = np.array([key in terms for key in keys])
    assert followed[inside.to_numpy()].mean() > 0.95  # a few are cut short by a program or a move
    share = len(pending) / (len(pending) + (txn["trans_type"] == "TERM").sum() - followed.sum())
    assert share == pytest.approx(small_world.config.backfill.notice_share, abs=0.03)
    ahead = linked(small_world)
    assert (
        ahead.loc[ahead["trans_type"] == "TERM_PENDING", "day"]
        < pd.to_datetime(ahead.loc[ahead["trans_type"] == "TERM_PENDING", "event_date"])
    ).all()


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (BackfillTruth(base_probability=1.5), "backfill.base_probability"),
        (BackfillTruth(level_shift={1: 0.0}), "backfill.level_shift has no value"),
        (BackfillTruth(notice_months=(0, 1)), "backfill.notice_months"),
    ],
)
def test_a_bad_backfill_truth_is_refused(truth: BackfillTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(backfill=truth)
