"""Tests for offers, pending starts, no-shows, and hires with no requisition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import OfferTruth, WorldConfig
from hcsim.synth.offers import next_weekday
from hcsim.synth.world import World
from tests.support.stat_check import stat_check

END = pd.Timestamp("2025-12-31")


def offers(world: World) -> pd.DataFrame:
    """Every accepted offer: an external fill, with when and whether its person started."""
    reqs = world.tables["req_interval"]
    made = reqs[reqs["close_reason"] == "filled_external"]
    txn = world.tables["workforce_txn"]
    starts = txn[txn["trans_type"] == "HIRE"].set_index("employee_id")["event_date"]
    frame = pd.DataFrame(
        {
            "employee_id": made["filled_by"].astype(int),
            "req_id": made["req_id"],
            "offer": pd.to_datetime(made["closed_date"]),
            "job_code": made["job_code"].astype(str),
            "started": pd.to_datetime(made["filled_by"].astype(int).map(starts)),
        }
    )
    return frame.astype({"offer": "datetime64[ns]", "started": "datetime64[ns]"}).reset_index(
        drop=True
    )


def test_people_start_on_a_monday_two_to_nine_weeks_after_their_offer(small_world: World) -> None:
    o = offers(small_world).dropna(subset=["started"])
    lag = (o["started"] - o["offer"]).dt.days
    assert lag.between(14, 66).all()
    assert (o["started"].dt.weekday == 0).all()
    txn = small_world.tables["workforce_txn"]
    hires = txn[txn["trans_type"] == "HIRE"].set_index("employee_id")["job_code"].astype(str)
    assert (hires.loc[o["employee_id"]].to_numpy() == o["job_code"].to_numpy()).all()


def test_a_share_of_offers_never_start(small_world: World) -> None:
    o = offers(small_world)
    # Only offers whose start fell inside the world can be judged.
    judged = o[o["offer"] <= pd.Timestamp("2025-10-22")]  # 70 days before the end
    missed = int(judged["started"].isna().sum())
    p = small_world.config.offers.no_show_probability
    stat_check(missed, len(judged) * p, float(np.sqrt(len(judged) * p * (1 - p))), label="no-shows")


def test_pending_starts_cover_every_offer_waiting_at_a_month_end(small_world: World) -> None:
    pending = small_world.tables["pending_start"].assign(
        snapshot=lambda f: pd.to_datetime(f["snapshot_date"]),
        offer=lambda f: pd.to_datetime(f["offer_date"]),
        start=lambda f: pd.to_datetime(f["start_date"]),
    )
    assert (pending["offer"] <= pending["snapshot"]).all()
    assert (pending["snapshot"] < pending["start"]).all()
    assert set(pending["snapshot"].dt.is_month_end) == {True}
    o = offers(small_world).set_index("employee_id")
    started = pending["employee_id"].map(o["started"])
    assert (started.isna() | (started == pending["start"])).all()
    # Every month-end between an offer and its start has a pending row for it.
    rows = set(zip(pending["employee_id"], pending["snapshot"], strict=True))
    waited = o.dropna(subset=["started"])
    for e in pd.date_range("2021-01-31", END, freq="ME"):
        due = waited[(waited["offer"] <= e) & (waited["started"] > e)]
        assert all((employee, e) in rows for employee in due.index)


def test_some_people_join_with_no_requisition(small_world: World) -> None:
    txn = small_world.tables["workforce_txn"]
    hires = txn[txn["trans_type"] == "HIRE"]
    reqs = small_world.tables["req_interval"]
    filled = set(
        reqs.loc[
            reqs["close_reason"].isin(["filled_external", "filled_campus"]), "filled_by"
        ].astype(int)
    )
    direct = int((~hires["employee_id"].isin(filled)).sum())
    regular = txn[(txn["trans_type"] == "ACTIVE") & (txn["account"] == "ACCT-100")]
    person_months = len(regular) - (regular["event_date"] == regular["event_date"].max()).sum()
    expected = small_world.config.offers.no_requisition_hire_rate * person_months
    stat_check(direct, expected, float(np.sqrt(expected)), label="direct hires")


def test_next_weekday_lands_on_or_after_the_date() -> None:
    dates = np.array(["2024-05-01", "2024-05-06", "2024-05-12"], dtype="datetime64[D]")
    assert next_weekday(dates, 0).astype(str).tolist() == ["2024-05-06", "2024-05-06", "2024-05-13"]


@pytest.mark.parametrize(
    ("truth", "message"),
    [
        (OfferTruth(start_lag_days=(30, 10)), "offers.start_lag_days"),
        (OfferTruth(no_show_probability=1.5), "offers.no_show_probability"),
        (OfferTruth(no_requisition_hire_rate=-0.1), "offers.no_requisition_hire_rate"),
    ],
)
def test_a_bad_offer_truth_is_refused(truth: OfferTruth, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(offers=truth)
