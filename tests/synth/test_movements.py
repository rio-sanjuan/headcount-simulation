"""Tests for reclassifications, reorganizations, and the contingent pool."""

from __future__ import annotations

import pandas as pd
import pytest

from hcsim.synth.config import MovementTruth, WorldConfig
from hcsim.synth.world import World

REGULAR, CONTINGENT = "ACCT-100", "ACCT-200"


def txn(world: World) -> pd.DataFrame:
    frame = world.tables["workforce_txn"].astype(
        {
            "trans_type": str,
            "account": str,
            "cost_center": str,
            "cost_center_previous": str,
            "job_code": str,
            "job_code_previous": str,
        }
    )
    return frame.assign(month=pd.to_datetime(frame["event_date"]).dt.to_period("M"))


def legs(frame: pd.DataFrame, first: str, second: str, key: str = "first") -> pd.DataFrame:
    """Pair two legs by employee and date. Every row of the `key` leg must find its partner;
    the other code may also occur alone, as contingent starts and ends do."""
    a = frame[frame["trans_type"] == first].set_index(["employee_id", "event_date"])
    b = frame[frame["trans_type"] == second].set_index(["employee_id", "event_date"])
    paired = a.join(b, lsuffix="_1", rsuffix="_2", how="inner")
    assert len(paired) == len(a if key == "first" else b) > 0
    return paired


def test_regular_stock_moves_only_with_regular_flows(small_world: World) -> None:
    """The headcount identity on regular employees alone: every flow on the regular account,
    including reorganizations and conversions, and nothing from the contingent pool."""
    t = txn(small_world)
    regular = t[t["account"] == REGULAR]
    stock = (
        regular[regular["trans_type"] == "ACTIVE"]
        .groupby(["month", "cost_center"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    flows = regular[regular["trans_type"] != "ACTIVE"]
    net = (
        flows.groupby(["month", "cost_center"])["amount"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=stock.index, columns=stock.columns, fill_value=0)
    )
    pd.testing.assert_frame_equal(stock.diff().iloc[1:].astype(int), net.iloc[1:].astype(int))


def test_the_contingent_pool_books_to_its_own_account(small_world: World) -> None:
    t = txn(small_world)
    pool = t[t["account"] == CONTINGENT]
    assert set(pool["trans_type"]) <= {"ACTIVE", "CTG_START", "CTG_END", "ADMIN"}
    assert set(t.loc[t["trans_type"].isin(["CTG_START", "CTG_END"]), "account"]) == {CONTINGENT}
    sizes = pool[pool["trans_type"] == "ACTIVE"].groupby("month").size()
    regular = t[(t["account"] == REGULAR) & (t["trans_type"] == "ACTIVE")].groupby("month").size()
    share = sizes / regular
    assert share.iloc[0] == pytest.approx(small_world.config.movements.contingent_share, abs=0.005)
    assert share.between(0.03, 0.2).all()


def test_a_conversion_out_leaves_regular_headcount_for_the_pool(small_world: World) -> None:
    paired = legs(txn(small_world), "CONV_OUT", "CTG_START")
    assert (paired["account_1"] == REGULAR).all()
    assert (paired["account_2"] == CONTINGENT).all()
    assert (paired["amount_1"] == -1).all()


def test_a_conversion_in_fills_a_requisition(small_world: World) -> None:
    t = txn(small_world)
    paired = legs(t, "CTG_END", "CONV_IN", key="second")
    assert (paired["account_1"] == CONTINGENT).all()
    assert (paired["account_2"] == REGULAR).all()
    reqs = small_world.tables["req_interval"]
    filled = reqs[reqs["close_reason"] == "filled_conversion"]
    keys = set(
        zip(filled["filled_by"].astype(int), pd.to_datetime(filled["closed_date"]), strict=True)
    )
    arrivals = t[t["trans_type"] == "CONV_IN"]
    assert (
        set(zip(arrivals["employee_id"], pd.to_datetime(arrivals["event_date"]), strict=True))
        == keys
    )
    # The converted worker takes the requisition's role. Someone can convert in twice, so
    # match on the date too.
    roles = pd.DataFrame(
        {
            "employee_id": filled["filled_by"].astype(int),
            "event_date": pd.to_datetime(filled["closed_date"]),
            "role": filled["job_code"].astype(str),
        }
    )
    joined = arrivals.assign(event_date=pd.to_datetime(arrivals["event_date"])).merge(
        roles, on=["employee_id", "event_date"], validate="one_to_one"
    )
    assert len(joined) == len(arrivals)
    assert (joined["job_code"] == joined["role"]).all()


def test_a_reorganization_moves_people_within_one_org(small_world: World) -> None:
    t = txn(small_world)
    paired = legs(t, "REORG_OUT", "REORG_IN")
    assert (t["trans_type"] == "REORG_IN").sum() == len(paired)
    org = small_world.tables["dim_org"].set_index("cost_center")["org_level1"]
    assert (paired["cost_center_1"] != paired["cost_center_2"]).all()
    assert (paired["cost_center_1"].map(org) == paired["cost_center_2"].map(org)).all()
    assert (paired["job_code_1"] == paired["job_code_2"]).all()
    batches = small_world.truth["processes"]["movements"]["reorganizations"]
    assert sum(b["people"] for b in batches) == len(paired)


def test_a_reclassification_changes_only_the_family(small_world: World) -> None:
    t = txn(small_world)
    jobs = small_world.tables["dim_job"].set_index("job_code")
    reclass = t[t["trans_type"] == "RECLASS"]
    batches = small_world.truth["processes"]["movements"]["reclassifications"]
    assert len(reclass) == sum(b["people"] for b in batches) > 0
    assert (reclass["amount"] == 0).all()
    before, after = (
        reclass["job_code_previous"].map(jobs["job_family"]),
        reclass["job_code"].map(jobs["job_family"]),
    )
    assert (before != after).all()
    assert (
        reclass["job_code_previous"].map(jobs["job_level"])
        == reclass["job_code"].map(jobs["job_level"])
    ).all()
    assert (reclass["cost_center"] == reclass["cost_center_previous"]).all()
    assert set(pd.to_datetime(reclass["event_date"]).dt.day) == {1}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"movements": MovementTruth(reorg_probability=2.0)}, "movements.reorg_probability"),
        ({"movements": MovementTruth(reorg_share=(0.8, 0.2))}, "movements.reorg_share"),
        ({"accounts": (("A", "regular"),)}, "need a contingent account"),
    ],
)
def test_bad_movement_settings_are_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(**overrides)  # type: ignore[arg-type]
