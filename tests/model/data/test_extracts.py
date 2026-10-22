"""Tests for the per-domain training extracts."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator

import duckdb
import pandas as pd
import pytest

from hcsim.model.core.schema import ORG_LEVEL
from hcsim.model.data.extracts import (
    CHANNELS,
    DOMAINS,
    EMPLOYEE_FEATURES,
    OPENING_FEATURES,
    Extract,
    ExtractError,
    Grain,
    extract,
    extracts,
)
from hcsim.model.data.population import EVERY_ACCOUNT, REGULAR
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import generate_world, write_world

START, END = dt.date(2021, 12, 31), dt.date(2025, 12, 31)


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "aliased"])
def con(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[duckdb.DuckDBPyConnection]:
    quirks = QuirkTruth(cost_center_aliases=request.param)
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(dataclasses.replace(preset("tiny", seed=8), quirks=quirks)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def every(con: duckdb.DuckDBPyConnection) -> dict[str, Extract]:
    return extracts(con, REGULAR, START, END)


def _frame(con: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    window = {"start": START, "end": END}
    return con.execute(query, {k: v for k, v in window.items() if f"${k}" in query}).df()


def test_every_domain_has_rows_an_outcome_and_features(every: dict[str, Extract]) -> None:
    assert set(every) == set(DOMAINS)
    for domain, ex in every.items():
        assert ex.domain == domain
        assert ex.rows > 0
        assert ex.population == REGULAR.stamp
        assert (ex.start, ex.end) == (START, END)
        assert set(ex.features) <= set(ex.frame.columns)
        assert ex.label not in ex.features
    for domain in ("attrition", "promotion", "backfill", "cancellation", *CHANNELS):
        labels = every[domain].labels.astype(bool)
        assert 0 < labels.mean() < 1, domain  # both outcomes, to fit a classifier
    assert every["attrition"].features == EMPLOYEE_FEATURES
    assert every["external_fill"].features == OPENING_FEATURES
    assert (every["growth"].frame["opened"] >= 0).all()


def test_person_months_are_labelled_by_the_next_month_s_net_events(
    con: duckdb.DuckDBPyConnection, every: dict[str, Extract]
) -> None:
    attrition, promotion = every["attrition"].frame, every["promotion"].frame
    assert len(attrition) == len(promotion)
    months = pd.to_datetime(attrition["month_end"])
    assert months.min() == pd.Timestamp(START)
    assert months.max() < pd.Timestamp(END)
    nets = _frame(
        con,
        "SELECT employee_id, last_day(event_date) AS m, "
        "sum(CASE trans_type WHEN 'TERM' THEN 1 ELSE -1 END) AS net FROM workforce_txn "
        "WHERE trans_type IN ('TERM', 'TERM_REV') GROUP BY ALL HAVING net > 0",
    )
    leavers = set(zip(nets["employee_id"], pd.to_datetime(nets["m"]), strict=True))
    following = months + pd.offsets.MonthEnd(1)
    expected = [k in leavers for k in zip(attrition["employee_id"], following, strict=True)]
    assert attrition["label"].tolist() == expected
    assert (attrition["next_month"] == following.dt.month).all()
    assert (attrition["tenure_months"] >= 0).all()


def test_a_person_month_carries_the_latest_rating_before_it(
    con: duckdb.DuckDBPyConnection, every: dict[str, Extract]
) -> None:
    rows = every["attrition"].frame
    ratings = _frame(con, "SELECT employee_id, cycle_date, rating_rank FROM rating")
    ratings["cycle_date"] = pd.to_datetime(ratings["cycle_date"])
    sample = rows.sample(200, random_state=0)
    for employee, month, rank in sample[["employee_id", "month_end", "rating_rank"]].itertuples(
        index=False
    ):
        before = ratings[
            (ratings["employee_id"] == employee) & (ratings["cycle_date"] <= pd.Timestamp(month))
        ]
        latest = before.sort_values("cycle_date")["rating_rank"].iloc[-1] if len(before) else 0
        assert rank == latest
    assert rows.loc[rows["rating_rank"] == 0, "rating_scale"].isna().all()


def test_requisition_months_start_at_approval_and_reconcile_with_the_intervals(
    con: duckdb.DuckDBPyConnection, every: dict[str, Extract]
) -> None:
    rows = every["cancellation"].frame
    assert not rows.duplicated(["req_id", "month_end"]).any()
    intervals = _frame(
        con,
        "SELECT req_id, interval_number, opened_date, closed_date, close_reason FROM req_interval "
        "WHERE closed_date > $start AND last_day(closed_date) <= $end",
    )
    fills = intervals[intervals["close_reason"].str.startswith("filled", na=False)]
    assert rows["fills"].sum() == len(fills)
    assert rows["closed_unfilled"].sum() == (intervals["close_reason"] == "unfilled").sum()
    assert (rows["openings"] >= rows["fills"]).all()
    assert (rows["age_months"] >= 0).all()
    # A requisition approved inside the window has a row in its approval month, at age 0.
    opened = _frame(
        con,
        "SELECT req_id, last_day(min(opened_date)) AS m FROM req_interval GROUP BY 1 "
        "HAVING min(opened_date) > $start AND last_day(min(opened_date)) <= $end",
    )
    first = rows.groupby("req_id").agg(m=("month_end", "min"), age=("age_months", "min"))
    joined = opened.set_index("req_id").join(first, rsuffix="_row", how="inner")
    assert len(joined) == len(opened)
    assert (pd.to_datetime(joined["m"]) == pd.to_datetime(joined["m_row"])).all()
    assert (joined["age"] == 0).all()


def test_the_earliest_fill_names_the_month_s_channel(
    con: duckdb.DuckDBPyConnection, every: dict[str, Extract]
) -> None:
    rows = every["cancellation"].frame
    fills = _frame(
        con,
        "SELECT req_id, last_day(closed_date) AS month_end, interval_number, close_reason "
        "FROM req_interval WHERE starts_with(close_reason, 'filled')",
    )
    fills["month_end"] = pd.to_datetime(fills["month_end"])
    earliest = fills.sort_values("interval_number").groupby(["req_id", "month_end"]).first()
    filled = rows[rows["fills"] > 0].assign(month_end=lambda f: pd.to_datetime(f["month_end"]))
    expected = earliest.loc[list(zip(filled["req_id"], filled["month_end"], strict=True))]
    channels = expected["close_reason"].str.removeprefix("filled_").to_numpy()
    assert (filled["fill_channel"].to_numpy() == channels).all()
    assert rows.loc[rows["fills"] == 0, "fill_channel"].isna().all()
    # The fill domains share the frame, each labelling its own channels.
    for domain, channel_set in CHANNELS.items():
        frame = every[domain].frame
        assert frame["label"].tolist() == rows["fill_channel"].isin(channel_set).tolist()
    assert every["cancellation"].frame["label"].equals(rows["closed_unfilled"])


def test_each_backfill_labels_the_departure_nearest_it(
    con: duckdb.DuckDBPyConnection, every: dict[str, Extract]
) -> None:
    rows = every["backfill"].frame
    assert set(rows["departure"]) <= {"TERM", "XFER_OUT"}
    backfills = _frame(
        con,
        "SELECT req_id, any_value(backfill_of) AS employee_id, min(opened_date) AS opened "
        "FROM req_interval WHERE backfill_of IS NOT NULL GROUP BY 1",
    )
    same_day = backfills.merge(
        rows, left_on=["employee_id", "opened"], right_on=["employee_id", "event_date"]
    )
    assert len(same_day) > 0
    assert same_day["label"].all()
    # No backfill is counted twice, so labels never outnumber backfills.
    assert rows["label"].sum() <= len(backfills)
    unbackfilled = rows[~rows["employee_id"].isin(backfills["employee_id"])]
    assert not unbackfilled["label"].any()


def test_growth_counts_every_staffed_cost_center_month(
    con: duckdb.DuckDBPyConnection, every: dict[str, Extract]
) -> None:
    growth = every["growth"]
    rows = growth.frame
    assert growth.grain is Grain.COST_CENTER_MONTH
    assert not rows.duplicated(["month_end", "cost_center"]).any()
    assert (rows["opened"] == 0).any()
    staffed = _frame(con, _staffed_query())
    cells = set(zip(pd.to_datetime(staffed["event_date"]), staffed["cost_center"], strict=True))
    assert set(zip(pd.to_datetime(rows["month_end"]), rows["cost_center"], strict=True)) == cells
    # Every growth requisition opened in a staffed cost center's month is counted there.
    roles = every["job_family"].frame
    months = pd.to_datetime(roles["opened_date"]) + pd.offsets.MonthEnd(0)
    assert rows["opened"].sum() == sum(
        k in cells for k in zip(months, roles["cost_center"], strict=True)
    )


def _staffed_query() -> str:
    return (
        "SELECT DISTINCT t.event_date, t.cost_center FROM workforce_txn t "
        "JOIN dim_account a ON a.account = t.account "
        "WHERE t.trans_type = 'ACTIVE' AND a.account_group = 'regular' "
        "AND t.event_date > $start AND t.event_date <= $end"
    )


def test_job_family_reads_its_org_context_finest_first(every: dict[str, Extract]) -> None:
    roles = every["job_family"]
    assert roles.label == "job_family"
    assert roles.features[0] == "cost_center"
    assert all(ORG_LEVEL.fullmatch(c) for c in roles.features[1:])
    levels = [int(c.removeprefix("org_level")) for c in roles.features[1:]]
    assert levels == sorted(levels, reverse=True)
    assert levels[-1] == 1
    for level in roles.features[1:]:
        assert not roles.frame[level].equals(roles.frame["cost_center"])
    opened = pd.to_datetime(roles.frame["opened_date"])
    assert opened.min() > pd.Timestamp(START)
    assert opened.max() <= pd.Timestamp(END)


def test_ratings_pair_each_with_the_one_before(
    con: duckdb.DuckDBPyConnection, every: dict[str, Extract]
) -> None:
    rows = every["performance"].frame
    history = _frame(con, "SELECT employee_id, cycle_date, rating_rank FROM rating")
    history = history.sort_values(["employee_id", "cycle_date"])
    history["previous"] = history.groupby("employee_id")["rating_rank"].shift().fillna(0)
    joined = rows.merge(history, on=["employee_id", "cycle_date"], suffixes=("", "_all"))
    assert len(joined) == len(rows)
    assert (joined["previous_rank"] == joined["previous"]).all()
    firsts = rows[rows["previous_rank"] == 0]
    assert len(firsts) > 0
    assert firsts["previous_scale"].isna().all()
    assert rows["cost_center"].notna().all()


def test_a_cutoff_splits_a_window_with_no_outcome_on_both_sides(
    every: dict[str, Extract],
) -> None:
    cutoff = dt.date(2024, 6, 30)
    for ex in every.values():
        early, late = ex.before(cutoff), ex.after(cutoff)
        assert early.rows + late.rows == ex.rows
        assert (early.start, early.end, late.start, late.end) == (START, cutoff, cutoff, END)
    # A person-month at the cutoff's month-end has its outcome after it, so it's held out.
    held_out = every["attrition"].after(cutoff).frame
    assert pd.to_datetime(held_out["month_end"]).min() == pd.Timestamp(cutoff)
    with pytest.raises(ExtractError, match="must lie in the window"):
        every["growth"].before(dt.date(2030, 1, 31))


def test_a_population_narrows_every_extract(con: duckdb.DuckDBPyConnection) -> None:
    wide = extracts(con, EVERY_ACCOUNT, START, END, ["attrition", "external_fill"])
    narrow = extracts(con, REGULAR, START, END, ["attrition", "external_fill"])
    assert wide["attrition"].rows > narrow["attrition"].rows
    # Account rules don't constrain requisitions, which book to no account.
    assert wide["external_fill"].rows == narrow["external_fill"].rows
    assert wide["attrition"].population == EVERY_ACCOUNT.stamp


def test_bad_requests_are_refused(con: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(ExtractError, match=r"no extract for domains \['tenure'\]"):
        extracts(con, REGULAR, START, END, ["attrition", "tenure"])
    with pytest.raises(ExtractError, match="start before its end"):
        extract(con, "growth", REGULAR, END, START)
    frame = pd.DataFrame({"label": [True]})
    with pytest.raises(ExtractError, match=r"no columns \['month_end'\]"):
        Extract("x", Grain.PERSON_MONTH, frame, "label", (), "month_end", REGULAR.stamp, START, END)
