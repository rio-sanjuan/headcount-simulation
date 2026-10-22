"""Tests for the population compiler and the extraction queries."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator

import duckdb
import pandas as pd
import pytest

from hcsim.model.core.contract import role_column
from hcsim.model.core.measures import NET_DEPARTURES, NET_EXTERNAL_HIRES
from hcsim.model.data.population import REGULAR, PopulationError, PopulationSpec, population
from hcsim.model.sql.extract import (
    TRANSACTIONS,
    person_months,
    predicate,
    realized_flows,
    requisitions,
    seed_employees,
    seed_openings,
)
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import generate_world, write_world

SNAPSHOT = dt.date(2024, 6, 30)
WINDOW = {"start": dt.date(2023, 12, 31), "end": dt.date(2024, 12, 31)}


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "aliased"])
def con(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[duckdb.DuckDBPyConnection]:
    quirks = QuirkTruth(cost_center_aliases=request.param)
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(dataclasses.replace(preset("tiny", seed=8), quirks=quirks)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_a_population_compiles_to_one_predicate() -> None:
    assert predicate(REGULAR, TRANSACTIONS) == "(a.account_group IN ('regular'))"
    spec = PopulationSpec.of(account_group=["regular", "contingent"], org_level1=["ORG-A"])
    assert predicate(spec, TRANSACTIONS) == (
        "(a.account_group IN ('contingent', 'regular') AND o.org_level1 IN ('ORG-A'))"
    )
    assert predicate(PopulationSpec.of(), TRANSACTIONS) == "TRUE"


def test_a_rule_the_join_can_t_carry_is_refused_or_skipped() -> None:
    join = requisitions()
    with pytest.raises(PopulationError, match=r"rules on \['account_group'\], which requisitions"):
        predicate(REGULAR, join)
    assert predicate(REGULAR, join, unavailable="skip") == "TRUE"
    assert '"department" AS cost_center' in requisitions('"department"').sql


def test_seed_employees_are_the_population_s_stock(con: duckdb.DuckDBPyConnection) -> None:
    seeded = con.execute(seed_employees(REGULAR), {"snapshot": SNAPSHOT}).df()
    (expected,) = con.execute(
        "SELECT count(*) FROM workforce_txn WHERE trans_type = 'ACTIVE' AND event_date = ? "
        "AND account IN (SELECT account FROM dim_account WHERE account_group = 'regular')",
        [SNAPSHOT],
    ).fetchone() or (0,)
    assert len(seeded) == expected > 0
    assert seeded["employee_id"].is_unique
    everyone = con.execute(seed_employees(population("every_account")), {"snapshot": SNAPSHOT}).df()
    assert len(everyone) > len(seeded)
    org = PopulationSpec.of(account_group=["regular"], org_level1=["ORG-A"])
    subset = con.execute(seed_employees(org), {"snapshot": SNAPSHOT}).df()
    orgs = con.execute("SELECT cost_center, org_level1 FROM dim_org").df()
    assert set(subset["cost_center"].map(orgs.set_index("cost_center")["org_level1"])) == {"ORG-A"}


def test_seed_openings_match_the_snapshot_view(con: duckdb.DuckDBPyConnection) -> None:
    sql = seed_openings(REGULAR, role_column(con, "req_interval"))
    seeded = con.execute(sql, {"snapshot": SNAPSHOT}).df()
    view = con.execute(
        "SELECT req_id, cost_center, openings FROM req_snapshot WHERE snapshot_date = ? "
        "ORDER BY req_id",
        [SNAPSHOT],
    ).df()
    assert len(seeded) > 0
    pd.testing.assert_frame_equal(
        seeded[["req_id", "cost_center", "openings"]].reset_index(drop=True), view
    )


def test_person_months_are_labelled_by_next_month_s_event(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        person_months(REGULAR, ["TERM"], attributes=["job_level", "cost_center"]), WINDOW
    ).df()
    assert list(rows.columns) == ["employee_id", "month_end", "job_level", "cost_center", "label"]
    assert rows.groupby("month_end").size().nunique() > 1  # one stock per month-end
    # Each label is a departure the next month, by someone in the stock at this month-end.
    terms = con.execute(
        "SELECT employee_id, last_day(event_date) AS m FROM workforce_txn WHERE trans_type = 'TERM'"
    ).df()
    labelled = rows[rows["label"]]
    keys = set(zip(terms["employee_id"], pd.to_datetime(terms["m"]), strict=True))
    nxt = pd.to_datetime(labelled["month_end"]) + pd.offsets.MonthEnd(1)
    assert all(k in keys for k in zip(labelled["employee_id"], nxt, strict=True))
    assert 0 < rows["label"].mean() < 0.1
    with pytest.raises(PopulationError, match="at least one event code"):
        person_months(REGULAR, [])


def test_realized_flows_count_the_measure(con: duckdb.DuckDBPyConnection) -> None:
    flows = con.execute(realized_flows(NET_EXTERNAL_HIRES, REGULAR, "org_level1"), WINDOW).df()
    (hires,) = con.execute(
        "SELECT count(*) FROM workforce_txn WHERE trans_type = 'HIRE' "
        "AND event_date > $start AND event_date <= $end AND account = 'ACCT-100'",
        WINDOW,
    ).fetchone() or (0,)
    assert flows["value"].sum() == hires > 0
    assert set(flows["group"]) <= {"ORG-A", "ORG-B"}
    leaves = con.execute(realized_flows(NET_DEPARTURES, REGULAR, "cost_center"), WINDOW).df()
    assert leaves["group"].str.startswith("CC-").all()
