"""Tests for realized totals: head flows from the employee-grain fact, requisitions from the
snapshots, and the artifact a `guided_actuals` run reads."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from hcsim.cli import main as hcsim
from hcsim.guidance.ingest import ingest
from hcsim.guidance.realized import QUOTA_CATEGORIES, RealizedError, problems, realized
from hcsim.model.core.categories import CATEGORIES, Grain, GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import generate_world, write_world

C = GuidanceCategory
FEB = pd.Period("2025-02", "M")
EVERY_QUIRK = QuirkTruth(
    backdated_share=0.2,
    correction_share=0.2,
    no_show_pair_share=1.0,
    pending_hire_rows=True,
    aggregate_only_share=0.2,
    snapshot_gap_share=0.2,
    state_seam_month="2023-07",
    cost_center_aliases=True,
)
HEADS = [c for c, spec in CATEGORIES.items() if spec.grain is not Grain.REQUISITION]


def world(path: Path, quirks: QuirkTruth | None = None) -> Path:
    config = preset("tiny", seed=4)
    if quirks is not None:
        config = dataclasses.replace(config, quirks=quirks)
    write_world(generate_world(config), path)
    return path


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return world(tmp_path_factory.mktemp("world") / "plain.duckdb")


@pytest.fixture(scope="module")
def quirky(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return world(tmp_path_factory.mktemp("world") / "quirky.duckdb", EVERY_QUIRK)


# --- a hand-made warehouse, for known answers ----------------------------------------------------

DIMENSIONS = ("dim_trans_type", "dim_account", "dim_org", "dim_job", "dim_location", "dim_company")
ROLE = {"job_code": "ENG-1", "location": "LOC-1", "company": "CO-1"}
REGULAR, CONTINGENT = "ACCT-100", "ACCT-200"
ENDS = ("2025-01-31", "2025-02-28", "2025-03-31")


def txn(employee: int, date: str, code: str, cc: str, account: str = REGULAR) -> dict[str, Any]:
    return {
        "employee_id": employee,
        "event_date": pd.Timestamp(date).date(),
        "trans_type": code,
        "account": account,
        "cost_center": cc,
        **ROLE,
    }


def snap(
    date: str, req: int, cc: str, openings: int, opened: str, state: str = "open"
) -> dict[str, Any]:
    return {
        "snapshot_date": pd.Timestamp(date).date(),
        "req_id": req,
        "cost_center": cc,
        "openings": openings,
        "state": state,
        "opened_date": pd.Timestamp(opened).date(),
        **ROLE,
    }


def offer(date: str, employee: int, req: int, cc: str, offered: str, start: str) -> dict[str, Any]:
    return {
        "snapshot_date": pd.Timestamp(date).date(),
        "employee_id": employee,
        "req_id": req,
        "cost_center": cc,
        "offer_date": pd.Timestamp(offered).date(),
        "start_date": pd.Timestamp(start).date(),
        **ROLE,
    }


TXN = [
    *(txn(e, d, "ACTIVE", "CC-01") for e in (1, 2, 3) for d in ENDS),
    *(txn(4, d, "ACTIVE", "CC-02") for d in ENDS),
    txn(1, "2025-02-15", "TERM", "CC-01"),
    txn(10, "2025-02-10", "HIRE", "CC-01"),  # offered and started within February
    txn(11, "2025-02-12", "HIRE", "CC-01"),  # a no-show pair, which nets to nothing
    txn(11, "2025-02-12", "HIRE_REV", "CC-01"),
    txn(12, "2025-02-20", "HIRE", "CC-02"),  # waited at January's month-end
    txn(13, "2025-02-11", "HIRE", "CC-01", CONTINGENT),  # not a regular hire
    txn(14, "2025-02-18", "XFER_IN", "CC-01"),  # a starter, though they waited for an earlier start
]
SNAPSHOTS = [
    snap(ENDS[0], 100, "CC-01", 2, "2025-01-05"),  # closes in February, filled by two offers
    snap(ENDS[0], 101, "CC-01", 1, "2025-01-06"),  # closes, filled by February's one starter
    snap(ENDS[0], 102, "CC-01", 1, "2025-01-07"),  # closes, with no starter left: unfilled
    *(
        snap(d, 103, "CC-02", 1, "2025-01-08", state)
        for d, state in zip(ENDS, ("open", "OPEN", "OPEN"), strict=True)
    ),
    snap(ENDS[1], 104, "CC-02", 1, "2025-02-02"),  # opens with one opening and one offer
    snap(ENDS[2], 104, "CC-02", 1, "2025-02-02"),
    # Stays open, one opening filled with no offer waiting: it takes one of February's starters.
    *(snap(d, 107, "CC-01", n, "2025-01-09") for d, n in zip(ENDS, (2, 1, 1), strict=True)),
]
OFFERS = [
    offer("2024-05-31", 14, 90, "CC-01", "2024-05-10", "2024-06-03"),
    offer(ENDS[0], 12, 103, "CC-02", "2025-01-20", "2025-02-20"),
    offer(ENDS[1], 20, 100, "CC-01", "2025-02-03", "2025-03-10"),
    offer(ENDS[1], 21, 100, "CC-01", "2025-02-05", "2025-03-15"),
    offer(ENDS[1], 22, 104, "CC-02", "2025-02-20", "2025-03-05"),
    offer(ENDS[1], 23, 105, "CC-02", "2025-02-25", "2025-03-02"),  # never seen open
]


def warehouse(
    source: Path,
    snapshots: Sequence[dict[str, Any]] = tuple(SNAPSHOTS),
    offers: Sequence[dict[str, Any]] = tuple(OFFERS),
) -> duckdb.DuckDBPyConnection:
    """The world's dimensions beside hand-made facts, in memory."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{source}' AS w (READ_ONLY)")
    for table in DIMENSIONS:
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM w.{table}")
    con.execute("DETACH w")
    for name, rows in (
        ("workforce_txn", TXN),
        ("req_snapshot", snapshots),
        ("pending_start", offers),
    ):
        con.register("rows", pd.DataFrame(list(rows)))
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM rows")
        con.unregister("rows")
    return con


@pytest.fixture
def hand(plain: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    with warehouse(plain) as con:
        yield con


CATEGORIES_ASKED = [
    C.ATTRITION,
    C.EXTERNAL_HIRES,
    C.HEADCOUNT,
    C.REQUISITIONS_OPENED,
    C.REQUISITIONS_CLOSED_UNFILLED,
]


def cells(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    nonzero = frame[frame["value"] != 0]
    return {
        (str(g), str(c)): float(v)
        for g, c, v in zip(nonzero["group"], nonzero["category"], nonzero["value"], strict=True)
    }


def test_each_count_is_known_by_hand(hand: duckdb.DuckDBPyConnection) -> None:
    result = realized(hand, FEB, 1, categories=CATEGORIES_ASKED)
    assert cells(result.frame) == {
        ("CC-01", "attrition"): 1.0,
        # One starter; a no-show pair nets to nothing, and a contingent hire isn't regular.
        ("CC-01", "external_hires"): 1.0,
        ("CC-02", "external_hires"): 1.0,
        ("CC-01", "headcount"): 3.0,
        ("CC-02", "headcount"): 1.0,
        # 104 with its opening and the offer accepted before its first sighting, and 105, seen
        # only by its offer.
        ("CC-02", "requisitions_opened"): 3.0,
        # February's two starters fill 107's opening and 101; 100 closed by its offers, and 102
        # unfilled.
        ("CC-01", "requisitions_closed_unfilled"): 1.0,
    }
    assert result.report["closures"] == {
        "requisitions": 3,
        "filled_by_offers": 1,
        "filled_by_starters": 1,
        "unfilled": 1,
    }
    assert result.report["missing_snapshots"] == []


def test_a_missing_month_end_moves_a_closure_to_the_next(plain: Path) -> None:
    gap = [s for s in SNAPSHOTS if str(s["snapshot_date"]) != ENDS[1]]
    with warehouse(plain, gap) as con:
        with pytest.raises(RealizedError, match="missing the month-ends of 2025-02"):
            realized(con, FEB, 2, categories=[C.REQUISITIONS_CLOSED_UNFILLED])
        assert realized(con, FEB, 2, categories=[C.ATTRITION]).report["missing_snapshots"]
        result = realized(con, FEB, 2, categories=[C.REQUISITIONS_CLOSED_UNFILLED], allow_gaps=True)
    assert result.report["missing_snapshots"] == ["2025-02"]
    closed = result.frame.groupby("month")["value"].sum()
    assert closed.to_dict() == {FEB: 0.0, FEB + 1: 1.0}


def test_a_state_that_doesn_t_mean_open_is_refused(plain: Path) -> None:
    closed = [*SNAPSHOTS, snap(ENDS[1], 106, "CC-01", 1, "2025-01-09", "closed")]
    with warehouse(plain, closed) as con, pytest.raises(RealizedError, match=r"\['closed'\]"):
        realized(con, FEB, 1)


def test_a_window_the_data_doesn_t_hold_is_refused(hand: duckdb.DuckDBPyConnection) -> None:
    assert problems(hand, FEB, 1) == []
    late = problems(hand, FEB, 3)
    assert late == ["the window runs to 2025-04, past the last month-end the data holds, 2025-03"]
    early = problems(hand, pd.Period("2025-01", "M"), 1)
    assert "the data's first is in 2025-01" in early[0]
    assert problems(hand, FEB, 0) == ["a window needs at least one month, not 0"]
    assert "team" in problems(hand, FEB, 1, grouping="team")[0]
    hand.execute("DROP TABLE pending_start")
    assert "the source has no pending_start" in problems(hand, FEB, 1)[0]


# --- a world -------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["plain", "quirky"])
def test_head_flows_and_the_stock_match_the_aggregate_fact(
    which: str, request: pytest.FixtureRequest
) -> None:
    # Backdated rows, corrections, no-show pairs, pending-hire rows, aggregate-only rows, and
    # aliased columns all leave the employee grain counting what the aggregate fact does.
    path = request.getfixturevalue(which)
    start = pd.Period("2025-01", "M")
    with duckdb.connect(str(path), read_only=True) as con:
        found = realized(con, start, 6, categories=HEADS).frame
        series = ingest(con, "cost_center", HEADS).frame
    within = series[(series["month"] >= start) & (series["month"] < start + 6)]
    keys = ["month", "group", "category"]
    both = pd.concat(
        [found.set_index(keys)["value"].rename("realized"), within.set_index(keys)["value"]],
        axis=1,
    ).fillna(0.0)
    assert both["realized"].sum() > 0
    assert (both["realized"] == both["value"]).all()


def test_the_snapshots_are_read_through_their_gaps_and_spellings(quirky: Path) -> None:
    with duckdb.connect(str(quirky), read_only=True) as con:
        (seam,) = con.execute(
            "SELECT count(*) FROM req_snapshot WHERE state = 'OPEN'"
        ).fetchone() or (0,)
        result = realized(con, pd.Period("2024-01", "M"), 24, allow_gaps=True)
    assert seam > 0  # the state's spelling changed partway through
    assert result.report["missing_snapshots"]
    totals = result.report["totals"]
    assert totals["requisitions_opened"] > 0
    assert totals["requisitions_closed_unfilled"] > 0


def test_the_artifact_is_guidance_for_a_run_from_the_window_s_start(
    plain: Path, tmp_path: Path
) -> None:
    with duckdb.connect(str(plain), read_only=True) as con:
        result = realized(con, pd.Period("2025-01", "M"), 6, groups=["CC-99"])
    artifact = result.artifact
    assert artifact.snapshot_date == "2024-12-31"
    assert artifact.horizon == 6
    assert artifact.grouping_column == "cost_center"
    assert set(artifact.categories) == set(QUOTA_CATEGORIES)
    assert sorted(artifact.frame["forecast_month"].unique()) == list(range(6))
    # Every group on every month, zeros included, and a group the caller adds.
    assert "CC-99" in artifact.groups
    assert len(artifact.frame) == 6 * len(artifact.groups) * len(QUOTA_CATEGORIES)
    artifact.save(tmp_path / "actuals")
    assert GuidanceForecast.load(tmp_path / "actuals").fingerprint() == artifact.fingerprint()


def test_inner_fill_nodes_are_their_leaves(plain: Path) -> None:
    with duckdb.connect(str(plain), read_only=True) as con:
        result = realized(con, pd.Period("2025-01", "M"), 6, categories=[C.FILLS, C.HIRES])
    totals = result.report["totals"]
    assert totals["fills"] >= totals["hires"] > 0


def test_the_command_writes_the_window_s_artifact(plain: Path, tmp_path: Path) -> None:
    args = ["guidance", "realized", "--world", str(plain), "--start", "2025-01", "--months", "6"]
    assert hcsim([*args, "--out", str(tmp_path / "a")]) == 0
    assert GuidanceForecast.load(tmp_path / "a").horizon == 6
    assert (tmp_path / "a" / "report.json").is_file()
    with pytest.raises(SystemExit):
        hcsim([*args[:5], "January", "--months", "6", "--out", str(tmp_path / "b")])
    with pytest.raises(SystemExit):
        hcsim([*args[:-1], "60", "--out", str(tmp_path / "c")])
