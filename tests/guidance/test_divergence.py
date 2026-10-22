"""Tests for declared divergences between the realized totals and the guidance measures."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.cli import main as hcsim
from hcsim.guidance import cli
from hcsim.guidance.divergence import (
    DIVERGENCES,
    Check,
    Comparison,
    Tolerance,
    against_measures,
    compare,
)
from hcsim.guidance.realized import QUOTA_CATEGORIES, problems, realized
from hcsim.model.core.categories import CATEGORIES, Grain, GuidanceCategory
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import generate_world, write_world

C = GuidanceCategory
EVERY = [*QUOTA_CATEGORIES, C.HEADCOUNT, C.HIRES, C.FILLS]
WINDOWS = [("2023-01", 12), ("2024-07", 6), ("2025-04", 6)]


def frame(totals: dict[str, float]) -> pd.DataFrame:
    rows = [("2025-01", "G", c, v) for c, v in totals.items()]
    return pd.DataFrame(rows, columns=["month", "group", "category", "value"])


def test_a_tolerance_bounds_its_reference_by_shares_and_slack() -> None:
    low, high = Tolerance(below=0.25, above=0.4, slack=6.0).bounds(100.0)
    assert (low, high) == pytest.approx((69.0, 146.0))
    with pytest.raises(ValueError, match="can't be negative"):
        Tolerance(below=-0.1)


def test_an_undeclared_category_must_match_exactly() -> None:
    same = compare(frame({"attrition": 10.0}), frame({"attrition": 10.0}))
    assert same.within
    off = compare(frame({"attrition": 11.0}), frame({"attrition": 10.0}))
    assert not off.within
    assert off.problems() == ["attrition: 11 against 10, which it must match exactly"]


@pytest.mark.parametrize(
    ("produced", "within"),
    [(80.0, True), (65.0, True), (64.0, False), (110.0, True), (111.0, False)],
)
def test_a_declared_divergence_holds_within_its_tolerance(produced: float, within: bool) -> None:
    # Opened requisitions may fall 25% short, and 10 either way: 65 to 110 against 100.
    result = compare(
        frame({"requisitions_opened": produced}), frame({"requisitions_opened": 100.0})
    )
    (check,) = result.checks
    assert check.within is within
    assert check.declared is DIVERGENCES[C.REQUISITIONS_OPENED]
    assert result.report()["categories"]["requisitions_opened"]["within"] is within


def test_only_the_categories_both_producers_count_are_compared() -> None:
    result = compare(frame({"attrition": 1.0, "promotions": 2.0}), frame({"attrition": 1.0}))
    assert [c.category for c in result.checks] == [C.ATTRITION]


def test_divergences_are_declared_for_requisitions_alone() -> None:
    assert set(DIVERGENCES) == {C.REQUISITIONS_OPENED, C.REQUISITIONS_CLOSED_UNFILLED}
    for category, declared in DIVERGENCES.items():
        assert CATEGORIES[category].grain is Grain.REQUISITION
        assert declared.category is category
        assert declared.reason


# --- on worlds -----------------------------------------------------------------------------------


def connect(path: Path, name: str, seed: int, quirks: QuirkTruth | None = None) -> Path:
    config = preset(name, seed=seed)
    if quirks is not None:
        config = dataclasses.replace(config, quirks=quirks)
    write_world(generate_world(config), path)
    return path


@pytest.fixture(scope="module", params=range(5), ids=lambda s: f"tiny-{s}")
def tiny(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return connect(tmp_path_factory.mktemp("world") / "w.duckdb", "tiny", request.param)


def within_every_window(path: Path) -> None:
    with duckdb.connect(str(path), read_only=True) as con:
        for start, months in WINDOWS:
            result = realized(con, pd.Period(start, "M"), months, categories=EVERY)
            comparison = against_measures(con, result)
            assert comparison.within, (start, comparison.problems())
            assert {c.category for c in comparison.checks} == set(EVERY)


def test_the_producers_agree_within_the_declarations_on_tiny_worlds(tiny: Path) -> None:
    within_every_window(tiny)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3))
def test_the_producers_agree_within_the_declarations_on_small_worlds(
    seed: int, tmp_path: Path
) -> None:
    within_every_window(connect(tmp_path / "w.duckdb", "small", seed))


@pytest.fixture(scope="module")
def quirky(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    quirks = QuirkTruth(
        backdated_share=0.2,
        correction_share=0.2,
        no_show_pair_share=1.0,
        pending_hire_rows=True,
        aggregate_only_share=0.2,
        snapshot_gap_share=0.2,
        state_seam_month="2023-07",
        cost_center_aliases=True,
    )
    path = connect(tmp_path_factory.mktemp("world") / "q.duckdb", "tiny", 4, quirks)
    with duckdb.connect(str(path), read_only=True) as con:
        yield con


def test_quirks_keep_the_producers_within_the_declarations_away_from_gaps(
    quirky: duckdb.DuckDBPyConnection,
) -> None:
    checked = 0
    for start in pd.period_range("2022-01", "2025-06", freq="M"):
        refused = problems(quirky, start, 6, categories=EVERY)
        if refused:
            assert all("missing the month-ends" in p for p in refused)
            continue
        comparison = against_measures(quirky, realized(quirky, start, 6, categories=EVERY))
        assert comparison.within, (start, comparison.problems())
        checked += 1
    assert checked >= 3


# --- the command ---------------------------------------------------------------------------------


def test_the_command_reports_the_divergences_and_refuses_beyond_them(
    tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = ["guidance", "realized", "--world", str(tiny), "--start", "2025-01", "--months", "6"]
    assert hcsim([*args, "--out", str(tmp_path / "a")]) == 0
    report = json.loads((tmp_path / "a" / "report.json").read_text())
    assert report["divergence"]["within"] is True
    assert report["divergence"]["categories"]["attrition"]["declared"] is None

    beyond = Comparison((Check(C.ATTRITION, 2.0, 1.0, 1.0, 1.0, None),))
    monkeypatch.setattr(cli, "against_measures", lambda con, result: beyond)
    with pytest.raises(SystemExit):
        hcsim([*args, "--out", str(tmp_path / "b")])
    assert not (tmp_path / "b").exists()
    assert hcsim([*args, "--out", str(tmp_path / "c"), "--allow-divergence"]) == 0
    assert json.loads((tmp_path / "c" / "report.json").read_text())["divergence"]["within"] is False
