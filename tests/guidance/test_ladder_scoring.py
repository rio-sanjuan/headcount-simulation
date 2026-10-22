"""Tests for scoring the guidance ladder: WAPE, the margin, and the block bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcsim.cli import main as hcsim
from hcsim.guidance.ladder import (
    Arm,
    ArmRun,
    LadderError,
    LadderRun,
    block_indices,
    markdown,
    score,
    wape,
)
from hcsim.model.core.categories import GuidanceCategory
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

C = GuidanceCategory
CUTOFF = pd.Period("2024-12", "M")
MONTHS = pd.period_range("2025-01", periods=12, freq="M")


def test_wape_is_the_summed_absolute_error_over_the_summed_actuals() -> None:
    assert wape(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])) == pytest.approx(1 / 3)
    with pytest.raises(LadderError, match="sum to more than zero"):
        wape(np.zeros(3), np.zeros(3))


def test_a_draw_strings_together_blocks_of_consecutive_months() -> None:
    rows = block_indices(12, 3, 200, seed=1)
    assert rows.shape == (200, 12)
    assert rows.min() >= 0
    assert rows.max() <= 11
    for row in rows:
        for block in row.reshape(4, 3):
            assert list(np.diff(block)) == [1, 1]
    assert np.array_equal(rows, block_indices(12, 3, 200, seed=1))
    assert not np.array_equal(rows, block_indices(12, 3, 200, seed=2))
    assert (block_indices(12, 12, 5) == np.arange(12)).all()
    assert block_indices(12, 5, 3).shape == (3, 12)  # three blocks, cut to a year
    with pytest.raises(LadderError, match="a block must be 1 to 12 months"):
        block_indices(12, 13, 5)


def arm_run(name: str, rank: int, error_share: float, series: C = C.HIRES) -> ArmRun:
    """An arm whose every month misses its actual of 10 by `error_share`, so every bootstrap
    draw gives the same WAPE."""
    frame = pd.DataFrame(
        {
            "group": "G",
            "month": MONTHS,
            "forecast": 10.0 * (1 + error_share),
            "actual": 10.0,
        }
    )
    return ArmRun(Arm(name, series, rank, ("native.SeasonalNaive",), null=rank == 0), frame)


def ladder_of(*runs: ArmRun) -> LadderRun:
    return LadderRun(CUTOFF, "TOPLINE", 12, runs)


def test_the_champion_passes_only_to_an_arm_that_clears_the_margin() -> None:
    ladder = ladder_of(
        arm_run("null", 0, 0.5),
        arm_run("better", 1, 0.4),  # 20% below 0.5: it clears
        arm_run("marginal", 2, 0.38),  # 5% below 0.4: it doesn't
        arm_run("best", 3, 0.3),  # 25% below 0.4: it clears
    )
    card = score(ladder)
    assert card.selected == {C.HIRES: "best"}
    by_name = {s.arm.name: s for s in card.scores}
    assert by_name["better"].challenged == "null"
    assert by_name["better"].gain == pytest.approx(0.2)
    assert by_name["marginal"].challenged == "better"
    assert not by_name["marginal"].cleared
    assert by_name["best"].challenged == "better"  # the champion when it was tried
    assert by_name["best"].gain == pytest.approx(0.25)
    # Every month misses by the same share, so every draw agrees with the point.
    for s in card.scores:
        assert s.low == pytest.approx(s.wape)
        assert s.high == pytest.approx(s.wape)
    assert by_name["best"].gain_low == pytest.approx(0.25)


def test_an_arm_exactly_at_the_margin_clears_it() -> None:
    card = score(ladder_of(arm_run("null", 0, 0.5), arm_run("edge", 1, 0.375)), margin=0.25)
    assert card.selected == {C.HIRES: "edge"}
    held = score(ladder_of(arm_run("null", 0, 0.5), arm_run("edge", 1, 0.375)), margin=0.3)
    assert held.selected == {C.HIRES: "null"}


def test_the_intervals_widen_with_the_months_errors() -> None:
    run = arm_run("null", 0, 0.0)
    noisy = run.frame.assign(forecast=10.0 + np.where(np.arange(12) % 4 == 0, 8.0, 0.0))
    card = score(ladder_of(ArmRun(run.arm, noisy)), draws=500)
    (s,) = card.scores
    assert s.wape == pytest.approx(24 / 120)
    assert s.low < s.wape < s.high


def test_the_scoring_settings_are_checked() -> None:
    ladder = ladder_of(arm_run("null", 0, 0.5))
    with pytest.raises(LadderError, match="a margin must be in"):
        score(ladder, margin=1.0)
    with pytest.raises(LadderError, match="level must be in"):
        score(ladder, level=1.0)


def test_the_report_gives_each_series_its_selected_arm() -> None:
    ladder = ladder_of(
        arm_run("null", 0, 0.5),
        arm_run("better", 1, 0.4),
        arm_run("null", 0, 0.2, C.ATTRITION),
        arm_run("worse", 1, 0.3, C.ATTRITION),
    )
    text = markdown(ladder, score(ladder))
    assert "## hires: better" in text
    assert "## attrition: null" in text
    assert "| 1 | worse | 0.300 | 0.300–0.300 | null | -50.0% | -50.0% to -50.0% | held |" in text
    frame = score(ladder).frame()
    assert frame.loc[frame["selected"], "arm"].tolist() == ["better", "null"]


# --- the command ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    return path


def test_a_ladder_study_runs_from_the_command(world: Path, tmp_path: Path) -> None:
    out = tmp_path / "ladder"
    assert hcsim(["guidance", "ladder", "--world", str(world), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["cutoff"] == "2024-12"  # a year before the tiny world ends
    assert set(report["selected"]) == {"attrition", "hires", "internal_starts"}
    scores = pd.read_csv(out / "scores.csv")
    assert len(scores) == 18
    assert scores["selected"].sum() == 3
    assert len(pd.read_csv(out / "forecasts.csv")) == 18 * 12 * 2  # two org nodes
    assert "# Guidance ladder" in (out / "ladder.md").read_text()
    with pytest.raises(SystemExit):
        hcsim(["guidance", "ladder", "--world", str(world), "--out", str(out), "--cutoff", "late"])
    with pytest.raises(SystemExit):
        hcsim(
            ["guidance", "ladder", "--world", str(world), "--out", str(out), "--cutoff", "2025-06"]
        )
