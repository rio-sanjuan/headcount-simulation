"""Tests for alignment onto plan targets: greedy re-derivation and the align step."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.guidance.alignment import AlignmentProblem, greedy, path, shortfall_of
from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.framework import PipelineError
from hcsim.guidance.pipeline import run
from hcsim.model.core.alignment import AlignmentTarget, marginal
from hcsim.model.core.categories import GuidanceCategory
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

SIGNS = {"hires": 1, "attrition": -1}
NAN = np.nan


def problem(
    target: list[float], band: list[float] | None = None, **kwargs: object
) -> AlignmentProblem:
    flows = {"hires": np.array([3.0, 3.0, 3.0]), "attrition": np.array([2.0, 2.0, 2.0])}
    anchor = np.array([101.0, 102.0, 103.0])
    bands = np.zeros(3) if band is None else np.array(band)
    return AlignmentProblem(anchor, 100.0, flows, SIGNS, np.array(target), bands, **kwargs)  # type: ignore[arg-type]


def balanced(result: object, p: AlignmentProblem) -> bool:
    rebuilt = path(p.last_actual, result.flows, p.signs, p.epsilon)  # type: ignore[attr-defined]
    return bool(np.allclose(rebuilt, result.anchor))  # type: ignore[attr-defined]


def test_greedy_moves_the_anchor_onto_its_targets_and_re_derives_the_flows() -> None:
    p = problem([NAN, 110.0, NAN])
    result = greedy(p)
    assert result.anchor[1] == pytest.approx(110.0)
    assert balanced(result, p)
    assert np.allclose(result.shortfall, 0.0)
    # Month 2 needs +9 against +1: the smallest flow, attrition, takes it first, to zero.
    assert result.flows["attrition"][1] == pytest.approx(0.0)
    assert result.flows["hires"][1] == pytest.approx(9.0)  # then hires carries the +9
    assert result.method == "greedy"


def test_a_band_aligns_only_as_far_as_its_edge() -> None:
    result = greedy(problem([NAN, 110.0, NAN], band=[0.0, 4.0, 0.0]))
    assert result.anchor[1] == pytest.approx(106.0)  # the band's near edge, not its centre


def test_fixed_flows_keep_their_forecast_and_an_unreachable_target_is_a_shortfall() -> None:
    p = problem([NAN, 90.0, NAN], fixed=frozenset({"attrition"}))
    result = greedy(p)
    assert result.flows["attrition"].tolist() == [2.0, 2.0, 2.0]
    # Hires alone can fall only to zero: month 2 reaches 101 - 2 = 99, not 90.
    assert result.flows["hires"][1] == pytest.approx(0.0)
    assert result.anchor[1] == pytest.approx(99.0)
    assert result.shortfall.tolist() == pytest.approx([0.0, 9.0, 0.0])
    assert balanced(result, p)


def test_declared_movements_enter_the_balance() -> None:
    p = problem([NAN, NAN, 103.0], movements=np.array([5.0, 0.0, 0.0]))
    result = greedy(p)
    assert balanced(result, p)
    assert result.anchor[2] == pytest.approx(103.0)
    # The movement already brings 5 of the 3 the target wants by month 3: the flows give 2 back.
    assert result.flows["hires"].sum() - result.flows["attrition"].sum() == pytest.approx(-2.0)


def test_a_shortfall_is_signed_and_zero_within_the_band() -> None:
    target, band = np.array([100.0, 100.0, NAN]), np.array([2.0, 2.0, 0.0])
    assert shortfall_of(np.array([103.0, 99.0, 50.0]), target, band).tolist() == [1.0, 0.0, 0.0]
    assert shortfall_of(np.array([97.0, 100.0, 0.0]), target, band).tolist() == [-1.0, 0.0, 0.0]


# --- the align step -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path_ = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path_)
    with duckdb.connect(str(path_), read_only=True) as connection:
        yield connection


def target_for(values: list[float], bands: list[float] | None = None) -> AlignmentTarget:
    # The tiny world's snapshot is December 2025, so the forecast runs from January 2026.
    dates = ["2026-03", "2026-06"]
    m = marginal("TOPLINE", ["ALL"], dates, [values], None if bands is None else [bands])
    return AlignmentTarget((m,), pd.Period("2026-01", "M"), months=12, planned_depth=6)


CONFIG = {"horizon": 6, "grouping": "TOPLINE", "alignment": {"method": "greedy"}}


def test_a_run_aligns_its_anchor_on_the_target_dates(con: duckdb.DuckDBPyConnection) -> None:
    result = run(con, GuidanceConfig.from_mapping(CONFIG), target_for([300.0, 320.0]))
    headcount = result.artifact.frame.query("category == 'headcount'").set_index("forecast_month")
    assert headcount.loc[2, "value"] == pytest.approx(300.0)  # March, the third month
    assert headcount.loc[5, "value"] == pytest.approx(320.0)  # June, the sixth
    report = result.report["alignment"]
    assert report["aligned_months"] == 2
    assert report["max_shortfall"] == 0.0
    assert report["target"].startswith("at1-")


def test_a_run_needs_a_target_with_its_grouping(con: duckdb.DuckDBPyConnection) -> None:
    config = GuidanceConfig.from_mapping(CONFIG)
    with pytest.raises(PipelineError, match="alignment needs a target"):
        run(con, config)
    m = marginal("cost_center", ["CC-01"], ["2026-03"], [[10.0]])
    other = AlignmentTarget((m,), pd.Period("2026-01", "M"), months=12, planned_depth=6)
    with pytest.raises(PipelineError, match="no marginal for the run's grouping 'TOPLINE'"):
        run(con, config, other)


def test_the_alignment_settings_are_checked_at_load() -> None:
    config = GuidanceConfig.from_mapping(
        {"alignment": {"method": "greedy", "fixed": ["attrition"], "weights": {"attrition": 2}}}
    )
    assert config.alignment.fixed == (GuidanceCategory.ATTRITION,)
    with pytest.raises(GuidanceConfigError) as error:
        GuidanceConfig.from_mapping(
            {
                "categories": ["attrition", "external_hires"],
                "reconciliation": {"method": None},
                "alignment": {"method": "simplex", "fixed": ["hires"], "weights": {"hires": 0}},
            }
        )
    message = str(error.value)
    assert "alignment.method: unknown alignment method 'simplex'" in message
    assert "aligning moves the headcount anchor" in message
    assert "alignment.fixed: ['hires'] aren't among the categories" in message
    assert "alignment.weights: every weight must be positive" in message


def test_a_run_aligns_by_the_qp_too(con: duckdb.DuckDBPyConnection) -> None:
    config = GuidanceConfig.from_mapping(
        {"horizon": 6, "grouping": "TOPLINE", "alignment": {"method": "qp"}}
    )
    result = run(con, config, target_for([300.0, 320.0], bands=[2.0, 2.0]))
    frame = result.artifact.frame
    headcount = frame[frame["category"] == "headcount"].sort_values("forecast_month")
    values = headcount["value"].to_numpy(dtype=float)
    assert abs(values[2] - 300.0) <= 2.0 + 1e-6
    assert abs(values[5] - 320.0) <= 2.0 + 1e-6
    assert result.report["alignment"]["method"] == "qp"
    assert result.report["alignment"]["max_shortfall"] == 0.0
