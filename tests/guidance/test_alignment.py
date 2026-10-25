"""Tests for alignment onto plan targets: greedy re-derivation and the align step."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.cli import main as hcsim
from hcsim.engine.config import ConfigError
from hcsim.guidance.alignment import AlignmentProblem, greedy, path, shortfall_of
from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.framework import PipelineError
from hcsim.guidance.pipeline import finish, fit, run
from hcsim.model.core.alignment import AlignmentTarget, marginal
from hcsim.model.core.categories import ANCHOR, GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast, movements_digest
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
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path_ = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path_)
    return path_


@pytest.fixture(scope="module")
def con(world: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(str(world), read_only=True) as connection:
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


# --- the aligned_to stamp, and precedence ------------------------------------------------------


def test_an_aligned_artifact_is_stamped_with_its_target(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    target = target_for([300.0, 320.0])
    aligned = run(con, GuidanceConfig.from_mapping(CONFIG), target).artifact
    assert aligned.aligned_to == target.digest()
    aligned.save(tmp_path / "g")
    assert GuidanceForecast.load(tmp_path / "g").aligned_to == target.digest()
    plain = run(con, GuidanceConfig(horizon=6, grouping="TOPLINE")).artifact
    assert plain.aligned_to is None
    # It records the declared movements it counted, none here, and so does its reloaded copy.
    assert aligned.movements == movements_digest(None)
    assert GuidanceForecast.load(tmp_path / "g").movements == aligned.movements
    assert plain.movements is None
    moved = {"ALL": [0.0, -3.0, 0.0, 0.0, 0.0, 0.0]}
    shifted = run(con, GuidanceConfig.from_mapping(CONFIG), target, moved).artifact
    assert shifted.movements == movements_digest(moved) != aligned.movements


def test_alignment_takes_precedence_over_reconciliation(con: duckdb.DuckDBPyConnection) -> None:
    config = GuidanceConfig.from_mapping(
        {**CONFIG, "reconciliation": {"method": "ols", "tolerance": 0.0}}
    )
    result = run(con, config, target_for([300.0, 320.0]))
    frame = result.artifact.frame
    values = frame[frame["category"] == "headcount"].sort_values("forecast_month")["value"]
    assert values.to_numpy()[[2, 5]].tolist() == pytest.approx([300.0, 320.0])
    assert result.report["alignment"]["precedence"] == "aligned after reconciliation"
    steps = [s["step"] for s in result.report["steps"]]
    assert steps.index("reconcile") < steps.index("align") < steps.index("format")


def test_one_fit_finishes_for_each_target_as_a_whole_run_would(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # Per cost center, its own target dates and bands, and one center the plan leaves out.
    config = GuidanceConfig.from_mapping({"horizon": 6, "alignment": {"method": "greedy"}})
    fitted = fit(con, config)
    groups = sorted({group for group, _ in fitted.context["forecasts"]})
    last = {g: float(fitted.context["series"].values(g, ANCHOR).iloc[-1]) for g in groups}
    dates = ["2026-02", "2026-04", "2026-06"]
    planned = groups[:-1]

    def plan(growth: float) -> AlignmentTarget:
        values = [
            [last[g] * (1 + growth * (k + 1)) if (k + i) % 2 == 0 else np.nan for k in range(3)]
            for i, g in enumerate(planned)
        ]
        bands = [[1.0 if not np.isnan(v) else np.nan for v in row] for row in values]
        m = marginal("cost_center", planned, dates, values, bands)
        return AlignmentTarget((m,), pd.Period("2026-01", "M"), months=6, planned_depth=6)

    for growth in (0.0, 0.1):  # finishing twice leaves the fit as it was
        target = plan(growth)
        finished = finish(fitted, target)
        whole = run(con, config, target)
        assert finished.artifact.fingerprint() == whole.artifact.fingerprint()
        assert finished.report["alignment"]["untargeted_groups"] == [groups[-1]]
        assert finished.report["alignment"]["aligned_months"] == sum(
            (k + i) % 2 == 0 for i in range(len(planned)) for k in range(3)
        )
        assert finished.report["alignment"]["max_shortfall"] == 0.0
        # Each center lands within its own band on its own dates.
        frame = finished.artifact.frame.query("category == 'headcount'")
        heads = frame.set_index(["cost_center", "forecast_month"])["value"]
        goals = target.marginals[0].values
        for i, g in enumerate(planned):
            for k, month in enumerate((1, 3, 5)):
                if not np.isnan(goals[i, k]):
                    assert abs(heads[g, month] - goals[i, k]) <= 1.0 + 1e-6, (g, month)
    assert "align" not in fitted.context.report
    steps = [s["step"] for s in finished.report["steps"]]
    assert steps.count("ingest") == 1
    assert steps[-1] == "format"


def totals(result: object) -> dict[str, np.ndarray]:
    """Each category's values by forecast month."""
    frame = result.artifact.frame  # type: ignore[attr-defined]
    table = frame.pivot_table(index="forecast_month", columns="category", values="value")
    return {str(c): table[c].to_numpy(dtype=float) for c in table.columns}


def test_alignment_re_derives_its_months_from_the_flows_as_generated(
    con: duckdb.DuckDBPyConnection,
) -> None:
    # The tiny world grew lately, so its forecast anchor starts well below the last actual, and
    # reconciliation books the fall as exits. Where the plan replaces that anchor, the flows
    # start from their forecasts as generated, so the fall doesn't stay as exits that as many
    # hires then cancel: hires the engine can only fill from the openings it has.
    plain = {"horizon": 6, "grouping": "TOPLINE"}
    generated = run(con, GuidanceConfig.from_mapping({**plain, "reconciliation": {"method": None}}))
    last = float(generated.series.values("ALL", GuidanceCategory.HEADCOUNT).iloc[-1])
    start = pd.Period("2026-01", "M")
    dates = [str(start + k) for k in range(6)]
    m = marginal("TOPLINE", ["ALL"], dates, [[last] * 6])
    flat = AlignmentTarget((m,), start, months=6, planned_depth=6)
    before, reconciled = totals(generated), totals(run(con, GuidanceConfig.from_mapping(plain)))
    assert reconciled["attrition"][0] > 10 * before["attrition"][0]  # the fall, as exits
    aligned = run(con, GuidanceConfig.from_mapping(CONFIG), flat)
    after = totals(aligned)
    assert after["headcount"] == pytest.approx([last] * 6)
    # Holding the plan flat, month 0's flows only close the gap between its hires and exits.
    flows = ("attrition", "external_hires")
    widest = max(before[c][0] for c in flows)
    assert max(after[c][0] for c in flows) == pytest.approx(widest)
    # Every month is aligned, so reconciling first changes nothing.
    without = {**CONFIG, "reconciliation": {"method": None}}
    unreconciled = run(con, GuidanceConfig.from_mapping(without), flat)
    pd.testing.assert_frame_equal(aligned.artifact.frame, unreconciled.artifact.frame)


def test_the_command_aligns_to_a_saved_target(
    con: duckdb.DuckDBPyConnection, world: Path, tmp_path: Path
) -> None:
    target = target_for([300.0, 320.0])
    target.save(tmp_path / "plan")
    (tmp_path / "guidance.yaml").write_text(
        "horizon: 6\ngrouping: TOPLINE\nalignment: {method: greedy}\n"
    )
    args = ["guidance", "fit", "--world", str(world), "--config", str(tmp_path / "guidance.yaml")]
    assert hcsim([*args, "--target", str(tmp_path / "plan"), "--out", str(tmp_path / "g")]) == 0
    assert GuidanceForecast.load(tmp_path / "g").aligned_to == target.digest()
    with pytest.raises(SystemExit):
        hcsim([*args, "--target", str(tmp_path / "nowhere"), "--out", str(tmp_path / "h")])


def test_the_engine_refuses_guidance_aligned_to_another_plan() -> None:
    from tests.engine.test_ceiling import plan
    from tests.engine.test_loop import config, guidance

    run_plan = plan({"CC-01": [90.0, 90.0]})
    other = plan({"CC-01": [95.0, 95.0]})
    aligned = dataclasses.replace(
        guidance({"attrition": {"CC-01": 1.0}}), aligned_to=other.digest()
    )
    with pytest.raises(ConfigError, match="guidance was aligned to the plan"):
        config(guidance=aligned, plan=run_plan)
    matching = dataclasses.replace(aligned, aligned_to=run_plan.digest())
    assert config(guidance=matching, plan=run_plan, alignment_method="demand").guidance is matching
