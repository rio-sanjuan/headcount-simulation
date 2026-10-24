"""Tests for the plan-alignment evaluator: bias, spread, and D at t*, by cost center and valve."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from hcsim.engine.loop import simulate
from hcsim.studies.evaluators import PLAN_COLUMNS, plan_alignment, plan_alignment_table
from tests.engine.test_alignment_known_answers import STILL, aligned, plan, run
from tests.engine.test_loop import config, initial_state


def metrics_of(result: object) -> dict[str, float]:
    return {m.name: m.value for m in plan_alignment(result)}  # type: ignore[arg-type]


def test_a_run_that_lands_on_its_plan_scores_zero() -> None:
    # Under supply at a gain of one, CC-01 reaches its 40 in February and holds it through June.
    result = run(plan=plan({"CC-01": [40.0]}))
    table = plan_alignment_table(result)
    assert table is not None
    assert list(table.columns) == list(PLAN_COLUMNS)
    assert table.to_dict("records") == [
        {
            "cost_center": "CC-01",
            "target": 40.0,
            "mean": 40.0,
            "bias": 0.0,
            "spread": 0.0,
            "deviation": 0.0,
            "valve": "supply",
            "route": "G-A",
        }
    ]
    found = metrics_of(result)
    assert found["evaluation_step"] == 5.0
    assert found["targets_scored"] == 1.0
    assert (found["bias_mean"], found["deviation_max"], found["bias_mean_supply"]) == (
        0.0,
        0.0,
        0.0,
    )
    assert all(m.threshold is None for m in plan_alignment(result))


def rows(table: pd.DataFrame | None, *columns: str) -> dict[str, tuple[object, ...]]:
    """Each cost center's `columns`, from a plan-alignment table."""
    assert table is not None
    return {r["cost_center"]: tuple(r[c] for c in columns) for r in table.to_dict("records")}


def test_the_regimes_say_how_each_target_was_driven() -> None:
    target = plan({"CC-01": [40.0], "CC-02": [40.0]})
    covered = aligned(target)
    hybrid = rows(
        plan_alignment_table(run(plan=target, guidance=covered, alignment_method="hybrid")),
        "valve",
        "route",
        "mean",
        "bias",
    )
    # Guidance's 2 a month carries CC-01 to 44 by June's end; supply holds CC-02 at 40.
    assert hybrid == {"CC-01": ("flow", "G-B", 44.0, 4.0), "CC-02": ("supply", "G-A", 40.0, 0.0)}
    demand = rows(
        plan_alignment_table(run(plan=target, guidance=covered, alignment_method="demand")),
        "valve",
        "route",
        "bias",
    )
    assert demand["CC-02"] == ("undriven", "G-C", 6.0)
    found = metrics_of(run(plan=target, guidance=covered, alignment_method="hybrid"))
    assert (found["bias_mean_flow"], found["bias_mean_supply"], found["bias_max"]) == (
        4.0,
        0.0,
        4.0,
    )
    none = rows(plan_alignment_table(run(plan=target, alignment_method="none")), "valve")
    assert set(none.values()) == {("none",)}


def test_spread_and_d_are_taken_over_trajectories() -> None:
    # With attrition back, trajectories part: spread is their sd at t*, and D the furthest one.
    noisy = dataclasses.replace(STILL, attrition=config().models.attrition)
    result = simulate(
        config(models=noisy, plan=plan({"CC-01": [40.0]}), num_trajectories=20), initial_state()
    )
    table = plan_alignment_table(result)
    assert table is not None
    (row,) = table.to_dict("records")
    assert row["spread"] > 0
    heads = result.headcount[:, 6, 0]
    assert row["deviation"] == pytest.approx(abs(heads - 40).max())
    assert row["bias"] == pytest.approx(abs(heads.mean() - 40))


def test_a_run_it_cant_score_reports_nothing() -> None:
    assert plan_alignment(run()) == []  # no plan
    late = plan({"CC-01": [40.0]})
    short = run(plan=late, simulation_months=3)  # t* is June, past a three-month run
    assert plan_alignment(short) == []
