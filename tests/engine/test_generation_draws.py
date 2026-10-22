"""Tests for growth's model draws and the degenerate-row diagnostic."""

from __future__ import annotations

import dataclasses

import numpy as np

from hcsim.engine.domains import rate_counts
from hcsim.engine.loop import simulate
from hcsim.engine.roles import DEGENERATE_ROWS, fill_roles
from hcsim.model.core.layouts import OPENING
from hcsim.model.ml.generative import EmpiricalAttributeGenerator
from hcsim.model.ml.group import BackfillRateModel, NegativeBinomialGrowthModel, PoissonGrowthModel
from tests.engine.test_loop import config, initial_state, models


def test_a_poisson_model_draws_as_a_poisson_count_would() -> None:
    model = PoissonGrowthModel(3, min_months=1).fit(np.array([[2.0, 1.0, 0.5]]))
    own = rate_counts(model, 3, np.random.default_rng(4))
    plain = np.random.default_rng(4).poisson(model.rates(np.arange(3)))
    assert own.tolist() == plain.tolist()
    # A rate table with no draw of its own gets a Poisson count at its rate.
    table = BackfillRateModel.constant(0.5, 3)
    counts = rate_counts(table, 3, np.random.default_rng(4))
    assert counts.tolist() == np.random.default_rng(4).poisson([0.5, 0.5, 0.5]).tolist()


def test_a_negative_binomial_model_draws_overdispersed_and_seeded_counts() -> None:
    history = np.array([[0.0], [8.0], [1.0], [9.0], [0.0], [10.0], [2.0], [6.0]])
    model = NegativeBinomialGrowthModel(1, min_months=2).fit(history)
    draws = np.array([rate_counts(model, 1, np.random.default_rng(s))[0] for s in range(2_000)])
    assert draws.var() > 1.5 * draws.mean()
    again = rate_counts(model, 1, np.random.default_rng(7))
    assert again.tolist() == rate_counts(model, 1, np.random.default_rng(7)).tolist()


def test_roles_drawn_from_the_pooled_rows_are_degenerate() -> None:
    own = np.array([[1, 0, 0, 0, 0, 0]] * 3)
    generator = EmpiricalAttributeGenerator(min_rows=3).fit(np.zeros(3, dtype=np.int64), own)
    counts = np.array([2, 4, 0], dtype=np.int64)
    rows = OPENING.empty(6)
    degenerate = fill_roles(rows, counts, generator, np.random.default_rng(1))
    assert degenerate.tolist() == [0, 4, 0]  # CC-02 has no rows of its own
    unknown = EmpiricalAttributeGenerator(min_rows=1).fit(
        np.zeros(1, dtype=np.int64), np.array([[-1, 0, 0, 0, 0, 0]])
    )
    assert fill_roles(
        OPENING.empty(2), np.array([2]), unknown, np.random.default_rng(1)
    ).tolist() == [2]


def test_a_run_records_degenerate_rows_per_month_and_cost_center() -> None:
    covered = simulate(config(simulation_months=4), initial_state())
    for t in covered.trajectories:
        assert t.diagnostics[DEGENERATE_ROWS].shape == (4, 3)
        assert int(t.diagnostics[DEGENERATE_ROWS].sum()) == 0
    thin = EmpiricalAttributeGenerator(min_rows=2).fit(
        np.array([0, 0], dtype=np.int64), np.array([[1, 0, 0, 0, 0, 0]] * 2)
    )
    sparse = simulate(
        config(simulation_months=4, models=dataclasses.replace(models(), attributes=thin)),
        initial_state(),
    )
    totals = np.sum([t.diagnostics[DEGENERATE_ROWS] for t in sparse.trajectories], axis=0)
    assert int(totals[:, 0].sum()) == 0
    assert int(totals[:, 1:].sum()) > 0
