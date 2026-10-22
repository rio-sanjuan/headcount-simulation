"""Tests for the performance domain: rating cycles, transitions, and first ratings."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from hcsim.engine.config import DomainModels
from hcsim.engine.domains import StepContext
from hcsim.engine.eligibility import EligibilityRule
from hcsim.engine.loop import simulate
from hcsim.engine.performance import performance
from hcsim.engine.seeds import SeedManager
from hcsim.engine.settings import from_record, to_record
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.ml.markov import MarkovModel
from hcsim.model.ml.persistence import save_model
from tests.engine.test_config_records import _saved
from tests.engine.test_config_state_seeds import SCHEMA, awards
from tests.engine.test_config_state_seeds import config as small_config
from tests.engine.test_loop import config, initial_state, models

CYCLE_STEP = 2  # the run starts in January and cycles in March

#: Every rating moves up one, the top stays, and a first rating is always the middle.
STEP_UP = MarkovModel(5).fit(np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 4]), np.array([2]))


def with_ratings(ratings: list[int], tenures: list[int]) -> SimulationState:
    state = initial_state()
    rows = state.employees[: len(ratings)]
    rows[:, EmployeeColumn.PERF_RATING] = ratings
    rows[:, EmployeeColumn.HIRE_STEP] = CYCLE_STEP - np.array(tenures)
    return state


def performance_models() -> DomainModels:
    return dataclasses.replace(models(attrition=0.0), performance=STEP_UP)


def test_a_cycle_moves_ratings_and_gives_first_ratings_after_the_cutoff() -> None:
    state = with_ratings([1, 3, 5, 0, 0], [24, 24, 24, 6, 5])
    ctx = StepContext(config(models=performance_models()), CYCLE_STEP, 0, SeedManager(1))
    result = performance(state, ctx)
    written = dict(
        zip(
            result.employees.write_index.tolist(),
            result.employees.write_rows[:, EmployeeColumn.PERF_RATING].tolist(),
            strict=True,
        )
    )
    # Ranks 1, 3, 5 step up to 2, 4, 5; six months' tenure earns a first rating, five doesn't.
    assert {k: written[k] for k in range(5) if k in written} == {0: 2, 1: 4, 2: 5, 3: 3}
    assert 4 not in written
    # Everyone else in the toy state is unrated with no tenure set, and stays unrated.
    assert len(written) == 4 + int(
        ((state.live_employees[5:, EmployeeColumn.HIRE_STEP]) <= -4).sum()
    )


def test_ratings_change_only_in_the_cycle_month() -> None:
    state = with_ratings([1, 3], [24, 24])
    for step in (0, 1, 3, 11):
        ctx = StepContext(config(models=performance_models()), step, 0, SeedManager(1))
        assert performance(state, ctx).employees.is_empty
    later = config(models=performance_models(), performance_cycle_month=9)
    assert not performance(state, StepContext(later, 8, 0, SeedManager(1))).employees.is_empty


def test_eligibility_keeps_rows_out_of_a_cycle() -> None:
    state = with_ratings([1, 1], [24, 24])
    state.employees[1, EmployeeColumn.COST_CENTER] = 1
    rule = EligibilityRule.of("performance", cost_center=["CC-02"])
    ctx = StepContext(
        config(models=performance_models(), eligibility=(rule,)), CYCLE_STEP, 0, SeedManager(1)
    )
    written = performance(state, ctx).employees.write_index.tolist()
    assert 0 in written
    assert 1 not in written


def test_the_domain_registers_only_with_a_model() -> None:
    with_model = simulate(config(models=performance_models()), with_ratings([1], [24]))
    assert "performance" not in with_model.unregistered
    without = simulate(config(), initial_state())
    assert "performance" in without.unregistered


def test_a_performance_model_round_trips_by_version(tmp_path: Path) -> None:
    saved = _saved(tmp_path)
    version = save_model(STEP_UP, tmp_path, "performance", {}).version
    built = dataclasses.replace(
        saved, performance=STEP_UP, versions={**saved.versions, "performance": version}
    )
    record = json.loads(json.dumps(to_record(small_config(models=built))))
    assert record["models"]["performance"] == version
    rebuilt = from_record(record, schema=SCHEMA, models_root=tmp_path, awards=awards())
    assert isinstance(rebuilt.models.performance, MarkovModel)
    np.testing.assert_allclose(rebuilt.models.performance.transitions, STEP_UP.transitions)
