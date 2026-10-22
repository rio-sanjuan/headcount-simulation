"""Tests for the campus domain: its schedule, roles, isolation, and fill-model hook."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hcsim.engine.campus import CampusSchedule, campus, is_campus
from hcsim.engine.config import ConfigError
from hcsim.engine.domains import StepContext, cancellation
from hcsim.engine.fills import channel_hazards
from hcsim.engine.loop import FLOW_SYMBOLS, HEADCOUNT_IDENTITY, OPENING_IDENTITY, simulate
from hcsim.engine.seeds import SeedManager
from hcsim.engine.settings import from_record, to_record
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import REQ_TYPE, SimulationSchema
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.generative import EmpiricalAttributeGenerator
from hcsim.model.ml.persistence import save_model
from tests.engine.test_config_records import _saved
from tests.engine.test_config_state_seeds import awards
from tests.engine.test_config_state_seeds import config as small_config
from tests.engine.test_loop import SCHEMA, config, initial_state, models

SCHEDULE = CampusSchedule({2: {"CC-01": 3, "CC-03": 1}}, start_months=(4,))
ROLES = EmpiricalAttributeGenerator(min_rows=1).fit(
    np.zeros(2, dtype=np.int64), np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
)


def with_campus(**overrides: Any) -> Any:
    built = dataclasses.replace(
        overrides.pop("models", models(attrition=0.0)), campus_attributes=ROLES
    )
    return config(models=built, campus=overrides.pop("campus", SCHEDULE), **overrides)


def campus_state() -> SimulationState:
    """The toy state with its CC-02 requisitions turned into campus ones."""
    state = initial_state()
    code = SCHEMA.vocabulary(REQ_TYPE).code("campus")
    state.openings[[2, 3], OpeningColumn.REQ_TYPE] = code
    return state


def test_the_schedule_opens_campus_requisitions_by_month_and_cost_center() -> None:
    ctx = StepContext(with_campus(), 1, 0, SeedManager(2))  # February
    result = campus(initial_state(), ctx)
    opened = result.openings.append
    assert result.flows["G_camp"].tolist() == [3, 0, 1]
    assert (opened[:, OpeningColumn.REQ_TYPE] == SCHEMA.vocabulary(REQ_TYPE).code("campus")).all()
    assert (opened[:, OpeningColumn.LEVEL] == 1).all()  # the generator's roles
    assert result.ledger == {}  # outside the requisitions-opened ledger
    quiet = campus(initial_state(), StepContext(with_campus(), 0, 0, SeedManager(2)))
    assert quiet.openings.is_empty


def test_campus_requisitions_fill_in_a_start_month_by_the_hook() -> None:
    april = StepContext(with_campus(), 3, 0, SeedManager(2))
    result = campus(campus_state(), april)
    assert result.openings.drop.tolist() == [2, 3]
    assert result.flows["F_camp"].tolist() == [0, 4, 0]  # one and three openings
    hires = result.employees.append
    assert (hires[:, EmployeeColumn.COST_CENTER] == 1).all()
    none = dataclasses.replace(models(attrition=0.0), campus_fill=ConstantModel(0.0))
    held = campus(campus_state(), StepContext(with_campus(models=none), 3, 0, SeedManager(2)))
    assert held.openings.drop.size == 0


def test_campus_requisitions_are_left_to_their_schedule() -> None:
    state = campus_state()
    ctx = StepContext(with_campus(models=models(cancel=1.0)), 0, 0, SeedManager(1))
    assert is_campus(state.live_openings, ctx).tolist() == [False, False, True, True, False, False]
    hazards = channel_hazards(state.live_openings, ctx)
    assert (hazards[[2, 3]] == 0).all()
    assert cancellation(state, ctx).openings.drop.tolist() == [0, 1, 4, 5]
    # Without a schedule, a campus requisition is like any other.
    plain = StepContext(config(models=models(cancel=1.0)), 0, 0, SeedManager(1))
    assert not is_campus(state.live_openings, plain).any()
    assert cancellation(state, plain).openings.drop.size == 6


def test_a_campus_run_keeps_the_identities() -> None:
    result = simulate(with_campus(), campus_state(), validate=True)
    f = {s: result.flow(s) for s in FLOW_SYMBOLS}
    assert int(f["G_camp"].sum()) == 4 * 3  # every trajectory opens February's four
    assert int(f["F_camp"].sum()) > 0
    h, o = result.headcount, result.open_openings
    heads = sum(v * f[s] for s, v in HEADCOUNT_IDENTITY.items())
    openings = sum(v * f[s] for s, v in OPENING_IDENTITY.items())
    assert np.array_equal(h[:, 1:], h[:, :-1] + heads)
    assert np.array_equal(o[:, 1:], o[:, :-1] + openings)
    assert "campus" not in result.unregistered


def test_a_schedule_that_doesn_t_fit_is_refused() -> None:
    bad = CampusSchedule({13: {"CC-09": 1}, 2: {"CC-01": -1}}, start_months=(0,))
    with pytest.raises(ConfigError) as caught:
        config(campus=bad)
    assert caught.value.problems == [
        "campus: campus months must lie in 1..12: [0, 2, 13]",
        "campus: campus openings for CC-01 in month 2 are negative",
        "campus: no cost center 'CC-09' for campus in month 13",
        "campus: a campus schedule needs a campus attribute generator",
    ]
    no_type = SimulationSchema.build({"cost_center": ["CC-01"]})
    assert "needs 'campus' among the requisition types" in SCHEDULE.problems(no_type)[0]


def test_a_schedule_round_trips_through_records(tmp_path: Path) -> None:
    saved = _saved(tmp_path)
    version = save_model(ROLES, tmp_path, "campus_attributes", {}).version
    built_models = dataclasses.replace(
        saved, campus_attributes=ROLES, versions={**saved.versions, "campus_attributes": version}
    )
    schedule = CampusSchedule({7: {"CC-01": 2}}, start_months=(9,))
    schema = SimulationSchema.build(
        {
            **{n: list(v.labels) for n, v in small_config().schema.vocabularies.items()},
            REQ_TYPE: ["backfill", "campus", "growth"],
        }
    )
    built = small_config(schema=schema, models=built_models, campus=schedule)
    record = json.loads(json.dumps(to_record(built)))
    assert record["campus"] == {"openings": {"7": {"CC-01": 2}}, "start_months": [9]}
    rebuilt = from_record(record, schema=built.schema, models_root=tmp_path, awards=awards())
    assert rebuilt.campus == schedule


def test_without_a_schedule_the_domain_does_nothing() -> None:
    result = campus(initial_state(), StepContext(config(), 1, 0, SeedManager(2)))
    assert result.openings.is_empty
    assert result.employees.is_empty
