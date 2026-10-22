"""Tests for seeding's fixed order: encoding, then subset filtering, then pruning."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.seeding import _pipeline, schema_from_world, seed_state
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn
from hcsim.model.core.schema import COST_CENTER, JOB_LEVEL, SimulationSchema
from hcsim.model.data.population import EVERY_ACCOUNT
from hcsim.model.data.subset import SubsetSpec
from hcsim.synth.config import QuirkTruth, preset
from hcsim.synth.world import generate_world, write_world

RUN = dt.date(2026, 1, 1)


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "aliased"])
def con(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[duckdb.DuckDBPyConnection]:
    quirks = QuirkTruth(cost_center_aliases=request.param)
    path = tmp_path_factory.mktemp("world") / "w.duckdb"
    write_world(generate_world(dataclasses.replace(preset("tiny", seed=9), quirks=quirks)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def test_pruned_rows_are_counted_under_their_cause(con: duckdb.DuckDBPyConnection) -> None:
    full = schema_from_world(con)
    labels = {name: list(v.labels) for name, v in full.vocabularies.items()}
    labels[COST_CENTER] = [cc for cc in labels[COST_CENTER] if cc != "CC-01"]
    labels.pop("org_level1")
    state, report = seed_state(con, SimulationSchema.build(labels), RUN)
    assert set(report.employee_causes) == {"unknown_cost_center"}
    assert sum(report.employee_causes.values()) == report.employees_pruned > 0
    assert sum(report.opening_causes.values()) == report.openings_pruned
    assert state.counters_agree()
    _, clean = seed_state(con, full, RUN)
    assert clean.employee_causes == {}
    assert clean.opening_causes == {}


def test_a_subset_seeds_only_its_rows(con: duckdb.DuckDBPyConnection) -> None:
    schema = schema_from_world(con)
    subset = SubsetSpec.of(org_level1=["ORG-A"], job_level=["1", "2"])
    state, report = seed_state(con, schema, RUN, subset=subset)
    _, everyone = seed_state(con, schema, RUN)
    assert 0 < report.employees_seeded < everyone.employees_seeded
    rows = state.live_employees
    under = schema.rollup("org_level1", rows[:, EmployeeColumn.COST_CENTER].astype(np.int64))
    assert (under == schema.vocabulary("org_level1").code("ORG-A")).all()
    levels = schema.decode(JOB_LEVEL, rows[:, EmployeeColumn.LEVEL].astype(np.int64))
    assert set(levels) <= {"1", "2"}
    # Rules the warehouse can apply are pushed down, so nothing outside arrives to drop.
    assert "outside_subset" not in report.employee_causes
    assert state.counters_agree()


def test_the_population_decides_who_is_read(con: duckdb.DuckDBPyConnection) -> None:
    schema = schema_from_world(con)
    _, regular = seed_state(con, schema, RUN)
    _, everyone = seed_state(con, schema, RUN, population=EVERY_ACCOUNT)
    assert everyone.employees_seeded > regular.employees_seeded
    # Requisitions book to no account, so the account rules don't narrow them.
    assert everyone.openings_seeded == regular.openings_seeded


def test_subset_filtering_comes_before_pruning() -> None:
    """A row both outside the subset and carrying an unknown label is counted once, as
    outside the subset: the order decides the cause."""
    schema = SimulationSchema.build({COST_CENTER: ["CC-01", "CC-02"], JOB_LEVEL: [1, 2]})
    frame = pd.DataFrame(
        {
            COST_CENTER: ["CC-01", "CC-09", "CC-02", "CC-09"],
            JOB_LEVEL: [1, 1, 2, 2],
        }
    )
    columns = {COST_CENTER: int(EmployeeColumn.COST_CENTER), JOB_LEVEL: int(EmployeeColumn.LEVEL)}
    lowered = SubsetSpec.of(job_level=["1"]).lower(schema)
    rows, keep, causes = _pipeline(frame, schema, columns, EMPLOYEE, lowered)
    assert keep.tolist() == [True, False, False, False]
    assert causes == {"outside_subset": 2, "unknown_cost_center": 1}
    assert rows[1, EmployeeColumn.COST_CENTER] == -1
    _, keep_all, plain = _pipeline(frame, schema, columns, EMPLOYEE, None)
    assert plain == {"unknown_cost_center": 2}
    assert keep_all.sum() == 2
