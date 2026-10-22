"""Tests for subsets: declaring, lowering to leaves, and both kinds of filtering."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pytest

from hcsim.engine.seeding import schema_from_world
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import COMPANY, COST_CENTER, JOB_FAMILY, JOB_LEVEL, SimulationSchema
from hcsim.model.data.population import REGULAR, PopulationSpec
from hcsim.model.data.subset import SubsetError, SubsetSpec
from hcsim.model.sql.extract import seed_employees
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world


def schema() -> SimulationSchema:
    return SimulationSchema.build(
        {
            COST_CENTER: ["CC-01", "CC-02", "CC-03", "CC-04"],
            JOB_FAMILY: ["Data", "Sales"],
            JOB_LEVEL: [1, 2, 3],
            COMPANY: ["CO-1"],
        },
        org_levels={
            "org_level1": {"CC-01": "ORG-A", "CC-02": "ORG-A", "CC-03": "ORG-B", "CC-04": "ORG-B"},
            "org_level2": {"CC-01": "A-1", "CC-02": "A-2", "CC-03": "B-1", "CC-04": "B-1"},
        },
    )


def employees(rows: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    """(cost center, family, level, manager, rating) per row."""
    buffer = EMPLOYEE.empty(len(rows))
    for i, (cc, family, level, manager, rating) in enumerate(rows):
        buffer[
            i,
            [
                EmployeeColumn.COST_CENTER,
                EmployeeColumn.JOB_FAMILY,
                EmployeeColumn.LEVEL,
                EmployeeColumn.IS_MANAGER,
                EmployeeColumn.PERF_RATING,
            ],
        ] = (cc, family, level, manager, rating)
    return buffer


def test_org_rules_lower_to_leaf_cost_centers_and_intersect() -> None:
    s = schema()
    assert SubsetSpec.of(org_level1=["ORG-B"]).lower(s).codes == {"cost_center": {2, 3}}
    both = SubsetSpec.of(org_level1=["ORG-A"], org_level2=["A-2", "B-1"]).lower(s)
    assert both.codes == {"cost_center": {1}}
    narrowed = SubsetSpec.of(org_level1=["ORG-A"], cost_center=["CC-02", "CC-03"]).lower(s)
    assert narrowed.codes["cost_center"] == {1}
    assert SubsetSpec.of(org_level1=["ORG-A"], cost_center=["CC-03"]).lower(s).empty


def test_buffers_filter_in_memory_with_the_rating_on_employees_only() -> None:
    lowered = SubsetSpec.of(
        org_level1=["ORG-A"], job_level=["2", "3"], is_manager=["True"], perf_rating=[4, 5]
    ).lower(schema())
    rows = employees(
        [(0, 0, 0, 1, 4), (1, 1, 2, 1, 5), (2, 0, 2, 1, 5), (1, 0, 2, 0, 5), (0, 0, 2, 1, 3)]
    )
    assert lowered.mask(rows, EMPLOYEE).tolist() == [False, True, False, False, False]
    openings = OPENING.empty(2)
    openings[:, OpeningColumn.LEVEL] = 2
    openings[:, OpeningColumn.IS_MANAGER] = 1
    openings[:, OpeningColumn.COST_CENTER] = [0, 3]
    assert lowered.mask(openings, OPENING).tolist() == [True, False]
    assert lowered.in_memory == {"perf_rating"}


def test_pushdown_narrows_the_population(tmp_path: Path) -> None:
    world = generate_world(preset("tiny", seed=3))
    path = tmp_path / "w.duckdb"
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        s = schema_from_world(con)
        lowered = SubsetSpec.of(
            org_level1=["ORG-A"], job_level=["1", "2"], is_manager=["false"], perf_rating=["5"]
        ).lower(s)
        narrowed = lowered.pushdown(REGULAR)
        assert narrowed.rules["account_group"] == {"regular"}
        assert "perf_rating" not in narrowed.rules  # filtered in memory instead
        assert narrowed.rules["is_manager"] == {"false"}
        rows = con.execute(seed_employees(narrowed), {"snapshot": dt.date(2024, 6, 30)}).df()
    orgs = world.tables["dim_org"].set_index("cost_center")["org_level1"]
    assert len(rows) > 0
    assert set(rows["cost_center"].map(orgs)) == {"ORG-A"}
    assert set(rows["job_level"]) <= {1, 2}
    assert not rows["is_manager"].any()


def test_pushdown_intersects_a_column_both_constrain() -> None:
    s = schema()
    population = PopulationSpec.of(account_group=["regular"], cost_center=["CC-01", "CC-03"])
    narrowed = SubsetSpec.of(org_level1=["ORG-A"]).lower(s).pushdown(population)
    assert narrowed.rules["cost_center"] == {"CC-01"}
    with pytest.raises(SubsetError, match="leaves nothing of"):
        SubsetSpec.of(cost_center=["CC-02"]).lower(s).pushdown(population)


def test_bad_subsets_report_every_problem() -> None:
    with pytest.raises(SubsetError, match="can't constrain 'account'") as error:
        SubsetSpec.of(account=["ACCT-100"], job_family=[], is_manager=["maybe"], perf_rating=["R5"])
    message = str(error.value)
    assert "the rule on 'job_family' admits nothing" in message
    assert "is_manager admits true or false" in message
    assert "perf_rating admits rating ranks" in message
    with pytest.raises(SubsetError, match="org_level1 has no nodes \\['ORG-Z'\\]") as error:
        SubsetSpec.of(org_level1=["ORG-Z"], org_level9=["X"], job_family=["Legal"]).lower(schema())
    assert "no rollup 'org_level9'" in str(error.value)
    assert "job_family has no labels ['Legal']" in str(error.value)
