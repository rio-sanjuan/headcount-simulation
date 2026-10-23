"""Tests for planner scopes: which rows a planner rule selects, and which rule decides a row."""

from __future__ import annotations

import pytest

from hcsim.engine.planner import PlannerError, PlannerScope, most_specific, resolve_scopes
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from tests.engine.test_loop import SCHEMA, initial_state

EVERYONE = PlannerScope()
FIRST = PlannerScope.of(cost_center=["CC-01"])
FIRST_SENIOR = PlannerScope.of(cost_center=["CC-01"], job_level=[2])
TWO = PlannerScope.of(cost_center=["CC-01", "CC-02"])


def test_a_scope_is_more_specific_the_more_it_constrains_and_the_less_it_admits() -> None:
    resolved = resolve_scopes([EVERYONE, FIRST, FIRST_SENIOR, TWO], SCHEMA)
    assert [r.specificity for r in resolved] == [(0, 0), (1, -1), (2, -2), (1, -2)]
    assert str(EVERYONE) == "everyone"
    assert str(FIRST_SENIOR) == "cost_center in ['CC-01'], job_level in ['2']"


def test_the_most_specific_scope_decides_each_row() -> None:
    state = initial_state()  # 30 employees per cost center, levels alternating 1 and 2
    people = state.live_employees
    resolved = resolve_scopes([EVERYONE, FIRST, FIRST_SENIOR, TWO], SCHEMA)
    winner = most_specific(resolved, people, EMPLOYEE)
    cc = people[:, EmployeeColumn.COST_CENTER].astype(int)
    senior = people[:, EmployeeColumn.LEVEL].astype(int) == 1
    assert set(winner[(cc == 0) & senior]) == {2}
    assert set(winner[(cc == 0) & ~senior]) == {1}
    assert set(winner[cc == 1]) == {3}
    assert set(winner[cc == 2]) == {0}
    assert set(most_specific(resolve_scopes([FIRST], SCHEMA), people, EMPLOYEE)[cc > 0]) == {-1}
    reqs = state.live_openings
    openings = most_specific(resolve_scopes([FIRST, TWO], SCHEMA), reqs, OPENING)
    assert openings.tolist() == [0, 0, 1, 1, -1, -1]
    assert reqs[:, OpeningColumn.COST_CENTER].astype(int).tolist() == [0, 0, 1, 1, 2, 2]


def test_equally_specific_scopes_that_can_meet_are_ambiguous() -> None:
    junior = PlannerScope.of(job_level=[1])
    with pytest.raises(PlannerError) as caught:
        resolve_scopes([FIRST, junior], SCHEMA)
    assert caught.value.problems == [
        "rules 1 (cost_center in ['CC-01']) and 2 (job_level in ['1']) are ambiguous: equally "
        "specific, they can select the same rows"
    ]
    # Equally specific but disjoint scopes are fine.
    assert len(resolve_scopes([FIRST, PlannerScope.of(cost_center=["CC-02"])], SCHEMA)) == 2


def test_a_scope_is_read_and_resolved_with_every_problem_named() -> None:
    assert PlannerScope.from_dict(FIRST_SENIOR.to_dict()) == FIRST_SENIOR
    assert PlannerScope.from_dict({}) == EVERYONE
    with pytest.raises(PlannerError, match="can't read scope"):
        PlannerScope.from_dict({"weather": ["rain"]})
    with pytest.raises(PlannerError) as caught:
        resolve_scopes(
            [PlannerScope.of(cost_center=["CC-09"]), FIRST, PlannerScope.of(job_level=[7])], SCHEMA
        )
    assert [p.split(":")[0] for p in caught.value.problems] == ["rule 1", "rule 3"]
