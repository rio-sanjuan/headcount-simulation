"""Tests for transfer matching: employee cells, availability, and cell draws."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.domains import StepContext
from hcsim.engine.eligibility import EligibilityRule
from hcsim.engine.known_events import EventKind, KnownEvent, KnownEvents, Source
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import resolve_plan
from hcsim.engine.seeds import SeedManager
from hcsim.engine.transfers import available, cells, match
from hcsim.model.core.layouts import EmployeeColumn
from tests.engine.test_loop import SCHEMA, config, initial_state

# The toy state: 30 employees per cost center, levels alternating 0 and 1, one family. Its
# requisition rows 0 and 1 are in CC-01, at levels 0 and 1.


def matching(slots: list[int], seed: int = 1, **overrides: Any) -> Any:
    state = initial_state()
    known = overrides.pop("known", None)
    ctx = StepContext(config(**overrides), 0, 0, SeedManager(seed), known=known)
    return match(state.live_openings, np.array(slots), state.live_employees, ctx), state


def test_cells_group_free_employees_by_family_and_level() -> None:
    employees = initial_state().live_employees
    free = np.ones(len(employees), dtype=bool)
    free[:4] = False
    by_cell = cells(employees, free)
    assert set(by_cell) == {(0, 0), (0, 1)}
    assert len(by_cell[(0, 0)]) + len(by_cell[(0, 1)]) == 86
    assert 0 not in by_cell[(0, 0)].tolist()


def test_each_opening_draws_from_its_cell_outside_its_own_cost_center() -> None:
    result, state = matching([0, 0, 0], transfers_level_band=0)
    assert result.matched.all()
    moved = state.live_employees[result.employees]
    assert (moved[:, EmployeeColumn.LEVEL] == 0).all()
    assert (moved[:, EmployeeColumn.COST_CENTER] != 0).all()
    assert len(set(result.employees.tolist())) == 3  # without replacement


def test_the_level_band_widens_the_cells_an_opening_reads() -> None:
    narrow, state = matching([0] * 40, transfers_level_band=0)
    assert int(narrow.matched.sum()) == 30  # level 0 in CC-02 and CC-03
    wide, _ = matching([0] * 40, transfers_level_band=1)
    assert wide.matched.all()
    assert set(state.live_employees[wide.employees, EmployeeColumn.LEVEL].tolist()) == {0, 1}


def test_only_the_available_share_of_a_cell_can_move() -> None:
    none, _ = matching([0, 1], transfers_availability_factor=0.0)
    assert not none.matched.any()
    state = initial_state()
    ctx = StepContext(config(transfers_availability_factor=0.5), 0, 0, SeedManager(3))
    share = available(state.live_employees, ctx).mean()
    assert 0.3 < share < 0.7


def test_eligibility_and_known_events_keep_people_out_of_a_cell() -> None:
    rule = EligibilityRule.of("transfers", cost_center=["CC-02"])
    result, state = matching([0] * 20, transfers_level_band=0, eligibility=(rule,))
    assert int(result.matched.sum()) == 15  # only CC-03's level-0 employees remain
    moved = state.live_employees[result.employees[result.matched]]
    assert (moved[:, EmployeeColumn.COST_CENTER] == 2).all()
    waiting = [KnownEvent(EventKind.TERMINATION, 3, Source.DECLARED, e) for e in range(30, 90, 2)]
    known = resolve_plan(sanitize(KnownEvents(tuple(waiting)), range(90), []), SCHEMA)
    blocked, _ = matching([0], transfers_level_band=0, known=known)
    assert not blocked.matched.any()


def test_a_negative_level_band_is_refused() -> None:
    with pytest.raises(ConfigError, match="transfers_level_band must be non-negative"):
        config(transfers_level_band=-1)
