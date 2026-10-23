"""Tests for SBC's own steps: the price step, and the annual and anticipated award domains."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hcsim.engine.apply import apply
from hcsim.engine.awards import (
    ANNUAL_AWARD,
    ANTICIPATED_AWARD,
    annual_award,
    anticipated_award,
    trajectory_price,
)
from hcsim.engine.config import AwardSettings, ConfigError
from hcsim.engine.domains import StepContext
from hcsim.engine.loop import SimulationResult, simulate
from hcsim.engine.results import DomainResult, Grant
from hcsim.engine.seeds import SeedManager
from hcsim.financial.calculator import SizingError
from hcsim.financial.prices import GBMPrice, StaticPrice
from hcsim.financial.routing import (
    RECORDED_COLUMNS,
    AnticipatedAwards,
    Overridden,
    RecordedAwards,
)
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.core.schema import AWARD_TYPE
from tests.engine.test_grant_sizing import EVERY, sizer
from tests.engine.test_loop import awards, config, initial_state, models

MARCH = 3  # the run starts in January, so March is step 2
ANNUAL_STEP = 2


def anticipated(rows: list[tuple[int, int, str]]) -> AnticipatedAwards:
    return AnticipatedAwards.from_frame(
        pd.DataFrame(rows, columns=["employee_id", "step", "award_type"])
    )


def settings(**overrides: Any) -> AwardSettings:
    return dataclasses.replace(awards(), **{"sizers": {"annual": sizer("annual")}, **overrides})


def simulated(**overrides: Any) -> SimulationResult:
    cfg = config(schema=EVERY, models=models(attrition=0.0), awards=settings(**overrides))
    return simulate(cfg, initial_state(), validate=True)


# --- the price step -----------------------------------------------------------------------------


def test_each_trajectory_walks_its_own_price_path_and_a_seeded_run_walks_them_again() -> None:
    gbm = GBMPrice(100.0, drift=0.05, volatility=0.4)
    paths = [t.prices for t in simulated(price_model=gbm).trajectories]
    assert not np.array_equal(paths[0], paths[1])
    again = [t.prices for t in simulated(price_model=gbm).trajectories]
    assert all(np.array_equal(a, b) for a, b in zip(paths, again, strict=True))
    # A model that doesn't draw is copied as it is, so every trajectory has its path.
    static = StaticPrice(80.0)
    assert trajectory_price(static, SeedManager(1), 0) == static
    assert trajectory_price(static, SeedManager(1), 0) is not static


# --- the annual award ---------------------------------------------------------------------------


def context(step: int, **overrides: Any) -> StepContext:
    cfg = config(schema=EVERY, awards=settings(**overrides))
    return StepContext(cfg, step, 0, SeedManager(1))


def test_the_annual_award_lists_the_whole_workforce_in_its_month_alone() -> None:
    state = initial_state()
    (grant,) = annual_award(state, context(ANNUAL_STEP, annual_month=MARCH)).grants
    assert (grant.award_type, grant.rows) == ("annual", "listed")
    assert grant.index is not None
    assert grant.index.tolist() == list(range(state.n_employees))
    assert annual_award(state, context(ANNUAL_STEP + 1, annual_month=MARCH)).grants == ()
    assert annual_award(state, context(ANNUAL_STEP)).grants == ()  # no annual month


def test_a_run_grants_its_annual_award_once_a_year_to_the_workforce() -> None:
    result = simulated(annual_month=MARCH)
    code = EVERY.vocabulary(AWARD_TYPE).code("annual")
    for t in result.trajectories:
        assert t.tranches is not None
        annual = t.tranches.award_type == code
        assert set(t.tranches.award_month[annual].tolist()) == {ANNUAL_STEP}
        holders = set(t.tranches.holder[annual].tolist())
        assert set(range(90)) <= holders  # the seeded workforce, none of whom left
        assert len(holders) <= int(t.headcount[ANNUAL_STEP].sum())
        assert t.unsized_grants[ANNUAL_AWARD] == 0
    assert ANNUAL_AWARD not in result.unregistered


# --- anticipated awards -------------------------------------------------------------------------


def test_anticipated_awards_are_read_with_every_problem_named() -> None:
    awards_ = anticipated([(7, 3, "retention"), (5, 3, "retention"), (7, 4, "off_cycle")])
    assert len(awards_) == 3
    assert awards_.at(3)["retention"].tolist() == [5, 7]
    assert dict(awards_.at(9)) == {}
    assert awards_.award_types == {"retention", "off_cycle"}
    assert awards_.last_step == 4
    assert AnticipatedAwards().last_step == -1
    with pytest.raises(SizingError) as caught:
        anticipated([(7, -1, "retention"), (5, 3, "retention"), (5, 3, "retention")])
    assert str(caught.value) == (
        "1 anticipated awards fall before the run's first step; "
        "1 anticipated awards name an employee again for one type at one step"
    )
    with pytest.raises(SizingError, match=r"need the columns \['award_type'\]"):
        AnticipatedAwards.from_frame(pd.DataFrame({"employee_id": [1], "step": [0]}))


def test_an_anticipated_award_lists_its_employee_if_theyre_in_the_workforce() -> None:
    state = initial_state()
    named = anticipated([(7, 3, "retention"), (999, 3, "retention"), (4, 3, "off_cycle")])
    result = anticipated_award(state, context(3, anticipated=named))
    assert [(g.award_type, g.rows) for g in result.grants] == [
        ("off_cycle", "listed"),
        ("retention", "listed"),
    ]
    assert [g.index.tolist() for g in result.grants if g.index is not None] == [[4], [7]]
    assert anticipated_award(state, context(2, anticipated=named)).grants == ()
    assert anticipated_award(state, context(3)).grants == ()


def test_a_run_grants_each_anticipated_award_at_its_step() -> None:
    retention = Overridden(sizer("retention"), shares={7: [12, 0, 0, 0]})
    named = anticipated([(7, 3, "retention"), (999, 3, "retention")])
    result = simulated(anticipated=named, sizers={"retention": retention})
    code = EVERY.vocabulary(AWARD_TYPE).code("retention")
    for t in result.trajectories:
        assert t.tranches is not None
        mine = t.tranches.award_type == code
        assert t.tranches.holder[mine].tolist() == [7]  # employee 7's vest slot is 7
        assert t.tranches.award_month[mine].tolist() == [3]
        assert t.tranches.shares[mine].tolist() == [12.0]
    assert ANTICIPATED_AWARD not in result.unregistered


def test_each_trajectory_grants_a_lagged_recorded_award_in_its_own_month() -> None:
    # Employee 7's retention award is anticipated at step 3, and recorded a month later.
    history = RecordedAwards.from_frame(
        pd.DataFrame([(7, "retention", 4, 24, 12)], columns=list(RECORDED_COLUMNS))
    )
    named = anticipated([(7, 3, "retention")])
    result = simulated(anticipated=named, replayed=history, replay_lag={"retention": 1})
    code = EVERY.vocabulary(AWARD_TYPE).code("retention")
    assert len(result.trajectories) > 1
    for t in result.trajectories:
        assert t.tranches is not None
        mine = t.tranches.award_type == code
        assert t.tranches.award_month[mine].tolist() == [4]
        assert t.tranches.shares[mine].tolist() == [12.0]
        assert t.tranches.recorded[mine].all()


def test_award_settings_check_the_annual_month_and_the_anticipated_awards() -> None:
    late = anticipated([(7, 6, "retention"), (7, 2, "bonus")])
    lag = {"bonus": 1, "retention": -1}
    with pytest.raises(ConfigError) as caught:
        config(schema=EVERY, awards=settings(annual_month=13, anticipated=late, replay_lag=lag))
    assert caught.value.problems == [
        "the annual award's month is a calendar month, 1 to 12, not 13",
        "replay lags for award types the schema doesn't know: ['bonus']",
        "a replay lag is a number of months from 0, which ['retention'] isn't",
        "anticipated awards of types the schema doesn't know: ['bonus']",
        "an anticipated award at step 6 lies past the run's last, 5",
    ]


# --- listed grants ------------------------------------------------------------------------------


def test_a_listed_grant_sizes_the_rows_it_lists() -> None:
    state = initial_state()
    state.ledger, state.price = HolderLedger(), StaticPrice(100.0)
    listed = Grant("annual", rows="listed", index=np.array([1, 4]))
    applied = apply(state, DomainResult("test", grants=(listed,)), 0, EVERY, settings())
    assert applied.recorded_tranches > 0
    slots = state.live_employees[[1, 4], EmployeeColumn.VEST_INDEX].astype(int)
    assert set(state.ledger.tranches().holder.tolist()) == set(slots.tolist())
    assert listed == Grant("annual", rows="listed", index=np.array([2]))  # by type and rows
    with pytest.raises(ValueError, match="exactly when its rows are 'listed'"):
        Grant("annual", rows="listed")
    with pytest.raises(ValueError, match="exactly when its rows are 'listed'"):
        Grant("annual", index=np.array([0]))
