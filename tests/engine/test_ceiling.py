"""Tests for the effective ceiling and the backfill rules that read it."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from hcsim.engine.ceiling import Ceiling, CeilingError
from hcsim.engine.config import ConfigError
from hcsim.engine.domains import ATTRITION, Departures, StepContext, backfill
from hcsim.engine.seeds import SeedManager
from hcsim.engine.settings import from_record, to_record
from hcsim.model.core.alignment import AlignmentTarget, Interpolation, marginal
from hcsim.model.core.layouts import FILL_COLUMN_MAP, OpeningColumn
from hcsim.model.ml.group import BackfillRateModel
from tests.engine.test_config_records import _saved
from tests.engine.test_config_state_seeds import awards
from tests.engine.test_config_state_seeds import config as small_config
from tests.engine.test_loop import RUN_DATE, SCHEMA, config, guidance, initial_state, models

ALWAYS = BackfillRateModel.constant(1.0, 3)


def plan(values: dict[str, list[float]], months: int = 6) -> AlignmentTarget:
    """Targets at the run's first and fourth months, held between."""
    dates = ["2026-01", "2026-04"]
    m = marginal("cost_center", list(values), dates, list(values.values()))
    return AlignmentTarget((m,), m.dates[0], months, months, Interpolation.HOLD_LAST)


def test_the_ceiling_is_the_tighter_of_cap_and_plan() -> None:
    ceiling = Ceiling.build(
        SCHEMA, {"CC-01": 40}, plan({"CC-01": [50.0, 30.0], "CC-02": [20.0, 25.0]})
    )
    assert ceiling.binds
    assert ceiling.at(RUN_DATE, 0).tolist() == [40.0, 20.0, np.inf]
    assert ceiling.at(RUN_DATE, 3).tolist() == [30.0, 25.0, np.inf]
    # Past the plan's window only the cap is left.
    assert ceiling.at(RUN_DATE, 9).tolist() == [40.0, np.inf, np.inf]
    room = ceiling.room(RUN_DATE, 0, np.array([35, 25, 5]), np.zeros(3, dtype=np.int64))
    assert room.tolist() == [5.0, 0.0, np.inf]
    assert not Ceiling.build(SCHEMA).binds


def test_caps_and_plans_that_don_t_fit_are_refused() -> None:
    with pytest.raises(CeilingError) as caught:
        Ceiling.build(SCHEMA, {"CC-09": 5, "CC-01": -1}, plan({"CC-07": [1.0, 1.0]}))
    assert caught.value.problems == [
        "the cap on CC-01 must be a non-negative whole number, not -1",
        "no cost center 'CC-09' to cap",
        "the plan targets cost centers the run lacks: ['CC-07']",
    ]
    family = marginal("job_family", ["F"], ["2026-01", "2026-04"], [[1.0, 1.0]])
    with pytest.raises(CeilingError, match="the plan has no cost-center targets"):
        Ceiling.build(SCHEMA, None, AlignmentTarget((family,), family.dates[0], 6, 6))
    with pytest.raises(ConfigError, match="ceiling: no cost center 'CC-09' to cap"):
        config(capacity_caps={"CC-09": 5})


def _backfill(**overrides: object) -> np.ndarray:
    state = initial_state()  # 30 heads and 4 openings per cost center
    cfg = config(models=dataclasses.replace(models(), backfill=ALWAYS), **overrides)
    ctx = StepContext(cfg, 0, 0, SeedManager(9))
    rows = state.live_employees[[0, 1, 2, 30, 31, 60]].copy()
    flags = np.array([False, False, True, False, False, False])
    ctx.departures.append(Departures(ATTRITION, rows, flags))
    opened: np.ndarray = backfill(state, ctx).openings.append
    return opened


def test_known_terminations_open_no_backfill_by_default() -> None:
    assert _backfill()[:, OpeningColumn.COST_CENTER].tolist() == [0, 0, 1, 1, 2]
    allowed = _backfill(backfill_anticipated_terminations=True)
    assert allowed[:, OpeningColumn.COST_CENTER].tolist() == [0, 0, 0, 1, 1, 2]


def test_no_cost_center_opens_past_its_ceiling() -> None:
    # CC-01 has room for one more opening (30 + 4 + 1 = 35); CC-02 has none.
    opened = _backfill(capacity_caps={"CC-01": 35, "CC-02": 34})
    assert opened[:, OpeningColumn.COST_CENTER].tolist() == [0, 2]


def test_a_backfill_copies_the_role_verbatim() -> None:
    state = initial_state()
    departed = state.live_employees[[7]].copy()
    departed[0, list(FILL_COLUMN_MAP.values())] = [1, 0, 0, 0, 2, 0, 1]
    cfg = config(models=dataclasses.replace(models(), backfill=ALWAYS))
    ctx = StepContext(cfg, 0, 0, SeedManager(1))
    ctx.departures.append(Departures(ATTRITION, departed, np.zeros(1, dtype=bool)))
    (row,) = backfill(state, ctx).openings.append
    for opening_column, employee_column in FILL_COLUMN_MAP.items():
        assert row[opening_column] == departed[0, employee_column]


def test_caps_and_the_plan_go_in_the_record(tmp_path: Path) -> None:
    target = plan({"CC-01": [2.0, 3.0]})
    built = small_config(models=_saved(tmp_path), capacity_caps={"CC-01": 9})
    record = json.loads(json.dumps(to_record(built)))
    assert record["capacity_caps"] == {"CC-01": 9}
    assert record["plan"] is None
    rebuilt = from_record(record, schema=built.schema, models_root=tmp_path, awards=awards())
    assert dict(rebuilt.capacity_caps) == {"CC-01": 9}
    with pytest.raises(ConfigError, match="the record names plan None, but the call passes at1-"):
        from_record(record, schema=built.schema, models_root=tmp_path, awards=awards(), plan=target)


def test_the_ledger_counts_only_backfills_the_ceiling_lets_through() -> None:
    # Guidance allows two backfills in CC-01, but its ceiling leaves room for one.
    opened = _backfill(
        guidance=guidance({"requisitions_opened": {"CC-01": 2.0}}),
        capacity_caps={"CC-01": 35},
        backfill_anticipated_terminations=True,
    )
    assert opened[:, OpeningColumn.COST_CENTER].tolist().count(0) == 1
    state = initial_state()
    cfg = config(
        models=dataclasses.replace(models(), backfill=ALWAYS),
        guidance=guidance({"requisitions_opened": {"CC-01": 2.0}}),
        capacity_caps={"CC-01": 35},
    )
    ctx = StepContext(cfg, 0, 0, SeedManager(9))
    rows = state.live_employees[[0, 1, 2]].copy()
    ctx.departures.append(Departures(ATTRITION, rows, np.zeros(3, dtype=bool)))
    assert backfill(state, ctx).ledger["requisitions_opened"].tolist() == [1]
