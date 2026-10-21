"""Tests for the configuration, the trajectory state, and the seed manager."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from hcsim.engine.config import (
    AwardSettings,
    ConfigError,
    DomainModels,
    ForecastScope,
    SimulationConfig,
)
from hcsim.engine.seeds import SeedManager, stable_hash
from hcsim.engine.state import BufferCapacityError, SimulationState
from hcsim.financial.prices import StaticPrice
from hcsim.financial.ranges import RangeTable
from hcsim.financial.sizing import NewHireGapFill
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import (
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    SimulationSchema,
)
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.generative import EmpiricalAttributeGenerator
from hcsim.model.ml.group import BackfillRateModel

SCHEMA = SimulationSchema.build(
    {COST_CENTER: ["CC-01", "CC-02"], JOB_FAMILY: ["F"], JOB_LEVEL: [1], LOCATION: ["L"]}
)


def models() -> DomainModels:
    generator = EmpiricalAttributeGenerator(min_rows=1).fit(
        np.zeros(1, dtype=np.int64), np.zeros((1, 6), dtype=np.int64)
    )
    return DomainModels(
        ConstantModel(0.01),
        ConstantModel(0.2),
        ConstantModel(0.05),
        BackfillRateModel.constant(0.5, 2),
        BackfillRateModel.constant(0.3, 2),
        generator,
    )


def awards() -> AwardSettings:
    ranges = pd.DataFrame(
        [
            {
                "job_family": "F",
                "job_level": 1,
                "location": "L",
                "min_base": 1,
                "max_base": 2,
                "min_target": 3,
                "max_target": 4,
                "base_adj": 1.0,
            }
        ]
    )
    return AwardSettings(NewHireGapFill(RangeTable.from_frame(ranges, SCHEMA)), StaticPrice(10.0))


def guidance(**kw: object) -> GuidanceForecast:
    frame = pd.DataFrame(
        {"forecast_month": [0], "category": ["attrition"], "value": [1.0], "cost_center": ["CC-01"]}
    )
    args: dict[str, object] = {"snapshot_date": "2025-12-31", "horizon": 12} | kw
    return GuidanceForecast(
        frame, COST_CENTER, str(args["snapshot_date"]), int(str(args["horizon"]))
    )


def config(**kw: object) -> SimulationConfig:
    base: dict[str, object] = {
        "run_date": dt.date(2026, 1, 1),
        "schema": SCHEMA,
        "models": models(),
        "awards": awards(),
    }
    return SimulationConfig(**(base | kw))  # type: ignore[arg-type]


# --- configuration ------------------------------------------------------------------------------


def test_a_valid_configuration_is_frozen() -> None:
    c = config()
    assert c.computes_sbc
    with pytest.raises(AttributeError):
        c.simulation_months = 3  # type: ignore[misc]


def test_every_problem_is_reported_at_once() -> None:
    with pytest.raises(ConfigError) as error:
        config(run_date=dt.date(2026, 1, 15), simulation_months=0, num_trajectories=0, awards=None)
    assert len(error.value.problems) == 4
    assert "first of a month" in str(error.value)
    assert "needs awards settings" in str(error.value)


def test_headcount_scope_refuses_awards_and_skips_sbc() -> None:
    with pytest.raises(ConfigError, match="refuses awards"):
        config(forecast_scope=ForecastScope.HEADCOUNT)
    assert not config(forecast_scope="hcf", awards=None).computes_sbc


def test_guidance_must_cover_the_run_start_at_its_origin_and_fit_the_schema() -> None:
    assert config(guidance=guidance()).aligner is not None
    with pytest.raises(ConfigError) as error:
        config(guidance=guidance(horizon=6, snapshot_date="2025-11-30"))
    assert "covers 6 months" in str(error.value)
    assert "snapshot of 2025-12-31" in str(error.value)
    other = GuidanceForecast(
        pd.DataFrame(
            {
                "forecast_month": [0],
                "category": ["attrition"],
                "value": [1.0],
                "cost_center": ["CC-99"],
            }
        ),
        COST_CENTER,
        "2025-12-31",
        12,
    )
    with pytest.raises(ConfigError, match="doesn't fit the schema"):
        config(guidance=other)


def test_guided_requisition_generation_needs_cost_center_grain() -> None:
    frame = pd.DataFrame(
        {
            "forecast_month": [0],
            "category": ["requisitions_opened"],
            "value": [1.0],
            "TOPLINE": ["ALL"],
        }
    )
    with pytest.raises(ConfigError, match="cost-center grain"):
        config(guidance=GuidanceForecast(frame, "TOPLINE", "2025-12-31", 12))


# --- state --------------------------------------------------------------------------------------


def state(employees: int = 3, openings: int = 2) -> SimulationState:
    e, o = EMPLOYEE.empty(5), OPENING.empty(4)
    e[:employees, EmployeeColumn.COST_CENTER] = [0, 1, 1][:employees]
    o[:openings, OpeningColumn.COST_CENTER] = [1, 1][:openings]
    o[:openings, OpeningColumn.NUM_OPENINGS] = [1, 3][:openings]
    return SimulationState(e, employees, o, openings, n_cost_centers=2)


def test_counters_are_built_from_the_live_rows() -> None:
    s = state()
    assert s.headcount.tolist() == [1, 2]
    assert s.open_openings.tolist() == [0, 4]  # openings, not rows
    assert s.open_rows.tolist() == [0, 2]
    assert s.counters_agree()


def test_a_drifted_counter_is_detected() -> None:
    s = state()
    s.headcount[0] += 1
    assert not s.counters_agree()


def test_clones_are_independent() -> None:
    s = state()
    clone = s.clone()
    clone.employees[0, EmployeeColumn.COST_CENTER] = 1
    clone.headcount[0] = 9
    assert s.employees[0, EmployeeColumn.COST_CENTER] == 0
    assert s.headcount[0] == 1


def test_reserving_past_capacity_names_the_setting() -> None:
    s = state()
    s.reserve("employee", 2)
    with pytest.raises(BufferCapacityError, match="Raise the employee capacity"):
        s.reserve("employee", 3)
    with pytest.raises(BufferCapacityError, match="opening buffer holds 4"):
        s.reserve("opening", 3)


def test_buffers_must_use_the_declared_layouts() -> None:
    with pytest.raises(ValueError, match="declared layouts"):
        SimulationState(np.zeros((2, 3)), 0, OPENING.empty(1), 0, 1)


# --- seeds --------------------------------------------------------------------------------------


def test_streams_depend_on_seed_subsystem_and_trajectory_only() -> None:
    a = SeedManager(42)
    first = a.generator(3, "attrition").random(4)
    b = SeedManager(42)
    b.generator(0, "attrition").random(10)  # other trajectories and subsystems don't interfere
    b.generator(3, "fills").random(10)
    assert (b.generator(3, "attrition").random(4) == first).all()
    assert not (SeedManager(43).generator(3, "attrition").random(4) == first).all()


def test_a_generator_advances_from_step_to_step() -> None:
    seeds = SeedManager(1)
    first, second = seeds.generator(0, "x").random(), seeds.generator(0, "x").random()
    assert first != second
    assert seeds.reproducible
    assert not SeedManager(None).reproducible


def test_the_subsystem_hash_is_stable_across_processes() -> None:
    assert stable_hash("attrition") == stable_hash("attrition")
    assert stable_hash("attrition") != stable_hash("fills")
    assert 0 <= stable_hash("attrition") < 2**63
