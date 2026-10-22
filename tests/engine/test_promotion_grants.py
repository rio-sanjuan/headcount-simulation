"""Tests for a promotion's family change and its grant."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from hcsim.engine.apply import apply
from hcsim.engine.config import AwardSettings, ConfigError
from hcsim.engine.domains import StepContext
from hcsim.engine.promotion import PROMOTION, promotion
from hcsim.engine.results import DomainResult, Grant, RowChange
from hcsim.engine.seeds import SeedManager
from hcsim.financial.prices import StaticPrice
from hcsim.financial.sizing import NewHireGapFill
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn
from hcsim.model.core.schema import AWARD_TYPE, JOB_FAMILY, SimulationSchema
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.markov import SimilarityMatrix
from tests.engine.test_loop import SCHEMA, awards, config, initial_state, models

MARCH = 2
#: Two families, F and G, each the other's only neighbour.
WIDE = SimulationSchema.build(
    {**{n: list(v.labels) for n, v in SCHEMA.vocabularies.items()}, JOB_FAMILY: ["F", "G"]}
)
NEIGHBOURS = SimilarityMatrix(2).fit(np.array([0]), np.array([1]))


def promoting(**overrides: Any) -> Any:
    built = dataclasses.replace(
        models(attrition=0.0), promotion=ConstantModel(1.0), family_similarity=NEIGHBOURS
    )
    return config(schema=WIDE, models=built, awards=None, forecast_scope="hcf", **overrides)


def promote(cfg: Any, trajectory: int = 0) -> DomainResult:
    return promotion(initial_state(), StepContext(cfg, MARCH, trajectory, SeedManager(6)))


def test_a_promotion_can_move_to_a_similar_family() -> None:
    every = promote(promoting(promotion_family_change_probability=1.0))
    assert (every.employees.write_rows[:, EmployeeColumn.JOB_FAMILY] == 1).all()
    none = promote(promoting())
    assert (none.employees.write_rows[:, EmployeeColumn.JOB_FAMILY] == 0).all()
    half = promote(promoting(promotion_family_change_probability=0.5))
    moved = int((half.employees.write_rows[:, EmployeeColumn.JOB_FAMILY] == 1).sum())
    assert 10 < moved < 35
    again = promote(promoting(promotion_family_change_probability=0.5))
    assert np.array_equal(half.employees.write_rows, again.employees.write_rows)


def test_a_family_with_no_neighbour_stays_put() -> None:
    lonely = SimilarityMatrix(2).fit(np.array([1]), np.array([1]))
    cfg = promoting(promotion_family_change_probability=1.0)
    cfg = dataclasses.replace(cfg, models=dataclasses.replace(cfg.models, family_similarity=lonely))
    assert (promote(cfg).employees.write_rows[:, EmployeeColumn.JOB_FAMILY] == 0).all()


def test_a_family_change_needs_a_similarity_model() -> None:
    plain = dataclasses.replace(models(), promotion=ConstantModel(1.0))
    with pytest.raises(ConfigError) as caught:
        config(models=plain, promotion_family_change_probability=0.2)
    assert caught.value.problems == [
        "a promotion that changes family needs a family_similarity model"
    ]
    with pytest.raises(ConfigError, match="must lie in"):
        config(promotion_family_change_probability=1.5)


def test_every_promotion_emits_a_grant_on_its_written_rows() -> None:
    result = promote(promoting())
    assert result.grants == (Grant("promotion", rows="written"),)


def _promoted() -> DomainResult:
    state = initial_state()
    rows = state.live_employees[[0, 2]].copy()
    rows[:, EmployeeColumn.LEVEL] = 1
    return DomainResult(
        PROMOTION,
        employees=RowChange(EMPLOYEE, write_index=np.array([0, 2]), write_rows=rows),
        grants=(Grant("promotion", rows="written"),),
    )


def test_a_promotion_grant_is_sized_when_the_run_has_a_sizer() -> None:
    schema = SimulationSchema.build(
        {
            **{n: list(v.labels) for n, v in SCHEMA.vocabularies.items()},
            AWARD_TYPE: ["new_hire", "promotion"],
        }
    )
    base = awards()
    sizer = dataclasses.replace(base.new_hire, award_type="promotion")
    settings = dataclasses.replace(base, promotion=sizer)
    state = initial_state()
    state.ledger, state.price = HolderLedger(), StaticPrice(100.0)
    applied = apply(state, _promoted(), 0, schema, settings)
    assert applied.recorded_tranches > 0
    assert applied.unsized_grants == 0
    assert (state.employees[[0, 2], EmployeeColumn.BASE_SALARY] > 0).all()
    assert set(state.ledger.tranches().award_type.tolist()) == {
        schema.vocabulary(AWARD_TYPE).code("promotion")
    }


def test_without_a_sizer_a_promotion_grant_is_counted_unsized() -> None:
    state = initial_state()
    state.ledger, state.price = HolderLedger(), StaticPrice(100.0)
    applied = apply(state, _promoted(), 0, SCHEMA, awards())
    assert (applied.recorded_tranches, applied.unsized_grants) == (0, 2)


def test_a_promotion_sizer_must_grant_promotions_the_schema_knows() -> None:
    base = awards()
    wrong = dataclasses.replace(base, promotion=base.new_hire)
    with pytest.raises(ConfigError, match="the promotion sizer grants 'new_hire'"):
        config(awards=wrong)
    sizer = dataclasses.replace(base.new_hire, award_type="promotion")
    unknown = AwardSettings(base.new_hire, base.price_model, promotion=sizer)
    with pytest.raises(ConfigError, match="needs 'promotion' among the award types"):
        config(awards=unknown)
    assert isinstance(sizer, NewHireGapFill)


def test_a_promotion_sizer_the_schema_knows_is_accepted() -> None:
    schema = SimulationSchema.build(
        {
            **{n: list(v.labels) for n, v in SCHEMA.vocabularies.items()},
            AWARD_TYPE: ["new_hire", "promotion"],
        }
    )
    base = awards()
    sizer = dataclasses.replace(base.new_hire, award_type="promotion")
    built = config(schema=schema, awards=dataclasses.replace(base, promotion=sizer))
    assert built.awards is not None
    assert built.awards.sizer("promotion") is sizer
    assert built.awards.sizer("bonus") is None


def test_an_executive_level_above_every_level_excludes_no_one() -> None:
    built = config(attrition_executive_min_level=9)
    assert built.resolved_eligibility.for_domain("attrition") == []
