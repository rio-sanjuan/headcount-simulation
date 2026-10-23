"""Tests for grant sizing in the apply step, by award type, and for the run's forecast scope."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hcsim.engine.apply import apply
from hcsim.engine.config import AwardSettings, ConfigError, ForecastScope
from hcsim.engine.results import DomainResult, Grant, RowChange
from hcsim.financial.prices import StaticPrice
from hcsim.financial.sizing import NewHireGapFill
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.layouts import EMPLOYEE, EmployeeColumn
from hcsim.model.core.schema import AWARD_TYPE, SimulationSchema
from tests.engine.test_loop import SCHEMA, awards, config, initial_state, run

#: Every award type the data contract names.
EVERY = SimulationSchema.build(
    {
        **{n: list(v.labels) for n, v in SCHEMA.vocabularies.items()},
        AWARD_TYPE: ["new_hire", "annual", "promotion", "transfer", "retention", "off_cycle"],
    }
)


def sizer(award_type: str) -> NewHireGapFill:
    base = awards().new_hire
    assert isinstance(base, NewHireGapFill)
    return dataclasses.replace(base, award_type=award_type)


def written(index: list[int], *grants: Grant) -> DomainResult:
    """Employees `index` written unchanged, with `grants` on them."""
    rows = initial_state().live_employees[index].copy()
    return DomainResult(
        "test",
        employees=RowChange(EMPLOYEE, write_index=np.array(index), write_rows=rows),
        grants=grants,
    )


def test_each_award_type_is_sized_by_its_own_sizer() -> None:
    settings = dataclasses.replace(
        awards(), sizers={"transfer": sizer("transfer"), "retention": sizer("retention")}
    )
    state = initial_state()
    state.ledger, state.price = HolderLedger(), StaticPrice(100.0)
    transfer = apply(state, written([0, 2], Grant("transfer", rows="written")), 0, EVERY, settings)
    retention = apply(
        state,
        written([1, 3], Grant("retention", rows="written"), Grant("off_cycle", rows="written")),
        0,
        EVERY,
        settings,
    )
    assert transfer.unsized_grants == 0
    assert retention.unsized_grants == 2  # the off-cycle grants: the run has no sizer for them
    code = EVERY.vocabulary(AWARD_TYPE).code
    t = state.ledger.tranches()
    slots = state.live_employees[:, EmployeeColumn.VEST_INDEX].astype(int)
    for award_type, index in (("transfer", [0, 2]), ("retention", [1, 3])):
        holders = t.holder[t.award_type == code(award_type)]
        assert set(holders.tolist()) == set(slots[index].tolist())
    assert code("off_cycle") not in set(t.award_type.tolist())


def test_a_sizer_grants_the_type_its_given_for_one_the_schema_knows_and_only_once() -> None:
    base = awards()
    with pytest.raises(ConfigError) as caught:
        config(
            schema=EVERY,
            awards=dataclasses.replace(
                base,
                promotion=sizer("promotion"),
                sizers={
                    "new_hire": sizer("new_hire"),
                    "promotion": sizer("promotion"),
                    "annual": sizer("retention"),
                },
            ),
        )
    assert caught.value.problems == [
        "new_hire grants have two sizers",
        "promotion grants have two sizers",
        "the annual sizer grants 'retention', not 'annual'",
    ]
    with pytest.raises(ConfigError, match="sizing transfer grants needs 'transfer' among"):
        config(awards=dataclasses.replace(base, sizers={"transfer": sizer("transfer")}))
    given = AwardSettings(base.new_hire, base.price_model, sizers={"promotion": sizer("promotion")})
    assert config(schema=EVERY, awards=given).awards == given
    assert given.sizer("promotion") is given.sizers["promotion"]
    assert given.sizer("annual") is None


def test_the_forecast_scope_never_changes_the_headcount() -> None:
    # Sizing, award timing, and prices draw on their own streams, so a run's headcount is the
    # same, bit for bit, whether or not it computes SBC.
    with_sbc = run()
    headcount = run(awards=None, forecast_scope=ForecastScope.HEADCOUNT)
    assert any(t.tranches is not None and len(t.tranches) for t in with_sbc.trajectories)
    for a, b in zip(with_sbc.trajectories, headcount.trajectories, strict=True):
        assert np.array_equal(a.headcount, b.headcount)
        assert np.array_equal(a.open_openings, b.open_openings)
        assert all(np.array_equal(a.flows[k], b.flows[k]) for k in a.flows)
        assert b.tranches is None
        assert b.unsized_grants == dict.fromkeys(b.unsized_grants, 0)
