"""SBC's own steps in a month: the price step, and the annual and anticipated award domains.

**The price step** opens every month of a run that computes SBC, before any domain. The run's
price model advances to the month, and the price it reaches is the month's: the run records it,
and every grant sized that month reads it. Each trajectory has its own copy of the model
(`trajectory_price`). One that draws, as `GBMPrice` does, is reseeded from the trajectory's
`price` stream, so each trajectory walks its own path, and a seeded run walks the same ones
again.

**The annual award** (`annual_award`) grants the whole workforce in the award settings'
`annual_month`. It runs after the month's attrition, so the departed get none, and after its
promotions, so the promoted are sized at their new level. The annual eligibility cutoff
(`AwardSettings.eligibility`) decides who's hired early enough, and the annual sizer how much
each gets.

**The anticipated award** (`anticipated_award`) grants the awards a planner knows of in advance
(`AnticipatedAwards`), each to its named employee at its step, if they're in the workforce then.
One whose employee isn't, having left or not yet started, isn't granted. It's routed and sized
like any grant of its type, so an individual override (`routing.Overridden`) can set its shares
outright.

Neither domain changes a row: each lists the rows its grants go to, and the apply step sizes and
records them. Each is registered only in a run that computes SBC and configures it.
"""

from __future__ import annotations

import copy
from typing import Final

import numpy as np

from hcsim.engine.domains import StepContext
from hcsim.engine.results import DomainResult, Grant
from hcsim.engine.seeds import SeedManager
from hcsim.engine.state import SimulationState
from hcsim.financial.prices import StockPriceModel
from hcsim.model.core.layouts import EmployeeColumn

ANNUAL_AWARD: Final = "annual_award"
ANTICIPATED_AWARD: Final = "anticipated_award"
#: The price step's random stream.
PRICE: Final = "price"


def trajectory_price(
    model: StockPriceModel, seeds: SeedManager, trajectory: int
) -> StockPriceModel:
    """The trajectory's own copy of the run's price model, reseeded from its `price` stream
    when the model draws."""
    reseeded = getattr(model, "reseeded", None)
    if reseeded is None:
        return copy.deepcopy(model)
    own: StockPriceModel = reseeded(int(seeds.generator(trajectory, PRICE).integers(2**63)))
    return own


def annual_award(state: SimulationState, ctx: StepContext) -> DomainResult:
    awards = ctx.config.awards
    if awards is None or ctx.month != awards.annual_month:
        return DomainResult(ANNUAL_AWARD)
    index = np.arange(state.n_employees, dtype=np.int64)
    return DomainResult(ANNUAL_AWARD, grants=(Grant("annual", rows="listed", index=index),))


def anticipated_award(state: SimulationState, ctx: StepContext) -> DomainResult:
    awards = ctx.config.awards
    if awards is None or awards.anticipated is None:
        return DomainResult(ANTICIPATED_AWARD)
    ids = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
    grants = tuple(
        Grant(award_type, rows="listed", index=np.flatnonzero(np.isin(ids, named)))
        for award_type, named in sorted(awards.anticipated.at(ctx.step).items())
    )
    return DomainResult(ANTICIPATED_AWARD, grants=grants)
