"""The walking skeleton's domain handlers: attrition, backfill, external fill, cancellation,
and growth.

Each handler is pure: it reads settled state and the step's context, and returns a
`DomainResult`. Entity models score buffer rows as they are. Where the guidance forecast covers
a group, a domain realizes the group's quota exactly; elsewhere every row draws independently.

Growth and backfill share the `requisitions_opened` quota through the category ledger. Backfill
runs first and is capped at the quota's headroom, and growth opens exactly the residual. So
where guidance covers a cost center, the requisitions it opens in a month equal the quota.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.config import SimulationConfig
from hcsim.engine.results import DomainResult, Grant, QuotaRecord, RowChange
from hcsim.engine.seeds import SeedManager
from hcsim.engine.state import SimulationState
from hcsim.model.core.aligner import GroupAligner
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.core.layouts import (
    EMPLOYEE,
    FILL_COLUMN_MAP,
    OPENING,
    Buffer,
    BufferLayout,
    EmployeeColumn,
    OpeningColumn,
    copy_role,
)
from hcsim.model.core.schema import REQ_TYPE, UNATTRIBUTED
from hcsim.model.interface import GroupRateModel, Mask, Probabilities
from hcsim.model.ml.generative import ATTRIBUTES
from hcsim.model.ml.sampling import apportion, bernoulli, quota_select

Ints = NDArray[np.int64]

ATTRITION: Final = "attrition"
BACKFILL: Final = "backfill"
EXTERNAL_FILL: Final = "external_fill"
CANCELLATION: Final = "cancellation"
GROWTH: Final = "growth"

_OPENED = GuidanceCategory.REQUISITIONS_OPENED


@dataclass
class StepContext:
    """What a handler may read besides state: the configuration, the step, the random streams,
    and the departures earlier domains reported this step."""

    config: SimulationConfig
    step: int
    trajectory: int
    seeds: SeedManager
    departures: list[Buffer] = field(default_factory=list)
    _quotas: dict[GuidanceCategory, Ints] = field(default_factory=dict)

    def rng(self, domain: str) -> np.random.Generator:
        return self.seeds.generator(self.trajectory, domain)

    def guided(self, category: GuidanceCategory) -> bool:
        return self.config.guidance is not None and self.config.guidance.covers(category)

    def _guide(self) -> tuple[GuidanceForecast, GroupAligner]:
        guidance, aligner = self.config.guidance, self.config.aligner
        if guidance is None or aligner is None:
            raise RuntimeError("a guided draw needs guidance in the configuration")
        return guidance, aligner

    def quota(self, category: GuidanceCategory) -> Ints:
        """This step's integer quota per guidance group, apportioned once per step."""
        if category not in self._quotas:
            guidance, _ = self._guide()
            self._quotas[category] = apportion(guidance.get_quota(self.step, category))
        return self._quotas[category]

    def guided_groups(self, rows: Buffer, layout: BufferLayout, category: GuidanceCategory) -> Ints:
        """Each row's guidance group, or UNATTRIBUTED where no series covers the row."""
        return self._covered(self._guide()[1].groups(rows, layout), category)

    def guided_groups_of_codes(self, codes: Ints, category: GuidanceCategory) -> Ints:
        return self._covered(self._guide()[1].groups_of_codes(codes), category)

    def _covered(self, groups: Ints, category: GuidanceCategory) -> Ints:
        cover = self._guide()[0].group_coverage(category)
        keep = (groups >= 0) & cover[np.maximum(groups, 0)]
        return np.where(keep, groups, UNATTRIBUTED)


Handler = Callable[[SimulationState, StepContext], DomainResult]


def _draw(
    p: Probabilities,
    rows: Buffer,
    layout: BufferLayout,
    category: GuidanceCategory,
    ctx: StepContext,
    rng: np.random.Generator,
) -> tuple[Mask, dict[str, QuotaRecord]]:
    """Quota sampling for rows in covered groups, independent draws for the rest."""
    independent = bernoulli(p, rng)
    if not ctx.guided(category):
        return independent, {}
    groups = ctx.guided_groups(rows, layout, category)
    selection = quota_select(p, groups, ctx.quota(category), rng)
    mask = np.where(groups >= 0, selection.mask, independent)
    record = QuotaRecord(selection.quota, selection.eligible, selection.realized)
    return mask, {category.value: record}


def _per_cost_center_rates(model: GroupRateModel, n: int) -> NDArray[np.float64]:
    return np.array([model.rate_for(c) for c in range(n)], dtype=np.float64)


def _requisitions(n: int, cost_centers: Ints, req_type: int, step: int) -> Buffer:
    """New one-opening requisition rows, with ids to be assigned and no role yet."""
    rows = OPENING.empty(n)
    rows[:, OpeningColumn.REQ_ID] = -1
    rows[:, OpeningColumn.COST_CENTER] = cost_centers
    rows[:, OpeningColumn.CREATION_STEP] = step
    rows[:, OpeningColumn.REQ_TYPE] = req_type
    rows[:, OpeningColumn.NUM_OPENINGS] = 1
    return rows


# --- attrition --------------------------------------------------------------------------------


def attrition(state: SimulationState, ctx: StepContext) -> DomainResult:
    rows = state.live_employees
    p = ctx.config.models.attrition.compute_probability(rows)
    p = ctx.config.resolved_eligibility.restrict(ATTRITION, rows, p)
    leave, quotas = _draw(p, rows, EMPLOYEE, GuidanceCategory.ATTRITION, ctx, ctx.rng(ATTRITION))
    index = np.flatnonzero(leave)
    return DomainResult(
        ATTRITION,
        employees=RowChange(EMPLOYEE, drop=index),
        departures=rows[index].copy(),
        quotas=quotas,
    )


# --- backfill ---------------------------------------------------------------------------------


def backfill(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Each departure opens a backfill requisition for the same role with its cost center's
    rate. Where guidance covers the cost center, the count is capped at the quota's headroom."""
    departed = np.concatenate([EMPLOYEE.empty(0), *ctx.departures])
    if not len(departed):
        return DomainResult(BACKFILL)
    cost_centers = departed[:, EmployeeColumn.COST_CENTER].astype(np.int64)
    rates = _per_cost_center_rates(ctx.config.models.backfill, state.n_cost_centers)
    rng = ctx.rng(BACKFILL)
    p = ctx.config.resolved_eligibility.restrict(BACKFILL, departed, rates[cost_centers])
    opened = bernoulli(p, rng)
    ledger: dict[str, Ints] = {}
    if ctx.guided(_OPENED):
        groups = ctx.guided_groups(departed, EMPLOYEE, _OPENED)
        used = state.category_ledger.get(_OPENED.value, 0)
        headroom = np.maximum(0, ctx.quota(_OPENED) - used)
        capped = quota_select(opened.astype(np.float64), groups, headroom, rng)
        opened = np.where(groups >= 0, capped.mask, opened)
        ledger[_OPENED.value] = capped.realized
    roles = departed[opened]
    code = ctx.config.schema.vocabulary(REQ_TYPE).code(BACKFILL)
    rows = _requisitions(len(roles), cost_centers[opened], code, ctx.step)
    for source, target in FILL_COLUMN_MAP.items():
        rows[:, source] = roles[:, target]
    return DomainResult(BACKFILL, openings=RowChange(OPENING, append=rows), ledger=ledger)


# --- external fill ----------------------------------------------------------------------------


def external_fill(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Each open opening fills with the model's monthly hazard, or by the external hires quota.
    A fill hires an external candidate into the requisition's role and consumes one opening."""
    reqs = state.live_openings
    openings = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64)
    row_of = np.repeat(np.arange(len(reqs)), openings)  # one entry per fillable opening
    hazard = ctx.config.models.external_fill.compute_probability(reqs)
    p = ctx.config.resolved_eligibility.restrict(EXTERNAL_FILL, reqs, hazard)[row_of]
    filled, quotas = _draw(
        p, reqs[row_of], OPENING, GuidanceCategory.EXTERNAL_HIRES, ctx, ctx.rng(EXTERNAL_FILL)
    )
    source = row_of[filled]
    per_row = np.bincount(source, minlength=len(reqs)).astype(np.int64)
    touched = np.flatnonzero(per_row)
    closed = touched[per_row[touched] == openings[touched]]
    partial = touched[per_row[touched] < openings[touched]]
    remaining = reqs[partial].copy()
    remaining[:, OpeningColumn.NUM_OPENINGS] -= per_row[partial]

    hires = EMPLOYEE.empty(len(source))
    copy_role(reqs[source], hires)
    hires[:, EmployeeColumn.EMPLOYEE_ID] = -1
    hires[:, EmployeeColumn.HIRE_STEP] = ctx.step
    return DomainResult(
        EXTERNAL_FILL,
        employees=RowChange(EMPLOYEE, append=hires),
        openings=RowChange(OPENING, drop=closed, write_index=partial, write_rows=remaining),
        grants=(Grant("new_hire"),),
        quotas=quotas,
    )


# --- cancellation -----------------------------------------------------------------------------


def cancellation(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Requisitions that exit unfilled, with every opening they still carry."""
    reqs = state.live_openings
    p = ctx.config.models.cancellation.compute_probability(reqs)
    p = ctx.config.resolved_eligibility.restrict(CANCELLATION, reqs, p)
    exits = bernoulli(p, ctx.rng(CANCELLATION))
    return DomainResult(CANCELLATION, openings=RowChange(OPENING, drop=np.flatnonzero(exits)))


# --- growth -----------------------------------------------------------------------------------


def growth(state: SimulationState, ctx: StepContext) -> DomainResult:
    """New growth requisitions: the residual of the requisitions-opened quota where guidance
    covers the cost center, and a Poisson count at the historical rate elsewhere. Roles come
    from the attribute generator."""
    n = state.n_cost_centers
    rng = ctx.rng(GROWTH)
    counts = rng.poisson(_per_cost_center_rates(ctx.config.models.growth, n)).astype(np.int64)
    ledger: dict[str, Ints] = {}
    quotas: dict[str, QuotaRecord] = {}
    if ctx.guided(_OPENED):
        group = ctx.guided_groups_of_codes(np.arange(n, dtype=np.int64), _OPENED)
        covered = group >= 0
        quota = ctx.quota(_OPENED)
        used = state.category_ledger.get(_OPENED.value, np.zeros(len(quota), dtype=np.int64))
        residual = np.maximum(0, quota - used)
        counts[covered] = residual[group[covered]]
        mine = np.bincount(group[covered], weights=counts[covered], minlength=len(quota))
        ledger[_OPENED.value] = mine.astype(np.int64)
        # Generation can always supply its residual, so the whole quota is eligible.
        quotas[_OPENED.value] = QuotaRecord(quota, quota.copy(), used + ledger[_OPENED.value])

    cost_centers = np.repeat(np.arange(n, dtype=np.int64), counts)
    code = ctx.config.schema.vocabulary(REQ_TYPE).code(GROWTH)
    rows = _requisitions(len(cost_centers), cost_centers, code, ctx.step)
    start = 0
    for cc in np.flatnonzero(counts):
        k = int(counts[cc])
        rows[start : start + k][:, list(ATTRIBUTES)] = ctx.config.models.attributes.sample(
            int(cc), k, rng
        )
        start += k
    return DomainResult(
        GROWTH, openings=RowChange(OPENING, append=rows), ledger=ledger, quotas=quotas
    )


#: The step's domains, in order. The order is behavior: backfill reads this step's departures,
#: fills can fill this step's backfills, and growth needs the settled ledger.
DOMAINS: Final[tuple[tuple[str, Handler], ...]] = (
    (ATTRITION, attrition),
    (BACKFILL, backfill),
    (EXTERNAL_FILL, external_fill),
    (CANCELLATION, cancellation),
    (GROWTH, growth),
)
