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
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.campus import is_campus
from hcsim.engine.config import BackfillSource, SimulationConfig
from hcsim.engine.fills import FILLS, fills
from hcsim.engine.known_events import EventKind
from hcsim.engine.known_resolution import FiringLog, ResolvedEvent, ResolvedPlan
from hcsim.engine.results import DomainResult, QuotaRecord, RowChange
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
)
from hcsim.model.core.schema import REQ_TYPE, UNATTRIBUTED
from hcsim.model.interface import EntityModel, GroupRateModel, Mask, Probabilities
from hcsim.model.ml.generative import ATTRIBUTES
from hcsim.model.ml.sampling import apportion, bernoulli, quota_select

Ints = NDArray[np.int64]

ATTRITION: Final = "attrition"
BACKFILL: Final = "backfill"
BACKFILL_TRANSFER: Final = "backfill_transfer"
TRANSFERS: Final = "transfers"
CANCELLATION: Final = "cancellation"
GROWTH: Final = "growth"

_OPENED = GuidanceCategory.REQUISITIONS_OPENED


@dataclass(frozen=True)
class Departures:
    """Employees who left a role this step: the domain they left by, their rows as they were,
    and whether a known event anticipated each departure."""

    domain: str
    rows: Buffer
    anticipated: Mask


@dataclass
class StepContext:
    """What a handler may read besides state: the configuration, the step, the random streams,
    and the departures earlier domains reported this step."""

    config: SimulationConfig
    step: int
    trajectory: int
    seeds: SeedManager
    departures: list[Departures] = field(default_factory=list)
    #: Openings this month's fills assigned to the transfer channel, by requisition id.
    transfers: dict[int, int] = field(default_factory=dict)
    #: The run's known events, resolved, and the log of those fired.
    known: ResolvedPlan | None = None
    fired: FiringLog = field(default_factory=FiringLog)
    _quotas: dict[GuidanceCategory, Ints] = field(default_factory=dict)

    def rng(self, domain: str) -> np.random.Generator:
        return self.seeds.generator(self.trajectory, domain)

    @property
    def month(self) -> int:
        """The calendar month, 1-12, of this step."""
        return (self.config.run_date.month - 1 + self.step) % 12 + 1

    def departed(self, *domains: str) -> tuple[Buffer, Mask]:
        """This step's departures by `domains`, and their anticipated flags."""
        chosen = [d for d in self.departures if d.domain in domains]
        rows = np.concatenate([EMPLOYEE.empty(0), *(d.rows for d in chosen)])
        flags = np.concatenate([np.zeros(0, dtype=bool), *(d.anticipated for d in chosen)])
        return rows, flags

    def reserved(self, reqs: Buffer) -> Ints:
        """Openings held back on each requisition row for pending starts still to come."""
        if self.known is None:
            return np.zeros(len(reqs), dtype=np.int64)
        return self.known.plan.reserved(reqs, self.step)

    def fillable(self, reqs: Buffer) -> Ints:
        """The openings fills may take on each requisition row."""
        result: Ints = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64) - self.reserved(reqs)
        return result

    def known_at(self, kind: EventKind) -> list[ResolvedEvent]:
        """This step's known events of `kind`."""
        return [] if self.known is None else self.known.at(self.step, kind)

    def fire_known(self, rows: Buffer, kind: EventKind) -> Mask:
        """The rows this step's known events of `kind` take, each recorded as fired. An event
        whose employee isn't among `rows` stays unfired."""
        taken = np.zeros(len(rows), dtype=bool)
        events = self.known_at(kind)
        if not events:
            return taken
        ids = rows[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
        where = {int(e): k for k, e in enumerate(ids)}
        for e in events:
            row = None if e.event.employee_id is None else where.get(int(e.event.employee_id))
            if row is not None:
                taken[row] = True
                self.fired.record(e.event)
        return taken

    def sampled(self, rows: Buffer, p: Probabilities) -> Probabilities:
        """`p` with every employee waiting on a known event set to zero, so no sampled draw
        takes them first."""
        if self.known is None:
            return p
        result: Probabilities = np.where(self.known.plan.sampled(rows, self.step), p, 0.0)
        return result

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


def draw(
    p: Probabilities,
    rows: Buffer,
    layout: BufferLayout,
    category: GuidanceCategory,
    ctx: StepContext,
    rng: np.random.Generator,
    known: Mask,
) -> tuple[Mask, dict[str, QuotaRecord]]:
    """Quota sampling for rows in covered groups, independent draws for the rest. Rows in
    `known` already experience the event: they count against their group's quota, and sampling
    takes only what's left of it."""
    independent = bernoulli(p, rng)
    if not ctx.guided(category):
        return independent, {}
    groups = ctx.guided_groups(rows, layout, category)
    quota = ctx.quota(category)
    covered = known & (groups >= 0)
    taken = np.bincount(groups[covered], minlength=len(quota)).astype(np.int64)
    selection = quota_select(p, groups, np.maximum(quota - taken, 0), rng)
    mask = np.where(groups >= 0, selection.mask, independent)
    record = QuotaRecord(quota, selection.eligible + taken, selection.realized + taken)
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
    """The month's known terminations leave first. Then everyone else draws from the model,
    except those excluded by eligibility or waiting on a known event; under guidance, the known
    terminations count against their group's quota, and sampling realizes what's left."""
    rows = state.live_employees
    known = ctx.fire_known(rows, EventKind.TERMINATION)
    p = ctx.config.models.attrition.compute_probability(rows)
    p = ctx.sampled(rows, ctx.config.resolved_eligibility.restrict(ATTRITION, rows, p))
    p = np.where(known, 0.0, p)
    sampled, quotas = draw(
        p, rows, EMPLOYEE, GuidanceCategory.ATTRITION, ctx, ctx.rng(ATTRITION), known
    )
    index = np.flatnonzero(sampled | known)
    return DomainResult(
        ATTRITION,
        employees=RowChange(EMPLOYEE, drop=index),
        departures=rows[index].copy(),
        anticipated=known[index],
        quotas=quotas,
    )


# --- backfill ---------------------------------------------------------------------------------


def _backfill_probability(
    departed: Buffer, ctx: StepContext, model: object, n: int
) -> Probabilities:
    """Each departure's chance of a backfill: its cost center's rate, or a classifier's score
    of the departure's row, as `backfill_source` says."""
    if ctx.config.backfill_source is BackfillSource.CLASSIFIER:
        return cast(EntityModel, model).compute_probability(departed)
    rates = _per_cost_center_rates(cast(GroupRateModel, model), n)
    result: Probabilities = rates[departed[:, EmployeeColumn.COST_CENTER].astype(np.int64)]
    return result


def _backfill(
    state: SimulationState, ctx: StepContext, domain: str, source: str, model: object
) -> DomainResult:
    """Each departure by `source` opens a backfill for the same role, copied verbatim, with its
    probability, as a binomial draw over the departures. A known termination opens none unless
    the configuration says so. Where guidance covers the cost center, the count is capped at the
    requisitions-opened quota's headroom; and no cost center opens more than the room its
    effective ceiling leaves."""
    departed, anticipated = ctx.departed(source)
    if not len(departed):
        return DomainResult(domain)
    cost_centers = departed[:, EmployeeColumn.COST_CENTER].astype(np.int64)
    p = _backfill_probability(departed, ctx, model, state.n_cost_centers)
    p = ctx.config.resolved_eligibility.restrict(BACKFILL, departed, p)
    if not ctx.config.backfill_anticipated_terminations:
        p = np.where(anticipated, 0.0, p)
    rng = ctx.rng(domain)
    opened = bernoulli(p, rng)
    ledger: dict[str, Ints] = {}
    if ctx.guided(_OPENED):
        groups = ctx.guided_groups(departed, EMPLOYEE, _OPENED)
        used = state.category_ledger.get(_OPENED.value, 0)
        headroom = np.maximum(0, ctx.quota(_OPENED) - used)
        capped = quota_select(opened.astype(np.float64), groups, headroom, rng)
        opened = np.where(groups >= 0, capped.mask, opened)
        ledger[_OPENED.value] = capped.realized
    ceiling = ctx.config.ceiling
    if ceiling.binds:
        strength = state.headcount + state.open_openings
        room = ceiling.room(ctx.config.run_date, ctx.step, strength)
        quota = np.minimum(room, len(departed)).astype(np.int64)
        vetoed = quota_select(opened.astype(np.float64), cost_centers, quota, rng)
        if _OPENED.value in ledger:
            groups = ctx.guided_groups(departed, EMPLOYEE, _OPENED)
            kept = vetoed.mask & (groups >= 0)
            ledger[_OPENED.value] = np.bincount(
                groups[kept], minlength=len(ledger[_OPENED.value])
            ).astype(np.int64)
        opened = vetoed.mask
    roles = departed[opened]
    code = ctx.config.schema.vocabulary(REQ_TYPE).code(BACKFILL)
    rows = _requisitions(len(roles), cost_centers[opened], code, ctx.step)
    for source_column, target in FILL_COLUMN_MAP.items():
        rows[:, source_column] = roles[:, target]
    return DomainResult(domain, openings=RowChange(OPENING, append=rows), ledger=ledger)


def backfill(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Backfill for departures (`B_att`)."""
    return _backfill(state, ctx, BACKFILL, ATTRITION, ctx.config.models.backfill)


def backfill_transfer(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Backfill for transfers out (`B_tr`), at the transfer rate where the run has one."""
    models = ctx.config.models
    model = models.backfill if models.backfill_transfer is None else models.backfill_transfer
    return _backfill(state, ctx, BACKFILL_TRANSFER, TRANSFERS, model)


# --- external fill ----------------------------------------------------------------------------


# --- cancellation -----------------------------------------------------------------------------


def cancellation(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Requisitions that exit unfilled, with every opening they still carry."""
    reqs = state.live_openings
    p = ctx.config.models.cancellation.compute_probability(reqs)
    p = ctx.config.resolved_eligibility.restrict(CANCELLATION, reqs, p)
    p = np.where(ctx.reserved(reqs) > 0, 0.0, p)  # a reserved requisition waits for its start
    p = np.where(is_campus(reqs, ctx), 0.0, p)  # a campus requisition fills on its schedule
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


#: The step's sampled domains, in order. The order is behavior: backfill reads this step's
#: departures, fills can fill this step's backfills, and growth needs the settled ledger.
#: Backfill registers twice: for departures, and for transfers out, which transfer matching
#: (M4.12) records.
DOMAINS: Final[tuple[tuple[str, Handler], ...]] = (
    (ATTRITION, attrition),
    (BACKFILL, backfill),
    (FILLS, fills),
    (CANCELLATION, cancellation),
    (BACKFILL_TRANSFER, backfill_transfer),
    (GROWTH, growth),
)
