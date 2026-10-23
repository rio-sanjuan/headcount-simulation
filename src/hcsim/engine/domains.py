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
from enum import IntEnum
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.campus import is_campus
from hcsim.engine.config import BackfillSource, SimulationConfig
from hcsim.engine.fills import FILLS, fills
from hcsim.engine.known_events import EventKind
from hcsim.engine.known_resolution import FiringLog, ResolvedEvent, ResolvedPlan
from hcsim.engine.modes import RunMode
from hcsim.engine.replay import ReplayedRequisitions
from hcsim.engine.results import DomainResult, QuotaRecord, RowChange
from hcsim.engine.roles import DEGENERATE_ROWS, fill_roles
from hcsim.engine.seeds import SeedManager
from hcsim.engine.state import SimulationState
from hcsim.engine.supply import Route, SupplyRule, expected_fill, soft_bridge, strength
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
from hcsim.model.ml.sampling import apportion, bernoulli, quota_select

Ints = NDArray[np.int64]

ATTRITION: Final = "attrition"
BACKFILL: Final = "backfill"
BACKFILL_TRANSFER: Final = "backfill_transfer"
TRANSFERS: Final = "transfers"
CANCELLATION: Final = "cancellation"
GROWTH: Final = "growth"

_OPENED = GuidanceCategory.REQUISITIONS_OPENED
_CLOSED = GuidanceCategory.REQUISITIONS_CLOSED_UNFILLED


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
    #: The realized requisitions a replay keeps on history's track.
    replay: ReplayedRequisitions | None = None
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

    @property
    def pinned(self) -> bool:
        """Whether every movement is pinned to history, as in replay: then nothing is sampled."""
        return self.config.run_mode is RunMode.REPLAY

    def sampled(self, rows: Buffer, p: Probabilities) -> Probabilities:
        """`p` with every employee waiting on a known event set to zero, so no sampled draw
        takes them first; and all of it zero in replay."""
        if self.pinned:
            return np.zeros(len(rows))
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
    record = QuotaRecord(quota, selection.eligible + taken, selection.realized + taken, taken)
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
    effective ceiling leaves (see `ceiling`). The backfills the ceiling refuses are withheld, and
    counted per cost center as the diagnostic `backfill_withheld`."""
    departed, anticipated = ctx.departed(source)
    if not len(departed):
        return DomainResult(domain)
    cost_centers = departed[:, EmployeeColumn.COST_CENTER].astype(np.int64)
    p = _backfill_probability(departed, ctx, model, state.n_cost_centers)
    p = ctx.config.resolved_eligibility.restrict(BACKFILL, departed, p)
    if ctx.pinned:  # replay opens its realized backfills itself
        p = np.zeros(len(departed))
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
    diagnostics: dict[str, Ints] = {}
    if ceiling.binds:
        room = ceiling.room(ctx.config.run_date, ctx.step, state.headcount, state.open_openings)
        quota = np.minimum(room, len(departed)).astype(np.int64)
        vetoed = quota_select(opened.astype(np.float64), cost_centers, quota, rng)
        withheld = opened & ~vetoed.mask
        diagnostics[BACKFILL_WITHHELD] = np.bincount(
            cost_centers[withheld], minlength=state.n_cost_centers
        ).astype(np.int64)
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
    return DomainResult(
        domain, openings=RowChange(OPENING, append=rows), ledger=ledger, diagnostics=diagnostics
    )


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
    """Requisitions that exit unfilled, with every opening they still carry.

    Each requisition has a monthly hazard from the cancellation model; a reserved requisition,
    a campus one, and any an eligibility rule excludes have none. Where guidance covers the
    group with `requisitions_closed_unfilled`, its quota counts openings: the group's
    requisitions are taken in Gumbel-top-k order on their hazards, each weighing its openings,
    and one is taken when it still fits, so the group closes as close to its quota as its
    requisitions allow without passing it. Elsewhere each requisition draws on its own.
    """
    reqs = state.live_openings
    p = ctx.config.models.cancellation.compute_probability(reqs)
    p = ctx.config.resolved_eligibility.restrict(CANCELLATION, reqs, p)
    p = np.where(ctx.reserved(reqs) > 0, 0.0, p)  # a reserved requisition waits for its start
    p = np.where(is_campus(reqs, ctx), 0.0, p)  # a campus requisition fills on its schedule
    if ctx.pinned:  # replay closes its realized exits itself
        p = np.zeros(len(reqs))
    rng = ctx.rng(CANCELLATION)
    exits = bernoulli(p, rng)
    quotas: dict[str, QuotaRecord] = {}
    if ctx.guided(_CLOSED):
        groups = ctx.guided_groups(reqs, OPENING, _CLOSED)
        weights = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64)
        chosen, record = weighted_top_k(p, weights, groups, ctx.quota(_CLOSED), rng)
        exits = np.where(groups >= 0, chosen, exits)
        quotas[_CLOSED.value] = record
    return DomainResult(
        CANCELLATION, openings=RowChange(OPENING, drop=np.flatnonzero(exits)), quotas=quotas
    )


def weighted_top_k(
    p: Probabilities, weights: Ints, groups: Ints, quota: Ints, rng: np.random.Generator
) -> tuple[Mask, QuotaRecord]:
    """Per group, rows in Gumbel-top-k order on `p`, each taken when its weight still fits in
    what's left of the group's quota. Rows with no chance, or no group, are never taken."""
    with np.errstate(divide="ignore"):
        keys = np.log(np.clip(p, 0.0, 1.0)) + rng.gumbel(size=len(p))
    eligible_rows = np.flatnonzero((groups >= 0) & (p > 0))
    order = eligible_rows[np.lexsort((-keys[eligible_rows], groups[eligible_rows]))]
    left = quota.astype(np.int64).copy()
    chosen = np.zeros(len(p), dtype=bool)
    for row in order.tolist():
        g = int(groups[row])
        if 0 < weights[row] <= left[g]:
            chosen[row] = True
            left[g] -= int(weights[row])
    size = len(quota)
    eligible = np.bincount(groups[eligible_rows], weights=weights[eligible_rows], minlength=size)
    realized = np.bincount(groups[chosen], weights=weights[chosen], minlength=size)
    untaken = eligible_rows[~chosen[eligible_rows] & (weights[eligible_rows] > 0)]
    stuck = np.bincount(groups[untaken], minlength=size) > 0  # every row left is too heavy
    blocked = np.where(stuck, left, 0)
    record = QuotaRecord(
        quota, eligible.astype(np.int64), realized.astype(np.int64), blocked=blocked
    )
    return chosen, record


# --- growth -----------------------------------------------------------------------------------


#: Backfills the effective ceiling withheld, per cost center: the diagnostic's name.
BACKFILL_WITHHELD: Final = "backfill_withheld"


class Regime(IntEnum):
    """How growth sizes a cost center's new requisitions in a month."""

    RATE = 0  #: at the growth model's historical rate
    TARGET = 1  #: toward the plan's target, by the supply rule
    GUIDED = 2  #: to the residual of the requisitions-opened quota


def regimes(state: SimulationState, ctx: StepContext) -> NDArray[np.int64]:
    """Each cost center's regime this month: target-seeking where the plan drives its supply
    and has a target for it (see `alignment`), guided where guidance covers it otherwise, and
    the rate elsewhere."""
    n = state.n_cost_centers
    regime = np.full(n, Regime.RATE, dtype=np.int64)
    if ctx.guided(_OPENED):
        covered = ctx.guided_groups_of_codes(np.arange(n, dtype=np.int64), _OPENED) >= 0
        regime[covered] = Regime.GUIDED
    planned = ~np.isnan(ctx.config.ceiling.targets(ctx.config.run_date, ctx.step))
    regime[planned] = Regime.TARGET
    return regime


def growth(state: SimulationState, ctx: StepContext) -> DomainResult:
    """New growth requisitions, sized per cost center by its regime, the route it takes (see
    `supply`):

    - **target-seeking** (G-A): the run's supply rule, `strength`, `expected_fill`, or
      `soft_bridge`, never negative;
    - **guided** (G-B): the residual of the requisitions-opened quota, after backfill's use of it;
    - **rate** (G-C): a count from the growth model's own draw at its historical rate, negative
      binomial for the negative-binomial model, from the seeded growth stream.

    Every count is then clamped at the room the effective ceiling leaves, since the capacity
    veto always wins. The result carries each cost center's regime, which the run records.
    Roles come from the required attribute generator.
    """
    n = state.n_cost_centers
    rng = ctx.rng(GROWTH)
    counts = rate_counts(ctx.config.models.growth, n, rng)
    if ctx.pinned:  # replay opens its realized requisitions itself
        counts[:] = 0
    regime = regimes(state, ctx)
    seeking = regime == Regime.TARGET
    if not ctx.pinned and seeking.any():
        counts[seeking] = _supply(state, ctx)[seeking]
    ledger: dict[str, Ints] = {}
    quotas: dict[str, QuotaRecord] = {}
    guided = regime == Regime.GUIDED
    group = np.full(n, -1, dtype=np.int64)
    if guided.any():
        group = ctx.guided_groups_of_codes(np.arange(n, dtype=np.int64), _OPENED)
        quota = ctx.quota(_OPENED)
        used = state.category_ledger.get(_OPENED.value, np.zeros(len(quota), dtype=np.int64))
        counts[guided] = np.maximum(0, quota - used)[group[guided]]
    ceiling = ctx.config.ceiling
    if ceiling.binds:
        room = ceiling.room(ctx.config.run_date, ctx.step, state.headcount, state.open_openings)
        counts = np.minimum(counts, np.minimum(room, np.iinfo(np.int64).max)).astype(np.int64)
    if guided.any():
        mine = np.bincount(group[guided], weights=counts[guided], minlength=len(quota))
        ledger[_OPENED.value] = mine.astype(np.int64)
        # Generation can always supply its residual, so the whole quota is eligible.
        quotas[_OPENED.value] = QuotaRecord(quota, quota.copy(), used + ledger[_OPENED.value])

    cost_centers = np.repeat(np.arange(n, dtype=np.int64), counts)
    code = ctx.config.schema.vocabulary(REQ_TYPE).code(GROWTH)
    rows = _requisitions(len(cost_centers), cost_centers, code, ctx.step)
    degenerate = fill_roles(rows, counts, ctx.config.models.attributes, rng)
    return DomainResult(
        GROWTH,
        openings=RowChange(OPENING, append=rows),
        ledger=ledger,
        quotas=quotas,
        diagnostics={DEGENERATE_ROWS: degenerate},
        regimes=regime,
    )


#: Each regime's route, as the supply module names them.
ROUTES: Final = {Regime.TARGET: Route.SUPPLY, Regime.GUIDED: Route.GUIDED, Regime.RATE: Route.RATE}


def _supply(state: SimulationState, ctx: StepContext) -> Ints:
    """What the run's supply rule opens in each cost center this month."""
    config = ctx.config
    h, o = state.headcount, state.open_openings
    rule, gain = config.requisitions_supply_rule, config.requisitions_supply_gain
    if rule is SupplyRule.SOFT_BRIDGE:
        upcoming, ahead = config.ceiling.next_targets(config.run_date, ctx.step)
        return soft_bridge(upcoming, ahead, h, o)
    target = config.ceiling.targets(config.run_date, ctx.step)
    if rule is SupplyRule.EXPECTED_FILL:
        return expected_fill(target, h, o, gain, config.requisitions_fill_yield)
    return strength(target, h, o, gain)


def rate_counts(model: GroupRateModel, n: int, rng: np.random.Generator) -> Ints:
    """One month's counts per cost center from the model's own draw where it has one, as the
    growth models do, and a Poisson draw at its rate otherwise."""
    sample = getattr(model, "sample", None)
    if callable(sample):
        return np.asarray(sample(np.arange(n, dtype=np.int64), rng), dtype=np.int64)
    return rng.poisson(_per_cost_center_rates(model, n)).astype(np.int64)


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
