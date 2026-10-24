"""Fill channels and the unified fills domain.

**Channels.** Each open requisition has a monthly hazard of filling by each channel.

An opening fills, in a month, by at most one of three channels:

- **external**: an external hire (`F_ext`), scored by the external-fill model;
- **conversion**: a contingent worker converted into the role (`F_conv`), scored by the
  conversion-fill model;
- **transfer**: an employee moving in from another role (`F_tr`), scored by the transfer-fill
  model.

`channel_hazards` scores every requisition row by each channel's model, a channel without one
scoring zero, and zeroes the rows each channel's eligibility rules exclude (the domains
`external_fill`, `conversion_fill`, and `transfer_fill`). The planner's **fill-share targets**
then reshape the hazards toward each scope's channel mix (see `planner`), and a **planner-mix
hook** in code may reshape them further; guidance still fixes the channels it covers.
Last, a row whose hazards sum past 1 is **rescaled** to sum to exactly 1, keeping its channels'
proportions, since an opening can't fill twice in a month.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.campus import is_campus
from hcsim.engine.logit import logit_shift
from hcsim.engine.results import DomainResult, Grant, QuotaRecord, RowChange
from hcsim.engine.state import SimulationState
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.fill_tree import FILL_TREE, ancestors, leaves
from hcsim.model.core.layouts import (
    EMPLOYEE,
    OPENING,
    Buffer,
    EmployeeColumn,
    OpeningColumn,
    copy_role,
)
from hcsim.model.interface import Mask
from hcsim.model.ml.sampling import quota_select

if TYPE_CHECKING:
    from hcsim.engine.domains import StepContext

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

CHANNELS: Final = ("external", "conversion", "transfer")
EXTERNAL, CONVERSION, TRANSFER = range(3)
#: Each channel's model field and eligibility domain.
CHANNEL_DOMAINS: Final = ("external_fill", "conversion_fill", "transfer_fill")


class FillMix(Protocol):
    """A planner's reshaping of the channel hazards: (rows, channels) in, the same out."""

    def __call__(self, hazards: Floats, openings: Buffer, step: int) -> Floats: ...


def rescale(hazards: Floats) -> Floats:
    """Rows whose hazards sum past 1 scaled down to sum to 1; the rest as they are."""
    total = hazards.sum(axis=1, keepdims=True)
    result: Floats = np.where(total > 1.0, hazards / np.where(total > 0, total, 1.0), hazards)
    return result


def model_hazards(openings: Buffer, ctx: StepContext) -> Floats:
    """(rows, channels): each channel's model score of each requisition row, zero for a channel
    without a model, before any shift, eligibility rule, or rescaling."""
    hazards = np.zeros((len(openings), len(CHANNELS)))
    for k, domain in enumerate(CHANNEL_DOMAINS):
        model = getattr(ctx.config.models, domain)
        if model is not None and len(openings):
            hazards[:, k] = np.asarray(model.compute_probability(openings), dtype=np.float64)
    return hazards


def channel_hazards(openings: Buffer, ctx: StepContext, *, record: bool = True) -> Floats:
    """(rows, channels): each requisition row's monthly hazard of filling by each channel. Each
    channel's score takes the planner's tilts, then under rate calibration its cost center's fill
    shift, before the eligibility rules (see `planner` and `calibration`). `record` off leaves
    the tilts unrecorded, for a call that only scores the openings."""
    hazards = model_hazards(openings, ctx)
    for k, domain in enumerate(CHANNEL_DOMAINS):
        hazards[:, k] = ctx.tilt(domain, openings, hazards[:, k], record=record)
    if ctx.calibration is not None and len(openings):
        cost_centers = openings[:, OpeningColumn.COST_CENTER].astype(np.int64)
        hazards = logit_shift(hazards, ctx.calibration.fill[cost_centers][:, None])
    for k, domain in enumerate(CHANNEL_DOMAINS):
        hazards[:, k] = ctx.config.resolved_eligibility.restrict(domain, openings, hazards[:, k])
    hazards[is_campus(openings, ctx)] = 0.0  # campus requisitions fill on their schedule
    if ctx.pinned:  # replay fills only what history filled
        hazards[:] = 0.0
    if ctx.config.resolved_fill_shares is not None and len(openings):
        hazards = ctx.config.resolved_fill_shares.apply(hazards, openings)
    if ctx.config.fill_mix is not None:
        hazards = np.asarray(ctx.config.fill_mix(hazards, openings, ctx.step), dtype=np.float64)
        if hazards.shape != (len(openings), len(CHANNELS)):
            raise ValueError(f"a fill mix must return {(len(openings), len(CHANNELS))} hazards")
        if not np.isfinite(hazards).all() or (hazards < 0).any():
            raise ValueError("a fill mix must return finite, non-negative hazards")
    return rescale(hazards)


# --- the fills domain ----------------------------------------------------------------------------

FILLS: Final = "fills"
#: Each fill-tree leaf's channel.
LEAF_CHANNEL: Final[Mapping[GuidanceCategory, int]] = MappingProxyType(
    {
        GuidanceCategory.EXTERNAL_HIRES: EXTERNAL,
        GuidanceCategory.CONVERSION_HIRES: CONVERSION,
        GuidanceCategory.INTERNAL_STARTS: TRANSFER,
    }
)
#: The fill-tree nodes, roots first.
FILL_NODES: Final = (
    GuidanceCategory.FILLS,
    GuidanceCategory.HIRES,
    GuidanceCategory.INTERNAL_STARTS,
    GuidanceCategory.EXTERNAL_HIRES,
    GuidanceCategory.CONVERSION_HIRES,
)
#: The flow each channel's fills count as.
CHANNEL_FLOWS: Final = ("F_ext", "F_conv", "F_tr")


def channels_of(node: GuidanceCategory) -> list[int]:
    return [LEAF_CHANNEL[leaf] for leaf in leaves(node)]


@dataclass
class _Draw:
    """One month's fill draw over the fillable openings, one slot per opening."""

    ctx: StepContext
    slots: Buffer  # each slot's requisition row
    hazards: Floats  # (slots, channels)
    rng: np.random.Generator
    ledger: Mapping[str, Ints]
    channel: Ints = field(init=False)
    #: Each slot's uniform draw for the unguided channels, drawn before any quota selection.
    u: Floats = field(init=False)
    quotas: dict[str, QuotaRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.channel = np.full(len(self.slots), -1, dtype=np.int64)

    def select(self, node: GuidanceCategory, among: Mask) -> Mask:
        """Exactly the node's quota of slots among `among`, per group, by Gumbel-top-k on the
        node's channels' hazards, less what other domains already used of it this month."""
        groups = self.ctx.guided_groups(self.slots, OPENING, node)
        weight = self.hazards[:, channels_of(node)].sum(axis=1) * among
        quota = self.ctx.quota(node)
        used = self.ledger.get(node.value, np.zeros(len(quota), dtype=np.int64))
        chosen = quota_select(weight, groups, np.maximum(quota - used, 0), self.rng)
        self.quotas[node.value] = QuotaRecord(
            quota, chosen.eligible + used, chosen.realized + used, used
        )
        mask: Mask = chosen.mask & (groups >= 0)
        return mask

    def assign(self, node: GuidanceCategory, picked: Mask) -> None:
        """Give each picked slot a channel under `node`: carried children take their exact
        quotas, and the residual pool goes to the rest by their hazards."""
        children = FILL_TREE.get(node, ())
        if not children:
            self.channel[picked] = LEAF_CHANNEL[node]
            return
        remaining = picked.copy()
        for child in children:
            if self.ctx.guided(child):
                chosen = self.select(child, remaining)
                self.assign(child, chosen)
                remaining &= ~chosen
        free = [c for c in children if not self.ctx.guided(c)]
        pool = [ch for c in free for ch in channels_of(c)] if free else channels_of(node)
        self.categorical(remaining, pool, always=True)

    def categorical(self, rows: Mask, pool: list[int], *, always: bool = False) -> None:
        """Give the slots in `rows` a channel from `pool`, in proportion to its hazards: with
        `always`, every slot fills; otherwise a slot fills by its hazards' sum."""
        index = np.flatnonzero(rows)
        if not len(index):
            return
        weights = self.hazards[np.ix_(index, pool)]
        total = weights.sum(axis=1)
        if always:
            weights = np.where(total[:, None] > 0, weights, 1.0)
            weights = weights / weights.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(weights, axis=1)
        u = self.rng.random(len(index)) if always else self.u[index]
        fills = u < cumulative[:, -1]
        pick = np.argmax(u[:, None] < cumulative, axis=1)
        self.channel[index[fills]] = np.asarray(pool)[pick[fills]]


def fills(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Each fillable opening fills by at most one channel this month.

    Where guidance carries a fill-tree node, the node's carried roots each realize exactly
    their quota, less what pending starts used, by Gumbel-top-k on their channels' hazards; a
    root's slots then go to its carried children's exact quotas, and the residual pool to the
    rest by their hazards. Every slot no root took then draws on its own, categorically, over
    the channels no carried root covers for it. External and conversion fills hire into the
    requisition's role; transfer fills are assigned for transfer matching to place.
    """
    reqs = state.live_openings
    row_of = np.repeat(np.arange(len(reqs)), ctx.fillable(reqs))  # one slot per fillable opening
    slots = reqs[row_of]
    rng = ctx.rng(FILLS)
    draw = _Draw(ctx, slots, channel_hazards(reqs, ctx)[row_of], rng, state.category_ledger)
    draw.u = rng.random(len(slots))
    carried = [c for c in FILL_NODES if ctx.guided(c)]
    roots = [c for c in carried if not any(a in carried for a in ancestors(c))]
    taken = np.zeros(len(slots), dtype=bool)
    guided = np.zeros((len(slots), len(CHANNELS)), dtype=bool)
    for root in roots:
        covered = ctx.guided_groups(slots, OPENING, root) >= 0
        guided[np.ix_(covered, channels_of(root))] = True
        picked = draw.select(root, ~taken)
        draw.assign(root, picked)
        taken |= picked
    free = ~taken
    for k in range(len(CHANNELS)):
        draw.hazards[:, k] = np.where(guided[:, k], 0.0, draw.hazards[:, k])
    draw.categorical(free, list(range(len(CHANNELS))))
    return _result(state, ctx, reqs, row_of, draw)


def _result(
    state: SimulationState, ctx: StepContext, reqs: Buffer, row_of: Ints, draw: _Draw
) -> DomainResult:
    n = state.n_cost_centers
    hired = (draw.channel == EXTERNAL) | (draw.channel == CONVERSION)
    source = row_of[hired]
    per_row = np.bincount(source, minlength=len(reqs)).astype(np.int64)
    openings = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64)
    touched = np.flatnonzero(per_row)
    closed = touched[per_row[touched] == openings[touched]]
    partial = touched[per_row[touched] < openings[touched]]
    remaining = reqs[partial].copy()
    remaining[:, OpeningColumn.NUM_OPENINGS] -= per_row[partial]
    hires = EMPLOYEE.empty(len(source))
    copy_role(reqs[source], hires)
    hires[:, EmployeeColumn.EMPLOYEE_ID] = -1
    hires[:, EmployeeColumn.HIRE_STEP] = ctx.step
    cc = draw.slots[:, OpeningColumn.COST_CENTER].astype(np.int64)
    flows = {
        CHANNEL_FLOWS[k]: np.bincount(cc[draw.channel == k], minlength=n).astype(np.int64)
        for k in (EXTERNAL, CONVERSION)
    }
    transfers = draw.slots[draw.channel == TRANSFER, OpeningColumn.REQ_ID].astype(np.int64)
    assigned = dict(zip(*np.unique(transfers, return_counts=True), strict=True))
    return DomainResult(
        FILLS,
        employees=RowChange(EMPLOYEE, append=hires),
        openings=RowChange(OPENING, drop=closed, write_index=partial, write_rows=remaining),
        grants=(Grant("new_hire"),),
        quotas=draw.quotas,
        flows=flows,
        transfers={int(r): int(c) for r, c in assigned.items()},
    )
