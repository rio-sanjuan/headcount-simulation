"""Transfer matching: placing an employee into each opening the fills assigned to transfer.

The transfers domain runs right after the fills, and `transfers` describes what it does with a
match. How it finds one:

**Cells.** Employees are grouped into cells by job family and level, the role attributes a
transfer is matched on. An opening matches the cells of its own family whose level lies within
`transfers_level_band` of its own: 0 matches the same level only.

**Availability.** Only part of a cell is free to move in a month. Each employee is available
with `transfers_availability_factor`, drawn once a month from the transfers stream.

**Cell draws.** The openings are matched in order, each drawing uniformly, without replacement,
from the available employees of its matching cells. A candidate is never in the opening's own
cost center, never excluded by an eligibility rule on `transfers`, and never waiting on a known
event. An opening whose cells have no one left goes unmatched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.fills import CONVERSION, EXTERNAL, channel_hazards
from hcsim.engine.results import DomainResult, Grant, QuotaRecord, RowChange
from hcsim.engine.state import SimulationState
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.layouts import (
    EMPLOYEE,
    OPENING,
    Buffer,
    EmployeeColumn,
    OpeningColumn,
    copy_role,
)

if TYPE_CHECKING:
    from hcsim.engine.domains import StepContext

Ints = NDArray[np.int64]
TRANSFERS_DOMAIN = "transfers"


@dataclass(frozen=True)
class Match:
    """Each transfer opening's slot and the employee row it took, or -1 where none was left."""

    #: For each slot, the opening's row in the requisition buffer.
    openings: Ints
    #: For each slot, the employee row matched to it, or -1.
    employees: Ints = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def matched(self) -> NDArray[np.bool_]:
        result: NDArray[np.bool_] = self.employees >= 0
        return result


def available(employees: Buffer, ctx: StepContext) -> NDArray[np.bool_]:
    """Which employees are free to move this month: drawn at the availability factor, and
    allowed by eligibility and the known-event plan."""
    rng = ctx.rng(TRANSFERS_DOMAIN)
    free = rng.random(len(employees)) < ctx.config.transfers_availability_factor
    p = ctx.config.resolved_eligibility.restrict(
        TRANSFERS_DOMAIN, employees, np.ones(len(employees))
    )
    result: NDArray[np.bool_] = free & (ctx.sampled(employees, p) > 0)
    return result


def cells(employees: Buffer, free: NDArray[np.bool_]) -> dict[tuple[int, int], Ints]:
    """The free employees' rows, by cell: their family and level."""
    rows = np.flatnonzero(free)
    family = employees[rows, EmployeeColumn.JOB_FAMILY].astype(np.int64)
    level = employees[rows, EmployeeColumn.LEVEL].astype(np.int64)
    found: dict[tuple[int, int], list[int]] = {}
    for row, f, lv in zip(rows.tolist(), family.tolist(), level.tolist(), strict=True):
        found.setdefault((f, lv), []).append(row)
    return {key: np.asarray(members, dtype=np.int64) for key, members in found.items()}


def match(openings: Buffer, slots: Ints, employees: Buffer, ctx: StepContext) -> Match:
    """Match each slot, a row of `openings`, to an available employee of a matching cell."""
    by_cell = cells(employees, available(employees, ctx))
    taken = np.zeros(len(employees), dtype=bool)
    rng = ctx.rng(TRANSFERS_DOMAIN)
    band = ctx.config.transfers_level_band
    home = employees[:, EmployeeColumn.COST_CENTER].astype(np.int64)
    chosen = np.full(len(slots), -1, dtype=np.int64)
    for k, row in enumerate(np.asarray(slots, dtype=np.int64).tolist()):
        opening = openings[row]
        family, level = int(opening[OpeningColumn.JOB_FAMILY]), int(opening[OpeningColumn.LEVEL])
        pools = [by_cell.get((family, lv)) for lv in range(level - band, level + band + 1)]
        members = np.concatenate(
            [np.zeros(0, dtype=np.int64), *(p for p in pools if p is not None)]
        )
        candidates = members[
            ~taken[members] & (home[members] != int(opening[OpeningColumn.COST_CENTER]))
        ]
        if len(candidates):
            pick = int(candidates[rng.integers(len(candidates))])
            chosen[k] = pick
            taken[pick] = True
    return Match(np.asarray(slots, dtype=np.int64), chosen)


# --- the transfers domain ------------------------------------------------------------------------


class ShortfallPolicy(StrEnum):
    """What becomes of a transfer opening no one could be matched to."""

    #: The opening stays open, for a later month's fills.
    KEEP_OPEN = "keep_open"
    #: The opening spills to another free fill channel, external or conversion, by their
    #: hazards, and fills now.
    SPILL = "spill"


def transfers(state: SimulationState, ctx: StepContext) -> DomainResult:
    """Place an employee in each opening this month's fills assigned to transfer.

    A matched employee keeps their identity (id, tenure, rating, equity slot, and salary) and
    takes the opening's role, with a transfer grant; their old row is recorded as a departure
    from `transfers`, for the transfer backfill to read. Under internal-starts guidance, no
    group places more than its quota. An unmatched opening follows the shortfall policy.
    """
    reqs = state.live_openings
    rows = _slots(reqs, ctx)
    if not len(rows):
        return DomainResult(TRANSFERS_DOMAIN)
    found = match(reqs, rows, state.live_employees, ctx)
    matched = _capped(found, reqs, ctx)
    employees = state.live_employees
    index = found.employees[matched]
    moved = employees[index].copy()
    copy_role(reqs[found.openings[matched]], moved)
    unmatched = found.openings[~matched]
    spilled, hires = _spill(reqs, unmatched, ctx)
    used = np.bincount(
        np.concatenate([found.openings[matched], spilled]), minlength=len(reqs)
    ).astype(np.int64)
    n = state.n_cost_centers
    flows = {
        "F_tr": np.bincount(
            moved[:, EmployeeColumn.COST_CENTER].astype(np.int64), minlength=n
        ).astype(np.int64),
        "V_tr": np.bincount(
            employees[index, EmployeeColumn.COST_CENTER].astype(np.int64), minlength=n
        ).astype(np.int64),
    }
    for k, symbol in ((EXTERNAL, "F_ext"), (CONVERSION, "F_conv")):
        chosen = hires[1] == k
        flows[symbol] = np.bincount(
            hires[0][chosen, EmployeeColumn.COST_CENTER].astype(np.int64), minlength=n
        ).astype(np.int64)
    openings = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64)
    touched = np.flatnonzero(used)
    closed = touched[used[touched] >= openings[touched]]
    partial = touched[used[touched] < openings[touched]]
    remaining = reqs[partial].copy()
    remaining[:, OpeningColumn.NUM_OPENINGS] -= used[partial]
    grants = (Grant("transfer", rows="written"), Grant("new_hire"))
    return DomainResult(
        TRANSFERS_DOMAIN,
        employees=RowChange(EMPLOYEE, write_index=index, write_rows=moved, append=hires[0]),
        openings=RowChange(OPENING, drop=closed, write_index=partial, write_rows=remaining),
        grants=grants,
        departures=employees[index].copy(),
        quotas=_record(reqs, found, matched, ctx),
        flows=flows,
    )


def _slots(reqs: Buffer, ctx: StepContext) -> Ints:
    """One slot per transfer-assigned opening still open, as its requisition row."""
    where = {int(r): k for k, r in enumerate(reqs[:, OpeningColumn.REQ_ID])}
    slots: list[int] = []
    for req, count in sorted(ctx.transfers.items()):
        row = where.get(req)
        if row is not None:
            slots += [row] * min(count, int(reqs[row, OpeningColumn.NUM_OPENINGS]))
    return np.asarray(slots, dtype=np.int64)


def _groups(reqs: Buffer, slots: Ints, ctx: StepContext) -> Ints | None:
    if not ctx.guided(GuidanceCategory.INTERNAL_STARTS):
        return None
    result: Ints = ctx.guided_groups(reqs[slots], OPENING, GuidanceCategory.INTERNAL_STARTS)
    return result


def _capped(found: Match, reqs: Buffer, ctx: StepContext) -> NDArray[np.bool_]:
    """The matched slots, with no guided group placing more than its quota."""
    matched = found.matched.copy()
    groups = _groups(reqs, found.openings, ctx)
    if groups is None:
        return matched
    quota = ctx.quota(GuidanceCategory.INTERNAL_STARTS)
    placed = np.zeros(len(quota), dtype=np.int64)
    for k in np.flatnonzero(matched):
        g = groups[k]
        if g >= 0 and placed[g] >= quota[g]:
            matched[k] = False
        elif g >= 0:
            placed[g] += 1
    return matched


def _record(
    reqs: Buffer, found: Match, matched: NDArray[np.bool_], ctx: StepContext
) -> dict[str, QuotaRecord]:
    """Under guidance, each group's transfer slots, capped at its quota, against how many were
    placed: the shortfall is the slots no one could be matched to."""
    groups = _groups(reqs, found.openings, ctx)
    if groups is None:
        return {}
    quota = ctx.quota(GuidanceCategory.INTERNAL_STARTS)
    covered = groups >= 0
    assigned = np.bincount(groups[covered], minlength=len(quota)).astype(np.int64)
    placed = np.bincount(groups[covered & matched], minlength=len(quota)).astype(np.int64)
    record = QuotaRecord(np.minimum(assigned, quota), assigned, placed)
    return {GuidanceCategory.INTERNAL_STARTS.value: record}


def _spill(reqs: Buffer, unmatched: Ints, ctx: StepContext) -> tuple[Ints, tuple[Buffer, Ints]]:
    """Under the spill policy, fill each unmatched slot now by external or conversion hire, in
    proportion to their hazards. Returns the slots filled, and the hires with their channels."""
    nothing = (np.zeros(0, dtype=np.int64), (EMPLOYEE.empty(0), np.zeros(0, dtype=np.int64)))
    if ctx.config.transfers_shortfall_policy is not ShortfallPolicy.SPILL or not len(unmatched):
        return nothing
    hazards = channel_hazards(reqs[unmatched], ctx)[:, [EXTERNAL, CONVERSION]]
    total = hazards.sum(axis=1, keepdims=True)
    weights = np.where(total > 0, hazards / np.where(total > 0, total, 1), [1.0, 0.0])
    rng = ctx.rng(f"{TRANSFERS_DOMAIN}:spill")
    channel = np.where(rng.random(len(unmatched)) < weights[:, 0], EXTERNAL, CONVERSION)
    hires = EMPLOYEE.empty(len(unmatched))
    copy_role(reqs[unmatched], hires)
    hires[:, EmployeeColumn.EMPLOYEE_ID] = -1
    hires[:, EmployeeColumn.HIRE_STEP] = ctx.step
    return unmatched, (hires, channel.astype(np.int64))
