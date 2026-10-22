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
from hcsim.engine.known_events import EventKind
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
    """Fire the month's known transfers, then place an employee in each opening this month's
    fills assigned to transfer.

    A known transfer, as replay supplies, moves its employee to the role it names. When its
    requisition is open in the run, it fills one of its openings (`F_tr`, `V_tr`); when it
    isn't, the move consumes nothing and counts as another movement (`W_in`, `W_out`).

    A matched employee keeps their identity (id, tenure, rating, equity slot, and salary) and
    takes the opening's role, with a transfer grant; their old row is recorded as a departure
    from `transfers`, for the transfer backfill to read. Under internal-starts guidance, no
    group places more than its quota. An unmatched opening follows the shortfall policy.
    """
    reqs = state.live_openings
    employees = state.live_employees
    n = state.n_cost_centers
    known_rows, known_moved, known_reqs = _known(state, ctx)
    rows = _slots(reqs, ctx)
    found = match(reqs, rows, employees, ctx)
    matched = _capped(found, reqs, ctx)
    index = found.employees[matched]
    moved = employees[index].copy()
    copy_role(reqs[found.openings[matched]], moved)
    unmatched = found.openings[~matched]
    spilled, hires = _spill(reqs, unmatched, ctx)
    filled_known = known_reqs >= 0
    used = np.bincount(
        np.concatenate([found.openings[matched], spilled, known_reqs[filled_known]]),
        minlength=len(reqs),
    ).astype(np.int64)
    flows = {
        "F_tr": _count(moved, n) + _count(known_moved[filled_known], n),
        "V_tr": _count(employees[index], n) + _count(employees[known_rows[filled_known]], n),
        "W_in": _count(known_moved[~filled_known], n),
        "W_out": _count(employees[known_rows[~filled_known]], n),
    }
    for k, symbol in ((EXTERNAL, "F_ext"), (CONVERSION, "F_conv")):
        flows[symbol] = _count(hires[0][hires[1] == k], n)
    openings = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64)
    touched = np.flatnonzero(used)
    closed = touched[used[touched] >= openings[touched]]
    partial = touched[used[touched] < openings[touched]]
    remaining = reqs[partial].copy()
    remaining[:, OpeningColumn.NUM_OPENINGS] -= used[partial]
    write_index = np.concatenate([known_rows, index])
    write_rows = np.concatenate([known_moved, moved])
    grants = (Grant("transfer", rows="written"), Grant("new_hire"))
    return DomainResult(
        TRANSFERS_DOMAIN,
        employees=RowChange(
            EMPLOYEE, write_index=write_index, write_rows=write_rows, append=hires[0]
        ),
        openings=RowChange(OPENING, drop=closed, write_index=partial, write_rows=remaining),
        grants=grants,
        departures=employees[write_index].copy(),
        quotas=_record(reqs, found, matched, ctx),
        flows=flows,
    )


def _count(rows: Buffer, n: int) -> Ints:
    result: Ints = np.bincount(
        rows[:, EmployeeColumn.COST_CENTER].astype(np.int64), minlength=n
    ).astype(np.int64)
    return result


def _known(state: SimulationState, ctx: StepContext) -> tuple[Ints, Buffer, Ints]:
    """The month's known transfers: each employee's row, their row as moved, and the row of
    the requisition they fill, or -1 where the run doesn't hold one open."""
    events = ctx.known_at(EventKind.TRANSFER)
    employees, reqs = state.live_employees, state.live_openings
    people = {int(e): k for k, e in enumerate(employees[:, EmployeeColumn.EMPLOYEE_ID])}
    openings = {int(r): k for k, r in enumerate(reqs[:, OpeningColumn.REQ_ID])}
    left = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64).copy()
    rows, moved, filled = [], [], []
    for e in events:
        row = None if e.event.employee_id is None else people.get(int(e.event.employee_id))
        if row is None:
            continue
        new = employees[row].copy()
        for column, code in e.codes.items():
            new[column] = code
        req = None if e.event.req_id is None else openings.get(int(e.event.req_id))
        if req is not None and left[req] >= 1:
            left[req] -= 1
            filled.append(req)
        else:
            filled.append(-1)
        rows.append(row)
        moved.append(new)
        ctx.fired.record(e.event)
    return (
        np.asarray(rows, dtype=np.int64),
        np.stack(moved) if moved else EMPLOYEE.empty(0),
        np.asarray(filled, dtype=np.int64),
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
