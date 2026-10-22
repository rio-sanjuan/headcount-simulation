"""The apply step: the only writer of simulation state.

It applies a result in a fixed order:

1. Drop employees. A leaver's unvested tranches forfeit in the departure month.
2. Write employee rows in place. A changed cost center moves a head.
3. Append employees, with new ids and vest slots.
4. Drop requisitions.
5. Write requisition rows in place. A partial fill lowers the open openings; the row stays.
6. Append requisitions, with new ids.
7. Size and record grants for the appended employees, when the run sizes awards.
8. Add the result's use of shared guidance quota to the category ledger.

The per-cost-center counters move with every change, so a later domain in the same step reads
settled state. Buffers never grow: an append past capacity raises before anything is written.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.config import AwardSettings
from hcsim.engine.results import DomainResult, RowChange
from hcsim.engine.state import SimulationState
from hcsim.financial.sizing import SizingContext
from hcsim.model.core.layouts import Buffer, EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import AWARD_TYPE, SimulationSchema

Ints = NDArray[np.int64]


@dataclass(frozen=True)
class Applied:
    """What one apply step changed, with ids and vest slots as assigned."""

    dropped_employees: Buffer
    appended_employees: Buffer
    dropped_openings: Buffer
    appended_openings: Buffer
    forfeited_tranches: int
    recorded_tranches: int
    #: Grant recipients no sizer covered.
    unsized_grants: int = 0


def per_cost_center(rows: Buffer, column: int, n: int, weights: int | None = None) -> Ints:
    """Rows per cost center, or the sum of the `weights` column per cost center."""
    w = None if weights is None else rows[:, weights]
    counts = np.bincount(rows[:, column].astype(np.int64), weights=w, minlength=n)
    return counts.astype(np.int64)


# A write edits a row in place, so it keeps the row's identity: the ledger finds a holder's
# tranches by vest slot, and the results trace rows by id.
_IDENTITY: dict[str, tuple[int, ...]] = {
    "employee": (EmployeeColumn.EMPLOYEE_ID, EmployeeColumn.VEST_INDEX),
    "opening": (OpeningColumn.REQ_ID,),
}


def _check(change: RowChange, live: Buffer) -> None:
    name = change.layout.name
    for label, index in (("drop", change.drop), ("write", change.write_index)):
        if len(index) and (index.min() < 0 or index.max() >= len(live)):
            raise IndexError(f"{name} {label} indices must lie in 0..{len(live) - 1}")
        if len(np.unique(index)) != len(index):
            raise ValueError(f"duplicate {name} {label} indices")
    identity = list(_IDENTITY[name])
    if not np.array_equal(live[change.write_index][:, identity], change.write_rows[:, identity]):
        raise ValueError(f"a {name} write can't change a row's id or vest slot")


def _drop(buffer: Buffer, live: int, drop: Ints) -> int:
    """Remove rows in place, keeping the survivors' order, and zero the freed tail."""
    keep = np.ones(live, dtype=bool)
    keep[drop] = False
    remaining = int(keep.sum())
    buffer[:remaining] = buffer[:live][keep]
    buffer[remaining:live] = 0.0
    return remaining


def _after_drops(index: Ints, drop: Ints) -> Ints:
    """Shift pre-change indices past the dropped rows that preceded them."""
    shifted: Ints = index - np.searchsorted(np.sort(drop), index)
    return shifted


def _new_ids(rows: Buffer, column: int, start: int) -> int:
    """Number rows with a negative id from `start`, returning the next free id."""
    new = rows[:, column] < 0
    count = int(new.sum())
    rows[new, column] = start + np.arange(count)
    return start + count


def apply(
    state: SimulationState,
    result: DomainResult,
    step: int,
    schema: SimulationSchema,
    awards: AwardSettings | None = None,
    calendar_month: int = 1,
) -> Applied:
    """Apply `result` to `state` at simulation step `step`, in `calendar_month`."""
    n = state.n_cost_centers
    e, o = result.employees, result.openings
    _check(e, state.live_employees)
    _check(o, state.live_openings)
    # Both capacity checks run before any write, so a refused result leaves state untouched.
    state.reserve("employee", len(e.append) - len(e.drop))
    state.reserve("opening", len(o.append) - len(o.drop))
    ecc = EmployeeColumn.COST_CENTER
    occ, num = OpeningColumn.COST_CENTER, OpeningColumn.NUM_OPENINGS

    # 1. Drop employees, forfeiting their unvested equity.
    dropped = state.live_employees[e.drop].copy()
    forfeited = 0
    if len(dropped) and result.forfeit and state.ledger is not None:
        slots = dropped[:, EmployeeColumn.VEST_INDEX].astype(np.int64)
        forfeited = state.ledger.forfeit(slots, step)
    state.headcount -= per_cost_center(dropped, ecc, n)
    state.n_employees = _drop(state.employees, state.n_employees, e.drop)

    # 2. Write employee rows in place.
    written = np.zeros(0, dtype=np.int64)
    if len(e.write_index):
        index = _after_drops(e.write_index, e.drop)
        written = index
        state.headcount -= per_cost_center(state.employees[index], ecc, n)
        state.employees[index] = e.write_rows
        state.headcount += per_cost_center(e.write_rows, ecc, n)

    # 3. Append employees, with new ids and vest slots.
    appended = e.append.copy()
    if len(appended):
        state.next_employee_id = _new_ids(
            appended, EmployeeColumn.EMPLOYEE_ID, state.next_employee_id
        )
        appended[:, EmployeeColumn.VEST_INDEX] = state.next_vest_slot + np.arange(len(appended))
        state.next_vest_slot += len(appended)
        start = state.n_employees
        state.employees[start : start + len(appended)] = appended
        state.n_employees += len(appended)
        state.headcount += per_cost_center(appended, ecc, n)

    # 4. Drop requisitions.
    dropped_reqs = state.live_openings[o.drop].copy()
    state.open_openings -= per_cost_center(dropped_reqs, occ, n, num)
    state.open_rows -= per_cost_center(dropped_reqs, occ, n)
    state.n_openings = _drop(state.openings, state.n_openings, o.drop)

    # 5. Write requisition rows in place.
    if len(o.write_index):
        index = _after_drops(o.write_index, o.drop)
        before = state.openings[index]
        state.open_openings -= per_cost_center(before, occ, n, num)
        state.open_rows -= per_cost_center(before, occ, n)
        state.openings[index] = o.write_rows
        state.open_openings += per_cost_center(o.write_rows, occ, n, num)
        state.open_rows += per_cost_center(o.write_rows, occ, n)

    # 6. Append requisitions, with new ids.
    appended_reqs = o.append.copy()
    if len(appended_reqs):
        state.next_requisition_id = _new_ids(
            appended_reqs, OpeningColumn.REQ_ID, state.next_requisition_id
        )
        start = state.n_openings
        state.openings[start : start + len(appended_reqs)] = appended_reqs
        state.n_openings += len(appended_reqs)
        state.open_openings += per_cost_center(appended_reqs, occ, n, num)
        state.open_rows += per_cost_center(appended_reqs, occ, n)

    # 7. Size and record grants, for the appended employees or the written ones.
    recorded = unsized = 0
    if result.grants and awards is not None and state.ledger is not None:
        first = state.n_employees - len(appended)
        for grant in result.grants:
            positions = np.arange(first, state.n_employees) if grant.rows == "appended" else written
            if not len(positions):
                continue
            if state.price is None:
                raise RuntimeError("a run that sizes awards needs a price model in its state")
            rows = state.employees[positions]
            recorded_batch = (
                None
                if awards.replayed is None
                else awards.replayed.batch(rows, step, grant.award_type, state.price)
            )
            sizer = awards.sizer(grant.award_type)
            if recorded_batch is not None:
                batch = recorded_batch
            elif sizer is not None:
                context = SizingContext(step, calendar_month, state.ledger)
                batch = sizer.size(rows, state.price, context)
            else:
                unsized += len(positions)
                continue
            state.employees[positions, EmployeeColumn.BASE_SALARY] = batch.base_salary
            recorded += state.ledger.record(
                rows[:, EmployeeColumn.VEST_INDEX].astype(np.int64),
                step,
                batch.vest_schedule,
                schema.vocabulary(AWARD_TYPE).code(grant.award_type),
                rows[:, EmployeeColumn.LEVEL].astype(np.int64),
                rows[:, ecc].astype(np.int64),
            )
        appended = state.employees[first : state.n_employees].copy()

    # 8. The result's use of shared guidance quota.
    for category, used in result.ledger.items():
        current = state.category_ledger.get(category)
        state.category_ledger[category] = used.copy() if current is None else current + used

    return Applied(dropped, appended, dropped_reqs, appended_reqs, forfeited, recorded, unsized)
