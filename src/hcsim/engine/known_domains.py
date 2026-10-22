"""The known-event domains: conversion out, reorganization, and other movement.

Each fires the step's known events of its kind, as resolved against the schema, and runs before
any sampled domain, so the rest of the month sees the workforce they leave behind. Each event
it fires goes in the trajectory's firing log; an event whose employee isn't there to move is
left unfired, and the run's firing check names it.

- **Conversion out** (`V_conv`): the employee leaves the regular workforce. The row is dropped
  and its unvested equity forfeits. A conversion opens no backfill.
- **Reorganization** (`R_in`, `R_out`): the employee moves to the event's destination.
  - *within* the run: their row takes the destination's codes. A move between cost centers
    counts as `R_out` at the origin and `R_in` at the destination; a change of role alone
    counts neither;
  - *out* of the run: the row is dropped as `R_out`. The person hasn't left the company, so
    their equity doesn't forfeit;
  - *in* to the run: a row is added for them with the destination's whole role, under their
    own id, as `R_in`. They're new to the run, so their tenure starts at the step.
- **Other movement** (`W_in`, `W_out`): a recorded change that isn't a reorganization, applied
  like a move within. It takes part only in replay, where the realized history supplies it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.engine.domains import StepContext
from hcsim.engine.known_events import EventKind
from hcsim.engine.known_resolution import ResolvedEvent
from hcsim.engine.results import DomainResult, RowChange
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import EMPLOYEE, Buffer, EmployeeColumn

Ints = NDArray[np.int64]

CONVERSION_OUT: Final = "conversion_out"
REORGANIZATION: Final = "reorganization"
OTHER_MOVEMENT: Final = "other_movement"
#: The kind each known-event domain fires.
KINDS: Final = {
    CONVERSION_OUT: EventKind.CONVERSION_OUT,
    REORGANIZATION: EventKind.REORGANIZATION,
    OTHER_MOVEMENT: EventKind.OTHER_MOVEMENT,
}


def _rows(state: SimulationState, events: Sequence[ResolvedEvent]) -> list[int | None]:
    """Each event's live employee row, or None when the employee isn't there."""
    ids = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
    where = {int(e): k for k, e in enumerate(ids)}
    return [
        None if e.event.employee_id is None else where.get(int(e.event.employee_id)) for e in events
    ]


def _per_cost_center(codes: list[int], n: int) -> Ints:
    return np.bincount(np.asarray(codes, dtype=np.int64), minlength=n).astype(np.int64)


def conversion_out(state: SimulationState, ctx: StepContext) -> DomainResult:
    events = ctx.known_at(EventKind.CONVERSION_OUT)
    rows = [(e, k) for e, k in zip(events, _rows(state, events), strict=True) if k is not None]
    drop = np.array([k for _, k in rows], dtype=np.int64)
    for e, _ in rows:
        ctx.fired.record(e.event)
    origin = state.live_employees[drop, EmployeeColumn.COST_CENTER].astype(np.int64)
    return DomainResult(
        CONVERSION_OUT,
        employees=RowChange(EMPLOYEE, drop=drop),
        flows={"V_conv": _per_cost_center(origin.tolist(), state.n_cost_centers)},
    )


def _moves(
    state: SimulationState,
    ctx: StepContext,
    domain: str,
    kind: EventKind,
    symbols: tuple[str, str],
) -> DomainResult:
    events = ctx.known_at(kind)
    live = state.live_employees
    drop, write_index, write_rows, append = [], [], [], []
    inflow, outflow = [], []
    for e, row in zip(events, _rows(state, events), strict=True):
        if e.boundary == "in":
            if row is not None:
                continue  # already in the run: not a boundary-in after all
            new = EMPLOYEE.empty(1)[0]
            new[EmployeeColumn.EMPLOYEE_ID] = e.event.employee_id
            new[EmployeeColumn.HIRE_STEP] = ctx.step
            for column, code in e.codes.items():
                new[column] = code
            append.append(new)
            inflow.append(int(new[EmployeeColumn.COST_CENTER]))
        elif row is None:
            continue
        elif e.boundary == "out":
            drop.append(row)
            outflow.append(int(live[row, EmployeeColumn.COST_CENTER]))
        else:
            moved = live[row].copy()
            for column, code in e.codes.items():
                moved[column] = code
            before, after = (
                int(live[row, EmployeeColumn.COST_CENTER]),
                int(moved[EmployeeColumn.COST_CENTER]),
            )
            if before != after:
                outflow.append(before)
                inflow.append(after)
            write_index.append(row)
            write_rows.append(moved)
        ctx.fired.record(e.event)
    n = state.n_cost_centers
    return DomainResult(
        domain,
        employees=RowChange(
            EMPLOYEE,
            drop=np.array(drop, dtype=np.int64),
            write_index=np.array(write_index, dtype=np.int64),
            write_rows=_stack(write_rows),
            append=_stack(append),
        ),
        forfeit=False,
        flows={symbols[0]: _per_cost_center(inflow, n), symbols[1]: _per_cost_center(outflow, n)},
    )


def _stack(rows: list[Buffer]) -> Buffer:
    return np.stack(rows) if rows else EMPLOYEE.empty(0)


def reorganization(state: SimulationState, ctx: StepContext) -> DomainResult:
    return _moves(state, ctx, REORGANIZATION, EventKind.REORGANIZATION, ("R_in", "R_out"))


def other_movement(state: SimulationState, ctx: StepContext) -> DomainResult:
    return _moves(state, ctx, OTHER_MOVEMENT, EventKind.OTHER_MOVEMENT, ("W_in", "W_out"))
