"""Transfer matching: placing an employee into each opening the fills assigned to transfer.

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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import Buffer, EmployeeColumn, OpeningColumn

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
