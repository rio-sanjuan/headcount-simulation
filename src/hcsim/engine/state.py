"""One trajectory's state: two preallocated buffers, their live counts, and the counters every
gap and ceiling reads.

Only the apply step writes state. The per-cost-center counters move with the buffers, and
`recount` rebuilds them from the buffers, so a test or an invariant check can confirm they
agree. State has no trajectory axis: each trajectory is an independent deep copy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hcsim.financial.prices import StockPriceModel
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.layouts import EMPLOYEE, OPENING, Buffer, EmployeeColumn, OpeningColumn

Ints = NDArray[np.int64]


class BufferCapacityError(RuntimeError):
    """An append would overflow a buffer. Buffers never grow, because a silent overflow would
    advance the counters while dropping rows."""


@dataclass
class SimulationState:
    employees: Buffer
    n_employees: int
    openings: Buffer
    n_openings: int
    n_cost_centers: int
    headcount: Ints = field(init=False)
    open_openings: Ints = field(init=False)
    open_rows: Ints = field(init=False)
    next_employee_id: int = 0
    next_requisition_id: int = 0
    next_vest_slot: int = 0
    ledger: HolderLedger | None = None
    price: StockPriceModel | None = None
    #: Per step: how much of each shared guidance quota the domains have used, by group.
    category_ledger: dict[str, Ints] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.employees.shape[1] != EMPLOYEE.width or self.openings.shape[1] != OPENING.width:
            raise ValueError("buffers must use the declared layouts")
        self.headcount, self.open_openings, self.open_rows = self.recount()

    @property
    def live_employees(self) -> Buffer:
        return self.employees[: self.n_employees]

    @property
    def live_openings(self) -> Buffer:
        return self.openings[: self.n_openings]

    def recount(self) -> tuple[Ints, Ints, Ints]:
        """Headcount, open openings, and open requisition rows per cost center, from the buffers."""
        n = self.n_cost_centers
        cc = self.live_employees[:, EmployeeColumn.COST_CENTER].astype(np.int64)
        req_cc = self.live_openings[:, OpeningColumn.COST_CENTER].astype(np.int64)
        openings = self.live_openings[:, OpeningColumn.NUM_OPENINGS]
        return (
            np.bincount(cc, minlength=n).astype(np.int64),
            np.bincount(req_cc, weights=openings, minlength=n).astype(np.int64),
            np.bincount(req_cc, minlength=n).astype(np.int64),
        )

    def counters_agree(self) -> bool:
        h, o, rows = self.recount()
        return bool(
            (h == self.headcount).all()
            and (o == self.open_openings).all()
            and (rows == self.open_rows).all()
        )

    def reserve(self, buffer: str, rows: int) -> None:
        """Raise if `rows` more rows wouldn't fit, naming what to raise."""
        array, live = (
            (self.employees, self.n_employees)
            if buffer == "employee"
            else (self.openings, self.n_openings)
        )
        if live + rows > len(array):
            raise BufferCapacityError(
                f"the {buffer} buffer holds {len(array)} rows; appending {rows} to {live} live "
                f"rows would overflow it. Raise the {buffer} capacity at seeding."
            )

    def clone(self) -> SimulationState:
        return copy.deepcopy(self)
