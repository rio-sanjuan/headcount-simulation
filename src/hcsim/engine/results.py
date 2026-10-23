"""Domain results: a declarative description of everything one domain does in one step.

Handlers are pure. They read state and return a `DomainResult`, and only the apply step writes.
Every index in a result refers to the live rows as the handler saw them, before any change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from hcsim.model.core.layouts import EMPLOYEE, OPENING, Buffer, BufferLayout

Ints = NDArray[np.int64]


def _no_index() -> Ints:
    return np.zeros(0, dtype=np.int64)


def _unset() -> Buffer:
    return np.zeros((0, 0))


@dataclass(frozen=True)
class RowChange:
    """Drops, in-place writes, and appends for one buffer.

    Appended rows with a negative id get the next simulated id when applied.
    """

    layout: BufferLayout
    drop: Ints = field(default_factory=_no_index)
    write_index: Ints = field(default_factory=_no_index)
    write_rows: Buffer = field(default_factory=_unset)
    append: Buffer = field(default_factory=_unset)

    def __post_init__(self) -> None:
        for name in ("drop", "write_index"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.int64))
        for name in ("write_rows", "append"):
            rows = getattr(self, name)
            if rows.shape == (0, 0):
                object.__setattr__(self, name, self.layout.empty(0))
            elif rows.ndim != 2 or rows.shape[1] != self.layout.width:
                raise ValueError(f"{name} must use the {self.layout.name} layout")
        if len(self.write_index) != len(self.write_rows):
            raise ValueError("one written row per write index")
        if len(np.intersect1d(self.drop, self.write_index)):
            raise ValueError("a row can't be both dropped and written")

    @property
    def is_empty(self) -> bool:
        return not (len(self.drop) or len(self.write_index) or len(self.append))


def _no_employees() -> RowChange:
    return RowChange(EMPLOYEE)


def _no_openings() -> RowChange:
    return RowChange(OPENING)


@dataclass(frozen=True)
class Grant:
    """An award for each employee the result appends, as a hire's is, each it writes, as a
    promotion's is, or each it lists by live row (`index`), as an award domain's are. Handlers
    emit grants whatever the run's scope. A run sizes a grant only when its award settings have
    a sizer for the award type, and counts the rest as unsized."""

    award_type: str
    rows: Literal["appended", "written", "listed"] = "appended"
    #: The live rows a listed grant goes to. Grants compare by type and rows alone.
    index: Ints | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if (self.rows == "listed") != (self.index is not None):
            raise ValueError("a grant lists its rows (index) exactly when its rows are 'listed'")
        if self.index is not None:
            object.__setattr__(self, "index", np.asarray(self.index, dtype=np.int64))


@dataclass(frozen=True)
class QuotaRecord:
    """A guided draw's accounting, per group: what guidance asked, who could leave or fill,
    and what happened.

    A draw honors its quota when it realizes the smaller of quota and eligible, with two
    allowances. Known events fire whatever the quota, so `known`, the part of `realized` they
    took, can pass it. And a draw that takes whole rows weighing several units, as cancellation
    takes requisitions by their openings, stops short of the quota by `blocked`: what was left
    of it when every eligible row still untaken was too heavy to fit.
    """

    quota: Ints
    eligible: Ints
    realized: Ints
    known: Ints = field(default_factory=_no_index)
    blocked: Ints = field(default_factory=_no_index)

    def __post_init__(self) -> None:
        for name in ("known", "blocked"):
            if not len(getattr(self, name)):
                object.__setattr__(self, name, np.zeros(len(self.quota), dtype=np.int64))

    @property
    def shortfall(self) -> Ints:
        """What each group's quota asked for and didn't get: eligibility ran short."""
        result: Ints = np.maximum(self.quota - self.realized, 0)
        return result

    @property
    def expected(self) -> Ints:
        """What each group must realize to honor its quota."""
        result: Ints = np.maximum(self.known, np.minimum(self.quota, self.eligible) - self.blocked)
        return result

    @property
    def honored(self) -> NDArray[np.bool_]:
        """Whether each group realized exactly what its quota asked, as far as it could."""
        result: NDArray[np.bool_] = self.realized == self.expected
        return result


@dataclass(frozen=True)
class DomainResult:
    domain: str
    employees: RowChange = field(default_factory=_no_employees)
    openings: RowChange = field(default_factory=_no_openings)
    grants: tuple[Grant, ...] = ()
    #: Whether dropped employees forfeit their unvested equity.
    forfeit: bool = True
    #: Pre-departure rows of employees who left a role this step, for backfill to read.
    departures: Buffer = field(default_factory=lambda: EMPLOYEE.empty(0))
    #: For each departure, whether a known event anticipated it. Empty means none was.
    anticipated: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    #: Shared guidance quota this result used, by category, per group.
    ledger: Mapping[str, Ints] = field(default_factory=dict)
    #: Accounting for guided draws, by category.
    quotas: Mapping[str, QuotaRecord] = field(default_factory=dict)
    #: Flows per cost center, by symbol, for a domain with more than one. A domain with one
    #: flow leaves this empty, and the loop reads its flow from what the apply step changed.
    flows: Mapping[str, Ints] = field(default_factory=dict)
    #: Openings assigned to the transfer channel, by requisition id, for matching to place.
    transfers: Mapping[int, int] = field(default_factory=dict)
    #: Diagnostic counts per cost center, by name, such as degenerate generated rows.
    diagnostics: Mapping[str, Ints] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.employees.layout is not EMPLOYEE or self.openings.layout is not OPENING:
            raise ValueError("employees and openings changes must use their own layouts")
        if self.departures.ndim != 2 or self.departures.shape[1] != EMPLOYEE.width:
            raise ValueError("departures must use the employee layout")
        if not len(self.anticipated):
            object.__setattr__(self, "anticipated", np.zeros(len(self.departures), dtype=bool))
        elif self.anticipated.shape != (len(self.departures),):
            raise ValueError("one anticipated flag per departure")
