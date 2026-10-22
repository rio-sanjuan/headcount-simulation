"""The pending-starts domain: accepted offers that start during the run.

Each pending start fires in its start month, before the month's fills, and the person joins
under their own id, with a new-hire grant:

- **matched**: when their requisition is open in the run, they fill it, taking its role and
  one of its openings, the one reserved for them since the run began. They count as `F_pend`;
- **unmatched**: when it isn't, because it wasn't seeded or has closed, they still join, in
  the role their offer names. No opening is consumed, so they count as a boundary fill,
  `F_bdy`.

In replay, a realized hire is a pending start into the requisition it filled. One whose
requisition the run doesn't hold is a boundary fill only when `replay_boundary_fill` allows;
otherwise it's left out, with the rest of their known events.

An offer whose start passed before the run started is still expected for
`pending_starts_staleness_days`, and starts in the run's first month; an older one is stale and
never becomes a known event.

**The carried fill nodes.** Under guidance, a pending start is an external hire, so it counts
against the most specific fill node the guidance carries: external hires, or else hires, or
else fills. It's recorded in the category ledger, and external fill samples only the rest of
its quota.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from hcsim.engine.domains import StepContext
from hcsim.engine.known_events import EventKind, Source
from hcsim.engine.results import DomainResult, Grant, RowChange
from hcsim.engine.state import SimulationState
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.fill_tree import ancestors
from hcsim.model.core.layouts import EMPLOYEE, OPENING, EmployeeColumn, OpeningColumn, copy_role

PENDING_STARTS: Final = "pending_starts"
#: The fill nodes a pending start can count against, most specific first.
CARRIED: Final = (
    GuidanceCategory.EXTERNAL_HIRES,
    *ancestors(GuidanceCategory.EXTERNAL_HIRES),
)


def pending_starts(state: SimulationState, ctx: StepContext) -> DomainResult:
    events = ctx.known_at(EventKind.PENDING_START)
    if not events:
        return DomainResult(PENDING_STARTS)
    reqs = state.live_openings
    where = {int(r): k for k, r in enumerate(reqs[:, OpeningColumn.REQ_ID])}
    remaining = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64).copy()
    hires = EMPLOYEE.empty(len(events))
    matched = np.zeros(len(events), dtype=bool)
    joining = np.ones(len(events), dtype=bool)
    for k, e in enumerate(events):
        row = None if e.event.req_id is None else where.get(int(e.event.req_id))
        if row is not None and remaining[row] >= 1:
            copy_role(reqs[row : row + 1], hires[k : k + 1])
            remaining[row] -= 1
            matched[k] = True
        elif e.event.source is Source.REPLAY and not ctx.config.replay_boundary_fill:
            joining[k] = False  # a replayed hire with no requisition, and no boundary fills
            if e.event.employee_id is not None:
                ctx.fired.leave_out(e.event.employee_id)
        else:
            for column, code in e.codes.items():
                hires[k, column] = code
        hires[k, EmployeeColumn.EMPLOYEE_ID] = e.event.employee_id
        hires[k, EmployeeColumn.HIRE_STEP] = ctx.step
        ctx.fired.record(e.event)
    hires, matched = hires[joining], matched[joining]
    total = reqs[:, OpeningColumn.NUM_OPENINGS].astype(np.int64)
    touched = np.flatnonzero(remaining < total)
    closed = touched[remaining[touched] == 0]
    partial = touched[remaining[touched] > 0]
    written = reqs[partial].copy()
    written[:, OpeningColumn.NUM_OPENINGS] = remaining[partial]
    n = state.n_cost_centers
    cc = hires[:, EmployeeColumn.COST_CENTER].astype(np.int64)
    return DomainResult(
        PENDING_STARTS,
        employees=RowChange(EMPLOYEE, append=hires),
        openings=RowChange(OPENING, drop=closed, write_index=partial, write_rows=written),
        grants=(Grant("new_hire"),),
        ledger=_ledger(hires, ctx),
        flows={
            "F_pend": np.bincount(cc[matched], minlength=n).astype(np.int64),
            "F_bdy": np.bincount(cc[~matched], minlength=n).astype(np.int64),
        },
    )


def _ledger(hires: np.ndarray, ctx: StepContext) -> dict[str, np.ndarray]:
    """The starts per group of the most specific fill node the guidance carries."""
    for node in CARRIED:
        if ctx.guided(node):
            groups = ctx.guided_groups(hires, EMPLOYEE, node)
            size = len(ctx.quota(node))
            covered = groups[groups >= 0]
            return {node.value: np.bincount(covered, minlength=size).astype(np.int64)}
    return {}
