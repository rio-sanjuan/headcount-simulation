"""The promotion domain: who moves up a level, in the months promotions take effect.

Each month:

1. **Known promotions fire first**, in whatever month they're known for. Each moves its
   employee to its known destination, as resolved against the schema.
2. **Then, in a cycle month, everyone else draws.** Sampled promotions take effect only in
   `promotion_cycle_months`, through a stochastic step gate that opens with
   `promotion_gate_probability`, drawn from the gate's own seeded stream, so a cycle can be
   skipped at random. The promotion model scores each employee row. Eligibility rules and
   known-event exclusions zero out their rows, and so does the top level, from which there's
   nowhere to go. Where guidance covers the `promotions` category, quota sampling
   realizes the group's quota less the known promotions; elsewhere each row draws on its own.
3. **A promoted employee moves to the next level**, as the schema's level vocabulary orders
   them. With `promotion_family_change_probability`, a sampled promotion also moves to another
   job family, drawn in proportion to the similarity matrix's row for the current one, from its
   own seeded stream. A family with no similar family stays put.
4. **Every promotion emits a grant** on the rows it rewrote. The run sizes it when its award
   settings have a promotion sizer.

A promotion changes a role, not a stock, so it counts as the flow `Pi` per cost center and
appears in no accounting identity. Without a promotion model, only known promotions happen.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from hcsim.engine.domains import StepContext, draw
from hcsim.engine.gates import StepGate
from hcsim.engine.known_events import EventKind
from hcsim.engine.results import DomainResult, Grant, RowChange
from hcsim.engine.state import SimulationState
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.layouts import EMPLOYEE, Buffer, EmployeeColumn
from hcsim.model.core.schema import JOB_LEVEL
from hcsim.model.interface import Mask

PROMOTION: Final = "promotion"


def promotion(state: SimulationState, ctx: StepContext) -> DomainResult:
    rows = state.live_employees
    n = state.n_cost_centers
    model = ctx.config.models.promotion
    known_events = ctx.known_at(EventKind.PROMOTION)
    known = ctx.fire_known(rows, EventKind.PROMOTION)
    written = rows.copy()
    destination = {e.event.employee_id: e.codes for e in known_events}
    for row in np.flatnonzero(known):
        for column, code in destination[int(rows[row, EmployeeColumn.EMPLOYEE_ID])].items():
            written[row, column] = code
    sampled = np.zeros(len(rows), dtype=bool)
    cycle = StepGate(
        PROMOTION, ctx.config.promotion_cycle_months, ctx.config.promotion_gate_probability
    )
    if model is not None and cycle.opens(ctx.month, ctx.rng(f"gate:{PROMOTION}")):
        top = len(ctx.config.schema.vocabulary(JOB_LEVEL)) - 1
        p = model.compute_probability(rows)
        p = ctx.sampled(rows, ctx.config.resolved_eligibility.restrict(PROMOTION, rows, p))
        p = np.where(known | (rows[:, EmployeeColumn.LEVEL] >= top), 0.0, p)
        sampled, quotas = draw(
            p, rows, EMPLOYEE, GuidanceCategory.PROMOTIONS, ctx, ctx.rng(PROMOTION), known
        )
        written[sampled, EmployeeColumn.LEVEL] += 1
        _change_family(written, sampled, ctx)
    else:
        quotas = {}
    index = np.flatnonzero(sampled | known)
    promoted = written[index]
    counts = np.bincount(
        promoted[:, EmployeeColumn.COST_CENTER].astype(np.int64), minlength=n
    ).astype(np.int64)
    return DomainResult(
        PROMOTION,
        employees=RowChange(EMPLOYEE, write_index=index, write_rows=promoted),
        grants=(Grant("promotion", rows="written"),),
        quotas=quotas,
        flows={"Pi": counts},
    )


def _change_family(written: Buffer, sampled: Mask, ctx: StepContext) -> None:
    """Move a share of the sampled promotions to a similar job family, in place."""
    chance = ctx.config.promotion_family_change_probability
    similarity = ctx.config.models.family_similarity
    if chance <= 0 or similarity is None:
        return
    rng = ctx.rng(f"{PROMOTION}:family")
    rows = np.flatnonzero(sampled)
    changing = rows[rng.random(len(rows)) < chance]
    matrix = similarity.matrix
    for row in changing:
        family = int(written[row, EmployeeColumn.JOB_FAMILY])
        weights = matrix[family].copy()
        weights[family] = 0.0
        if weights.sum() > 0:
            written[row, EmployeeColumn.JOB_FAMILY] = rng.choice(
                len(weights), p=weights / weights.sum()
            )
