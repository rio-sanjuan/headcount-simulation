"""The step loop: every trajectory, every month, every domain, then the expense phase.

Each trajectory runs on its own deep copy of the seeded state, with its own price path and
holder ledger, and draws from its own random streams. A month advances the price, clears the
category ledger, then runs the registered domains in order, applying each result before the next
domain reads state. The known-event domains run first, so the sampled domains see the workforce
they leave, and then performance, so the month's draws read its new ratings. After the last
month, every recorded tranche is valued at the price in its award month and amortized, and every
known event of a registered kind must have fired.

Histories are per cost center: stocks at the start of every month plus the end of the last
(`months + 1` rows), and each flow per month, named as in the notation. They obey the runtime
accounting identities, `HEADCOUNT_IDENTITY` and `OPENING_IDENTITY`:

    H(t+1) = H(t) + F_ext - A - V_conv + R_in - R_out + W_in - W_out
    O(t+1) = O(t) + G + B_att + B_tr - U - F_ext
"""

from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.apply import Applied, apply, per_cost_center
from hcsim.engine.config import SimulationConfig
from hcsim.engine.domains import (
    ATTRITION,
    BACKFILL,
    BACKFILL_TRANSFER,
    CANCELLATION,
    DOMAINS,
    EXTERNAL_FILL,
    GROWTH,
    Departures,
    Handler,
    StepContext,
)
from hcsim.engine.gates import Registered, RegistrationGate, register
from hcsim.engine.known_domains import (
    CONVERSION_OUT,
    KINDS,
    OTHER_MOVEMENT,
    REORGANIZATION,
    conversion_out,
    other_movement,
    reorganization,
)
from hcsim.engine.known_events import EventKind
from hcsim.engine.known_resolution import FiringLog, ResolvedPlan, check_fired
from hcsim.engine.modes import RunMode
from hcsim.engine.performance import PERFORMANCE, performance
from hcsim.engine.promotion import PROMOTION, promotion
from hcsim.engine.results import QuotaRecord
from hcsim.engine.seeds import SeedManager
from hcsim.engine.state import SimulationState
from hcsim.financial.expense import ExpenseResult, compute_expense, sbc_frame
from hcsim.financial.tracker import HolderLedger, Tranches
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn

Ints = NDArray[np.int64]

#: The flow each single-flow domain contributes, by its symbol in the notation.
FLOWS: Final[Mapping[str, str]] = {
    ATTRITION: "A",
    BACKFILL: "B_att",
    BACKFILL_TRANSFER: "B_tr",
    EXTERNAL_FILL: "F_ext",
    CANCELLATION: "U",
    GROWTH: "G",
}
#: Every flow a run records.
FLOW_SYMBOLS: Final = (*FLOWS.values(), "V_conv", "R_in", "R_out", "W_in", "W_out", "Pi")
#: The runtime accounting identities: each flow's sign in the change of a stock.
HEADCOUNT_IDENTITY: Final[Mapping[str, int]] = {
    "F_ext": +1,
    "A": -1,
    "V_conv": -1,
    "R_in": +1,
    "R_out": -1,
    "W_in": +1,
    "W_out": -1,
}
OPENING_IDENTITY: Final[Mapping[str, int]] = {
    "G": +1,
    "B_att": +1,
    "B_tr": +1,
    "U": -1,
    "F_ext": -1,
}
#: Every domain, in step order: the known-event domains, then the sampled ones.
STEP_ORDER: Final[tuple[tuple[str, Handler], ...]] = (
    (CONVERSION_OUT, conversion_out),
    (REORGANIZATION, reorganization),
    (OTHER_MOVEMENT, other_movement),
    (PERFORMANCE, performance),
    (PROMOTION, promotion),
    *DOMAINS,
)
#: The known-event kind each domain fires.
FIRES: Final[Mapping[str, EventKind]] = {
    **KINDS,
    PROMOTION: EventKind.PROMOTION,
    ATTRITION: EventKind.TERMINATION,
}
#: Gates every run has: other movement comes only from replayed history.
DEFAULT_GATES: Final = (
    RegistrationGate(OTHER_MOVEMENT, modes=(RunMode.REPLAY,), reason="replay only"),
)


class InvariantError(AssertionError):
    """State whose counters no longer match its buffers."""


@dataclass(frozen=True)
class QuotaRow:
    """One guided draw: a domain, a category, a step, and its accounting per group."""

    step: int
    domain: str
    category: str
    record: QuotaRecord


@dataclass(frozen=True)
class TrajectoryResult:
    trajectory: int
    #: (months + 1, cost centers): headcount at the start of each month, then the end.
    headcount: Ints
    #: (months + 1, cost centers): open openings, likewise.
    open_openings: Ints
    #: Flow symbol to (months, cost centers).
    flows: Mapping[str, Ints]
    quotas: tuple[QuotaRow, ...]
    #: The price in force each month, after that month's step. Empty without SBC.
    prices: NDArray[np.float64]
    tranches: Tranches | None = None
    expense: ExpenseResult | None = None
    sbc: pd.DataFrame | None = None
    #: Months a step gate kept each registered domain from running.
    bypasses: Mapping[str, int] = field(default_factory=dict)
    #: Known events fired, by kind.
    known_fired: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    trajectories: tuple[TrajectoryResult, ...]
    #: Domains no registration gate admitted.
    unregistered: tuple[str, ...] = ()

    @property
    def headcount(self) -> Ints:
        """(trajectories, months + 1, cost centers)."""
        return np.stack([t.headcount for t in self.trajectories])

    @property
    def open_openings(self) -> Ints:
        return np.stack([t.open_openings for t in self.trajectories])

    def flow(self, symbol: str) -> Ints:
        """(trajectories, months, cost centers) for one flow symbol."""
        return np.stack([t.flows[symbol] for t in self.trajectories])


def calendar_month(run_date: dt.date, step: int) -> int:
    """The calendar month, 1-12, of simulation step `step`."""
    return (run_date.month - 1 + step) % 12 + 1


def _flow(domain: str, applied: Applied, n: int) -> Ints:
    cc, openings = OpeningColumn.COST_CENTER, OpeningColumn.NUM_OPENINGS
    if domain == ATTRITION:
        return per_cost_center(applied.dropped_employees, EmployeeColumn.COST_CENTER, n)
    if domain == EXTERNAL_FILL:
        return per_cost_center(applied.appended_employees, EmployeeColumn.COST_CENTER, n)
    if domain == CANCELLATION:
        return per_cost_center(applied.dropped_openings, cc, n, openings)
    return per_cost_center(applied.appended_openings, cc, n, openings)  # backfill and growth


def run_trajectory(
    config: SimulationConfig,
    initial: SimulationState,
    trajectory: int,
    seeds: SeedManager,
    *,
    validate: bool = False,
    registered: Registered | None = None,
    known: ResolvedPlan | None = None,
) -> TrajectoryResult:
    """Run one trajectory from a copy of `initial`. `validate` recounts state after every
    domain, which costs a pass over both buffers each time."""
    state = initial.clone()
    awards = config.awards if config.computes_sbc else None
    if awards is not None:
        state.price = copy.deepcopy(awards.price_model)
        if state.ledger is None:
            state.ledger = HolderLedger()
    months, n = config.simulation_months, state.n_cost_centers
    headcount = np.zeros((months + 1, n), dtype=np.int64)
    open_openings = np.zeros((months + 1, n), dtype=np.int64)
    flows = {symbol: np.zeros((months, n), dtype=np.int64) for symbol in FLOW_SYMBOLS}
    prices = np.zeros(months if awards is not None else 0)
    quotas: list[QuotaRow] = []
    headcount[0], open_openings[0] = state.headcount, state.open_openings
    registered = registered or _register(config)
    handlers = dict(STEP_ORDER)
    bypasses = dict.fromkeys(registered.steps, 0)
    fired = FiringLog()

    for step in range(months):
        if state.price is not None:
            state.price.step(calendar_month(config.run_date, step))
            prices[step] = state.price.price
        state.category_ledger = {}
        ctx = StepContext(config, step, trajectory, seeds, known=known, fired=fired)
        month = calendar_month(config.run_date, step)
        for domain in registered.domains:
            if not registered.opens(domain, month, ctx.rng(f"gate:{domain}")):
                bypasses[domain] += 1
                continue
            result = handlers[domain](state, ctx)
            applied = apply(state, result, step, config.schema, awards)
            if validate and not state.counters_agree():
                raise InvariantError(f"counters drifted after {domain} at step {step}")
            if len(result.departures):
                ctx.departures.append(Departures(domain, result.departures, result.anticipated))
            if result.flows:
                for symbol, counts in result.flows.items():
                    flows[symbol][step] = counts
            elif domain in FLOWS:
                flows[FLOWS[domain]][step] = _flow(domain, applied, n)
            quotas += [QuotaRow(step, domain, c, r) for c, r in result.quotas.items()]
        headcount[step + 1], open_openings[step + 1] = state.headcount, state.open_openings

    known_fired: dict[str, int] = {}
    if known is not None:
        kinds = [FIRES[d] for d in registered.domains if d in FIRES]
        check_fired(known, fired, kinds)
        for e in known.events:
            if e.event.key in fired.fired:
                known_fired[e.event.kind.value] = known_fired.get(e.event.kind.value, 0) + 1
    if awards is None or state.ledger is None:
        return TrajectoryResult(
            trajectory,
            headcount,
            open_openings,
            flows,
            tuple(quotas),
            prices,
            bypasses=bypasses,
            known_fired=known_fired,
        )
    tranches = state.ledger.tranches()
    if len(tranches) and (tranches.award_month.min() < 0 or tranches.award_month.max() >= months):
        raise ValueError("every tranche must be awarded during the run to be valued at its price")
    expense = compute_expense(tranches, prices[tranches.award_month], awards.expense)
    return TrajectoryResult(
        trajectory,
        headcount,
        open_openings,
        flows,
        tuple(quotas),
        prices,
        tranches,
        expense,
        sbc_frame(expense, tranches),
        bypasses,
        known_fired,
    )


def simulate(
    config: SimulationConfig,
    initial: SimulationState,
    *,
    known: ResolvedPlan | None = None,
    validate: bool = False,
) -> SimulationResult:
    """Run every trajectory of `config` from the seeded state `initial`, which stays untouched,
    with the run's resolved known events."""
    seeds = SeedManager(config.seed)
    registered = _register(config)
    trajectories = tuple(
        run_trajectory(
            config, initial, j, seeds, validate=validate, registered=registered, known=known
        )
        for j in range(config.num_trajectories)
    )
    return SimulationResult(config, trajectories, registered.unregistered)


def _register(config: SimulationConfig) -> Registered:
    mode = config.run_mode or RunMode.UNGUIDED
    names = [name for name, _ in STEP_ORDER]
    defaults = [g for g in DEFAULT_GATES if g.domain in names]
    if config.models.performance is None and PERFORMANCE in names:
        defaults.append(RegistrationGate(PERFORMANCE, enabled=False, reason="no model"))
    return register(names, (*defaults, *config.gates), mode)
