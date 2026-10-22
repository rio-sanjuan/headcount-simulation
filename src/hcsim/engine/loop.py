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

    H(t+1) = H(t) + F_ext + F_conv + F_tr + F_camp + F_pend + F_bdy
                  - A - V_tr - V_conv + R_in - R_out + W_in - W_out
    O(t+1) = O(t) + G + G_camp + B_att + B_tr - U - F_ext - F_conv - F_tr - F_camp - F_pend
"""

from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.apply import Applied, apply, per_cost_center
from hcsim.engine.campus import CAMPUS, campus
from hcsim.engine.config import SimulationConfig
from hcsim.engine.domains import (
    ATTRITION,
    BACKFILL,
    BACKFILL_TRANSFER,
    CANCELLATION,
    DOMAINS,
    GROWTH,
    Departures,
    Handler,
    StepContext,
)
from hcsim.engine.events import EventCells, cells_frame, decode
from hcsim.engine.events import empty as no_cells
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
from hcsim.engine.pending import PENDING_STARTS, pending_starts
from hcsim.engine.performance import PERFORMANCE, performance
from hcsim.engine.promotion import PROMOTION, promotion
from hcsim.engine.replay import REPLAY_REQUISITIONS, ReplayedRequisitions, replay_requisitions
from hcsim.engine.results import QuotaRecord
from hcsim.engine.seeds import SeedManager
from hcsim.engine.state import SimulationState
from hcsim.engine.transfers import TRANSFERS_DOMAIN as TRANSFERS
from hcsim.engine.transfers import transfers
from hcsim.financial.expense import ExpenseResult, compute_expense, sbc_frame
from hcsim.financial.tracker import HolderLedger, Tranches
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.core.schema import COST_CENTER

Ints = NDArray[np.int64]

#: The flow each single-flow domain contributes, by its symbol in the notation.
FLOWS: Final[Mapping[str, str]] = {
    ATTRITION: "A",
    BACKFILL: "B_att",
    BACKFILL_TRANSFER: "B_tr",
    CANCELLATION: "U",
    GROWTH: "G",
}
#: Every flow a run records.
FLOW_SYMBOLS: Final = (
    *FLOWS.values(),
    "F_ext",
    "F_conv",
    "F_tr",
    "F_camp",
    "F_pend",
    "F_bdy",
    "G_camp",
    "V_tr",
    "V_conv",
    "R_in",
    "R_out",
    "W_in",
    "W_out",
    "Pi",
)
#: The runtime accounting identities: each flow's sign in the change of a stock.
HEADCOUNT_IDENTITY: Final[Mapping[str, int]] = {
    "F_ext": +1,
    "F_conv": +1,
    "F_tr": +1,
    "F_camp": +1,
    "F_pend": +1,
    "F_bdy": +1,
    "A": -1,
    "V_tr": -1,
    "V_conv": -1,
    "R_in": +1,
    "R_out": -1,
    "W_in": +1,
    "W_out": -1,
}
OPENING_IDENTITY: Final[Mapping[str, int]] = {
    "G": +1,
    "G_camp": +1,
    "B_att": +1,
    "B_tr": +1,
    "U": -1,
    "F_ext": -1,
    "F_conv": -1,
    "F_tr": -1,
    "F_camp": -1,
    "F_pend": -1,
}
#: Every domain, in step order: the known-event domains, then the sampled ones.
STEP_ORDER: Final[tuple[tuple[str, Handler], ...]] = (
    (REPLAY_REQUISITIONS, replay_requisitions),  # replay's realized requisitions open first
    (CONVERSION_OUT, conversion_out),
    (REORGANIZATION, reorganization),
    (OTHER_MOVEMENT, other_movement),
    (PERFORMANCE, performance),
    (PROMOTION, promotion),
    *DOMAINS[:2],  # attrition, then backfill for its departures
    (PENDING_STARTS, pending_starts),  # starts take their reserved openings before fills
    DOMAINS[2],  # fills
    (TRANSFERS, transfers),  # places the fills' transfer assignments
    (CAMPUS, campus),  # opens and fills campus requisitions on their schedule
    *DOMAINS[3:],
)
#: Each domain's code in the event cells: its place in the step order.
DOMAIN_INDEX: Final[Mapping[str, int]] = {name: k for k, (name, _) in enumerate(STEP_ORDER)}
#: The known-event kind each domain fires.
FIRES: Final[Mapping[str, EventKind]] = {
    **KINDS,
    PROMOTION: EventKind.PROMOTION,
    ATTRITION: EventKind.TERMINATION,
    PENDING_STARTS: EventKind.PENDING_START,
    TRANSFERS: EventKind.TRANSFER,
}
#: Gates every run has: other movement comes only from replayed history.
DEFAULT_GATES: Final = (
    RegistrationGate(OTHER_MOVEMENT, modes=(RunMode.REPLAY,), reason="replay only"),
    RegistrationGate(REPLAY_REQUISITIONS, modes=(RunMode.REPLAY,), reason="replay only"),
)


#: The columns of a shortfall report.
SHORTFALL_COLUMNS: Final = (
    "trajectory",
    "step",
    "month",
    "domain",
    "category",
    "group",
    "quota",
    "eligible",
    "realized",
    "shortfall",
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
    #: Diagnostics by name, as (months, cost centers) counts, such as degenerate rows.
    diagnostics: Mapping[str, Ints] = field(default_factory=dict)
    #: The trajectory's event cells, as `events.EventCells` gathers them.
    event_cells: Ints = field(default_factory=no_cells)
    #: Grants each domain emitted that the run's award settings had no sizer for.
    unsized_grants: Mapping[str, int] = field(default_factory=dict)


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

    def shortfalls(self) -> pd.DataFrame:
        """Every group a guided draw left short of its quota: one row per trajectory, step,
        domain, category, and group, with the quota, what was eligible, what was realized,
        and the shortfall. A shortfall means eligibility ran out before the quota did, as
        when guidance asks for more fills than there are openings to fill."""
        groups = () if self.config.guidance is None else self.config.guidance.groups
        rows = [
            {
                "trajectory": t.trajectory,
                "step": q.step,
                "month": calendar_month(self.config.run_date, q.step),
                "domain": q.domain,
                "category": q.category,
                "group": groups[g],
                "quota": int(q.record.quota[g]),
                "eligible": int(q.record.eligible[g]),
                "realized": int(q.record.realized[g]),
                "shortfall": int(short),
            }
            for t in self.trajectories
            for q in t.quotas
            for g, short in enumerate(q.record.shortfall.tolist())
            if short > 0
        ]
        return pd.DataFrame(rows, columns=list(SHORTFALL_COLUMNS))

    def shortfall_totals(self) -> pd.DataFrame:
        """Shortfalls summed by domain and category, averaged over trajectories."""
        frame = self.shortfalls()
        totals = frame.groupby(["domain", "category"])["shortfall"].sum().reset_index()
        totals["per_trajectory"] = totals["shortfall"] / len(self.trajectories)
        return totals

    def event_cells(self, *, decoded: bool = False, trajectory: int | None = None) -> pd.DataFrame:
        """What every domain did, counted by cell (see `events`): one row per trajectory,
        step, domain, buffer, change, cost center, level, and job family. The attributes are
        codes, or with `decoded`, their labels; `trajectory` keeps one trajectory's cells."""
        cells = [(t.trajectory, t.event_cells) for t in self._of(trajectory)]
        return cells_frame(cells, list(DOMAIN_INDEX), self.config.schema, decoded=decoded)

    def channel_counts(
        self, *, decoded: bool = False, trajectory: int | None = None
    ) -> pd.DataFrame:
        """Every channel's count: one row per trajectory, step, flow symbol, and cost center
        the flow moved, as the notation names the channels by which headcount and openings
        change."""
        return self._long("flow", lambda t: t.flows, decoded, trajectory)

    def diagnostics(self, *, decoded: bool = False, trajectory: int | None = None) -> pd.DataFrame:
        """Every nonzero diagnostic: one row per trajectory, step, name, and cost center."""
        return self._long("name", lambda t: t.diagnostics, decoded, trajectory)

    def run_counts(self, *, trajectory: int | None = None) -> pd.DataFrame:
        """Each trajectory's counts for the whole run: months each domain's step gate
        bypassed it, known events fired by kind, and grants left unsized by domain."""
        rows = [
            {"trajectory": t.trajectory, "kind": kind, "name": name, "count": int(count)}
            for t in self._of(trajectory)
            for kind, counts in (
                ("bypassed", t.bypasses),
                ("known_fired", t.known_fired),
                ("unsized_grants", t.unsized_grants),
            )
            for name, count in counts.items()
            if count
        ]
        return pd.DataFrame(rows, columns=["trajectory", "kind", "name", "count"])

    def _of(self, trajectory: int | None) -> list[TrajectoryResult]:
        if trajectory is None:
            return list(self.trajectories)
        chosen = [t for t in self.trajectories if t.trajectory == trajectory]
        if not chosen:
            raise KeyError(f"no trajectory {trajectory} in this result")
        return chosen

    def _long(
        self,
        key: str,
        series: Callable[[TrajectoryResult], Mapping[str, Ints]],
        decoded: bool,
        trajectory: int | None,
    ) -> pd.DataFrame:
        """(months, cost centers) counts by name, as one row per nonzero count."""
        frames = []
        for t in self._of(trajectory):
            for name, counts in series(t).items():
                step, cc = np.nonzero(counts)
                frames.append(
                    pd.DataFrame(
                        {
                            "trajectory": t.trajectory,
                            "step": step.astype(np.int64),
                            key: name,
                            COST_CENTER: cc.astype(np.int64),
                            "count": counts[step, cc].astype(np.int64),
                        }
                    )
                )
        columns = ["trajectory", "step", key, COST_CENTER, "count"]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        frame = frame[columns]
        if decoded:
            codes = frame[COST_CENTER].to_numpy(dtype=np.int64)
            frame[COST_CENTER] = decode(self.config.schema, COST_CENTER, codes)
        return frame


def calendar_month(run_date: dt.date, step: int) -> int:
    """The calendar month, 1-12, of simulation step `step`."""
    return (run_date.month - 1 + step) % 12 + 1


def _flow(domain: str, applied: Applied, n: int) -> Ints:
    cc, openings = OpeningColumn.COST_CENTER, OpeningColumn.NUM_OPENINGS
    if domain == ATTRITION:
        return per_cost_center(applied.dropped_employees, EmployeeColumn.COST_CENTER, n)
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
    replay: ReplayedRequisitions | None = None,
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
    diagnostics: dict[str, Ints] = {}
    cells = EventCells(config.schema)
    unsized = dict.fromkeys(registered.domains, 0)

    for step in range(months):
        if state.price is not None:
            state.price.step(calendar_month(config.run_date, step))
            prices[step] = state.price.price
        state.category_ledger = {}
        ctx = StepContext(config, step, trajectory, seeds, known=known, fired=fired, replay=replay)
        month = calendar_month(config.run_date, step)
        for domain in registered.domains:
            if not registered.opens(domain, month, ctx.rng(f"gate:{domain}")):
                bypasses[domain] += 1
                continue
            result = handlers[domain](state, ctx)
            cells.capture(step, DOMAIN_INDEX[domain], result, state)
            applied = apply(state, result, step, config.schema, awards)
            unsized[domain] += applied.unsized_grants
            if validate and not state.counters_agree():
                raise InvariantError(f"counters drifted after {domain} at step {step}")
            for name, counts in result.diagnostics.items():
                diagnostics.setdefault(name, np.zeros((months, n), dtype=np.int64))[step] += counts
            for req, count in result.transfers.items():
                ctx.transfers[req] = ctx.transfers.get(req, 0) + count
            if len(result.departures):
                ctx.departures.append(Departures(domain, result.departures, result.anticipated))
            if result.flows:
                for symbol, counts in result.flows.items():
                    flows[symbol][step] += counts
            elif domain in FLOWS:
                flows[FLOWS[domain]][step] += _flow(domain, applied, n)
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
            diagnostics=diagnostics,
            event_cells=cells.array(),
            unsized_grants=unsized,
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
        diagnostics,
        cells.array(),
        unsized,
    )


def simulate(
    config: SimulationConfig,
    initial: SimulationState,
    *,
    known: ResolvedPlan | None = None,
    replay: ReplayedRequisitions | None = None,
    validate: bool = False,
) -> SimulationResult:
    """Run every trajectory of `config` from the seeded state `initial`, which stays untouched,
    with the run's resolved known events and, in replay, its realized requisitions."""
    seeds = SeedManager(config.seed)
    registered = _register(config)
    trajectories = tuple(
        run_trajectory(
            config,
            initial,
            j,
            seeds,
            validate=validate,
            registered=registered,
            known=known,
            replay=replay,
        )
        for j in range(config.num_trajectories)
    )
    return SimulationResult(config, trajectories, registered.unregistered)


def _register(config: SimulationConfig) -> Registered:
    mode = config.run_mode or RunMode.UNGUIDED
    names = [name for name, _ in STEP_ORDER]
    defaults = [g for g in DEFAULT_GATES if g.domain in names]
    replaying = config.run_mode is RunMode.REPLAY
    if (config.campus is None or replaying) and CAMPUS in names:
        defaults.append(RegistrationGate(CAMPUS, enabled=False, reason="no campus schedule"))
    if config.models.performance is None and PERFORMANCE in names:
        defaults.append(RegistrationGate(PERFORMANCE, enabled=False, reason="no model"))
    return register(names, (*defaults, *config.gates), mode)
