"""Gates: thresholds on evaluator metrics that mark a run provisional or stop it.

A **gate** bounds one metric, `evaluator.metric`, from above (`max`) or below (`min`), on its
value or its magnitude (`absolute`), and says what a failure does (`Action`):

- `stop` stops the study: the arm failed something that holds by construction, or that its
  mode guarantees, so nothing after it can be trusted;
- `provisional` marks the run provisional: its results stand, flagged.

**The built-in gates** (`BUILTIN`) hold in every study whose mode and window closure they apply
to (`gate_set`):

- the invariants stop any run: every quota honored, and the headcount, opening, and accounting
  identities exact, the last within its tolerance;
- a replay over a closed window reproduces history: its month-end headcount exactly, and every
  head event too, or it stops; its SBC within 5% of the reference ledger month by month, or
  it's provisional;
- guided actuals over a closed window lands its month-end headcount within 5% of history's,
  and a forecast, guided or not, within 10%, or it's provisional.

A profile adds its own gates and can leave the built-in ones out. Gates apply to the arms that
rank and the null arm, in pass B; a reference arm is never gated. A gate whose metric an arm
doesn't report, because its evaluator didn't run or had nothing to score, isn't applied, and
says so.

**The verdict** (`verdict`) is the study's: `stopped` when a stop gate failed, `provisional`
when a provisional one did or the inputs include a fixture, and `passed` otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from hcsim.engine.modes import RunMode
from hcsim.studies.evaluators import Metric


class Action(StrEnum):
    STOP = "stop"
    PROVISIONAL = "provisional"


class Verdict(StrEnum):
    PASSED = "passed"
    PROVISIONAL = "provisional"
    STOPPED = "stopped"


class GateError(ValueError):
    """A gate that can't be declared."""


#: The window closures a gate can apply in, by name.
CLOSURES: Final = ("wholly_open", "partially_closed", "wholly_closed")


@dataclass(frozen=True)
class Gate:
    """A bound on one metric, `evaluator.metric`, and what failing it does. `modes` and
    `closures` limit where it applies; empty applies everywhere."""

    name: str
    metric: str
    action: Action
    max: float | None = None
    min: float | None = None
    #: Bound the metric's magnitude, not its value.
    absolute: bool = False
    modes: frozenset[RunMode] = field(default_factory=frozenset)
    closures: frozenset[str] = field(default_factory=frozenset)
    doc: str = ""

    def __post_init__(self) -> None:
        problems = []
        if "." not in self.metric:
            problems.append(f"gate {self.name}'s metric is evaluator.metric, not {self.metric!r}")
        if (self.max is None) == (self.min is None):
            problems.append(f"gate {self.name} bounds its metric by exactly one of max and min")
        unknown = sorted(set(self.closures) - set(CLOSURES))
        if unknown:
            problems.append(f"gate {self.name} names no window closures {unknown}")
        if problems:
            raise GateError("; ".join(problems))

    @property
    def evaluator(self) -> str:
        return self.metric.split(".", 1)[0]

    def applies(self, mode: RunMode | str, closure: str) -> bool:
        """Whether the gate holds in a study of `mode` over a window of `closure`."""
        return (not self.modes or RunMode(mode) in self.modes) and (
            not self.closures or closure in self.closures
        )

    def holds(self, value: float) -> bool:
        x = abs(value) if self.absolute else value
        return x <= self.max if self.max is not None else x >= (self.min or 0.0)

    @property
    def bound(self) -> str:
        x = "|x|" if self.absolute else "x"
        return f"{x} <= {self.max:g}" if self.max is not None else f"{x} >= {self.min:g}"


@dataclass(frozen=True)
class GateResult:
    """One gate on one arm: the metric's value, whether it held, and why not, if it wasn't
    applied."""

    gate: str
    arm: str
    metric: str
    action: Action
    bound: str
    value: float | None
    passed: bool
    reason: str = ""

    @property
    def applied(self) -> bool:
        return self.value is not None

    @property
    def failed(self) -> bool:
        return self.applied and not self.passed


_REPLAY, _ACTUALS = frozenset({RunMode.REPLAY}), frozenset({RunMode.GUIDED_ACTUALS})
_FORECAST = frozenset({RunMode.GUIDED_FORECAST, RunMode.UNGUIDED})
_CLOSED = frozenset({"wholly_closed"})
#: The events a replay reproduces exactly.
_HEAD_EVENTS: Final = (
    "attrition",
    "hires",
    "internal_starts",
    "conversions_out",
    "promotions",
    "requisitions_closed_unfilled",
)
#: Every built-in gate.
BUILTIN: Final = (
    Gate("quota_honored", "invariants.quota_violations", Action.STOP, max=0, doc="every quota"),
    Gate("headcount_identity", "invariants.headcount_identity_max_gap", Action.STOP, max=0),
    Gate("opening_identity", "invariants.opening_identity_max_gap", Action.STOP, max=0),
    Gate(
        "accounting_identity",
        "invariants.accounting_identity_max_residual",
        Action.STOP,
        max=0.01,
        doc="every tranche's recognized expense ties to its vested value",
    ),
    Gate(
        "replay_headcount",
        "fidelity.headcount_wape",
        Action.STOP,
        max=0,
        modes=_REPLAY,
        closures=_CLOSED,
        doc="a replay reproduces history's month-end headcount",
    ),
    *(
        Gate(
            f"replay_{event}",
            f"fidelity.{event}_wape",
            Action.STOP,
            max=0,
            modes=_REPLAY,
            closures=_CLOSED,
            doc=f"a replay reproduces history's {event.replace('_', ' ')}",
        )
        for event in _HEAD_EVENTS
    ),
    Gate(
        "replay_sbc",
        "sbc.monthly_wape",
        Action.PROVISIONAL,
        max=0.05,
        modes=_REPLAY,
        closures=_CLOSED,
        doc="a replay's SBC ties to the reference ledger",
    ),
    Gate(
        "actuals_headcount",
        "fidelity.headcount_wape",
        Action.PROVISIONAL,
        max=0.05,
        modes=_ACTUALS,
        closures=_CLOSED,
        doc="correct volumes land near history's headcount",
    ),
    Gate(
        "forecast_headcount",
        "fidelity.headcount_wape",
        Action.PROVISIONAL,
        max=0.10,
        modes=_FORECAST,
        closures=_CLOSED,
        doc="a backtest lands near history's headcount",
    ),
)
#: Each built-in gate by name.
BUILTIN_BY_NAME: Final = {g.name: g for g in BUILTIN}


def gate_set(
    mode: RunMode | str, closure: str, extra: Sequence[Gate] = (), builtin: bool = True
) -> tuple[Gate, ...]:
    """The gates a study of `mode` over a window of `closure` holds: the built-in ones that
    apply, unless left out, and its own that do."""
    if closure not in CLOSURES:
        raise GateError(f"no window closure {closure!r}")
    names = [g.name for g in extra]
    clashing = sorted(
        {n for n in names if names.count(n) > 1 or (builtin and n in BUILTIN_BY_NAME)}
    )
    if clashing:
        raise GateError(f"gate names repeat: {clashing}")
    chosen = [*(BUILTIN if builtin else ()), *extra]
    return tuple(g for g in chosen if g.applies(mode, closure))


def apply(gates: Iterable[Gate], arm: str, metrics: Iterable[Metric]) -> tuple[GateResult, ...]:
    """Every gate on one arm's metrics."""
    by_name: Mapping[str, float] = {f"{m.evaluator}.{m.name}": m.value for m in metrics}
    results = []
    for g in gates:
        value = by_name.get(g.metric)
        if value is None:
            reason = f"arm {arm} reports no {g.metric}"
            results.append(GateResult(g.name, arm, g.metric, g.action, g.bound, None, True, reason))
        else:
            passed = g.holds(value)
            results.append(GateResult(g.name, arm, g.metric, g.action, g.bound, value, passed))
    return tuple(results)


def verdict(results: Iterable[GateResult], provisional_inputs: bool = False) -> Verdict:
    """The study's verdict from every gate it applied, and whether its inputs include a
    fixture."""
    failed = [r for r in results if r.failed]
    if any(r.action is Action.STOP for r in failed):
        return Verdict.STOPPED
    if failed or provisional_inputs:
        return Verdict.PROVISIONAL
    return Verdict.PASSED
