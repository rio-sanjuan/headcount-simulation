"""Planner rules: the controls a planner sets beside the plan, each over a scope.

**Scopes** (`PlannerScope`). A rule applies to the rows its scope selects: a subset's rules over
the buffer attributes and org rollups, in raw labels (see `model.data.subset`). So a scope of
`{"org_level1": ["ORG-A"], "job_level": [5]}` selects level-5 rows anywhere under ORG-A, and an
empty scope selects every row. A scope is declared once and resolved per schema, as eligibility
rules are, with every problem in every scope collected.

**Specificity.** Where several rules of one kind select a row, the most specific one decides
it: the scope that constrains the most attributes, an org level counting as one, and among
those, the narrowest, the one admitting the fewest codes in all. Two scopes that tie on both
and can select the same row are ambiguous, since neither is more specific, and resolution
refuses them, naming both.

**Fill-share targets** (`FillShareTarget`) set the share of fills each channel takes among the
openings in a scope: external, conversion, and transfer. They reshape the channel hazards
after eligibility, where the fills domain's planner-mix hook sits, so guidance still governs
the channels it fixes. Each month, each target's multiplicative solver (`solve_shares`) finds a
multiplier per channel at which its openings' expected fills split as the target says, with as
many expected fills as before: the target moves the mix, not the count. A target the scope's
openings can't meet is reported with its reason (`Unattainable`): a channel it needs that has
no hazard, or capped rows that can't carry the mix.

**Probability tilts** (`ProbabilityTilt`) multiply one domain's event probabilities for the
rows in a scope, in a window of calendar months. The window resolves to the run's steps, and
one the run doesn't reach never applies. A tilted probability is capped at the tilt's cap, one
at most, and where several of a domain's tilts select a row, the most specific decides it. A
run records, per cost center and month, the rows each domain's tilts moved and capped. Under
guidance, a domain realizes its quota whatever the probabilities, so a tilt moves who
experiences the event, not how many: only where nothing fixes the count does it move the
count too.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.fills import CHANNELS, rescale
from hcsim.model.core.layouts import EMPLOYEE, OPENING, Buffer, BufferLayout
from hcsim.model.core.schema import SimulationSchema
from hcsim.model.data.subset import LoweredSubset, SubsetError, SubsetSpec

Ints = NDArray[np.int64]
Floats = NDArray[np.float64]


class PlannerError(ValueError):
    """Planner rules that can't be declared or resolved, every problem listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("bad planner rules: " + "; ".join(problems))


@dataclass(frozen=True)
class PlannerScope:
    """The rows a planner rule applies to, in raw labels: every row, with no rules."""

    where: SubsetSpec = field(default_factory=lambda: SubsetSpec({}))

    @classmethod
    def of(cls, **where: Iterable[object]) -> PlannerScope:
        return cls(
            SubsetSpec({c: frozenset(str(v) for v in values) for c, values in where.items()})
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Iterable[object]]) -> PlannerScope:
        try:
            return cls.of(**{str(c): v for c, v in data.items()})
        except (TypeError, SubsetError) as error:
            raise PlannerError([f"can't read scope {dict(data)!r}: {error}"]) from None

    def to_dict(self) -> dict[str, Any]:
        return {c: sorted(v) for c, v in self.where.rules.items()}

    def resolve(self, schema: SimulationSchema) -> ResolvedScope:
        try:
            return ResolvedScope(self, self.where.lower(schema))
        except SubsetError as error:
            raise PlannerError([f"scope {self.to_dict()}: {error}"]) from None

    def __str__(self) -> str:
        if not self.where.rules:
            return "everyone"
        return ", ".join(f"{c} in {sorted(v)}" for c, v in self.where.rules.items())


@dataclass(frozen=True)
class ResolvedScope:
    """A scope lowered to codes through one schema."""

    scope: PlannerScope
    lowered: LoweredSubset

    @property
    def specificity(self) -> tuple[int, int]:
        """How specific the scope is, greater first: the attributes it constrains, then how
        few codes it admits."""
        admitted = sum(len(codes) for codes in self.lowered.codes.values())
        return len(self.scope.where.rules), -admitted

    def mask(self, rows: Buffer, layout: BufferLayout) -> NDArray[np.bool_]:
        return self.lowered.mask(rows, layout)

    def overlaps(self, other: ResolvedScope) -> bool:
        """Whether some row could fall in both: no attribute both constrain is disjoint."""
        mine, theirs = self.lowered.codes, other.lowered.codes
        return all(mine[c] & theirs[c] for c in set(mine) & set(theirs))


def resolve_scopes(scopes: Sequence[PlannerScope], schema: SimulationSchema) -> list[ResolvedScope]:
    """Every scope lowered through the schema, refusing, with every problem named, one that
    doesn't lower and any pair that's ambiguous."""
    problems, resolved = [], []
    for k, scope in enumerate(scopes):
        try:
            resolved.append(scope.resolve(schema))
        except PlannerError as error:
            problems += [f"rule {k + 1}: {p}" for p in error.problems]
    for i, a in enumerate(resolved):
        for j in range(i + 1, len(resolved)):
            b = resolved[j]
            if a.specificity == b.specificity and a.overlaps(b):
                problems.append(
                    f"rules {i + 1} ({a.scope}) and {j + 1} ({b.scope}) are ambiguous: equally "
                    "specific, they can select the same rows"
                )
    if problems:
        raise PlannerError(problems)
    return resolved


def most_specific(scopes: Sequence[ResolvedScope], rows: Buffer, layout: BufferLayout) -> Ints:
    """Each row's deciding scope: the index of the most specific scope that selects it, or -1
    where none does."""
    winner = np.full(len(rows), -1, dtype=np.int64)
    order = sorted(range(len(scopes)), key=lambda k: scopes[k].specificity, reverse=True)
    for k in order:
        open_rows = winner < 0
        if not open_rows.any():
            break
        winner[open_rows & scopes[k].mask(rows, layout)] = k
    return winner


# --- fill-share targets -----------------------------------------------------------------------

#: How close an expected share must come to its target to count as met.
SHARE_TOLERANCE: Final = 1e-6
_SOLVER_ITERATIONS: Final = 500
_MULTIPLIER_BOUNDS: Final = (1e-9, 1e9)


class Unattainable(StrEnum):
    """Why a fill-share target can't be met among its scope's openings."""

    #: A channel the target gives a share has no hazard in the scope: no model, or every row
    #: ineligible, so no multiplier can open it.
    CLOSED = "closed_channel"
    #: The openings' hazards are capped at one a month, and the capped rows can't carry the mix
    #: at the expected fills the scope already has.
    SATURATED = "saturated"


@dataclass(frozen=True)
class FillShareTarget:
    """The share of fills each channel takes among the openings in `scope`: external,
    conversion, and transfer, summing to one. A channel left out takes none."""

    scope: PlannerScope
    shares: Mapping[str, float]

    def __post_init__(self) -> None:
        problems = []
        unknown = sorted(set(self.shares) - set(CHANNELS))
        if unknown:
            problems.append(f"no fill channels {unknown}; the channels are {list(CHANNELS)}")
        values = [float(self.shares.get(c, 0.0)) for c in CHANNELS]
        if not all(np.isfinite(v) and v >= 0 for v in values):
            problems.append(f"shares are finite and non-negative, not {dict(self.shares)}")
        elif abs(sum(values) - 1.0) > 1e-9:
            problems.append(f"shares sum to one, not {sum(values):g}")
        if problems:
            raise PlannerError([f"fill shares for {self.scope}: {p}" for p in problems])
        object.__setattr__(
            self, "shares", MappingProxyType(dict(zip(CHANNELS, values, strict=True)))
        )

    @property
    def goal(self) -> Floats:
        return np.array([self.shares[c] for c in CHANNELS], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope.to_dict(), "shares": dict(self.shares)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FillShareTarget:
        try:
            return cls(PlannerScope.from_dict(data["scope"]), dict(data["shares"]))
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, PlannerError):
                raise
            raise PlannerError([f"can't read fill shares {dict(data)!r}: {error}"]) from None


@dataclass(frozen=True)
class ShareSolution:
    """One scope's solved multipliers per channel, the expected shares they reach, and why the
    target can't be met, if it can't."""

    multipliers: Floats
    achieved: Floats
    #: Expected fills a month among the scope's openings, before and after.
    fills: tuple[float, float]
    reason: Unattainable | None = None


def solve_shares(hazards: Floats, goal: Floats) -> ShareSolution:
    """Per-channel multipliers on `hazards` (rows, channels) whose expected fills, each row
    capped at one, split by channel as `goal` says, with as many fills as before.

    Multiplicative: each round scales every open channel by its target over its share, then
    scales them all to restore the expected fills, until both settle. A channel with no hazard
    can't be opened, so a target that needs one is solved over the open channels, renormalized,
    and reported `closed_channel`; one the capped rows can't reach is reported `saturated`."""
    hazards = np.asarray(hazards, dtype=np.float64)
    channels = hazards.shape[1]
    before = float(rescale(hazards).sum())
    ones = np.ones(channels)
    if before <= 0:
        return ShareSolution(ones, np.zeros(channels), (0.0, 0.0))
    available = hazards.sum(axis=0) > 0
    reason = Unattainable.CLOSED if ((goal > 0) & ~available).any() else None
    wanted = np.where(available, goal, 0.0)
    wanted = wanted / wanted.sum() if wanted.sum() > 0 else wanted
    m = ones.copy()
    low, high = _MULTIPLIER_BOUNDS
    for _ in range(_SOLVER_ITERATIONS):
        fills = rescale(hazards * m).sum(axis=0)
        share = fills / fills.sum()
        m = np.clip(
            m * np.where(available & (share > 0), wanted / np.where(share > 0, share, 1), 1),
            low,
            high,
        )
        m = np.clip(m * before / rescale(hazards * m).sum(), low, high)
        fills = rescale(hazards * m).sum(axis=0)
        settled = np.abs(fills / fills.sum() - wanted).max() < SHARE_TOLERANCE / 10
        if settled and abs(fills.sum() - before) <= SHARE_TOLERANCE * before:
            break
    fills = rescale(hazards * m).sum(axis=0)
    achieved = fills / fills.sum()
    if reason is None and (
        np.abs(achieved - goal).max() > SHARE_TOLERANCE
        or abs(fills.sum() - before) > SHARE_TOLERANCE * before
    ):
        reason = Unattainable.SATURATED
    return ShareSolution(m, achieved, (before, float(fills.sum())), reason)


@dataclass(frozen=True)
class FillShares:
    """Fill-share targets resolved against one schema: each opening is shaped by its most
    specific target's scope."""

    targets: tuple[FillShareTarget, ...]
    scopes: tuple[ResolvedScope, ...]

    def solve(self, hazards: Floats, openings: Buffer) -> list[ShareSolution | None]:
        """Each target's solution among the openings it decides, or None where it decides
        none."""
        winner = most_specific(self.scopes, openings, OPENING)
        return [
            solve_shares(hazards[winner == k], t.goal) if (winner == k).any() else None
            for k, t in enumerate(self.targets)
        ]

    def apply(self, hazards: Floats, openings: Buffer) -> Floats:
        """`hazards` with each target's multipliers on the openings it decides."""
        shaped = np.array(hazards, dtype=np.float64)
        winner = most_specific(self.scopes, openings, OPENING)
        for k, solution in enumerate(self.solve(shaped, openings)):
            if solution is not None:
                rows = winner == k
                shaped[rows] = shaped[rows] * solution.multipliers
        return shaped


def resolve_fill_shares(
    targets: Sequence[FillShareTarget], schema: SimulationSchema
) -> FillShares | None:
    """The targets resolved, or None for none; refusing ambiguous or unresolvable scopes."""
    if not targets:
        return None
    return FillShares(tuple(targets), tuple(resolve_scopes([t.scope for t in targets], schema)))


# --- probability tilts ------------------------------------------------------------------------

#: The domains a tilt may name, and the buffer whose rows each one scores.
TILTED: Final[Mapping[str, BufferLayout]] = MappingProxyType(
    {
        "attrition": EMPLOYEE,
        "promotion": EMPLOYEE,
        "backfill": EMPLOYEE,  # the departures, as employee rows
        "external_fill": OPENING,
        "conversion_fill": OPENING,
        "transfer_fill": OPENING,
        "cancellation": OPENING,
    }
)


@dataclass(frozen=True)
class ProbabilityTilt:
    """A planner's multiplier on one domain's event probabilities, for the rows in `scope`, in
    the calendar months from `start` through `end` (`YYYY-MM`, either open-ended), capped at
    `cap`."""

    domain: str
    scope: PlannerScope
    multiplier: float
    start: str | None = None
    end: str | None = None
    cap: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        problems = []
        if self.domain not in TILTED:
            problems.append(f"no domain {self.domain!r} to tilt; known: {', '.join(TILTED)}")
        if not (np.isfinite(self.multiplier) and self.multiplier >= 0):
            problems.append(f"a multiplier is finite and non-negative, not {self.multiplier}")
        if not 0.0 < self.cap <= 1.0:
            problems.append(f"a cap lies in (0, 1], not {self.cap}")
        try:
            first, last = self.window
        except ValueError as error:
            problems.append(f"the window's months are YYYY-MM: {error}")
        else:
            if first is not None and last is not None and first > last:
                problems.append(f"the window starts in {first}, after it ends in {last}")
        if problems:
            raise PlannerError([f"tilt on {self.domain} for {self.scope}: {p}" for p in problems])

    @property
    def window(self) -> tuple[pd.Period | None, pd.Period | None]:
        def month(text: str | None) -> pd.Period | None:
            return None if text is None else pd.Period(str(text), freq="M")

        return month(self.start), month(self.end)

    def steps(self, run_date: dt.date, months: int) -> range:
        """The run's steps inside the window: none for a window the run doesn't reach."""
        first, last = self.window
        origin = pd.Period(run_date, freq="M")
        lo = 0 if first is None else max(0, (first - origin).n)
        hi = months - 1 if last is None else min(months - 1, (last - origin).n)
        return range(lo, hi + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "scope": self.scope.to_dict(),
            "multiplier": self.multiplier,
            "start": self.start,
            "end": self.end,
            "cap": self.cap,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbabilityTilt:
        try:
            return cls(
                str(data["domain"]),
                PlannerScope.from_dict(data["scope"]),
                float(data["multiplier"]),
                data.get("start"),
                data.get("end"),
                float(data.get("cap", 1.0)),
                str(data.get("reason", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, PlannerError):
                raise
            raise PlannerError([f"can't read tilt {dict(data)!r}: {error}"]) from None


@dataclass(frozen=True)
class Tilted:
    """A domain's probabilities after its tilts, and which rows a tilt moved or capped."""

    p: Floats
    tilted: NDArray[np.bool_]
    capped: NDArray[np.bool_]


@dataclass(frozen=True)
class _DomainTilts:
    tilts: tuple[ProbabilityTilt, ...]
    scopes: tuple[ResolvedScope, ...]
    steps: tuple[frozenset[int], ...]


@dataclass(frozen=True)
class Tilts:
    """Probability tilts resolved against one schema and run: per domain, its tilts, their
    scopes, and the steps each applies in."""

    by_domain: Mapping[str, _DomainTilts]

    def apply(self, domain: str, rows: Buffer, step: int, p: Floats) -> Tilted:
        """`p` for `rows` of `domain` at `step`, each row multiplied by the most specific tilt
        in force that selects it, and capped at its cap."""
        p = np.asarray(p, dtype=np.float64)
        none = np.zeros(len(p), dtype=bool)
        found = self.by_domain.get(domain)
        if found is None or not len(rows):
            return Tilted(p, none, none)
        tilts, scopes, steps = found.tilts, found.scopes, found.steps
        live = [k for k in range(len(tilts)) if step in steps[k]]
        if not live:
            return Tilted(p, none, none)
        winner = most_specific([scopes[k] for k in live], rows, TILTED[domain])
        multiplier = np.ones(len(p))
        cap = np.ones(len(p))
        for j, k in enumerate(live):
            mine = winner == j
            multiplier[mine], cap[mine] = tilts[k].multiplier, tilts[k].cap
        raised = p * multiplier
        tilted = winner >= 0
        capped = tilted & (raised > cap)
        return Tilted(np.where(tilted, np.minimum(raised, cap), p), tilted, capped)


def resolve_tilts(
    tilts: Sequence[ProbabilityTilt], schema: SimulationSchema, run_date: dt.date, months: int
) -> Tilts | None:
    """The tilts resolved, or None for none: each domain's scopes lowered, ambiguous ones
    refused, and each window resolved to the run's steps."""
    if not tilts:
        return None
    problems: list[str] = []
    by_domain = {}
    for domain in TILTED:
        mine = tuple(t for t in tilts if t.domain == domain)
        if not mine:
            continue
        try:
            scopes = tuple(resolve_scopes([t.scope for t in mine], schema))
        except PlannerError as error:
            problems += [f"tilts on {domain}: {p}" for p in error.problems]
            continue
        steps = tuple(frozenset(t.steps(run_date, months)) for t in mine)
        by_domain[domain] = _DomainTilts(mine, scopes, steps)
    if problems:
        raise PlannerError(problems)
    return Tilts(MappingProxyType(by_domain))
