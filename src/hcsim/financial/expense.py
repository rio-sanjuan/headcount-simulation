"""Phase 3: amortize every tranche into monthly expense, after the simulation.

**Preparation** (`prepare`) turns what a run leaves, its ledger's tranches, their values, and its
calendar, into one table, in four steps, every problem with the inputs refused at once:

1. **Windows**: each tranche's accrual window, from its award month through its last month, its
   vest, or its forfeit when that comes first.
2. **Values**: each tranche's value, its shares times its value per share.
3. **Calendar**: the calendar month of the run's first step, its origin, and each tranche's
   cohort, the calendar year it was awarded in.
4. **Survival**: for tranches whose holder the run doesn't simulate, the chance, month by month
   from the run's first, that the holder still holds it.

**The calculator** (`calculate`) is stateless: it reads only the prepared table and the expense
settings, so the same table always gives the same expense, and it can be checked on hand-built
schedules. Vectorized over tranches, for each tranche:

1. **Gross expense** accrues from the award month through the vest month, each month's share of
   the value in proportion to its calendar days, given an origin, or in equal shares without.
2. **Survival.** A month accrues only if the holder still holds the tranche when it starts. For a
   simulated holder that's certain until the month they leave, the forfeit month the ledger
   records; for the rest, it's their survival path's chance, so their expense is its
   expectation.
3. **Net expense** is gross times `1 - e`, for the effective forfeiture rate `e`: flat, or with
   `forfeiture="cohort"`, each cohort's own, from a pre-pass that recomputes it every month as the
   share of the cohort's decided shares, forfeited or vested so far, that forfeited, the flat
   rate standing in until any are decided.
4. **The three true-ups.** When a cohort's rate changes, the **rate** true-up adjusts what each
   still-accruing tranche accrued before, by the fall in the rate. A holder leaving before the
   vest reverses everything the tranche recognized: the **attrition** true-up, weighted by the
   chance of leaving that month. At the vest, the **vesting** true-up is the residual between the
   value and what was recognized, weighted by the chance of getting there. That's what makes the
   accounting identity hold by construction, in expectation where survival is uncertain:

       cumulative net + true-ups = vested shares * value per share

A residual above the tolerance therefore means a defect in this code, and it raises.

**Pieces.** Given its holders' intervals, each row is booked in the cost center its holder sat
in that month, a 9-D piece of the tranche (see `tracker`). A forfeit then takes the
**forfeited-piece path**: the attrition true-up reverses each piece's recognized expense in that
piece's own cost center, so every cost center's books net to nothing for a forfeited tranche.

**Chunks.** `chunk_size` calculates the tranches a chunk at a time, so a large run's rows are
never all built at once. The results are bit-identical to an unchunked calculation: a tranche's
arithmetic never reads another's, and the cohort pre-pass sums its shares in exact scaled
integers.

**Toggles.** `trueups` names the true-ups to make. With all three, the identity is enforced; with
any left out it can't hold, so each tranche's residual is reported and nothing raises.

`compute_expense` is preparation and the calculator together.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.financial.tracker import Intervals, Tranches, cost_centers_at

IDENTITY_TOLERANCE: Final = 0.01
#: The cohort pre-pass's exact scale: shares are summed as integers in units of 2**-24.
SCALE: Final = 2**24
Floats = NDArray[np.float64]
Ints = NDArray[np.int64]
SBC_KEYS: Final = ["month", "cost_center", "award_type", "level"]
SBC_VALUES: Final = [
    "gross",
    "net",
    "trueup_attrition",
    "trueup_rate",
    "trueup_vesting",
    "trueup",
    "recognized",
]
#: The true-ups, in the order the rows carry them.
TRUEUPS: Final = ("trueup_attrition", "trueup_rate", "trueup_vesting")


class AccountingIdentityError(AssertionError):
    """A tranche whose recognized expense doesn't tie to its vested value."""


@dataclass(frozen=True)
class ExpenseConfig:
    #: Flat effective forfeiture rate, and the prior a cohort's rate starts from. Invented
    #: default.
    forfeiture_rate: float = 0.10
    #: `flat`, one rate throughout, or `cohort`, each cohort's own, recomputed every month.
    forfeiture: Literal["flat", "cohort"] = "flat"
    #: The true-ups to make. With any left out, the identity can't hold, so it isn't enforced.
    trueups: frozenset[str] = frozenset(TRUEUPS)
    identity_tolerance: float = IDENTITY_TOLERANCE
    #: Tranches calculated at a time, so a run's rows never all pile up; the results don't
    #: depend on it.
    chunk_size: int = 2_000

    def __post_init__(self) -> None:
        if not 0.0 <= self.forfeiture_rate < 1.0:
            raise ValueError("forfeiture_rate must lie in [0, 1)")
        if self.forfeiture not in ("flat", "cohort"):
            raise ValueError(f"forfeiture is 'flat' or 'cohort', not {self.forfeiture!r}")
        if self.chunk_size < 1:
            raise ValueError(f"a chunk holds at least one tranche, not {self.chunk_size}")
        unknown = sorted(set(self.trueups) - set(TRUEUPS))
        if unknown:
            raise ValueError(f"true-ups are {list(TRUEUPS)}, not {unknown}")
        object.__setattr__(self, "trueups", frozenset(self.trueups))

    @property
    def enforces_identity(self) -> bool:
        return self.trueups == frozenset(TRUEUPS)


@dataclass(frozen=True)
class ExpenseResult:
    """Monthly expense per tranche, in long form, each tranche's identity residual, and what each
    recognized before the first month the rows cover, its opening balance."""

    rows: pd.DataFrame
    residual: NDArray[np.float64]
    opening: NDArray[np.float64] | None = None
    #: Whether the identity was enforced, as it is when every true-up is made.
    identity_enforced: bool = True

    @property
    def worst_tranche(self) -> int | None:
        """The tranche with the largest identity residual, or None without tranches."""
        return int(np.abs(self.residual).argmax()) if len(self.residual) else None

    @property
    def max_identity_residual(self) -> float:
        return float(np.abs(self.residual).max(initial=0.0))


@dataclass(frozen=True)
class Prepared:
    """The table the calculator reads: one row per tranche, from the four preparation steps."""

    tranches: Tranches
    #: 2. Each tranche's value: shares times value per share.
    value: NDArray[np.float64]
    #: 1. Whether it forfeits before it vests, and its last accrual month.
    forfeited: NDArray[np.bool_]
    last: NDArray[np.int64]
    #: 3. The calendar month of the run's first step, and each tranche's award year, or -1
    #: without an origin.
    origin: pd.Period | None
    cohort: NDArray[np.int64]
    #: 4. (tranches, months): the chance each tranche's holder still holds it at each month's
    #: end from the run's first, or None when the run simulates every holder.
    survival: NDArray[np.float64] | None
    #: The holders' intervals, which cut each tranche into pieces by cost center, or None to
    #: book every tranche where it was granted.
    intervals: Intervals | None = None

    def __len__(self) -> int:
        return len(self.tranches)


def prepare(
    tranches: Tranches,
    value_per_share: NDArray[np.float64],
    origin: pd.Period | None = None,
    survival: NDArray[np.float64] | None = None,
    intervals: Intervals | None = None,
) -> Prepared:
    """The calculator's table, in four steps, refusing every problem with the inputs at once."""
    n = len(tranches)
    problems = []
    vps = np.asarray(value_per_share, dtype=np.float64)
    if vps.shape != (n,):
        problems.append("one value per share per tranche")
    elif not np.isfinite(vps).all() or (vps < 0).any():
        problems.append("values per share must be finite and non-negative")
    a, v, f = tranches.award_month, tranches.vest_month, tranches.forfeit_month
    if (v < a).any():
        problems.append("a tranche can't vest before its award month")
    if survival is not None:
        held = np.asarray(survival, dtype=np.float64)
        if held.ndim != 2 or len(held) != n:
            problems.append(f"survival is (tranches, months): {n} rows, not {held.shape}")
        elif not (np.isfinite(held) & (held >= 0) & (held <= 1)).all():
            problems.append("survival is a chance in [0, 1]")
        elif (np.diff(held, axis=1) > 1e-12).any():
            problems.append("survival never rises")
    if problems:
        raise ValueError("; ".join(problems))
    # 1. Windows.
    forfeited = (f >= 0) & (f < v)
    last = np.where(forfeited, f, v)
    # 2. Values.
    value = tranches.shares * vps
    # 3. Calendar.
    if origin is None:
        cohort = np.full(n, -1, dtype=np.int64)
    else:
        origin = pd.Period(origin, "M")
        cohort = ((origin.year * 12 + origin.month - 1 + a) // 12).astype(np.int64)
    # 4. Survival.
    kept = None if survival is None else np.asarray(survival, dtype=np.float64)
    return Prepared(tranches, value, forfeited, last, origin, cohort, kept, intervals)


def _within(values: Floats, groups: Ints) -> Floats:
    """Running sums of `values` within each group, each group's alone, so a tranche's never
    depends on the rows beside it."""
    summed = pd.Series(values).groupby(groups, sort=False).cumsum()
    out: Floats = np.asarray(summed, dtype=np.float64)
    return out


def _survival(prepared: Prepared, tranche: Ints, month: Ints) -> Floats:
    """Each (tranche, month)'s chance its holder still holds it at the month's end: their
    survival path's, 1 before the run, its last value after it; and nothing from the month a
    simulated holder leaves."""
    out = np.ones(len(tranche))
    if prepared.survival is not None and prepared.survival.shape[1]:
        path = prepared.survival
        column = np.clip(month, 0, path.shape[1] - 1)
        out = np.where(month >= 0, path[tranche, column], 1.0)
    t = prepared.tranches
    left = prepared.forfeited[tranche] & (month >= t.forfeit_month[tranche])
    out = np.where(left, 0.0, out)
    return out


@dataclass(frozen=True)
class _Rows:
    """One chunk's rows before they're valued: tranche, month, and survival either side."""

    tranche: Ints
    month: Ints
    alive: Floats
    prior_alive: Floats


def _rows(prepared: Prepared) -> _Rows:
    t = prepared.tranches
    a = t.award_month
    # Rows run through the vest, or through the forfeit for a simulated holder who leaves; a
    # tranche whose holder left before its award has none.
    count = np.maximum(0, prepared.last - a + 1)
    tranche = np.repeat(np.arange(len(t)), count)
    starts = np.cumsum(count) - count
    month = a[tranche] + (np.arange(len(tranche)) - starts[tranche])
    return _Rows(
        tranche,
        month,
        _survival(prepared, tranche, month),
        _survival(prepared, tranche, month - 1),
    )


def _chunk(prepared: Prepared, lo: int, hi: int) -> Prepared:
    t = prepared.tranches
    names = ("holder", "award_month", "vest_month", "shares", "forfeit_month", "award_type")
    part = Tranches(
        **{n: getattr(t, n)[lo:hi] for n in (*names, "level", "cost_center")},
        value=None if t.value is None else t.value[lo:hi],
    )
    survival = None if prepared.survival is None else prepared.survival[lo:hi]
    return Prepared(
        part,
        prepared.value[lo:hi],
        prepared.forfeited[lo:hi],
        prepared.last[lo:hi],
        prepared.origin,
        prepared.cohort[lo:hi],
        survival,
        prepared.intervals,
    )


@dataclass(frozen=True)
class _Rates:
    """The cohort pre-pass's result: each cohort's rate in every month from `first`."""

    cohorts: Ints
    table: Floats
    first: int


def _cohort_rates(prepared: Prepared, bounds: list[tuple[int, int]], prior: float) -> _Rates:
    """The cohort pre-pass, a chunk at a time: each cohort's effective forfeiture rate in every
    month, from the shares decided by then, forfeited or vested, and `prior` until any are.
    Shares are summed as integers in units of 1/SCALE, so the sum is exact in any order."""
    cohorts = np.unique(prepared.cohort)
    t = prepared.tranches
    accrues = prepared.last >= t.award_month
    if not accrues.any():
        return _Rates(cohorts, np.full((len(cohorts), 1), prior), 0)
    first = int(t.award_month[accrues].min())
    width = int(prepared.last[accrues].max()) - first + 1
    grid = np.zeros((2, len(cohorts), width), dtype=np.int64)
    for lo, hi in bounds:
        chunk = _chunk(prepared, lo, hi)
        r = _rows(chunk)
        t = chunk.tranches
        code = np.searchsorted(cohorts, chunk.cohort[r.tranche])
        shares = t.shares[r.tranche]
        vesting = r.month == t.vest_month[r.tranche]
        gone = np.where(vesting, 0.0, (r.prior_alive - r.alive) * shares)
        kept = np.where(vesting, r.prior_alive * shares, 0.0)
        for k, amount in enumerate((gone, kept)):
            np.add.at(grid[k], (code, r.month - first), np.rint(amount * SCALE).astype(np.int64))
    gone_total, kept_total = np.cumsum(grid[0], axis=1), np.cumsum(grid[1], axis=1)
    decided = gone_total + kept_total
    with np.errstate(divide="ignore", invalid="ignore"):
        table = np.where(decided > 0, gone_total / np.maximum(decided, 1), prior)
    return _Rates(cohorts, np.minimum(table, 1.0 - 1e-12), first)


def _calculate(
    prepared: Prepared, config: ExpenseConfig, first_month: int | None, rates: _Rates | None
) -> ExpenseResult:
    t = prepared.tranches
    n = len(t)
    a, v = t.award_month, t.vest_month
    value = prepared.value
    r = _rows(prepared)
    tranche, month, alive, prior_alive = r.tranche, r.month, r.alive, r.prior_alive
    # 1. Gross, by calendar days.
    if prepared.origin is None:
        days = np.ones(len(month))
        window = (v - a + 1).astype(np.float64)
    else:
        base = prepared.origin.start_time
        lo = min(int(a.min(initial=0)), 0)
        hi = int(v.max(initial=0)) + 1
        calendar = np.array(
            [(base + pd.DateOffset(months=m)).days_in_month for m in range(lo, hi)],
            dtype=np.float64,
        )
        cumulative = np.concatenate([[0.0], np.cumsum(calendar)])
        days = calendar[month - lo] if len(month) else np.zeros(0)
        window = cumulative[v - lo + 1] - cumulative[a - lo]
    gross = value[tranche] * days / window[tranche]
    # 2. Survival is in the rows: alive at each month's end, and at its start.
    # 3. Net, at the flat rate or each cohort's.
    if rates is not None and len(month):
        code = np.searchsorted(rates.cohorts, prepared.cohort[tranche])
        width = rates.table.shape[1]
        rate = rates.table[code, np.clip(month - rates.first, 0, width - 1)]
        before = rates.table[code, np.clip(month - 1 - rates.first, 0, width - 1)]
        before = np.where(month == a[tranche], rate, before)
    else:
        rate = before = np.full(len(month), config.forfeiture_rate)
    accrued_before = _within(gross, tranche) - gross
    net = gross * (1 - rate)
    # 4. The true-ups: rate on what accrued before, then attrition and vesting.
    made = config.trueups
    rate_up = accrued_before * (before - rate) if "trueup_rate" in made else np.zeros(len(month))
    step = net + rate_up
    recognized = _within(step, tranche)
    vests = month == v[tranche]
    attrition = np.where(vests, 0.0, -(prior_alive - alive) * recognized)
    if "trueup_attrition" not in made:
        attrition = np.zeros(len(month))
    vesting = np.where(vests, prior_alive * (value[tranche] - recognized), 0.0)
    if "trueup_vesting" not in made:
        vesting = np.zeros(len(month))
    columns = {
        "gross": gross * prior_alive,
        "net": net * prior_alive,
        "trueup_attrition": attrition,
        "trueup_rate": rate_up * prior_alive,
        "trueup_vesting": vesting,
    }
    # Pieces: each row in its holder's cost center that month.
    granted = t.cost_center[tranche]
    cost_center = (
        granted
        if prepared.intervals is None
        else cost_centers_at(prepared.intervals, t.holder[tranche], month, granted)
    )
    frame = pd.DataFrame({"tranche": tranche, "month": month, "cost_center": cost_center} | columns)
    frame = _forfeited_pieces(frame, step)
    amount = frame[["net", *TRUEUPS]].sum(axis=1).to_numpy()
    total = np.zeros(n)
    np.add.at(total, frame["tranche"].to_numpy(), amount)
    expected = value * _survival(prepared, np.arange(n), v - 1)
    tolerance = config.identity_tolerance if config.enforces_identity else np.inf
    residual = check_identity(total, expected, tolerance)
    opening = np.zeros(n)
    if first_month is not None:
        early = frame["month"].to_numpy() < first_month
        np.add.at(opening, frame["tranche"].to_numpy()[early], amount[early])
        frame = frame[~early].reset_index(drop=True)
    return ExpenseResult(
        frame,
        residual,
        opening if first_month is not None else None,
        config.enforces_identity,
    )


def _forfeited_pieces(frame: pd.DataFrame, step: Floats) -> pd.DataFrame:
    """The forfeited-piece path: for a tranche whose rows span cost centers, each attrition
    true-up is split among them in proportion to what each had recognized by then, and booked in
    each, so each cost center's books net to nothing for a forfeited tranche."""
    moved = frame.groupby("tranche", sort=False)["cost_center"].transform("nunique") > 1
    if not moved.any():
        return frame
    spread = frame[moved].copy()
    spread["step"] = step[moved.to_numpy()]
    pieces = []
    for _, rows in spread.groupby("tranche", sort=False):
        reversed_ = rows["trueup_attrition"].to_numpy()
        months = rows["month"].to_numpy()
        centers = rows["cost_center"].to_numpy()
        for k in np.flatnonzero(reversed_ != 0):
            booked = pd.Series(rows["step"].to_numpy()[: k + 1]).groupby(centers[: k + 1]).sum()
            whole = float(booked.sum())
            if whole == 0:
                continue
            for center, part in booked.items():
                if center == centers[k]:
                    continue
                share = reversed_[k] * float(part) / whole
                pieces.append((rows["tranche"].iloc[0], months[k], center, share))
                reversed_[k] -= share
        spread.loc[rows.index, "trueup_attrition"] = reversed_
    out = frame.copy()
    out.loc[spread.index, "trueup_attrition"] = spread["trueup_attrition"]
    if pieces:
        extra = pd.DataFrame(
            pieces, columns=["tranche", "month", "cost_center", "trueup_attrition"]
        )
        for name in ("gross", "net", "trueup_rate", "trueup_vesting"):
            extra[name] = 0.0
        out = pd.concat([out, extra[out.columns]], ignore_index=True)
        out = out.sort_values(["tranche", "month"], kind="stable", ignore_index=True)
    return out


def calculate_chunks(
    prepared: Prepared,
    config: ExpenseConfig | None = None,
    first_month: int | None = None,
    chunk_size: int | None = None,
) -> Iterator[tuple[int, Prepared, ExpenseResult]]:
    """The calculation a chunk of `chunk_size` tranches at a time, the settings' own by default:
    each chunk's first tranche, its prepared table, and its result, whose rows number its
    tranches from 0. The cohort pre-pass reads every chunk before the first is calculated, so
    each chunk's result is what a whole calculation gives its tranches."""
    config = config or ExpenseConfig()
    n = len(prepared)
    size = config.chunk_size if chunk_size is None else chunk_size
    if size < 1:
        raise ValueError(f"a chunk holds at least one tranche, not {chunk_size}")
    bounds = [(lo, min(lo + size, n)) for lo in range(0, n, size)] or [(0, 0)]
    rates = (
        _cohort_rates(prepared, bounds, config.forfeiture_rate)
        if config.forfeiture == "cohort"
        else None
    )
    for lo, hi in bounds:
        chunk = _chunk(prepared, lo, hi)
        yield lo, chunk, _calculate(chunk, config, first_month, rates)


def calculate(
    prepared: Prepared,
    config: ExpenseConfig | None = None,
    first_month: int | None = None,
    chunk_size: int | None = None,
) -> ExpenseResult:
    """Every tranche's monthly expense and true-ups, from the prepared table alone, calculated a
    chunk at a time (see `calculate_chunks`) and put together. With `first_month`, as a run's
    first, the rows start there: what a tranche recognized before it, accrual and any true-up,
    is its opening balance instead, and the identity still holds over the tranche's whole
    life."""
    config = config or ExpenseConfig()
    rows, residuals, openings = [], [], []
    for lo, _, part in calculate_chunks(prepared, config, first_month, chunk_size):
        frame = part.rows
        frame["tranche"] += lo
        rows.append(frame)
        residuals.append(part.residual)
        if part.opening is not None:
            openings.append(part.opening)
    return ExpenseResult(
        pd.concat(rows, ignore_index=True),
        np.concatenate(residuals),
        np.concatenate(openings) if first_month is not None else None,
        config.enforces_identity,
    )


def compute_expense(
    tranches: Tranches,
    value_per_share: NDArray[np.float64],
    config: ExpenseConfig | None = None,
    first_month: int | None = None,
    origin: pd.Period | None = None,
) -> ExpenseResult:
    """Prepare and calculate in one call (see `prepare` and `calculate`)."""
    return calculate(prepare(tranches, value_per_share, origin), config, first_month)


def check_identity(
    recognized: NDArray[np.float64], vested_value: NDArray[np.float64], tolerance: float
) -> NDArray[np.float64]:
    """Each tranche's residual, raising on the worst one if it exceeds `tolerance`."""
    residual: NDArray[np.float64] = recognized - vested_value
    if len(residual):
        worst = int(np.abs(residual).argmax())
        if abs(residual[worst]) > tolerance:
            raise AccountingIdentityError(
                f"tranche {worst} recognizes {recognized[worst]:.6f} against a vested value of "
                f"{vested_value[worst]:.6f}"
            )
    return residual


def sbc_frame(result: ExpenseResult, tranches: Tranches) -> pd.DataFrame:
    """Monthly SBC by cost center, award type, and level, in schema codes.

    The frame runs from the earliest award month to the last vest month, past the run's end:
    grants recorded during the run keep expensing after it.
    """
    rows = result.rows
    t = rows["tranche"].to_numpy()
    keyed = rows.assign(
        cost_center=rows["cost_center"].to_numpy()
        if "cost_center" in rows
        else tranches.cost_center[t],
        award_type=tranches.award_type[t],
        level=tranches.level[t],
    )
    frame = keyed.groupby(SBC_KEYS, as_index=False)[["gross", "net", *TRUEUPS]].sum()
    frame["trueup"] = frame["trueup_attrition"] + frame["trueup_rate"] + frame["trueup_vesting"]
    frame["recognized"] = frame["net"] + frame["trueup"]
    return frame[SBC_KEYS + SBC_VALUES].sort_values(SBC_KEYS, ignore_index=True)
