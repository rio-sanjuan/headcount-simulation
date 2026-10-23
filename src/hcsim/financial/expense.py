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

`compute_expense` is the two together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.financial.tracker import Tranches

IDENTITY_TOLERANCE: Final = 0.01
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
    identity_tolerance: float = IDENTITY_TOLERANCE

    def __post_init__(self) -> None:
        if not 0.0 <= self.forfeiture_rate < 1.0:
            raise ValueError("forfeiture_rate must lie in [0, 1)")
        if self.forfeiture not in ("flat", "cohort"):
            raise ValueError(f"forfeiture is 'flat' or 'cohort', not {self.forfeiture!r}")


@dataclass(frozen=True)
class ExpenseResult:
    """Monthly expense per tranche, in long form, each tranche's identity residual, and what each
    recognized before the first month the rows cover, its opening balance."""

    rows: pd.DataFrame
    residual: NDArray[np.float64]
    opening: NDArray[np.float64] | None = None

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

    def __len__(self) -> int:
        return len(self.tranches)


def prepare(
    tranches: Tranches,
    value_per_share: NDArray[np.float64],
    origin: pd.Period | None = None,
    survival: NDArray[np.float64] | None = None,
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
    return Prepared(tranches, value, forfeited, last, origin, cohort, kept)


def _within(values: Floats, groups: Ints, starts: Ints) -> Floats:
    """Cumulative sums of `values` within contiguous groups beginning at `starts`."""
    total = np.cumsum(values)
    offset = np.zeros(len(starts))
    offset[1:] = total[starts[1:] - 1]
    out: Floats = total - offset[groups]
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


def _cohort_rates(
    prepared: Prepared,
    tranche: Ints,
    month: Ints,
    alive: Floats,
    prior_alive: Floats,
    prior: float,
) -> tuple[Floats, Ints, Floats]:
    """The cohort pre-pass: each cohort's effective forfeiture rate in every month, from the
    shares decided by then, forfeited or vested, and `prior` until any are. Returns the
    (cohorts, months) table, its cohorts' codes, and its first month."""
    t = prepared.tranches
    cohorts, code = np.unique(prepared.cohort, return_inverse=True)
    first, last = int(month.min()), int(month.max())
    width = last - first + 1
    shares = t.shares[tranche]
    vesting = month == t.vest_month[tranche]
    forfeited = np.where(vesting, 0.0, (prior_alive - alive) * shares)
    vested = np.where(vesting, prior_alive * shares, 0.0)
    grid = np.zeros((2, len(cohorts), width))
    np.add.at(grid[0], (code[tranche], month - first), forfeited)
    np.add.at(grid[1], (code[tranche], month - first), vested)
    gone, kept = np.cumsum(grid[0], axis=1), np.cumsum(grid[1], axis=1)
    decided = gone + kept
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(decided > 0, gone / decided, prior)
    return np.minimum(rates, 1.0 - 1e-12), code, np.array([first])


def calculate(
    prepared: Prepared, config: ExpenseConfig | None = None, first_month: int | None = None
) -> ExpenseResult:
    """Every tranche's monthly expense and true-ups, from the prepared table alone. With
    `first_month`, as a run's first, the rows start there: what a tranche recognized before it,
    accrual and any true-up, is its opening balance instead, and the identity still holds over
    the tranche's whole life."""
    config = config or ExpenseConfig()
    t = prepared.tranches
    n = len(t)
    a, v = t.award_month, t.vest_month
    value = prepared.value
    # Rows run through the vest, or through the forfeit for a simulated holder who leaves; a
    # tranche whose holder left before its award has none.
    count = np.maximum(0, prepared.last - a + 1)
    tranche = np.repeat(np.arange(n), count)
    starts = np.cumsum(count) - count
    month = a[tranche] + (np.arange(len(tranche)) - starts[tranche])
    # 1. Gross, by calendar days.
    if prepared.origin is None:
        days = np.ones(len(month))
        window = (v - a + 1).astype(np.float64)
    else:
        base = prepared.origin.start_time
        offsets = (
            np.arange(int(a.min(initial=0)), int(v.max(initial=0)) + 2) if n else np.zeros(1, int)
        )
        calendar = np.array(
            [(base + pd.DateOffset(months=int(m))).days_in_month for m in offsets[:-1]],
            dtype=np.float64,
        )
        cumulative = np.concatenate([[0.0], np.cumsum(calendar)])
        lo = int(offsets[0])
        days = calendar[month - lo] if len(month) else np.zeros(0)
        window = cumulative[v - lo + 1] - cumulative[a - lo]
    gross = value[tranche] * days / window[tranche]
    # 2. Survival: the chance the holder holds the tranche at each month's start, and end.
    alive = _survival(prepared, tranche, month)
    prior_alive = _survival(prepared, tranche, month - 1)
    # 3. Net, at the flat rate or each cohort's.
    if config.forfeiture == "cohort" and len(month):
        rates, code, first = _cohort_rates(
            prepared, tranche, month, alive, prior_alive, config.forfeiture_rate
        )
        rate = rates[code[tranche], month - first[0]]
        before = rates[code[tranche], np.maximum(month - 1 - first[0], 0)]
        before = np.where(month == a[tranche], rate, before)
    else:
        rate = before = np.full(len(month), config.forfeiture_rate)
    accrued_before = _within(gross, tranche, starts) - gross
    net = gross * (1 - rate)
    # 4. The true-ups: rate on what accrued before, then attrition and vesting.
    rate_up = accrued_before * (before - rate)
    recognized = _within(net + rate_up, tranche, starts)
    vests = month == v[tranche]
    attrition = np.where(vests, 0.0, -(prior_alive - alive) * recognized)
    vesting = np.where(vests, prior_alive * (value[tranche] - recognized), 0.0)
    columns = {
        "gross": gross * prior_alive,
        "net": net * prior_alive,
        "trueup_attrition": attrition,
        "trueup_rate": rate_up * prior_alive,
        "trueup_vesting": vesting,
    }
    amount = columns["net"] + columns["trueup_attrition"] + columns["trueup_rate"]
    amount = amount + columns["trueup_vesting"]
    total = np.zeros(n)
    np.add.at(total, tranche, amount)
    expected = value * _survival(prepared, np.arange(n), v - 1)
    residual = check_identity(total, expected, config.identity_tolerance)
    opening = np.zeros(n)
    keep = np.ones(len(month), dtype=bool)
    if first_month is not None:
        keep = month >= first_month
        np.add.at(opening, tranche[~keep], amount[~keep])
    rows = pd.DataFrame(
        {"tranche": tranche[keep], "month": month[keep]}
        | {name: values[keep] for name, values in columns.items()}
    )
    return ExpenseResult(rows, residual, opening if first_month is not None else None)


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
        cost_center=tranches.cost_center[t],
        award_type=tranches.award_type[t],
        level=tranches.level[t],
    )
    frame = keyed.groupby(SBC_KEYS, as_index=False)[["gross", "net", *TRUEUPS]].sum()
    frame["trueup"] = frame["trueup_attrition"] + frame["trueup_rate"] + frame["trueup_vesting"]
    frame["recognized"] = frame["net"] + frame["trueup"]
    return frame[SBC_KEYS + SBC_VALUES].sort_values(SBC_KEYS, ignore_index=True)
