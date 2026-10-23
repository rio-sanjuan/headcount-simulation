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

1. Gross expense accrues in equal monthly shares from the award month through the vest month.
2. Net expense is gross times `1 - e`, for a flat effective forfeiture rate `e`.
3. A forfeited tranche (the holder left before it vested) accrues through its forfeiture month,
   where the attrition true-up reverses every net dollar recognized.
4. A tranche that vests gets the vesting true-up at its vest month: the residual between its
   value and the net recognized. That's what makes the accounting identity hold by construction:

       cumulative net + true-ups = vested shares * value per share

A residual above the tolerance therefore means a defect in this code, and it raises.

`compute_expense` is the two together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.financial.tracker import Tranches

IDENTITY_TOLERANCE: Final = 0.01
SBC_KEYS: Final = ["month", "cost_center", "award_type", "level"]
SBC_VALUES: Final = [
    "gross",
    "net",
    "trueup_attrition",
    "trueup_vesting",
    "trueup",
    "recognized",
]


class AccountingIdentityError(AssertionError):
    """A tranche whose recognized expense doesn't tie to its vested value."""


@dataclass(frozen=True)
class ExpenseConfig:
    #: Flat effective forfeiture rate. Invented default.
    forfeiture_rate: float = 0.10
    identity_tolerance: float = IDENTITY_TOLERANCE

    def __post_init__(self) -> None:
        if not 0.0 <= self.forfeiture_rate < 1.0:
            raise ValueError("forfeiture_rate must lie in [0, 1)")


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


def calculate(
    prepared: Prepared, config: ExpenseConfig | None = None, first_month: int | None = None
) -> ExpenseResult:
    """Every tranche's monthly expense and true-ups, from the prepared table alone. With
    `first_month`, as a run's first, the rows start there: what a tranche recognized before it,
    net accrual and any true-up, is its opening balance instead, and the identity still holds
    over the tranche's whole life."""
    config = config or ExpenseConfig()
    t = prepared.tranches
    n = len(t)
    a, v, f = t.award_month, t.vest_month, t.forfeit_month
    e = config.forfeiture_rate
    value, forfeited, last = prepared.value, prepared.forfeited, prepared.last
    per_month = value / (v - a + 1)
    months_accrued = np.maximum(0, last - a + 1)
    start = a if first_month is None else np.maximum(a, first_month)
    before = months_accrued - np.maximum(0, last - start + 1)
    emitted = months_accrued - before

    index = np.repeat(np.arange(n), emitted)
    starts = np.repeat(np.cumsum(emitted) - emitted, emitted)
    month = start[index] + (np.arange(len(index)) - starts)
    gross = per_month[index]
    net_recognized = per_month * months_accrued * (1 - e)
    attrition = np.where(forfeited, -net_recognized, 0.0)
    vesting = np.where(forfeited, 0.0, value - net_recognized)
    trueup_month = np.where(forfeited, np.maximum(f, a), v)
    late = np.ones(n, dtype=bool) if first_month is None else trueup_month >= first_month
    opening = per_month * before * (1 - e) + np.where(late, 0.0, attrition + vesting)
    rows = pd.DataFrame(
        {
            "tranche": np.concatenate([index, np.flatnonzero(late)]),
            "month": np.concatenate([month, trueup_month[late]]),
            "gross": np.concatenate([gross, np.zeros(int(late.sum()))]),
            "net": np.concatenate([gross * (1 - e), np.zeros(int(late.sum()))]),
            "trueup_attrition": np.concatenate([np.zeros(len(index)), attrition[late]]),
            "trueup_vesting": np.concatenate([np.zeros(len(index)), vesting[late]]),
        }
    )
    total = opening.copy()
    for column in ("net", "trueup_attrition", "trueup_vesting"):
        np.add.at(total, rows["tranche"].to_numpy(), rows[column].to_numpy())
    residual = check_identity(total, np.where(forfeited, 0.0, value), config.identity_tolerance)
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
    frame = keyed.groupby(SBC_KEYS, as_index=False)[
        ["gross", "net", "trueup_attrition", "trueup_vesting"]
    ].sum()
    frame["trueup"] = frame["trueup_attrition"] + frame["trueup_vesting"]
    frame["recognized"] = frame["net"] + frame["trueup"]
    return frame[SBC_KEYS + SBC_VALUES].sort_values(SBC_KEYS, ignore_index=True)
