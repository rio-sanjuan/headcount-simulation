"""Phase 3: amortize every tranche into monthly expense, after the simulation.

The calculator is stateless and vectorized over tranches. For each tranche, worth
`shares * value_per_share`:

1. Gross expense accrues in equal monthly shares from the award month through the vest month.
   Day-weighted amortization arrives in M6.
2. Net expense is gross times `1 - e`, for a flat effective forfeiture rate `e`. The cohort rate
   path arrives in M6.
3. A forfeited tranche (the holder left before it vested) accrues through its forfeiture month,
   where the attrition true-up reverses every net dollar recognized.
4. A tranche that vests gets the vesting true-up at its vest month: the residual between its
   value and the net recognized. That's what makes the accounting identity hold by construction:

       cumulative net + true-ups = vested shares * value per share

A residual above the tolerance therefore means a defect in this code, and it raises.
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


def compute_expense(
    tranches: Tranches,
    value_per_share: NDArray[np.float64],
    config: ExpenseConfig | None = None,
    first_month: int | None = None,
) -> ExpenseResult:
    """Every tranche's monthly expense and true-ups. With `first_month`, as a run's first, the
    rows start there: what a tranche recognized before it, net accrual and any true-up, is its
    opening balance instead, and the identity still holds over the tranche's whole life."""
    config = config or ExpenseConfig()
    n = len(tranches)
    vps = np.asarray(value_per_share, dtype=np.float64)
    if vps.shape != (n,):
        raise ValueError("one value per share per tranche")
    if not np.isfinite(vps).all() or (vps < 0).any():
        raise ValueError("values per share must be finite and non-negative")
    a, v, f = tranches.award_month, tranches.vest_month, tranches.forfeit_month
    if (v < a).any():
        raise ValueError("a tranche can't vest before its award month")
    e = config.forfeiture_rate
    value = tranches.shares * vps
    per_month = value / (v - a + 1)
    forfeited = (f >= 0) & (f < v)
    last = np.where(forfeited, f, v)
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
