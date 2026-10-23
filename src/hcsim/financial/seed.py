"""The seed stream: equity granted before the run that's still vesting when it starts.

Most of a year's SBC comes from awards already granted, so a run starts from them: the seed
tranches, each with its award and vest month as steps from the run's first month (an award
month before the run is negative), its shares, its grant's fair value per share, its holder's
level at grant, and its holder's **cost center at the origin**, the run's first month, which the
seed's expense is attributed to.

**Holder status.** Each seed tranche's holder is classified at the origin (`classify`):

- `active`: in the run's seeded workforce. Their tranches join the run's ledger, so a simulated
  departure forfeits them and annual sizing counts their vests.
- `outside`: in post at the origin, but outside the run's subset. Their tranches wait in the
  ledger's scratch, never published.
- `future`: a known future entrant, holding equity before they start. Their tranches wait in
  scratch until their start month.
- `departed`: neither. Their tranches have nothing left to vest in the run, so they're left out.

**The opening balance.** A seed tranche has accrued expense before the run; `opening_balance`
gives what each recognized through the month before it, so the run's months continue from it.

**The departure hazard.** The run simulates only its active holders. For the rest the seed
stream needs each holder's chance of leaving: in a replay, the realized departures, each
tranche forfeiting in the month history says; in a forecast, a model-averaged monthly hazard.
`DepartureHazard.survival` gives each tranche's chance of still being held at each month's end.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.financial.expense import ExpenseConfig, compute_expense
from hcsim.financial.tracker import NOT_FORFEITED, Tranches

Ints = NDArray[np.int64]
Floats = NDArray[np.float64]


class HolderStatus(StrEnum):
    ACTIVE = "active"
    OUTSIDE = "outside"
    FUTURE = "future"
    DEPARTED = "departed"


#: The columns a seed frame holds, one row per tranche.
COLUMNS: Final = (
    "award_id",
    "holder",
    "award_type",
    "award_month",
    "vest_month",
    "shares",
    "value_per_share",
    "level",
    "cost_center",
    "status",
    "start_month",
    "forfeit_month",
)


class SeedError(ValueError):
    """Seed tranches that don't hold together."""


def classify(holders: Ints, active: Ints, outside: Ints, future: Ints) -> NDArray[np.str_]:
    """Each holder's status at the origin: active when in `active`, else outside when in
    `outside`, else a future entrant when in `future`, else departed."""
    holders = np.asarray(holders, dtype=np.int64)
    out = np.full(len(holders), HolderStatus.DEPARTED.value, dtype=object)
    out[np.isin(holders, future)] = HolderStatus.FUTURE.value
    out[np.isin(holders, outside)] = HolderStatus.OUTSIDE.value
    out[np.isin(holders, active)] = HolderStatus.ACTIVE.value
    return out.astype(str)


@dataclass(frozen=True)
class SeedTranches:
    """The seed's tranches, one row per tranche (`COLUMNS`).

    `start_month` is a future entrant's start, -1 for the rest; `forfeit_month` the month
    history records the holder leaving after the origin, -1 when it records none, which only a
    replay may read."""

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        absent = [c for c in COLUMNS if c not in self.frame.columns]
        if absent:
            raise SeedError(f"seed tranches need the columns {absent}")
        f = self.frame
        problems = []
        if (f["award_month"] >= 0).any():
            problems.append("a seed tranche is awarded before the run, in a negative month")
        if (f["vest_month"] < 0).any():
            problems.append("a seed tranche still vests at or after the run's first month")
        if (f["shares"] <= 0).any():
            problems.append("a seed tranche has at least one share")
        values = f["value_per_share"].to_numpy(dtype=np.float64)
        if not (np.isfinite(values) & (values >= 0)).all():
            problems.append("a seed tranche's value per share is finite and non-negative")
        unknown = sorted(set(f["status"]) - {s.value for s in HolderStatus})
        if unknown:
            problems.append(f"holder statuses are {[s.value for s in HolderStatus]}, not {unknown}")
        if problems:
            raise SeedError("; ".join(problems))

    def __len__(self) -> int:
        return len(self.frame)

    def of(self, *statuses: HolderStatus) -> pd.DataFrame:
        """The tranches whose holders have one of `statuses`."""
        return self.frame[self.frame["status"].isin([s.value for s in statuses])]

    def counts(self) -> dict[str, int]:
        """Tranches per holder status."""
        return {s.value: int((self.frame["status"] == s.value).sum()) for s in HolderStatus}

    def tranches(self, *statuses: HolderStatus) -> Tranches:
        """The tranches of `statuses` (every status by default) as the expense calculator reads
        them, live at the origin, valued at their grants' fair values."""
        rows = self.of(*statuses) if statuses else self.frame

        def column(name: str) -> Ints:
            return rows[name].to_numpy(dtype=np.int64)

        return Tranches(
            holder=column("holder"),
            award_month=column("award_month"),
            vest_month=column("vest_month"),
            shares=rows["shares"].to_numpy(dtype=np.float64),
            forfeit_month=np.full(len(rows), NOT_FORFEITED, dtype=np.int64),
            award_type=column("award_type"),
            level=column("level"),
            cost_center=column("cost_center"),
            value=rows["value_per_share"].to_numpy(dtype=np.float64),
        )

    def opening_balance(self, config: ExpenseConfig | None = None) -> Floats:
        """What each tranche recognized before the run, through the month before its first:
        its net expense from its award month, at the calculator's flat forfeiture rate."""
        t = self.tranches()
        if not len(t):
            return np.zeros(0)
        opening = compute_expense(t, t.values, config, first_month=0).opening
        assert opening is not None
        return opening


@dataclass(frozen=True)
class DepartureHazard:
    """Each unsimulated holder's chance of leaving: realized departures, a forfeit month per
    tranche (-1 for none), in a replay; or a model-averaged monthly hazard in a forecast."""

    realized: Ints | None = None
    monthly: float = 0.0

    def __post_init__(self) -> None:
        if self.realized is None and not 0.0 <= self.monthly < 1.0:
            raise SeedError(f"a monthly departure hazard lies in [0, 1), not {self.monthly}")

    @classmethod
    def replayed(cls, forfeit_months: Ints) -> DepartureHazard:
        """History's departures, each tranche forfeiting in the month it records."""
        return cls(realized=np.asarray(forfeit_months, dtype=np.int64))

    @classmethod
    def averaged(cls, monthly: float) -> DepartureHazard:
        """A model-averaged hazard, the same each month for every holder."""
        return cls(monthly=float(monthly))

    def survival(self, n: int, months: int) -> Floats:
        """(tranches, months): each of `n` tranches' chance its holder still holds it at the end
        of each of the run's months."""
        if self.realized is not None:
            if len(self.realized) != n:
                raise SeedError(
                    f"one realized forfeit month per tranche: {n}, not {len(self.realized)}"
                )
            left = self.realized[:, None]
            month = np.arange(months)[None, :]
            out: Floats = np.where((left >= 0) & (month >= left), 0.0, 1.0)
            return out
        return np.repeat((1.0 - self.monthly) ** np.arange(1, months + 1)[None, :], n, axis=0)
