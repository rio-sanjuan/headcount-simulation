"""The offline and manual streams: SBC the simulation doesn't make.

**The offline stream** (`OfflineStream`) is the equity of offline subsidiaries, valued by
appraisal with no market price. Their tranches outstanding at the run's start run through the
same calculator as the simulation's, with no attrition: none forfeits, whatever the run's people
do, and each is valued at its latest appraisal before the run. Its expense is attributed to the
holder's cost center at the origin.

In a replay, the stream is history's. It also holds the awards the offline plans granted during
the run, each at its grant's appraisal, and a tranche forfeits in the month history records
(`forfeit_month`). Those awards, as recorded (`recorded`), route a grant as the primary plan's
recorded awards do (see `financial.routing`): a recipient who holds one takes it from the
stream, so the run neither replays nor sizes their grant.

**Manual adjustments** (`ManualAdjustments`) are expense a planner adds by hand, as for a known
grant the run can't make: a month, a cost center, an award type, a level, an amount, and a note.
Each adds its amount to the month's recognized expense, and nothing else.

Both give SBC in the same keyed form as the simulation's (`expense.SBC_KEYS`), so the parts can
be added up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from hcsim.financial.expense import (
    SBC_KEYS,
    SBC_VALUES,
    ExpenseConfig,
    ExpenseResult,
    compute_expense,
    sbc_frame,
)
from hcsim.financial.routing import RecordedAwards
from hcsim.financial.tracker import NOT_FORFEITED, Tranches
from hcsim.model.core.schema import AWARD_TYPE, COST_CENTER, JOB_LEVEL, SimulationSchema

#: The columns an offline frame holds, one row per tranche. A replay's adds `forfeit_month`.
OFFLINE_COLUMNS: Final = (
    "award_id",
    "holder",
    "award_type",
    "award_month",
    "vest_month",
    "shares",
    "value_per_share",
    "level",
    "cost_center",
)
#: The columns a manual adjustment gives, in labels.
MANUAL_COLUMNS: Final = ("month", "cost_center", "award_type", "level", "amount", "note")


class StreamError(ValueError):
    """A stream whose rows don't hold together."""


@dataclass(frozen=True)
class OfflineStream:
    """Offline subsidiaries' tranches (`OFFLINE_COLUMNS`): those outstanding at the run's start,
    each at its latest appraisal before it, and in a replay, those granted during it, each at its
    grant's appraisal, with the month each tranche forfeits and the awards as `recorded`."""

    frame: pd.DataFrame
    recorded: RecordedAwards | None = None

    def __post_init__(self) -> None:
        absent = [c for c in OFFLINE_COLUMNS if c not in self.frame.columns]
        if absent:
            raise StreamError(f"offline tranches need the columns {absent}")
        f = self.frame
        problems = []
        if (f["vest_month"] < 0).any():
            problems.append("an offline tranche still vests at or after the run's first month")
        if (f["vest_month"] < f["award_month"]).any():
            problems.append("an offline tranche can't vest before its award month")
        values = f["value_per_share"].to_numpy(dtype=np.float64)
        if not (np.isfinite(values) & (values >= 0)).all():
            problems.append("an offline tranche's appraisal is finite and non-negative")
        forfeits = f["forfeit_month"] if "forfeit_month" in f else pd.Series(dtype=np.int64)
        if ((forfeits < 0) & (forfeits != NOT_FORFEITED)).any():
            problems.append("an offline tranche forfeits during the run, or never (-1)")
        if self.recorded is not None:
            held = set(zip(f["award_month"].tolist(), f["holder"].tolist(), strict=True))
            missing = sum(
                (step, employee) not in held
                for (_, step), by_id in self.recorded.schedules.items()
                for employee in by_id
            )
            if missing:
                problems.append(
                    f"{missing} recorded awards have no tranche awarded to their holder in their "
                    "month"
                )
        if problems:
            raise StreamError("; ".join(problems))

    def __len__(self) -> int:
        return len(self.frame)

    def tranches(self) -> Tranches:
        """The tranches as the calculator reads them: none forfeits, unless a replay's
        `forfeit_month` gives the month it does."""
        f = self.frame

        def column(name: str) -> np.ndarray:
            return f[name].to_numpy(dtype=np.int64)

        forfeits = (
            column("forfeit_month")
            if "forfeit_month" in f
            else np.full(len(f), NOT_FORFEITED, dtype=np.int64)
        )
        return Tranches(
            holder=column("holder"),
            award_month=column("award_month"),
            vest_month=column("vest_month"),
            shares=f["shares"].to_numpy(dtype=np.float64),
            forfeit_month=forfeits,
            award_type=column("award_type"),
            level=column("level"),
            cost_center=column("cost_center"),
            value=f["value_per_share"].to_numpy(dtype=np.float64),
        )

    def expense(self, config: ExpenseConfig | None = None) -> ExpenseResult:
        """The stream through the calculator, from the run's first month."""
        t = self.tranches()
        return compute_expense(t, t.values, config, first_month=0)

    def sbc(self, config: ExpenseConfig | None = None) -> pd.DataFrame:
        """Its SBC by month, cost center, award type, and level."""
        if not len(self):
            return pd.DataFrame(columns=[*SBC_KEYS, *SBC_VALUES])
        return sbc_frame(self.expense(config), self.tranches())


@dataclass(frozen=True)
class ManualAdjustments:
    """Expense added by hand: one row per adjustment, in schema codes, with its note."""

    frame: pd.DataFrame

    @classmethod
    def from_labels(
        cls, frame: pd.DataFrame, schema: SimulationSchema, months: int
    ) -> ManualAdjustments:
        """Adjustments given in labels, for a run of `months`, refused with every problem named:
        a month outside the run, a label the schema doesn't know, or an amount that isn't
        finite."""
        absent = [c for c in MANUAL_COLUMNS if c not in frame.columns]
        if absent:
            raise StreamError(f"manual adjustments need the columns {absent}")
        problems = []
        month = frame["month"].to_numpy(dtype=np.int64)
        outside = sorted({int(m) for m in month if not 0 <= m < months})
        if outside:
            problems.append(f"months {outside} lie outside the run's 0..{months - 1}")
        coded = frame.copy()
        for column, vocabulary in (
            ("cost_center", COST_CENTER),
            ("award_type", AWARD_TYPE),
            ("level", JOB_LEVEL),
        ):
            codes = schema.encode(vocabulary, frame[column].astype(str), unknown="unattributed")
            unknown = sorted(set(frame[column].astype(str)[codes < 0]))
            if unknown:
                problems.append(f"{column} labels the schema doesn't know: {unknown}")
            coded[column] = codes
        amounts = frame["amount"].to_numpy(dtype=np.float64)
        if not np.isfinite(amounts).all():
            problems.append(f"{int((~np.isfinite(amounts)).sum())} amounts aren't finite")
        if problems:
            raise StreamError("manual adjustments: " + "; ".join(problems))
        return cls(coded[list(MANUAL_COLUMNS)].reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.frame)

    def sbc(self) -> pd.DataFrame:
        """Their SBC by month, cost center, award type, and level: each adjustment's amount in
        the month's recognized expense, nothing else."""
        out: pd.DataFrame = self.frame.groupby(SBC_KEYS, as_index=False).agg(
            recognized=("amount", "sum")
        )
        for column in SBC_VALUES:
            if column != "recognized":
                out[column] = 0.0
        return out[[*SBC_KEYS, *SBC_VALUES]].sort_values(SBC_KEYS, ignore_index=True)
