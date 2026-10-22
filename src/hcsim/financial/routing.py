"""Routing grants between recorded awards and sizing, and overriding what sizing gives.

- **Replay routing** (`route`): a grant's recipients who hold a recorded award of its type in
  its month take that award as recorded; the rest are sized by the strategy. Every recipient's
  base is the strategy's imputation, since a recorded award carries no salary.
- **Hybrid windows** (`HybridWindow`): recorded awards route only through a last step, and
  every grant after it is sized, as when a run starts before its data ends: its first months
  replay what was granted, and the rest forecast it.
- **Overrides** (`Overridden`): an aggregate override scales every grant a strategy sizes by
  one factor, as for a budget change. Individual overrides set named employees' shares per
  forward year outright, and aren't scaled.

A routed batch's shares per year are its schedule's, in twelve-month buckets over the whole
72-month buffer, so recorded and sized rows line up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.financial.calculator import SizingError
from hcsim.financial.prices import StockPriceModel
from hcsim.financial.sizing import AwardBatch, SizingContext, SizingStrategy
from hcsim.financial.vesting import CANONICAL_VEST_MONTHS, VestTemplate
from hcsim.model.core.layouts import Buffer, EmployeeColumn

Ints = NDArray[np.int64]
#: The columns a recorded tranche needs: who, which award type, the award month as a step, how
#: many months after it the tranche vests, and its shares.
RECORDED_COLUMNS: Final = ("employee_id", "award_type", "step", "after", "shares")
YEARS: Final = CANONICAL_VEST_MONTHS // 12


def per_year(schedule: Ints) -> Ints:
    """A (rows, 72) schedule's shares in each of its six twelve-month forward years."""
    out: Ints = schedule.reshape(len(schedule), YEARS, 12).sum(axis=2)
    return out


@dataclass(frozen=True)
class RecordedAwards:
    """Awards as recorded: per award type and month (a step), each recipient's monthly vest
    schedule, column c vesting c + 1 months after the award month."""

    schedules: Mapping[tuple[str, int], Mapping[int, Ints]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, by_id in self.schedules.items():
            for employee, schedule in by_id.items():
                if schedule.shape != (CANONICAL_VEST_MONTHS,) or (schedule < 0).any():
                    raise SizingError(
                        f"a recorded schedule is {CANONICAL_VEST_MONTHS} non-negative months: "
                        f"employee {employee}'s {key[0]} award at step {key[1]} isn't"
                    )
        object.__setattr__(
            self,
            "schedules",
            MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.schedules.items()}),
        )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> RecordedAwards:
        """From one row per recorded tranche (`RECORDED_COLUMNS`). An employee's awards of one
        type in one month add up. A tranche vesting in its award month vests in the month after,
        the buffer's first; one beyond the buffer is refused."""
        absent = [c for c in RECORDED_COLUMNS if c not in frame.columns]
        if absent:
            raise SizingError(f"recorded tranches need the columns {absent}")
        after = np.maximum(frame["after"].to_numpy(dtype=np.int64), 1)
        beyond = after > CANONICAL_VEST_MONTHS
        if beyond.any():
            raise SizingError(
                f"{int(beyond.sum())} recorded tranches vest more than {CANONICAL_VEST_MONTHS} "
                "months after their award"
            )
        schedules: dict[tuple[str, int], dict[int, Ints]] = {}
        keys = zip(
            frame["award_type"].astype(str),
            frame["step"].astype(int),
            frame["employee_id"].astype(int),
            after,
            frame["shares"].astype(int),
            strict=True,
        )
        for award_type, step, employee, months, shares in keys:
            by_id = schedules.setdefault((award_type, step), {})
            schedule = by_id.setdefault(employee, np.zeros(CANONICAL_VEST_MONTHS, dtype=np.int64))
            schedule[months - 1] += shares
        return cls(schedules)

    def lookup(self, ids: Ints, step: int, award_type: str) -> tuple[NDArray[np.bool_], Ints]:
        """Which of `ids` hold a recorded award of `award_type` at `step`, and their schedules
        (zeros where they hold none)."""
        by_id = self.schedules.get((award_type, step), {})
        covered = np.array([int(i) in by_id for i in ids], dtype=bool)
        schedules = np.zeros((len(ids), CANONICAL_VEST_MONTHS), dtype=np.int64)
        for row in np.flatnonzero(covered):
            schedules[row] = by_id[int(ids[row])]
        return covered, schedules

    def __len__(self) -> int:
        return sum(len(v) for v in self.schedules.values())


@dataclass(frozen=True)
class HybridWindow:
    """Recorded awards route through step `through`, and grants after it are sized. With no
    last step, they route at every step."""

    through: int | None = None

    def covers(self, step: int) -> bool:
        return self.through is None or step <= self.through


@dataclass(frozen=True)
class Routed:
    """A routed grant: its batch, which rows took a recorded award, and which were sized."""

    batch: AwardBatch
    replayed: NDArray[np.bool_]
    sized: NDArray[np.bool_]


def route(
    strategy: SizingStrategy | None,
    award_type: str,
    rows: Buffer,
    prices: StockPriceModel,
    context: SizingContext,
    recorded: RecordedAwards | None = None,
    window: HybridWindow | None = None,
) -> Routed | None:
    """Route one grant's recipients: a recorded award where they hold one and the window covers
    the step, sizing by `strategy` otherwise. None when no recipient is covered either way.
    Without a strategy, recorded recipients keep the base they have."""
    ids = np.asarray(rows[:, EmployeeColumn.EMPLOYEE_ID], dtype=np.int64)
    n = len(rows)
    covered = np.zeros(n, dtype=bool)
    schedule = np.zeros((n, CANONICAL_VEST_MONTHS), dtype=np.int64)
    if recorded is not None and (window or HybridWindow()).covers(context.step):
        covered, schedule = recorded.lookup(ids, context.step, award_type)
    if strategy is None and not covered.any():
        return None
    base = rows[:, EmployeeColumn.BASE_SALARY].astype(np.float64)
    sized = np.zeros(n, dtype=bool)
    if strategy is not None:
        batch = strategy.size(rows, prices, context)
        base = batch.base_salary
        sized = ~covered
        schedule = np.where(covered[:, None], schedule, batch.vest_schedule)
    else:
        schedule = np.where(covered[:, None], schedule, 0)
    return Routed(AwardBatch(award_type, base, per_year(schedule), schedule), covered, sized)


@dataclass(frozen=True)
class Overridden:
    """A strategy with an aggregate override, scaling every grant it sizes by `scale`, and
    individual overrides, setting named employees' shares per forward year of the strategy's
    template outright. The template is the strategy's own unless given."""

    strategy: SizingStrategy
    scale: float = 1.0
    shares: Mapping[int, Sequence[int]] = field(default_factory=dict)
    template: VestTemplate | None = None

    def __post_init__(self) -> None:
        if self.template is None:
            own = getattr(self.strategy, "template", None)
            if not isinstance(own, VestTemplate):
                raise SizingError("an override needs a template: its strategy has none")
            object.__setattr__(self, "template", own)
        assert self.template is not None
        problems = []
        if not (np.isfinite(self.scale) and self.scale >= 0):
            problems.append(f"an aggregate override is a non-negative factor, not {self.scale}")
        years = self.template.n_years
        bad = sorted(
            e for e, s in self.shares.items() if len(s) != years or any(int(x) < 0 for x in s)
        )
        if bad:
            problems.append(
                f"individual overrides give {years} non-negative yearly share counts, which "
                f"employees {bad} don't"
            )
        if problems:
            raise SizingError("; ".join(problems))
        object.__setattr__(self, "shares", MappingProxyType(dict(self.shares)))

    @property
    def award_type(self) -> str:
        return self.strategy.award_type

    def size(
        self, rows: Buffer, prices: StockPriceModel, context: SizingContext | None = None
    ) -> AwardBatch:
        assert self.template is not None
        batch = self.strategy.size(rows, prices, context)
        if batch.shares_per_year.shape[1] != self.template.n_years:
            raise SizingError(
                f"the override's template has {self.template.n_years} years, and the "
                f"strategy's grants {batch.shares_per_year.shape[1]}"
            )
        shares = np.rint(batch.shares_per_year * self.scale).astype(np.int64)
        ids = rows[:, EmployeeColumn.EMPLOYEE_ID].astype(np.int64)
        for row, employee in enumerate(ids):
            if int(employee) in self.shares:
                shares[row] = np.asarray(self.shares[int(employee)], dtype=np.int64)
        return AwardBatch(batch.award_type, batch.base_salary, shares, self.template.expand(shares))
