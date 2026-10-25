"""SBC output: a run's expense by part and basis, aligned to actuals.

A trajectory's SBC frame keeps every row the calculator made, summed by month, cost center,
award type, and level (`expense.SBC_KEYS`), and labelled two ways:

- **part**: `sim`, the grants the run made; `seed`, the equity granted before it; `offline`, the
  offline subsidiaries' stream; or `manual`, the adjustments added by hand;
- **basis**: `replayed`, an award as recorded, which the seed's and the offline stream's always
  are and a run's grant is when a recorded award routed it; or `sized`, one a strategy sized,
  which a manual adjustment counts as too.

**Publication.** A subset run publishes SBC only for holders inside its subset, month by month
(see `model.data.publication`): `publish` splits a result's rows by their holder's membership in
each row's month, into the published rows and the rest, the run's scratch, which stays out of
its SBC output and reconciles it with the whole.

**Aligned to actuals.** `align` sets a forecast's SBC beside the actual SBC history recorded for
the same months, part, and keys, a row per combination either holds, with the gap between them,
so a run's SBC can be scored against the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.financial.expense import SBC_KEYS, SBC_VALUES, TRUEUPS, ExpenseResult
from hcsim.financial.tracker import Tranches

PARTS: Final = ("sim", "seed", "offline", "manual")
BASES: Final = ("replayed", "sized")
#: The keys a parts frame sums by.
PART_KEYS: Final = ["part", "basis", *SBC_KEYS]


class OutputError(ValueError):
    """SBC parts that don't line up."""


def publish(
    result: ExpenseResult, inside: NDArray[np.bool_]
) -> tuple[ExpenseResult, ExpenseResult]:
    """The result's rows split by the publication rule: those whose holder is inside the subset
    in the row's month (`inside`, one flag per row), and the rest, the run's scratch. Each half
    keeps the result's residuals and opening balances."""
    flags = np.asarray(inside, dtype=bool)
    if flags.shape != (len(result.rows),):
        raise OutputError(f"one membership flag per row: {len(result.rows)}, not {flags.shape}")
    rows = result.rows
    return (
        replace(result, rows=rows[flags].reset_index(drop=True)),
        replace(result, rows=rows[~flags].reset_index(drop=True)),
    )


def _labelled(frame: pd.DataFrame, part: str, basis: str) -> pd.DataFrame:
    return frame.assign(part=part, basis=basis)[[*PART_KEYS, *SBC_VALUES]]


#: The values a coded frame sums: each row's own, before the true-ups' total and the recognized.
SUMMED: Final = ["gross", "net", *TRUEUPS]
_PART_LABELS: Final = np.array(["sim", "seed"], dtype=object)
_BASIS_LABELS: Final = np.array(["replayed", "sized"], dtype=object)


def sbc_parts(
    result: ExpenseResult,
    tranches: Tranches,
    offline: pd.DataFrame | None = None,
    manual: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """A trajectory's SBC by part and basis (`PART_KEYS`): its ledger's rows, `seed` for a
    tranche awarded before the run and `sim` for the rest, beside the offline stream's and the
    manual adjustments' SBC frames, when given. Rows are grouped by integer codes and labelled
    after, so no row carries text."""
    labelled = label_parts(coded_parts(result, tranches))
    return combine_parts([labelled, *streams(offline, manual)])


def coded_parts(result: ExpenseResult, tranches: Tranches) -> pd.DataFrame:
    """The ledger's rows summed by `PART_KEYS`, the part and basis as integer codes (`sim` 0,
    `seed` 1; `replayed` 0, `sized` 1), the values those `SUMMED` sums. Coded frames add up by
    `sum_coded`, and `label_parts` labels them."""
    rows = result.rows
    t = rows["tranche"].to_numpy()
    seed = tranches.award_month[t] < 0
    keyed = pd.DataFrame(
        {
            "part": seed.astype(np.int8),
            "basis": (~(tranches.recorded[t] | seed)).astype(np.int8),
            "month": rows["month"].to_numpy(),
            "cost_center": rows["cost_center"].to_numpy()
            if "cost_center" in rows
            else tranches.cost_center[t],
            "award_type": tranches.award_type[t],
            "level": tranches.level[t],
        }
    )
    for column in SUMMED:
        keyed[column] = rows[column].to_numpy()
    return sum_coded([keyed])


def sum_coded(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Coded frames summed into one by `PART_KEYS`, in no particular order."""
    present = [f for f in frames if len(f)]
    if not present:
        return pd.DataFrame(columns=[*PART_KEYS, *SUMMED])
    joined = present[0] if len(present) == 1 else pd.concat(present, ignore_index=True)
    return joined.groupby(PART_KEYS, as_index=False, sort=False)[SUMMED].sum()


def label_parts(coded: pd.DataFrame) -> pd.DataFrame:
    """A coded frame labelled: the part and basis by name, the true-ups' total, and the
    recognized expense, the net plus it."""
    # Object arrays of the labels, so every row shares two strings rather than making its own.
    columns: dict[str, Any] = {
        "part": _PART_LABELS[coded["part"].to_numpy(dtype=np.int64)],
        "basis": _BASIS_LABELS[coded["basis"].to_numpy(dtype=np.int64)],
        **{k: coded[k].to_numpy() for k in SBC_KEYS},
        **{k: coded[k].to_numpy() for k in SUMMED},
    }
    trueup = sum(coded[k].to_numpy() for k in TRUEUPS)
    columns["trueup"] = trueup
    columns["recognized"] = coded["net"].to_numpy() + trueup
    return pd.DataFrame(columns, columns=[*PART_KEYS, *SBC_VALUES], copy=False)


def streams(offline: pd.DataFrame | None, manual: pd.DataFrame | None) -> list[pd.DataFrame]:
    """The offline stream's and the manual adjustments' SBC frames, labelled as their parts."""
    out = []
    if offline is not None and len(offline):
        out.append(_labelled(offline, "offline", "replayed"))
    if manual is not None and len(manual):
        out.append(_labelled(manual, "manual", "sized"))
    return out


def combine_parts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Labelled SBC frames, as a run's chunks give them, summed into one by `PART_KEYS`."""
    present = [f for f in frames if len(f)]
    if not present:
        return pd.DataFrame(columns=[*PART_KEYS, *SBC_VALUES])
    joined = pd.concat(present, ignore_index=True)
    summed = joined.groupby(PART_KEYS, as_index=False)[SBC_VALUES].sum()
    return summed.sort_values(PART_KEYS, ignore_index=True)


#: The rows a running total lets wait before summing them, at least.
SUMMED_ROWS: Final = 200_000


@dataclass
class PartsTotal:
    """Coded SBC frames (`coded_parts`) summed as they come, so a run's many chunks never pile
    up, and labelled once, at the end. Once the frames waiting hold more rows than the limit,
    they're summed into one. The limit is at least `limit`, and twice the last sum's rows, the
    sum itself counting among the frames waiting: so a sum waits until the rows added since
    match its own, summing stays linear however many keys the total holds, and a sum never
    holds much more than twice its total's rows. The total is the sum of every frame
    added, up to the order floats are summed in."""

    limit: int = SUMMED_ROWS
    _frames: list[pd.DataFrame] = field(default_factory=list)
    _rows: int = 0

    def add(self, frame: pd.DataFrame) -> None:
        if not len(frame):
            return
        self._frames.append(frame)
        self._rows += len(frame)
        if self._rows > self.limit:
            summed = sum_coded(self._frames)
            self._frames, self._rows = [summed], len(summed)
            self.limit = max(self.limit, 2 * len(summed))

    def total(self, more: list[pd.DataFrame] | None = None) -> pd.DataFrame:
        """Every frame added, labelled, and `more`, labelled frames of other parts, summed into
        one and sorted by `PART_KEYS`, as `combine_parts` sorts. The ledger's keys are already
        unique once summed, and no other part shares them, so the ledger's frame is sorted by
        its codes, in its labels' order, and labelled after, rather than grouped again by
        text."""
        summed = sum_coded(self._frames)
        others = combine_parts(list(more or []))
        if not len(summed):
            return others
        clash = sorted(set(others["part"]) & {"sim", "seed"})
        if clash:
            raise OutputError(f"only the ledger's rows are parts {clash}")
        # "seed" sorts before "sim", though its code is the larger.
        keys = [summed[k].to_numpy() for k in reversed(SBC_KEYS)]
        order = np.lexsort([*keys, summed["basis"].to_numpy(), 1 - summed["part"].to_numpy()])
        ledger = label_parts(summed.iloc[order].reset_index(drop=True))
        if not len(others):
            return ledger
        return pd.concat([others, ledger], ignore_index=True)


def align(forecast: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """The forecast's recognized SBC beside the actual, by part and `SBC_KEYS`, over the months
    the actual covers: a row per combination either holds, a missing side counting zero, with
    the gap, forecast less actual."""
    keys = ["part", *SBC_KEYS]
    for name, frame in (("forecast", forecast), ("actual", actual)):
        absent = [c for c in (*keys, "recognized") if c not in frame.columns]
        if absent:
            raise OutputError(f"the {name} SBC needs the columns {absent}")
    months = set(actual["month"])
    ours = forecast[forecast["month"].isin(months)].groupby(keys)["recognized"].sum()
    theirs = actual.groupby(keys)["recognized"].sum()
    joined = pd.concat([ours.rename("forecast"), theirs.rename("actual")], axis=1).fillna(0.0)
    joined["gap"] = joined["forecast"] - joined["actual"]
    return joined.reset_index().sort_values(keys, ignore_index=True)
