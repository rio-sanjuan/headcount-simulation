"""SBC output: a run's expense by part and basis, aligned to actuals.

A trajectory's SBC frame keeps every row the calculator made, summed by month, cost center,
award type, and level (`expense.SBC_KEYS`), and labelled two ways:

- **part**: `sim`, the grants the run made; `seed`, the equity granted before it; `offline`, the
  offline subsidiaries' stream; or `manual`, the adjustments added by hand;
- **basis**: `replayed`, an award as recorded, which the seed's and the offline stream's always
  are and a run's grant is when a recorded award routed it; or `sized`, one a strategy sized,
  which a manual adjustment counts as too.

**Aligned to actuals.** `align` sets a forecast's SBC beside the actual SBC history recorded for
the same months, part, and keys, a row per combination either holds, with the gap between them,
so a run's SBC can be scored against the record.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from hcsim.financial.expense import SBC_KEYS, SBC_VALUES, TRUEUPS, ExpenseResult
from hcsim.financial.tracker import Tranches

PARTS: Final = ("sim", "seed", "offline", "manual")
BASES: Final = ("replayed", "sized")
#: The keys a parts frame sums by.
PART_KEYS: Final = ["part", "basis", *SBC_KEYS]


class OutputError(ValueError):
    """SBC parts that don't line up."""


def _labelled(frame: pd.DataFrame, part: str, basis: str) -> pd.DataFrame:
    return frame.assign(part=part, basis=basis)[[*PART_KEYS, *SBC_VALUES]]


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
    for column in ("gross", "net", *TRUEUPS):
        keyed[column] = rows[column].to_numpy()
    frame = keyed.groupby(PART_KEYS, as_index=False)[["gross", "net", *TRUEUPS]].sum()
    frame["part"] = np.array(["sim", "seed"])[frame["part"].to_numpy()]
    frame["basis"] = np.array(["replayed", "sized"])[frame["basis"].to_numpy()]
    frame["trueup"] = frame[list(TRUEUPS)].sum(axis=1)
    frame["recognized"] = frame["net"] + frame["trueup"]
    return combine_parts([frame[[*PART_KEYS, *SBC_VALUES]], *streams(offline, manual)])


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
