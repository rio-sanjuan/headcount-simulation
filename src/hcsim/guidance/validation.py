"""Validation: the series checked and settled before any model sees them.

Ingestion counts what the warehouse holds; validation decides what a forecast starts from.

- **Month-end dates.** The aggregate fact's periods are month-end dates. Any other date is a
  defect in the feed, and ingestion refuses the source before counting anything.
- **Zero-filling.** Every group has every month and category, zero where nothing happened, as
  ingestion's grid lays them out.
- **The snapshot rule.** The history ends at the snapshot: by default, the last month in which
  any measure counted anything. Months after it with nothing at all are trimmed, because they
  usually mark months not yet loaded. A configuration can fix the snapshot at an earlier month,
  as a backtest does, and the months after it are dropped. A snapshot past the data is refused.
- **Interior gaps.** A month up to the snapshot in which no measure has any row at all is a gap
  in the feed, not a month of zeros. Validation refuses it by default. With `gaps: zero` it
  stays in as zeros. Either way the report names it.
- **All-zero series.** A series that's zero for its whole history gets no forecast. It's
  dropped and named in the report, unless the configuration keeps it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from hcsim.guidance.ingest import CATEGORY, GROUP, MONTH, VALUE, Series


@dataclass(frozen=True)
class Validated:
    series: Series
    #: Months up to the snapshot with no rows at all, in order.
    gaps: tuple[pd.Period, ...]
    #: The (group, category) series dropped for being zero throughout.
    dropped: tuple[tuple[str, str], ...]


def snapshot_of(series: Series, month: str | None) -> pd.Period:
    """The snapshot: `month` when the configuration fixes one, else the snapshot rule's."""
    return series.snapshot if month is None else pd.Period(month, freq="M")


def gaps(series: Series, snapshot: pd.Period) -> tuple[pd.Period, ...]:
    """Months from the first through `snapshot` with no rows at all in any measure."""
    months = pd.period_range(series.frame[MONTH].min(), snapshot, freq="M")
    observed = set(series.observed)
    return tuple(m for m in months if m not in observed)


def problems(series: Series, month: str | None, gap_policy: str) -> list[str]:
    """What keeps `series` from validating, as the configuration asks."""
    found = []
    snapshot = snapshot_of(series, month)
    if snapshot > series.snapshot:
        found.append(f"the snapshot {snapshot} is past the data, which ends at {series.snapshot}")
        return found
    if snapshot < series.frame[MONTH].min():
        found.append(
            f"the snapshot {snapshot} is before the data, which starts at "
            f"{series.frame[MONTH].min()}"
        )
        return found
    missing = gaps(series, snapshot)
    if missing and gap_policy == "refuse":
        listed = ", ".join(str(m) for m in missing)
        found.append(
            f"{len(missing)} month(s) inside the history have no rows at all: {listed}. A gap "
            "in the feed isn't a month of zeros; reload it, or set validation.gaps to 'zero'"
        )
    return found


def validate(series: Series, month: str | None, drop_all_zero: bool = True) -> Validated:
    """`series` cut at its snapshot, with its gaps found and its all-zero series dropped."""
    snapshot = snapshot_of(series, month)
    frame = series.frame[series.frame[MONTH] <= snapshot]
    missing = gaps(series, snapshot)
    dropped: list[tuple[str, str]] = []
    if drop_all_zero:
        live = frame.loc[frame[VALUE] != 0, [GROUP, CATEGORY]].drop_duplicates()
        alive = set(zip(live[GROUP].astype(str), live[CATEGORY].astype(str), strict=True))
        every = frame[[GROUP, CATEGORY]].drop_duplicates()
        pairs = zip(every[GROUP].astype(str), every[CATEGORY].astype(str), strict=True)
        dropped = [pair for pair in pairs if pair not in alive]
        keys = pd.MultiIndex.from_frame(frame[[GROUP, CATEGORY]])
        frame = frame[~keys.isin(dropped)]
    cut = replace(series, frame=frame.reset_index(drop=True), snapshot=snapshot)
    return Validated(cut, missing, tuple(sorted(dropped)))
