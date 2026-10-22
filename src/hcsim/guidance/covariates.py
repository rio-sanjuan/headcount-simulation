"""Calendar covariates: what each month is, for the models that can read it.

Every month from the first of the history through the last of the horizon gets one row:

- `month_of_year` (1–12) and `quarter` (1–4);
- `days`, the days in the month, and `weekdays`, its Monday-to-Friday days, which bound how
  many working days a month's events can fall on;
- `sin1`, `cos1`, `sin2`, and `cos2`: the first two annual harmonics, the smooth form of the
  seasonal cycle that count models with harmonics fit;
- `future`, true for the horizon's months.

Covariates depend only on the calendar, so the horizon's are known when the forecast is made.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from hcsim.guidance.ingest import MONTH, Series

#: Every covariate column, in order.
CALENDAR_COLUMNS: Final = (
    "month_of_year",
    "quarter",
    "days",
    "weekdays",
    "sin1",
    "cos1",
    "sin2",
    "cos2",
)


def calendar(months: pd.PeriodIndex) -> pd.DataFrame:
    """One row of calendar covariates per month, indexed by month."""
    if months.freqstr != "M":
        raise ValueError(f"calendar covariates are monthly, not {months.freqstr!r}")
    month = months.month.to_numpy()
    starts = months.start_time.to_numpy().astype("datetime64[D]")
    ends = months.end_time.to_numpy().astype("datetime64[D]") + np.timedelta64(1, "D")
    angle = 2 * np.pi * (month - 1) / 12
    frame = pd.DataFrame(
        {
            "month_of_year": month,
            "quarter": months.quarter.to_numpy(),
            "days": months.days_in_month.to_numpy(),
            "weekdays": np.busday_count(starts, ends),
            "sin1": np.sin(angle),
            "cos1": np.cos(angle),
            "sin2": np.sin(2 * angle),
            "cos2": np.cos(2 * angle),
        },
        index=pd.PeriodIndex(months, name=MONTH),
    )
    return frame.astype(dict.fromkeys(CALENDAR_COLUMNS[:4], np.int64))


def covariates(series: Series, horizon: int) -> pd.DataFrame:
    """The calendar covariates for `series`' history and the `horizon` months after it."""
    if horizon < 0:
        raise ValueError(f"the horizon can't be negative, got {horizon}")
    first = series.frame[MONTH].min()
    months = pd.period_range(first, series.snapshot + horizon, freq="M")
    frame = calendar(months)
    frame["future"] = frame.index > series.snapshot
    return frame
