"""The world's compensation and equity: ranges, prices, awards, vests, and a reference ledger.

Every parameter is invented. The world's own grant policy (a value by level, with noise) differs
from the engine's gap-fill sizing on purpose, so the engine has something to estimate. The
reference ledger is a deliberately simple loop per tranche, independent of hcsim.financial, so
the engine's vectorized calculator can be scored against it.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from hcsim.synth.config import CompensationTruth, GrantTruth, PriceTruth


def compensation_ranges(
    families: tuple[str, ...],
    levels: tuple[int, ...],
    locations: tuple[str, ...],
    truth: CompensationTruth,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """A range for every family, level, and location, executive levels included: full
    coverage, no gaps. With the `missing_range_share` quirk on, a share of cells is dropped."""
    rows: list[dict[str, Any]] = []
    w = truth.range_half_width
    for family in families:
        for level in levels:
            for i, location in enumerate(locations):
                factor = truth.family_factor[family] * truth.location_factor(i)
                base = truth.base_midpoint_by_level[level] * factor
                target = base * (1 + truth.equity_share_by_level[level])
                rows.append(
                    {
                        "job_family": family,
                        "job_level": level,
                        "location": location,
                        "min_base": round(base * (1 - w)),
                        "max_base": round(base * (1 + w)),
                        "min_target": round(target * (1 - w)),
                        "max_target": round(target * (1 + w)),
                        "base_adj": truth.base_adjustment,
                    }
                )
    frame = pd.DataFrame(rows)
    if truth.missing_range_share > 0 and rng is not None:
        frame = frame[rng.random(len(frame)) >= truth.missing_range_share].reset_index(drop=True)
    return frame


def daily_prices(
    month_ends: list[pd.Timestamp], truth: PriceTruth, rng: np.random.Generator
) -> pd.DataFrame:
    """A close on every business day of the world, and each day's grant-date fair value: the
    trailing average close over `averaging_days` business days (fewer at the start)."""
    first = month_ends[0].to_period("M").start_time
    days = pd.bdate_range(first, month_ends[-1])
    n = truth.trading_days_per_month
    drift, sigma = truth.monthly_drift / n, truth.monthly_volatility / np.sqrt(n)
    shocks = rng.normal(drift - sigma**2 / 2, sigma, len(days) - 1)
    closes = truth.start_price * np.exp(np.r_[0.0, np.cumsum(shocks)])
    fair = pd.Series(closes).rolling(truth.averaging_days, min_periods=1).mean().to_numpy()
    return pd.DataFrame(
        {"price_date": days.date, "close": closes.round(4), "fair_value": fair.round(4)}
    )


def month_end_closes(daily: pd.DataFrame, month_ends: list[pd.Timestamp]) -> pd.DataFrame:
    """Each month's close: the close of its last business day."""
    month = pd.to_datetime(daily["price_date"]).dt.to_period("M")
    last = {str(k): float(v) for k, v in daily.groupby(month)["close"].last().items()}
    return pd.DataFrame(
        {
            "month_end": [d.date() for d in month_ends],
            "close": [last[str(d.to_period("M"))] for d in month_ends],
        }
    )


def truth_dict(comp: CompensationTruth, price: PriceTruth, grant: GrantTruth) -> dict[str, Any]:
    return {"compensation": asdict(comp), "prices": asdict(price), "grants": asdict(grant)}
