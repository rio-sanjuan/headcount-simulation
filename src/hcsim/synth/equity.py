"""The world's compensation and equity: ranges, prices, awards, vests, and a reference ledger.

Every parameter is invented. The world's own grant policy (a value by level, with noise) differs
from the engine's gap-fill sizing on purpose, so the engine has something to estimate. The
reference ledger is a deliberately simple loop per tranche, independent of hcsim.financial, so
the engine's vectorized calculator can be scored against it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from hcsim.synth.config import CompensationTruth, GrantTruth, PriceTruth

LEDGER_COLUMNS = [
    "award_id",
    "period",
    "gross",
    "net",
    "trueup_attrition",
    "trueup_vesting",
    "recognized",
]


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


def business_day_on_or_after(date: dt.date) -> dt.date:
    while date.weekday() >= 5:
        date += dt.timedelta(days=1)
    return date


def add_months(date: dt.date, months: int) -> dt.date:
    """The same day `months` later, clamped to the end of a shorter month."""
    period = pd.Period(date, freq="M") + months
    return dt.date(period.year, period.month, min(date.day, period.days_in_month))


def new_hire_awards(
    hires: pd.DataFrame,
    terms: pd.DataFrame,
    jobs: pd.DataFrame,
    daily: pd.DataFrame,
    truth: GrantTruth,
    world_end: dt.date,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One award per hire still employed on the award date, a business day, and its vest
    tranches. Its fair value is that day's. Awards after the world ends are never made."""
    level = dict(zip(jobs["job_code"], jobs["job_level"], strict=True))
    left = dict(zip(terms["employee_id"], terms["event_date"], strict=True))
    fair_values = dict(zip(daily["price_date"], daily["fair_value"], strict=True))
    awards: list[dict[str, Any]] = []
    vests: list[dict[str, Any]] = []
    for award_id, hire in enumerate(hires.to_dict("records"), start=900_001):
        lag = int(rng.integers(truth.award_lag_days[0], truth.award_lag_days[1] + 1))
        award_date = business_day_on_or_after(hire["event_date"] + dt.timedelta(days=lag))
        employee_id = int(hire["employee_id"])
        if award_date > world_end or (employee_id in left and left[employee_id] <= award_date):
            continue  # only employees still there on the award date receive it
        fair_value = float(fair_values[award_date])
        job_level = int(level[hire["job_code"]])
        value = truth.new_hire_value_by_level[job_level] * float(
            np.exp(rng.normal(0, truth.value_sigma))
        )
        shares = max(1, round(value / fair_value))
        awards.append(
            {
                "award_id": award_id,
                "employee_id": employee_id,
                "award_type": "new_hire",
                "award_date": award_date,
                "shares": shares,
                "fair_value": round(fair_value, 4),
                "cost_center": hire["cost_center"],
                "job_level": job_level,
            }
        )
        split = np.floor(np.array(truth.vest_shares) * shares).astype(int)
        split[-1] += shares - int(split.sum())  # the last tranche absorbs rounding
        vests.extend(
            {
                "award_id": award_id,
                "vest_number": n,
                "vest_date": add_months(award_date, months),
                "shares": int(s),
            }
            for n, (months, s) in enumerate(zip(truth.vest_months, split, strict=True), start=1)
        )
    return pd.DataFrame(awards), pd.DataFrame(vests)


def vest_status(
    vests: pd.DataFrame, awards: pd.DataFrame, terms: pd.DataFrame, world_end: dt.date
) -> pd.DataFrame:
    """Mark each tranche vested, forfeited (the holder left first), or still unvested."""
    left = dict(zip(terms["employee_id"], terms["event_date"], strict=True))
    holder = dict(zip(awards["award_id"], awards["employee_id"], strict=True))
    status: list[str] = []
    forfeit: list[dt.date | None] = []
    for tranche in vests.to_dict("records"):
        departed = left.get(holder[tranche["award_id"]])
        vest_date: dt.date = tranche["vest_date"]
        if departed is not None and departed < vest_date:
            status.append("forfeited")
            forfeit.append(departed)
        else:
            status.append("vested" if vest_date <= world_end else "unvested")
            forfeit.append(None)
    return vests.assign(
        status=status, forfeit_date=pd.Series(forfeit, index=vests.index, dtype=object)
    )


def reference_ledger(
    awards: pd.DataFrame, vests: pd.DataFrame, month_ends: list[pd.Timestamp]
) -> pd.DataFrame:
    """Monthly recognized expense per award, from a plain loop per tranche.

    A tranche expenses an equal share of its value each month from its award month through its
    vest month. A forfeited tranche accrues through its forfeiture month, which then reverses
    everything recognized. The world books no forfeiture estimate, so net equals gross. Only
    months inside the world's window are recorded.
    """
    index = {pd.Period(d, freq="M"): i for i, d in enumerate(month_ends)}
    fair_value = dict(zip(awards["award_id"], awards["fair_value"], strict=True))
    award_month = {
        a: pd.Period(d, freq="M")
        for a, d in zip(awards["award_id"], awards["award_date"], strict=True)
    }
    cells: dict[tuple[int, int], list[float]] = {}

    def add(award_id: int, month: pd.Period, gross: float, trueup: float) -> None:
        if month in index:
            cell = cells.setdefault((award_id, index[month]), [0.0, 0.0])
            cell[0] += gross
            cell[1] += trueup

    for tranche in vests.to_dict("records"):
        award_id = int(tranche["award_id"])
        value = int(tranche["shares"]) * float(fair_value[award_id])
        first = award_month[award_id]
        vest = pd.Period(tranche["vest_date"], freq="M")
        per_month = value / ((vest - first).n + 1)
        forfeited = tranche["status"] == "forfeited"
        last = pd.Period(tranche["forfeit_date"], freq="M") if forfeited else vest
        accrued = (last - first).n + 1
        for k in range(accrued):
            add(award_id, first + k, per_month, 0.0)
        if forfeited:
            add(award_id, last, 0.0, -per_month * accrued)

    return pd.DataFrame(
        [
            {
                "award_id": award_id,
                "period": month_ends[i].date(),
                "gross": g,
                "net": g,
                "trueup_attrition": t,
                "trueup_vesting": 0.0,
                "recognized": g + t,
            }
            for (award_id, i), (g, t) in sorted(cells.items())
        ],
        columns=LEDGER_COLUMNS,
    )


def truth_dict(comp: CompensationTruth, price: PriceTruth, grant: GrantTruth) -> dict[str, Any]:
    return {"compensation": asdict(comp), "prices": asdict(price), "grants": asdict(grant)}
