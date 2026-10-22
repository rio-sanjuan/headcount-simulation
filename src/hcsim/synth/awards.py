"""The world's equity awards: triggers, sizing, vesting, forfeiture, and a reference ledger.

Every award type in the compensation policy (see `GrantTruth`) has a trigger:

- `new_hire`: each hire.
- `annual`: everyone with enough tenure in the stock before the annual month, on the month's
  first business day, scaled by their rating.
- `promotion`: each promotion, at the new level.
- `transfer`: a share of transfers.
- `retention` and `off_cycle`: monthly chances, in the stock before the month.

An award falls on the first business day after its trigger and lag, if the person is still
employed then. Its value is the type's value by level, times its noise and any rating factor;
its shares are that value over the day's fair value. Its tranches vest on the type's months.
A tranche forfeits when its holder leaves (a departure, a termination program, or a conversion
out) before it vests.

The reference ledger is deliberately simple and independent of `hcsim.financial`, so the
engine's calculator can be scored against it: a loop per award, straight-line per tranche.
"""

from __future__ import annotations

import datetime as dt
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.config import WorldConfig

LEDGER_COLUMNS: Final = [
    "award_id",
    "period",
    "gross",
    "net",
    "trueup_attrition",
    "trueup_rate",
    "trueup_vesting",
    "recognized",
]
EXITS: Final = ("TERM", "RIF", "CONV_OUT")
FIRST_AWARD_ID: Final = 900_001


def add_months(
    dates: NDArray[np.datetime64], months: NDArray[np.integer]
) -> NDArray[np.datetime64]:
    """The same day `months` later, clamped to the end of a shorter month."""
    day = dates.astype("datetime64[D]")
    month = day.astype("datetime64[M]")
    offset = (day - month.astype("datetime64[D]")).astype(np.int64)
    target = month + np.asarray(months).astype("timedelta64[M]")
    length = (
        (target + np.timedelta64(1, "M")).astype("datetime64[D]") - target.astype("datetime64[D]")
    ).astype(np.int64)
    result: NDArray[np.datetime64] = target.astype("datetime64[D]") + np.minimum(
        offset, length - 1
    ).astype("timedelta64[D]")
    return result


def _world_end(config: WorldConfig) -> np.datetime64:
    last = pd.Period(config.start_month, freq="M") + config.months - 1
    return np.datetime64(last.end_time.date(), "D")


def _business_day(dates: NDArray[np.datetime64]) -> NDArray[np.datetime64]:
    result: NDArray[np.datetime64] = np.busday_offset(
        dates.astype("datetime64[D]"), 0, roll="forward"
    )
    return result


class _History:
    """What the triggers read from the world's tables, in plain columns."""

    def __init__(self, txn: pd.DataFrame, ratings: pd.DataFrame, jobs: pd.DataFrame) -> None:
        level = jobs.set_index("job_code")["job_level"]
        # Labels stay categorical: copying millions of rows as Python strings would cost
        # gigabytes at the larger presets.
        self.txn = pd.DataFrame(
            {
                "employee_id": txn["employee_id"].to_numpy(),
                "trans_type": txn["trans_type"],
                "date": txn["event_date"].to_numpy().astype("datetime64[D]"),
                "cost_center": txn["cost_center"],
                "company": txn["company"],
                "job_level": txn["job_code"].map(level).to_numpy(np.int64),
                "account": txn["account"],
                "hire_date": txn["hire_date"].to_numpy().astype("datetime64[D]"),
            }
        )
        exits = self.txn[self.txn["trans_type"].isin(EXITS)]
        self.exits = exits[["employee_id", "date"]].rename(columns={"date": "exit_date"})
        self.ratings = ratings.sort_values("cycle_date")

    def events(self, code: str) -> pd.DataFrame:
        return self.txn[self.txn["trans_type"] == code]

    def stock(self, month_end: np.datetime64, regular: str) -> pd.DataFrame:
        rows = self.txn[(self.txn["trans_type"] == "ACTIVE") & (self.txn["date"] == month_end)]
        return rows[rows["account"] == regular]

    def rating_on(self, frame: pd.DataFrame, when: str) -> NDArray[np.int64]:
        """Each row's latest rating code on its date, or -1 if unrated."""
        if frame.empty:
            return np.zeros(0, dtype=np.int64)
        keyed = frame[["employee_id", when]].reset_index().sort_values(when)
        keyed[when] = pd.to_datetime(keyed[when])
        found = pd.merge_asof(
            keyed,
            self.ratings.assign(cycle_date=pd.to_datetime(self.ratings["cycle_date"])),
            left_on=when,
            right_on="cycle_date",
            by="employee_id",
        ).set_index("index")
        result: NDArray[np.int64] = (
            found["rating"].reindex(frame.index).fillna(-1).to_numpy(np.int64)
        )
        return result

    def next_exit(self, frame: pd.DataFrame, when: str) -> NDArray[np.datetime64]:
        """Each row's first exit on or after its date, or NaT."""
        if frame.empty:
            return np.zeros(0, dtype="datetime64[D]")
        keyed = frame[["employee_id", when]].reset_index()
        keyed[when] = pd.to_datetime(keyed[when])
        exits = self.exits.assign(exit_date=pd.to_datetime(self.exits["exit_date"])).sort_values(
            "exit_date"
        )
        found = pd.merge_asof(
            keyed.sort_values(when),
            exits,
            left_on=when,
            right_on="exit_date",
            by="employee_id",
            direction="forward",
        ).set_index("index")
        result: NDArray[np.datetime64] = (
            found["exit_date"].reindex(frame.index).to_numpy().astype("datetime64[D]")
        )
        return result


def _triggers(config: WorldConfig, history: _History, rng: np.random.Generator) -> pd.DataFrame:
    """Every candidate award: who, which type, when its trigger fell, and its sizing inputs."""
    g = config.grants
    regular = next(name for name, group in config.accounts if group == "regular")
    columns = ["employee_id", "cost_center", "company", "job_level"]
    parts = []

    def add(
        rows: pd.DataFrame,
        award_type: str,
        dates: NDArray[np.datetime64],
        factor: NDArray[np.float64] | float = 1.0,
    ) -> None:
        if len(rows):
            parts.append(
                rows[columns]
                .assign(
                    award_type=award_type,
                    trigger=np.asarray(dates).astype("datetime64[s]"),
                    factor=factor,
                )
                .reset_index(drop=True)
            )

    hires = history.events("HIRE")
    add(hires, "new_hire", hires["date"].to_numpy())
    promotions = history.events("PROMO")
    add(promotions, "promotion", promotions["date"].to_numpy())
    transfers = history.events("XFER_IN")
    picked = transfers[rng.random(len(transfers)) < g.transfer_award_probability]
    add(picked, "transfer", picked["date"].to_numpy())

    first = pd.Period(config.start_month, freq="M")
    rating_factor = np.asarray(g.annual_rating_factor, dtype=np.float64)
    top = len(g.annual_rating_factor) - 2  # the top two ratings
    for k in range(1, config.months):
        month = first + k
        before = np.datetime64((month - 1).end_time.date(), "D")
        stock = history.stock(before, regular)
        start = np.datetime64(month.start_time.date(), "D")
        if month.month == g.annual_month:
            tenure = (start - stock["hire_date"].to_numpy()).astype(np.int64) / 30.4
            eligible = stock[tenure >= g.annual_min_tenure_months].assign(trigger=start)
            code = history.rating_on(eligible, "trigger")
            factor = np.where(code >= 0, rating_factor[np.maximum(code, 0)], 0.0)
            chosen = eligible[factor > 0]
            add(chosen, "annual", np.full(len(chosen), start), factor[factor > 0])
        rated = stock.assign(trigger=start)
        code = history.rating_on(rated, "trigger")
        retained = rated[(code >= top) & (rng.random(len(rated)) < g.retention_monthly_rate)]
        add(retained, "retention", np.full(len(retained), start))
        off = stock[rng.random(len(stock)) < g.off_cycle_monthly_rate]
        add(off, "off_cycle", np.full(len(off), start))
    return pd.concat(parts, ignore_index=True)


def schedules(config: WorldConfig) -> dict[str, tuple[tuple[int, ...], tuple[float, ...]]]:
    """Every vest schedule: each award type's own, and the executive one."""
    g = config.grants
    found = {
        name: (g.award_type(name).vest_months, g.award_type(name).vest_shares) for name in g.TYPES
    }
    found["executive"] = (g.executive_vest_months, g.executive_vest_shares)
    return found


def schedule_frame(config: WorldConfig) -> pd.DataFrame:
    """`dim_vest_schedule`: each schedule's tranches, in months after the award."""
    rows = [
        {"schedule": name, "vest_number": k + 1, "months_after_award": m, "share": share}
        for name, (months, shares) in schedules(config).items()
        for k, (m, share) in enumerate(zip(months, shares, strict=True))
    ]
    return pd.DataFrame(rows)


def _appraisals(config: WorldConfig, monthly: pd.DataFrame) -> pd.Series:
    """The offline plans' appraised value per share, set at each quarter-end."""
    ends = pd.to_datetime(monthly["month_end"])
    quarter = ends.dt.month.isin([3, 6, 9, 12])
    value = monthly["close"].to_numpy() * config.grants.offline_appraisal_discount
    return pd.Series(
        value[quarter.to_numpy()], index=ends[quarter].to_numpy().astype("datetime64[D]")
    )


def _appraisal_on(
    appraisals: pd.Series, dates: NDArray[np.datetime64], opening: float
) -> NDArray[np.float64]:
    """The latest appraisal on or before each date, or the opening one before the first."""
    index = appraisals.index.to_numpy().astype("datetime64[D]")
    position = np.searchsorted(index, dates.astype("datetime64[D]"), side="right") - 1
    values = appraisals.to_numpy()
    result: NDArray[np.float64] = np.where(position >= 0, values[np.maximum(position, 0)], opening)
    return result


def make_awards(
    config: WorldConfig,
    txn: pd.DataFrame,
    ratings: pd.DataFrame,
    jobs: pd.DataFrame,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """The equity tables: `equity_award`, `equity_vest`, `equity_valuation`, `equity_holder`,
    and `dim_vest_schedule`."""
    g = config.grants
    world_end = _world_end(config)
    history = _History(txn, ratings, jobs)
    candidates = _triggers(config, history, rng)
    lag = np.zeros(len(candidates), dtype=np.int64)
    value = np.zeros(len(candidates))
    for name in g.TYPES:
        mine = (candidates["award_type"] == name).to_numpy()
        t = g.award_type(name)
        lag[mine] = rng.integers(t.lag_days[0], t.lag_days[1] + 1, int(mine.sum()))
        base = candidates.loc[mine, "job_level"].map(t.value_by_level).to_numpy(np.float64)
        value[mine] = base * np.exp(rng.normal(0, t.value_sigma, int(mine.sum())))
    trigger = candidates["trigger"].to_numpy().astype("datetime64[D]")
    award_date = _business_day(trigger + lag.astype("timedelta64[D]"))
    candidates = candidates.assign(
        award_date=award_date.astype("datetime64[s]"),
        value=value * candidates["factor"].to_numpy(),
    )
    left = history.next_exit(candidates, "trigger")
    kept = (award_date <= world_end) & (np.isnat(left) | (left > award_date))
    awards = (
        candidates[kept]
        .sort_values(["award_date", "employee_id", "award_type"], kind="stable")
        .reset_index(drop=True)
    )
    # Plans and schedules: offline subsidiaries grant under their own plan, and executives'
    # new-hire and annual awards vest on the executive schedule.
    offline = set(config.companies[len(config.companies) - config.offline_subsidiaries :])
    plan = np.where(awards["company"].astype(str).isin(offline), "offline", "primary")
    executive = (awards["job_level"] >= config.executive_min_level) & awards["award_type"].isin(
        ["new_hire", "annual"]
    )
    schedule = np.where(executive, "executive", awards["award_type"])
    fair = daily.assign(
        price_date=pd.to_datetime(daily["price_date"]).to_numpy().astype("datetime64[D]")
    ).set_index("price_date")["fair_value"]
    dates = awards["award_date"].to_numpy().astype("datetime64[D]")
    grant_value = fair.reindex(dates).to_numpy()
    appraisals = _appraisals(config, monthly)
    opening = float(monthly["close"].iloc[0]) * g.offline_appraisal_discount
    grant_value = np.where(
        plan == "offline", _appraisal_on(appraisals, dates, opening), grant_value
    )
    shares = np.maximum(1, np.rint(awards["value"].to_numpy() / grant_value)).astype(np.int64)
    awards = awards.assign(
        award_id=FIRST_AWARD_ID + np.arange(len(awards)),
        shares=shares,
        fair_value=grant_value.round(4),
        plan=plan,
        schedule=schedule,
    )
    vests = _vests(config, awards, history)
    settles = _settle_dates(awards, vests, history, world_end)
    return {
        "equity_award": _frame(awards),
        "equity_vest": vests,
        "equity_valuation": _valuations(awards, settles, appraisals, world_end),
        "equity_holder": _holders(awards, settles, history),
        "dim_vest_schedule": schedule_frame(config),
    }


def _settle_dates(
    awards: pd.DataFrame, vests: pd.DataFrame, history: _History, world_end: np.datetime64
) -> NDArray[np.datetime64]:
    """When each award stops being held: its last vest, its holder's exit, or the world's end,
    whichever comes first."""
    last_vest = vests.groupby("award_id")["vest_date"].max().reindex(awards["award_id"]).to_numpy()
    left = history.next_exit(awards, "award_date")
    end = np.minimum(last_vest.astype("datetime64[D]"), world_end)
    result: NDArray[np.datetime64] = np.where(~np.isnat(left) & (left < end), left, end)
    return result


def _valuations(
    awards: pd.DataFrame,
    settles: NDArray[np.datetime64],
    appraisals: pd.Series,
    world_end: np.datetime64,
) -> pd.DataFrame:
    """`equity_valuation`: each award's grant-date value per share, and for offline plans,
    every quarter-end appraisal while the award is held."""
    grant = pd.DataFrame(
        {
            "award_id": awards["award_id"],
            "valuation_date": awards["award_date"].to_numpy().astype("datetime64[s]"),
            "value_per_share": awards["fair_value"],
            "basis": np.where(awards["plan"] == "offline", "appraisal", "grant_date_fair_value"),
        }
    )
    offline = awards["plan"].to_numpy() == "offline"
    dates = appraisals.index.to_numpy().astype("datetime64[D]")
    start = awards["award_date"].to_numpy().astype("datetime64[D]")[offline]
    later = (dates[None, :] > start[:, None]) & (dates[None, :] <= settles[offline][:, None])
    award, quarter = np.nonzero(later)
    remeasured = pd.DataFrame(
        {
            "award_id": awards["award_id"].to_numpy()[offline][award],
            "valuation_date": dates[quarter].astype("datetime64[s]"),
            "value_per_share": appraisals.to_numpy()[quarter].round(4),
            "basis": "appraisal",
        }
    )
    frames = [f for f in (grant, remeasured) if len(f)]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["award_id", "valuation_date"], ignore_index=True
    )


def _holders(
    awards: pd.DataFrame, settles: NDArray[np.datetime64], history: _History
) -> pd.DataFrame:
    """`equity_holder`: each award's holder link over time, a new interval whenever the holder
    moves cost center or company, until the award settles."""
    moves = history.txn[history.txn["trans_type"].isin(["XFER_IN", "REORG_IN", "CONV_IN"])][
        ["employee_id", "date", "cost_center", "company"]
    ]
    base = pd.DataFrame(
        {
            "award_id": awards["award_id"].to_numpy(),
            "employee_id": awards["employee_id"].to_numpy(),
            "valid_from": awards["award_date"].to_numpy().astype("datetime64[D]"),
            "settles": settles,
            "cost_center": awards["cost_center"].to_numpy(),
            "company": awards["company"].to_numpy(),
        }
    )
    joined = base[["award_id", "employee_id", "valid_from", "settles"]].merge(
        moves, on="employee_id"
    )
    within = (joined["date"] > joined["valid_from"]) & (joined["date"] < joined["settles"])
    changes = joined[within].assign(valid_from=lambda f: f["date"])[base.columns]
    links = pd.concat([base, changes], ignore_index=True).sort_values(
        ["award_id", "valid_from"], ignore_index=True
    )
    following = links.groupby("award_id")["valid_from"].shift(-1)
    valid_to = following.fillna(pd.Series(links["settles"].to_numpy(), index=links.index))
    return pd.DataFrame(
        {
            "award_id": links["award_id"],
            "employee_id": links["employee_id"],
            "cost_center": links["cost_center"].astype(str),
            "company": links["company"].astype(str),
            "valid_from": links["valid_from"].to_numpy().astype("datetime64[s]"),
            "valid_to": pd.to_datetime(valid_to).to_numpy().astype("datetime64[s]"),
        }
    )


def _frame(awards: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "award_id": awards["award_id"],
            "employee_id": awards["employee_id"],
            "award_type": awards["award_type"],
            "award_date": awards["award_date"].to_numpy().astype("datetime64[s]"),
            "shares": awards["shares"],
            "fair_value": awards["fair_value"],
            "cost_center": awards["cost_center"],
            "company": awards["company"],
            "job_level": awards["job_level"].astype(np.int64),
            "plan": awards["plan"],
            "schedule": awards["schedule"],
        }
    )


def _vests(config: WorldConfig, awards: pd.DataFrame, history: _History) -> pd.DataFrame:
    """One row per tranche, with its status: vested, forfeited (the holder left first), or
    still unvested at the world's end."""
    parts = []
    for name, (vest_months, vest_shares) in schedules(config).items():
        mine = awards[awards["schedule"] == name]
        k = len(vest_months)
        shares = mine["shares"].to_numpy()
        split = np.floor(np.outer(shares, vest_shares)).astype(np.int64)
        split[:, -1] += shares - split.sum(axis=1)  # the last tranche absorbs rounding
        if mine.empty:
            continue
        dates = np.repeat(mine["award_date"].to_numpy().astype("datetime64[D]"), k)
        vest_dates = add_months(dates, np.tile(np.asarray(vest_months), len(mine)))
        parts.append(
            pd.DataFrame(
                {
                    "award_id": np.repeat(mine["award_id"].to_numpy(), k),
                    "employee_id": np.repeat(mine["employee_id"].to_numpy(), k),
                    "award_date": dates.astype("datetime64[s]"),
                    "vest_number": np.tile(np.arange(1, k + 1), len(mine)),
                    "vest_date": vest_dates.astype("datetime64[s]"),
                    "shares": split.ravel(),
                }
            )
        )
    vests = pd.concat(parts, ignore_index=True).sort_values(
        ["award_id", "vest_number"], ignore_index=True
    )
    world_end = _world_end(config)
    left = history.next_exit(vests, "award_date")
    vest_date = vests["vest_date"].to_numpy().astype("datetime64[D]")
    forfeited = ~np.isnat(left) & (left < vest_date)
    status = np.where(
        forfeited, "forfeited", np.where(vest_date <= world_end, "vested", "unvested")
    )
    return pd.DataFrame(
        {
            "award_id": vests["award_id"],
            "vest_number": vests["vest_number"],
            "vest_date": vest_date.astype("datetime64[s]"),
            "shares": vests["shares"],
            "status": status,
            "forfeit_date": np.where(forfeited, left, np.datetime64("NaT", "D")).astype(
                "datetime64[s]"
            ),
        }
    )


def forfeiture_estimates(
    config: WorldConfig, awards: pd.DataFrame, vests: pd.DataFrame
) -> pd.DataFrame:
    """`equity_forfeiture_estimate`: each cohort's estimate, revised every January.

    A cohort is an award year. Its estimate is the prior until the first January after the
    cohort starts. Each revision blends in the share of the cohort's shares forfeited so far,
    with a weight that rises to one over the credibility years, capped at the cap.
    """
    g = config.grants
    cohort = pd.to_datetime(awards.set_index("award_id")["award_date"]).dt.year
    tranche = vests.assign(cohort=vests["award_id"].map(cohort))
    first, last = int(cohort.min()), config_last_year(config)
    rows = []
    for c in range(first, last + 1):
        mine = tranche[tranche["cohort"] == c]
        total = float(mine["shares"].sum())
        rows.append(
            {"cohort": c, "effective_date": dt.date(c, 1, 1), "estimate": g.forfeiture_prior}
        )
        for year in range(c + 1, last + 1):
            revision = pd.Timestamp(year, 1, 1)
            lost = float(mine.loc[pd.to_datetime(mine["forfeit_date"]) < revision, "shares"].sum())
            observed = lost / total if total else 0.0
            weight = min(1.0, (year - c) / g.forfeiture_credibility_years)
            estimate = min(g.forfeiture_cap, (1 - weight) * g.forfeiture_prior + weight * observed)
            rows.append(
                {"cohort": c, "effective_date": revision.date(), "estimate": round(estimate, 6)}
            )
    return pd.DataFrame(rows)


def config_last_year(config: WorldConfig) -> int:
    return int((pd.Period(config.start_month, freq="M") + config.months - 1).year)


def reference_ledger(
    config: WorldConfig,
    awards: pd.DataFrame,
    vests: pd.DataFrame,
    estimates: pd.DataFrame,
    month_ends: list[pd.Timestamp],
) -> pd.DataFrame:
    """Monthly ASC 718 expense per award, from a plain loop per award.

    For each tranche, worth its shares times the award's grant value per share:

    - gross accrues straight-line from the award month through the vest month;
    - net is gross times one less the cohort's forfeiture estimate in effect that month;
    - a January revision books a rate true-up: the gross accrued so far, times the drop in the
      estimate, for every tranche still accruing;
    - a forfeiture reverses everything recognized for the tranche, as the attrition true-up;
    - a vest books the vesting true-up: the tranche's value less everything recognized, so
      cumulative net plus true-ups equals the vested value for every settled tranche.

    Only months inside the world's window are recorded. It stays independent of the engine's
    calculator, so that calculator can be scored against it.
    """
    first = pd.Period(month_ends[0], freq="M")
    window = len(month_ends)

    def month_of(values: pd.Series) -> NDArray[np.int64]:
        months = pd.DatetimeIndex(pd.to_datetime(values)).to_period("M")
        result: NDArray[np.int64] = np.array([(m - first).n for m in months], dtype=np.int64)
        return result

    award_month = dict(zip(awards["award_id"], month_of(awards["award_date"]), strict=True))
    grant_value = dict(zip(awards["award_id"], awards["fair_value"], strict=True))
    years = pd.to_datetime(awards["award_date"]).dt.year
    cohort_of = {int(a): int(y) for a, y in zip(awards["award_id"], years, strict=True)}
    # The estimate in effect for each cohort and month index.
    horizon = int(month_of(vests["vest_date"]).max(initial=0)) + 1
    rate: dict[int, NDArray[np.float64]] = {}
    for c, rows in estimates.groupby("cohort"):
        series = np.full(max(horizon, window) + 1, float(rows["estimate"].iloc[0]))
        for effective, estimate in zip(
            month_of(rows["effective_date"]), rows["estimate"], strict=True
        ):
            series[max(int(effective), 0) :] = float(estimate)
        rate[int(str(c))] = series
    forfeit = vests["forfeit_date"]
    vests = vests.assign(
        vest_month=month_of(vests["vest_date"]),
        forfeit_month=np.where(
            forfeit.isna(), -1, month_of(forfeit.fillna(pd.Timestamp(month_ends[0])))
        ),
    )
    parts: dict[str, list[NDArray[np.float64]]] = {
        k: [] for k in ("gross", "net", "trueup_attrition", "trueup_rate", "trueup_vesting")
    }
    ids: list[NDArray[np.int64]] = []
    months: list[NDArray[np.int64]] = []
    for award_id, tranches in vests.groupby("award_id", sort=True):
        a = int(award_month[award_id])
        value = tranches["shares"].to_numpy() * float(grant_value[award_id])
        vest = tranches["vest_month"].to_numpy()
        lost_at = tranches["forfeit_month"].to_numpy()
        lost = lost_at >= 0
        last = np.where(lost, lost_at, vest)
        span = np.arange(a, int(max(last.max(), vest.max())) + 1)
        e = rate[cohort_of[int(str(award_id))]][np.clip(span, 0, None)]
        accrues = (span[None, :] <= last[:, None]).astype(np.float64)
        per_month = value / (vest - a + 1)
        gross = accrues * per_month[:, None]
        net = gross * (1 - e)[None, :]
        cum_gross_before = np.cumsum(gross, axis=1) - gross
        drop = np.r_[0.0, e[:-1] - e[1:]]  # the estimate's fall at each month
        rate_up = cum_gross_before * drop[None, :] * accrues
        booked = np.cumsum(net + rate_up, axis=1)
        column = last - a
        at = np.arange(len(value))
        attrition = np.zeros_like(gross)
        attrition[at[lost], column[lost]] = -booked[at[lost], column[lost]]
        vesting = np.zeros_like(gross)
        vesting[at[~lost], column[~lost]] = value[~lost] - booked[at[~lost], column[~lost]]
        inside = (span >= 0) & (span < window)
        ids.append(np.full(int(inside.sum()), award_id, dtype=np.int64))
        months.append(span[inside])
        for name, matrix in (
            ("gross", gross),
            ("net", net),
            ("trueup_attrition", attrition),
            ("trueup_rate", rate_up),
            ("trueup_vesting", vesting),
        ):
            parts[name].append(matrix.sum(axis=0)[inside])
    if not ids:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    period = np.array([d.date() for d in month_ends], dtype="datetime64[D]")
    frame = pd.DataFrame(
        {
            "award_id": np.concatenate(ids),
            "period": period[np.concatenate(months)].astype("datetime64[s]"),
            **{name: np.concatenate(values) for name, values in parts.items()},
        }
    )
    frame["recognized"] = (
        frame["net"] + frame["trueup_attrition"] + frame["trueup_rate"] + frame["trueup_vesting"]
    )
    return frame[LEDGER_COLUMNS]
