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


def reference_ledger(
    awards: pd.DataFrame, vests: pd.DataFrame, month_ends: list[pd.Timestamp]
) -> pd.DataFrame:
    """Monthly recognized expense per award, from a plain loop per award.

    A tranche expenses an equal share of its value each month from its award month through its
    vest month. A forfeited tranche accrues through its forfeiture month, which then reverses
    everything recognized. The world books no forfeiture estimate, so net equals gross. Only
    months inside the world's window are recorded.
    """
    first = pd.Period(month_ends[0], freq="M")
    window = len(month_ends)
    month_of = lambda d: (pd.DatetimeIndex(d).to_period("M") - first).map(lambda x: x.n).to_numpy()  # noqa: E731
    fair_value = dict(zip(awards["award_id"], awards["fair_value"], strict=True))
    award_month = dict(zip(awards["award_id"], month_of(awards["award_date"]), strict=True))
    vests = vests.assign(
        vest_month=month_of(vests["vest_date"]),
        forfeit_month=np.where(
            vests["forfeit_date"].isna(),
            -1,
            month_of(vests["forfeit_date"].fillna(pd.Timestamp("2000-01-01"))),
        ),
    )
    ids: list[NDArray[np.int64]] = []
    months: list[NDArray[np.int64]] = []
    grosses: list[NDArray[np.float64]] = []
    reversals: list[NDArray[np.float64]] = []
    for award_id, tranches in vests.groupby("award_id", sort=True):
        a = int(award_month[award_id])
        value = tranches["shares"].to_numpy() * float(fair_value[award_id])
        vest = tranches["vest_month"].to_numpy()
        forfeit = tranches["forfeit_month"].to_numpy()
        lost = forfeit >= 0
        last = np.where(lost, forfeit, vest)
        per_month = value / (vest - a + 1)
        span = np.arange(a, int(last.max()) + 1)
        accrues = span[None, :] <= last[:, None]
        gross = (accrues * per_month[:, None]).sum(axis=0)
        reversal = np.zeros(len(span))
        np.add.at(reversal, last[lost] - a, -per_month[lost] * (last[lost] - a + 1))
        inside = (span >= 0) & (span < window)
        ids.append(np.full(int(inside.sum()), award_id, dtype=np.int64))
        months.append(span[inside])
        grosses.append(gross[inside])
        reversals.append(reversal[inside])
    frame = pd.DataFrame(
        {
            "award_id": np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64),
            "month": np.concatenate(months) if months else np.zeros(0, dtype=np.int64),
            "gross": np.concatenate(grosses) if grosses else np.zeros(0),
            "trueup_attrition": np.concatenate(reversals) if reversals else np.zeros(0),
        }
    )
    return pd.DataFrame(
        {
            "award_id": frame["award_id"],
            "period": np.array([d.date() for d in month_ends], dtype="datetime64[D]")[
                frame["month"].to_numpy()
            ].astype("datetime64[s]"),
            "gross": frame["gross"],
            "net": frame["gross"],
            "trueup_attrition": frame["trueup_attrition"],
            "trueup_vesting": 0.0,
            "recognized": frame["gross"] + frame["trueup_attrition"],
        },
        columns=LEDGER_COLUMNS,
    )
