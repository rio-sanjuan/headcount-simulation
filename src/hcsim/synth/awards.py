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
from collections.abc import Iterator
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.columns import categorical
from hcsim.synth.config import GrantTruth, WorldConfig

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
    """What the triggers read from the world's tables, in plain columns.

    Only the events are copied: the month-end stocks, nearly every row of the transactions,
    are read from the table itself through an index by month-end, since a copy of them would
    cost gigabytes at the larger presets.
    """

    #: The events the triggers, forfeitures, and holder links read.
    EVENTS: Final = ("HIRE", "PROMO", "XFER_IN", "REORG_IN", "CONV_IN", *EXITS)

    def __init__(self, txn: pd.DataFrame, ratings: pd.DataFrame, jobs: pd.DataFrame) -> None:
        self._level = jobs.set_index("job_code")["job_level"]
        self._source = txn
        codes = txn["trans_type"]
        self.txn = self._plain(txn[codes.isin(self.EVENTS).to_numpy()])
        exits = self.txn[self.txn["trans_type"].isin(EXITS)]
        self.exits = exits[["employee_id", "date"]].rename(columns={"date": "exit_date"})
        self.ratings = ratings.sort_values("cycle_date")
        # Stock rows grouped by month-end, each group in the table's own order.
        active = np.flatnonzero((codes == "ACTIVE").to_numpy())
        dates = txn["event_date"].to_numpy()[active]
        order = np.argsort(dates, kind="stable")
        self._stock_rows = active[order]
        days, first = np.unique(dates[order], return_index=True)
        self._stock_span = dict(
            zip(
                days.astype("datetime64[D]").tolist(),
                zip(first.tolist(), [*first[1:].tolist(), len(order)], strict=True),
                strict=True,
            )
        )

    def _plain(self, rows: pd.DataFrame) -> pd.DataFrame:
        # Labels stay categorical: copying millions of rows as Python strings would cost
        # gigabytes at the larger presets.
        return pd.DataFrame(
            {
                "employee_id": rows["employee_id"].to_numpy(),
                "trans_type": rows["trans_type"],
                "date": rows["event_date"].to_numpy().astype("datetime64[D]"),
                "cost_center": rows["cost_center"],
                "company": rows["company"],
                "job_level": rows["job_code"].map(self._level).to_numpy(np.int64),
                "account": rows["account"],
                "hire_date": rows["hire_date"].to_numpy().astype("datetime64[D]"),
            },
            index=rows.index,
        )

    def events(self, code: str) -> pd.DataFrame:
        return self.txn[self.txn["trans_type"] == code]

    def stock(self, month_end: np.datetime64, regular: str) -> pd.DataFrame:
        lo, hi = self._stock_span.get(month_end.astype("datetime64[D]").item(), (0, 0))
        rows = self._plain(self._source.iloc[self._stock_rows[lo:hi]])
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
    # A holder's next exit is the same for every tranche of an award: find it once per award.
    left = history.next_exit(awards, "award_date")
    vests = _vests(config, awards, left)
    settles = _settle_dates(awards, vests, left, world_end)
    return {
        "equity_award": _frame(awards),
        "equity_vest": vests,
        "equity_valuation": _valuations(awards, settles, appraisals, world_end),
        "equity_holder": _holders(awards, settles, history),
        "dim_vest_schedule": schedule_frame(config),
    }


def _settle_dates(
    awards: pd.DataFrame,
    vests: pd.DataFrame,
    left: NDArray[np.datetime64],
    world_end: np.datetime64,
) -> NDArray[np.datetime64]:
    """When each award stops being held: its last vest, its holder's exit (`left`), or the
    world's end, whichever comes first."""
    last_vest = vests.groupby("award_id")["vest_date"].max().reindex(awards["award_id"]).to_numpy()
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
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["award_id", "valuation_date"], ignore_index=True)
        .astype({"basis": pd.CategoricalDtype(["grant_date_fair_value", "appraisal"])})
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
            "cost_center": links["cost_center"].astype("category"),
            "company": links["company"].astype("category"),
            "valid_from": links["valid_from"].to_numpy().astype("datetime64[s]"),
            "valid_to": pd.to_datetime(valid_to).to_numpy().astype("datetime64[s]"),
        }
    )


def _frame(awards: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "award_id": awards["award_id"],
            "employee_id": awards["employee_id"],
            "award_type": pd.Categorical(awards["award_type"], categories=list(GrantTruth.TYPES)),
            "award_date": awards["award_date"].to_numpy().astype("datetime64[s]"),
            "shares": awards["shares"],
            "fair_value": awards["fair_value"],
            "cost_center": awards["cost_center"],
            "company": awards["company"],
            "job_level": awards["job_level"].astype(np.int64),
            "plan": pd.Categorical(awards["plan"], categories=["primary", "offline"]),
            "schedule": pd.Categorical(
                awards["schedule"], categories=[*GrantTruth.TYPES, "executive"]
            ),
        }
    )


#: Tranches whose vest dates are computed at a time, to bound the working arrays.
_VEST_BLOCK: Final = 1_000_000


def _vests(config: WorldConfig, awards: pd.DataFrame, left: NDArray[np.datetime64]) -> pd.DataFrame:
    """One row per tranche, with its status: vested, forfeited (the holder left first), or
    still unvested at the world's end. `left` is each award's holder's next exit, or NaT.

    Tranches are built in award order straight from each schedule's table of months and
    shares, so there's nothing to sort, and no array outlives its use: at scale, the working
    arrays would otherwise cost several times the finished table."""
    plans = schedules(config)
    names = list(plans)
    widest = max(len(months) for months, _ in plans.values())
    months_of = np.zeros((len(names), widest), dtype=np.int64)
    share_of = np.zeros((len(names), widest), dtype=np.float64)
    for row, (months, shares) in enumerate(plans.values()):
        months_of[row, : len(months)] = months
        share_of[row, : len(shares)] = shares
    count = np.array([len(months) for months, _ in plans.values()], dtype=np.int64)
    code = pd.Categorical(awards["schedule"], categories=names).codes
    k = count[code]  # tranches per award
    first = np.cumsum(k) - k  # each award's first tranche
    position = np.repeat(np.arange(len(awards), dtype=np.int64), k)
    number = np.arange(len(position), dtype=np.int64) - first[position] + 1
    plan = code[position]
    award_shares = awards["shares"].to_numpy(np.int64)
    split = np.floor(award_shares[position] * share_of[plan, number - 1]).astype(np.int64)
    if len(split):  # the last tranche absorbs rounding
        split[first + k - 1] += award_shares - np.add.reduceat(split, first)
    award_dates = awards["award_date"].to_numpy().astype("datetime64[D]")
    vest_date = np.empty(len(position), dtype="datetime64[D]")
    for lo in range(0, len(position), _VEST_BLOCK):
        hi = lo + _VEST_BLOCK
        vest_date[lo:hi] = add_months(
            award_dates[position[lo:hi]], months_of[plan[lo:hi], number[lo:hi] - 1]
        )
    del plan
    world_end = _world_end(config)
    exit_date = left[position]
    forfeited = ~np.isnat(exit_date) & (exit_date < vest_date)
    status = categorical(
        np.where(forfeited, 2, np.where(vest_date <= world_end, 0, 1)).astype(np.int8),
        ["vested", "unvested", "forfeited"],
    )
    exit_date[~forfeited] = np.datetime64("NaT", "D")
    award_id = awards["award_id"].to_numpy(np.int64)[position]
    del position, forfeited
    return pd.DataFrame(
        {
            "award_id": award_id,
            "vest_number": number,
            "vest_date": vest_date.astype("datetime64[s]"),
            "shares": split,
            "status": status,
            "forfeit_date": exit_date.astype("datetime64[s]"),
        },
        copy=False,
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
    # Plain arrays: a copy of the tranche table per cohort would cost gigabytes at scale.
    award_ids = awards["award_id"].to_numpy(np.int64)
    years = pd.to_datetime(awards["award_date"]).dt.year.to_numpy(np.int64)
    order = np.argsort(award_ids, kind="stable")
    cohort = years[order][np.searchsorted(award_ids[order], vests["award_id"].to_numpy(np.int64))]
    shares = vests["shares"].to_numpy(np.int64)
    forfeit = vests["forfeit_date"].to_numpy().astype("datetime64[D]")
    first, last = int(years.min()), config_last_year(config)
    rows = []
    for c in range(first, last + 1):
        mine = cohort == c
        mine_shares, mine_forfeit = shares[mine], forfeit[mine]
        total = float(mine_shares.sum())
        rows.append(
            {"cohort": c, "effective_date": dt.date(c, 1, 1), "estimate": g.forfeiture_prior}
        )
        for year in range(c + 1, last + 1):
            revision = np.datetime64(dt.date(year, 1, 1), "D")
            lost = float(mine_shares[mine_forfeit < revision].sum())
            observed = lost / total if total else 0.0
            weight = min(1.0, (year - c) / g.forfeiture_credibility_years)
            estimate = min(g.forfeiture_cap, (1 - weight) * g.forfeiture_prior + weight * observed)
            rows.append(
                {"cohort": c, "effective_date": dt.date(year, 1, 1), "estimate": round(estimate, 6)}
            )
    return pd.DataFrame(rows)


def config_last_year(config: WorldConfig) -> int:
    return int((pd.Period(config.start_month, freq="M") + config.months - 1).year)


def _month_index(values: pd.Series, first: pd.Period) -> NDArray[np.int64]:
    """Months from `first` to each date's month."""
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    months: NDArray[np.int64] = (
        (dates.year.to_numpy(np.int64) - first.year) * 12
        + dates.month.to_numpy(np.int64)
        - first.month
    )
    return months


#: Awards per chunk of the ledger. A chunk's working arrays stay small however big the world.
LEDGER_CHUNK_AWARDS: Final = 20_000


def ledger_chunks(
    config: WorldConfig,
    awards: pd.DataFrame,
    vests: pd.DataFrame,
    estimates: pd.DataFrame,
    month_ends: list[pd.Timestamp],
    chunk_awards: int = LEDGER_CHUNK_AWARDS,
) -> Iterator[dict[str, NDArray[Any]]]:
    """The reference ledger's columns, a chunk of awards at a time, in award order.

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
    award_ids = awards["award_id"].to_numpy(np.int64)
    order = np.argsort(award_ids, kind="stable")
    sorted_ids = award_ids[order]
    award_month = _month_index(awards["award_date"], first)[order]
    grant_value = awards["fair_value"].to_numpy(np.float64)[order]
    cohort_of = pd.to_datetime(awards["award_date"]).dt.year.to_numpy(np.int64)[order]
    # The estimate in effect for each cohort and month index.
    vest_month = _month_index(vests["vest_date"], first)
    horizon = int(vest_month.max(initial=0)) + 1
    rate: dict[int, NDArray[np.float64]] = {}
    for c, rows in estimates.groupby("cohort"):
        series = np.full(max(horizon, window) + 1, float(rows["estimate"].iloc[0]))
        for effective, estimate in zip(
            _month_index(rows["effective_date"], first), rows["estimate"], strict=True
        ):
            series[max(int(effective), 0) :] = float(estimate)
        rate[int(str(c))] = series
    forfeit = vests["forfeit_date"]
    forfeit_month = np.where(
        forfeit.isna(), -1, _month_index(forfeit.fillna(pd.Timestamp(month_ends[0])), first)
    )
    # Tranches grouped by award, in award order, each award's tranches in their own order.
    tranche_ids = vests["award_id"].to_numpy(np.int64)
    by_award = np.argsort(tranche_ids, kind="stable")
    grouped = tranche_ids[by_award]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]][: len(grouped)])
    bounds = np.r_[starts, len(grouped)].astype(np.int64)
    position = np.searchsorted(sorted_ids, grouped[bounds[:-1]])
    shares = vests["shares"].to_numpy()[by_award]
    vest_month, forfeit_month = vest_month[by_award], forfeit_month[by_award]
    period = np.array([d.date() for d in month_ends], dtype="datetime64[D]").astype("datetime64[s]")
    names = ("gross", "net", "trueup_attrition", "trueup_rate", "trueup_vesting")
    for begin in range(0, len(bounds) - 1, chunk_awards):
        parts: dict[str, list[NDArray[np.float64]]] = {k: [] for k in names}
        ids: list[NDArray[np.int64]] = []
        months: list[NDArray[np.int64]] = []
        for g in range(begin, min(begin + chunk_awards, len(bounds) - 1)):
            lo, hi, k = bounds[g], bounds[g + 1], position[g]
            award_id = int(sorted_ids[k])
            a = int(award_month[k])
            value = shares[lo:hi] * float(grant_value[k])
            vest = vest_month[lo:hi]
            lost_at = forfeit_month[lo:hi]
            lost = lost_at >= 0
            last = np.where(lost, lost_at, vest)
            span = np.arange(a, int(max(last.max(), vest.max())) + 1)
            e = rate[int(cohort_of[k])][np.clip(span, 0, None)]
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
            for name, matrix in zip(names, (gross, net, attrition, rate_up, vesting), strict=True):
                parts[name].append(matrix.sum(axis=0)[inside])
        chunk: dict[str, NDArray[Any]] = {
            "award_id": np.concatenate(ids),
            "period": period[np.concatenate(months)],
            **{name: np.concatenate(values) for name, values in parts.items()},
        }
        chunk["recognized"] = (
            chunk["net"]
            + chunk["trueup_attrition"]
            + chunk["trueup_rate"]
            + chunk["trueup_vesting"]
        )
        yield chunk


def ledger_frames(
    config: WorldConfig,
    awards: pd.DataFrame,
    vests: pd.DataFrame,
    estimates: pd.DataFrame,
    month_ends: list[pd.Timestamp],
) -> Iterator[pd.DataFrame]:
    """The reference ledger as frames, a chunk of awards each, and one empty frame if there
    are no awards."""
    empty = True
    for chunk in ledger_chunks(config, awards, vests, estimates, month_ends):
        empty = False
        yield pd.DataFrame({name: chunk[name] for name in LEDGER_COLUMNS}, copy=False)
    if empty:
        yield pd.DataFrame(columns=LEDGER_COLUMNS)
