"""Data quirks in `workforce_txn` (see `QuirkTruth`): backdated rows, date corrections,
no-show pairs, and pending hires with future dates.

Each quirk rewrites or adds rows after the world has run, from its own random stream, so the
processes and every other table are untouched. Corrections and no-show pairs add rows whose
amounts net out within the month, so the month-end stocks still agree with the summed flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcsim.synth.config import WorldConfig
from hcsim.synth.offers import Offers
from hcsim.synth.structure import cost_center_names, job_code


def _days(values: np.ndarray) -> pd.TimedeltaIndex:
    return pd.to_timedelta(values, unit="D")


def _offer_rows(
    config: WorldConfig,
    offers: Offers,
    index: np.ndarray,
    code: str,
    amount: int,
    event: np.ndarray,
    reported: np.ndarray,
) -> pd.DataFrame:
    """Transaction rows for accepted offers, in each offer's role, with no previous role."""
    book = offers.book.column
    names = cost_center_names(config)
    cost_center = [names[c] for c in book("cost_center")[index]]
    job = [
        job_code(config.families[f], int(lv), bool(m))
        for f, lv, m in zip(
            book("family")[index], book("level")[index], book("is_manager")[index], strict=True
        )
    ]
    location = [config.locations[i] for i in book("location")[index]]
    company = [config.companies[i] for i in book("company")[index]]
    account = [config.accounts[i][0] for i in book("account")[index]]
    when = pd.to_datetime(event.astype("datetime64[D]"))
    return pd.DataFrame(
        {
            "employee_id": book("employee_id")[index],
            "reporting_date": pd.to_datetime(reported.astype("datetime64[D]")),
            "event_date": when,
            "trans_type": code,
            "amount": amount,
            "cost_center": cost_center,
            "cost_center_previous": cost_center,
            "job_code": job,
            "job_code_previous": job,
            "location": location,
            "location_previous": location,
            "company": company,
            "company_previous": company,
            "account": account,
            "account_previous": account,
            "is_manager": book("is_manager")[index],
            "is_manager_previous": book("is_manager")[index],
            "hire_date": when,
        }
    )


def apply(
    config: WorldConfig, txn: pd.DataFrame, offers: Offers, rng: np.random.Generator
) -> pd.DataFrame:
    q = config.quirks
    frame = txn.copy()
    extra: list[pd.DataFrame] = []

    if q.backdated_share > 0:
        late = (frame["trans_type"] != "ACTIVE") & (rng.random(len(frame)) < q.backdated_share)
        lag = rng.integers(30, 91, int(late.sum()))
        frame.loc[late, "reporting_date"] = frame.loc[late, "event_date"] + _days(lag)

    if q.correction_share > 0:
        # A hire or departure first recorded on a wrong day in its month, then reversed and
        # recorded on the right day: three rows that net to the one true event.
        chosen = frame["trans_type"].isin(["HIRE", "TERM"]) & (
            rng.random(len(frame)) < q.correction_share
        )
        true = frame[chosen]
        month = true["event_date"].dt.to_period("M")
        start = month.dt.start_time
        span = month.dt.days_in_month.to_numpy()
        wrong = start + _days(rng.integers(0, span))
        later = true["reporting_date"] + _days(rng.integers(1, 15, len(true)))
        reverse = true["trans_type"].astype(str).map({"HIRE": "HIRE_REV", "TERM": "TERM_REV"})
        extra += [
            true.assign(event_date=wrong),
            true.assign(
                event_date=wrong, reporting_date=later, trans_type=reverse, amount=-true["amount"]
            ),
            true.assign(reporting_date=later),
        ]
        frame = frame[~chosen]

    done = np.flatnonzero(offers.book.column("done"))
    if q.no_show_pair_share > 0:
        missed = done[offers.book.column("no_show")[done]]
        pairs = missed[rng.random(len(missed)) < q.no_show_pair_share]
        start = offers.book.column("start_date")[pairs]
        month_end = (start.astype("datetime64[M]") + np.timedelta64(1, "M")).astype(
            "datetime64[D]"
        ) - np.timedelta64(1, "D")
        undone = np.minimum(
            start + rng.integers(1, 15, len(pairs)).astype("timedelta64[D]"), month_end
        )
        extra += [
            _offer_rows(config, offers, pairs, "HIRE", 1, start, start),
            _offer_rows(config, offers, pairs, "HIRE_REV", -1, undone, undone),
        ]
    if q.pending_hire_rows:
        made = np.arange(len(offers.book))
        extra.append(
            _offer_rows(
                config,
                offers,
                made,
                "HIRE_PENDING",
                0,
                offers.book.column("start_date")[made],
                offers.book.column("offer_date")[made],
            )
        )
    if not extra:
        return frame
    return _concat(frame, extra)


def snapshot_gaps(
    config: WorldConfig, month_ends: list[pd.Timestamp], rng: np.random.Generator
) -> list[str]:
    """The month-ends a missed feed run leaves out of `req_snapshot`, as ISO dates."""
    share = config.quirks.snapshot_gap_share
    if share <= 0:
        return []
    missed = rng.random(len(month_ends)) < share
    return [str(d.date()) for d, gone in zip(month_ends, missed, strict=True) if gone]


def aggregate_only(
    config: WorldConfig, agg: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Add `SYS_ADJ` rows to the aggregate fact: a share of its regular cost-center months gets
    one, with a row count but no amount, as a system adjustment would. They have no
    transactions behind them."""
    share = config.quirks.aggregate_only_share
    if share <= 0:
        return agg
    stock = agg[agg["trans_type"] == "ACTIVE"]
    cells = stock.drop_duplicates(["period", "cost_center"])
    chosen = cells[rng.random(len(cells)) < share]
    rows = chosen.assign(trans_type="SYS_ADJ", headcount=rng.integers(1, 21, len(chosen)), amount=0)
    combined = pd.concat([agg, rows], ignore_index=True)
    for column, dtype in agg.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype):
            combined[column] = pd.Categorical(combined[column], categories=dtype.categories)
    return combined


def _concat(frame: pd.DataFrame, extra: list[pd.DataFrame]) -> pd.DataFrame:
    """Append rows, keeping the frame's column order and dtypes, categorical labels included."""
    dates = {
        c: frame[c].dtype for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])
    }
    labels = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]
    parts = [frame, *[e[frame.columns].astype(dates) for e in extra]]
    combined = pd.concat(
        [p.astype(dict.fromkeys(labels, object)) for p in parts], ignore_index=True
    )
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            combined[column] = pd.Categorical(combined[column], categories=dtype.categories)
        else:
            combined[column] = combined[column].astype(dtype)
    return combined.sort_values(["event_date", "employee_id"], kind="stable", ignore_index=True)
