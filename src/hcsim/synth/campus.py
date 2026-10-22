"""Campus hiring: requisitions opened on a seasonal schedule at entry levels, and filled only
through the campus channel in the start months (see `CampusTruth`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcsim.synth.config import WorldConfig
from hcsim.synth.dates import MonthDays
from hcsim.synth.population import draw_roles
from hcsim.synth.requisitions import REQ_TYPES, RequisitionBook
from hcsim.synth.workforce import Workforce

CAMPUS = REQ_TYPES.index("campus")


def campus_requisitions(book: RequisitionBook) -> np.ndarray:
    open_reqs = book.open_requisitions()
    return open_reqs[book.column("req_type")[open_reqs] == CAMPUS]


def campus_month(
    config: WorldConfig,
    workforce: Workforce,
    book: RequisitionBook,
    month: pd.Period,
    m: int,
    days: MonthDays,
    rng: np.random.Generator,
) -> None:
    """Open this year's campus requisitions in the open month, and fill or close them in the
    start months."""
    t = config.campus
    if month.month == t.open_month:
        headcount = np.bincount(
            workforce.column("cost_center")[workforce.active()], minlength=config.cost_centers
        )
        cost_centers = np.repeat(
            np.arange(config.cost_centers), rng.poisson(t.positions_per_head * headcount)
        )
        levels = rng.choice(
            np.array(list(t.level_mix)), size=len(cost_centers), p=list(t.level_mix.values())
        )
        roles = draw_roles(config, cost_centers, levels, rng)
        book.open("campus", roles, days.sample("business", len(cost_centers), rng), m)
    if month.month not in t.start_months:
        return
    k = t.start_months.index(month.month)
    reqs = campus_requisitions(book)
    filled = reqs[rng.random(len(reqs)) < t.fill_probability[k]]
    opened = book.column("opened_date")[filled]
    starts = days.sample("start", len(filled), rng, opened)
    hired = workforce.hire(book.roles(filled), starts)
    book.close(filled, starts, "filled_campus", hired)
    if k == len(t.start_months) - 1:
        rest = campus_requisitions(book)
        book.close(
            rest,
            days.sample("business", len(rest), rng, book.column("opened_date")[rest]),
            "unfilled",
        )
