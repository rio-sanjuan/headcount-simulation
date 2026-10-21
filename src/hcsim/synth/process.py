"""The world's monthly loop: departures, requisitions, fills, and the month-end stock."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from hcsim.synth.config import AttritionTruth, RequisitionTruth
from hcsim.synth.requisitions import Requisition, RequisitionBook
from hcsim.synth.workforce import Workforce, random_day, tenure_months

RoleDrawer = Callable[[int, int, np.random.Generator], dict[str, Any]]


def day_on_or_after(start: dt.date, month_end: pd.Timestamp, rng: np.random.Generator) -> dt.date:
    """A day in `month_end`'s month, no earlier than `start`."""
    first = start.day if (start.year, start.month) == (month_end.year, month_end.month) else 1
    return dt.date(month_end.year, month_end.month, int(rng.integers(first, month_end.day + 1)))


def transfer_candidate(
    workforce: Workforce, req: Requisition, band: int, rng: np.random.Generator
) -> int | None:
    """An employee elsewhere in the same family, within `band` levels of the role."""
    role = req.role
    pool = [
        employee_id
        for employee_id, person in workforce
        if person["job_family"] == role["job_family"]
        and abs(int(person["job_level"]) - int(role["job_level"])) <= band
        and person["cost_center"] != role["cost_center"]
    ]
    return pool[int(rng.integers(len(pool)))] if pool else None


def simulate(
    initial: pd.DataFrame,
    month_ends: list[pd.Timestamp],
    cost_centers: list[str],
    next_id: int,
    rng: np.random.Generator,
    draw_role: RoleDrawer,
    attrition: AttritionTruth | None = None,
    requisitions: RequisitionTruth | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the world over `month_ends`. The first month-end is the initial stock.

    Returns the transaction fact, the requisition intervals, and the truth that drove them.
    """
    attrition = attrition or AttritionTruth()
    reqs = requisitions or RequisitionTruth()
    workforce = Workforce(initial, next_id)
    workforce.snapshot(month_ends[0])
    book = RequisitionBook()
    cc_effect = np.exp(rng.normal(0, attrition.cost_center_sigma, len(cost_centers)))
    growth_rate = reqs.growth_requisitions_per_month * np.exp(
        rng.normal(0, reqs.growth_sigma, len(cost_centers))
    )
    cc_index = {cc: i for i, cc in enumerate(cost_centers)}
    levels, level_p = zip(*reqs.growth_level_mix.items(), strict=True)

    for m, month_end in enumerate(month_ends[1:], start=1):
        # Departures, and the backfill each one may open.
        for employee_id, person in workforce:
            h = attrition.hazard(
                int(person["job_level"]),
                tenure_months(person, month_end),
                month_end.month,
                float(cc_effect[cc_index[person["cost_center"]]]),
            )
            if rng.random() < h:
                date = random_day(month_end, rng)
                departed = workforce.terminate(employee_id, date)
                if rng.random() < reqs.backfill_probability:
                    book.open_requisition("backfill", departed, date, m)
        # Growth requisitions.
        for i, n in enumerate(rng.poisson(growth_rate)):
            for _ in range(int(n)):
                level = int(rng.choice(levels, p=level_p))
                book.open_requisition(
                    "growth", draw_role(i, level, rng), random_day(month_end, rng), m
                )
        # Fills and unfilled exits. Requisitions opened below, by transfers, wait a month.
        for req in list(book.open.values()):
            h_ext = reqs.external_hazard(m - req.opened_month)
            u = rng.random()
            date = day_on_or_after(req.opened_date, month_end, rng)
            if u < h_ext:
                book.close(req, date, "filled_external", workforce.hire(req.role, date))
            elif u < h_ext + reqs.transfer_fill_hazard:
                mover = transfer_candidate(workforce, req, reqs.transfer_level_band, rng)
                if mover is not None:
                    vacated = workforce.reassign(mover, req.role, date)
                    book.close(req, date, "filled_transfer", mover)
                    if rng.random() < reqs.backfill_probability:
                        book.open_requisition("backfill", vacated, date, m + 1)
            elif u < h_ext + reqs.transfer_fill_hazard + reqs.unfilled_exit_hazard:
                book.close(req, date, "unfilled")
        workforce.snapshot(month_end)

    truth = {
        "attrition": asdict(attrition)
        | {"cost_center_effect": dict(zip(cost_centers, cc_effect.round(6).tolist(), strict=True))},
        "requisitions": reqs.to_dict()
        | {"growth_rate": dict(zip(cost_centers, growth_rate.round(6).tolist(), strict=True))},
    }
    return pd.DataFrame(workforce.rows), book.frame(), truth
