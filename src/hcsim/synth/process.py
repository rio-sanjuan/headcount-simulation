"""The world's monthly loop: departures, requisitions, fills, and the month-end stock.

Each month, in order:

0. In a cycle month, a calibration rates everyone eligible (see `ratings`), and then a
   promotion cycle promotes everyone ready (see `promotions`).
1. Every active employee may leave, with the true departure hazard. A departure may open a
   backfill requisition for the same role.
2. Each cost center opens growth requisitions at its own Poisson rate.
3. Every open requisition, including this month's, may fill externally (a hazard by age), fill
   by transfer when an internal candidate exists, fill by converting a contingent worker, exit
   unfilled, or stay open. A transfer
   vacates the mover's old role, which may open a backfill of its own from the next month.
4. The month-end stock is recorded.

Draws are made over whole arrays. Transfers are the one sequential step: each picks a mover
from a pool grouped by family and level, so no requisition scans the workforce.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.config import WorldConfig
from hcsim.synth.movements import MovementProcess
from hcsim.synth.population import draw_roles
from hcsim.synth.promotions import PromotionProcess
from hcsim.synth.ratings import RatingProcess
from hcsim.synth.requisitions import RequisitionBook
from hcsim.synth.structure import cost_center_names, level_one_names
from hcsim.synth.workforce import Workforce, tenure_months

Dates = NDArray[np.datetime64]


def month_start(month_end: pd.Timestamp) -> np.datetime64:
    return np.datetime64(month_end.date().replace(day=1), "D")


def random_days(
    month_end: pd.Timestamp, n: int, rng: np.random.Generator, earliest: Dates | None = None
) -> Dates:
    """A day in `month_end`'s month for each of `n` events, no earlier than `earliest`."""
    first = month_start(month_end)
    low = np.zeros(n, dtype=np.int64)
    if earliest is not None:
        low = np.maximum(0, (earliest - first).astype(np.int64))
    return first + rng.integers(low, month_end.day).astype("timedelta64[D]")


class Shocks:
    """The attrition regime and org shocks, stepped once a month.

    The company-wide regime is a two-state Markov chain, calm or turbulent. Each level-1 org
    may start a shock that multiplies its hazard for a few months. Both are drawn before the
    month's departures, and both are recorded for the truth manifest.
    """

    def __init__(self, config: WorldConfig, org_of_cost_center: NDArray[np.int64]) -> None:
        self.truth = config.attrition
        self.org = org_of_cost_center
        self.turbulent = False
        self.remaining = np.zeros(int(org_of_cost_center.max(initial=-1)) + 1, dtype=np.int64)
        self.regime: list[bool] = []
        self.org_shocks: list[tuple[int, int]] = []  # (month, org)

    def step(self, month: int, rng: np.random.Generator) -> NDArray[np.float64]:
        """Advance one month and return each cost center's multiplier."""
        t = self.truth
        flip = t.turbulent_to_calm if self.turbulent else t.calm_to_turbulent
        if rng.random() < flip:
            self.turbulent = not self.turbulent
        self.remaining = np.maximum(self.remaining - 1, 0)
        start = (rng.random(len(self.remaining)) < t.org_shock_probability) & (self.remaining == 0)
        self.remaining[start] = t.org_shock_months
        self.regime.append(self.turbulent)
        self.org_shocks += [(month, int(org)) for org in np.flatnonzero(start)]
        org_factor = np.where(self.remaining > 0, t.org_shock_factor, 1.0)
        regime = t.turbulent_factor if self.turbulent else 1.0
        multiplier: NDArray[np.float64] = regime * org_factor[self.org]
        return multiplier


class TransferPool:
    """Active employees grouped by family and level, for drawing transfer candidates.

    A candidate shares the requisition's family, sits within `band` levels of it, and works in
    another cost center. Each pick is uniform over the valid candidates.
    """

    def __init__(
        self,
        workforce: Workforce,
        band: int,
        members: NDArray[np.int64] | None = None,
        other_cost_center: bool = True,
    ) -> None:
        self.workforce = workforce
        self.band = band
        self.other_cost_center = other_cost_center
        active = workforce.active() if members is None else members
        family, level = workforce.column("family")[active], workforce.column("level")[active]
        self.groups: dict[tuple[int, int], list[int]] = {}
        for f, lv, i in zip(family.tolist(), level.tolist(), active.tolist(), strict=True):
            self.groups.setdefault((f, lv), []).append(i)

    def pick(
        self, family: int, level: int, cost_center: int, rng: np.random.Generator
    ) -> int | None:
        groups = [
            self.groups.get((family, lv), [])
            for lv in range(level - self.band, level + self.band + 1)
        ]
        sizes = np.array([len(g) for g in groups])
        total = int(sizes.sum())
        if total == 0:
            return None
        bounds = np.cumsum(sizes)
        cost_centers = self.workforce.column("cost_center")
        for _ in range(8):  # rejection is uniform over the valid candidates
            r = int(rng.integers(total))
            g = int(np.searchsorted(bounds, r, side="right"))
            candidate = groups[g][r - (int(bounds[g - 1]) if g else 0)]
            if not self.other_cost_center or cost_centers[candidate] != cost_center:
                return candidate
        pool = np.array([i for g in groups for i in g])
        valid = pool[cost_centers[pool] != cost_center] if self.other_cost_center else pool
        return int(valid[rng.integers(len(valid))]) if len(valid) else None

    def move(self, index: int, old: tuple[int, int], new: tuple[int, int] | None) -> None:
        """Regroup a member whose job changed, or drop one that left the pool (`new=None`)."""
        self.groups[old].remove(index)
        if new is not None:
            self.groups.setdefault(new, []).append(index)


def simulate(
    config: WorldConfig,
    workforce: Workforce,
    book: RequisitionBook,
    month_ends: list[pd.Timestamp],
    rng: np.random.Generator,
    ratings: RatingProcess | None = None,
    org_of_cost_center: NDArray[np.int64] | None = None,
    promotions: PromotionProcess | None = None,
    movements: MovementProcess | None = None,
) -> dict[str, Any]:
    """Run the world over `month_ends`. The first month-end is the initial stock. Returns the
    random effects the run drew, for the truth manifest."""
    attrition, reqs = config.attrition, config.requisitions
    n = config.cost_centers
    cc_effect = np.exp(rng.normal(0, attrition.cost_center_sigma, n))
    growth_rate = reqs.growth_requisitions_per_month * np.exp(rng.normal(0, reqs.growth_sigma, n))
    growth_levels = np.array(list(reqs.growth_level_mix))
    growth_p = np.array(list(reqs.growth_level_mix.values()))
    orgs = (
        org_of_cost_center
        if org_of_cost_center is not None
        else np.arange(n, dtype=np.int64) % config.org_level1_nodes
    )
    shocks = Shocks(config, orgs)
    workforce.snapshot(np.datetime64(month_ends[0].date(), "D"))

    if ratings is not None:
        ratings.seed(pd.Period(month_ends[0], freq="M"))
    if promotions is not None:
        promotions.seed()
    for m, month_end in enumerate(month_ends[1:], start=1):
        end = np.datetime64(month_end.date(), "D")
        month = pd.Period(month_end, freq="M")
        # 0. A calibration takes effect on the first day of a cycle month.
        if ratings is not None and ratings.is_cycle(month):
            ratings.calibrate(month)
        # 0b. Promotions take effect on the same day, reading the new ratings.
        if promotions is not None and promotions.is_cycle(month):
            promotions.promote(month)
        # 0c. Reclassification and reorganization batches.
        if movements is not None:
            movements.reclassify(month)
            movements.reorganize(month)
        # 1. Departures, and the backfill each may open.
        multiplier = shocks.step(m, rng)
        active = workforce.active()
        cost_center = workforce.column("cost_center")[active]
        h = attrition.hazards(
            workforce.column("level")[active],
            tenure_months(workforce.column("hire_date")[active], end),
            month_end.month,
            cc_effect[cost_center],
            workforce.column("rating")[active],
            multiplier[cost_center],
        )
        leavers = active[rng.random(len(active)) < h]
        dates = random_days(month_end, len(leavers), rng)
        workforce.terminate(leavers, dates)
        backfill = rng.random(len(leavers)) < reqs.backfill_probability
        book.open("backfill", workforce.roles(leavers[backfill]), dates[backfill], m)
        # 1b. Conversions out, and the contingent pool's own ends and starts.
        if movements is not None:
            movements.contingent(month)

        # 2. Growth requisitions.
        cost_centers = np.repeat(np.arange(n), rng.poisson(growth_rate))
        levels = rng.choice(growth_levels, size=len(cost_centers), p=growth_p)
        roles = draw_roles(config, cost_centers, levels, rng)
        book.open("growth", roles, random_days(month_end, len(cost_centers), rng), m)

        # 3. Fills and unfilled exits. Backfills opened by transfers wait a month.
        open_reqs = book.open_requisitions()
        h_ext = reqs.external_hazards(m - book.column("opened_month")[open_reqs])
        u = rng.random(len(open_reqs))
        dates = random_days(month_end, len(open_reqs), rng, book.column("opened_date")[open_reqs])
        h_conv = config.movements.conversion_fill_hazard if movements is not None else 0.0
        external = u < h_ext
        transfer = ~external & (u < h_ext + reqs.transfer_fill_hazard)
        conversion = ~external & ~transfer & (u < h_ext + reqs.transfer_fill_hazard + h_conv)
        unfilled = (
            ~external
            & ~transfer
            & ~conversion
            & (u < h_ext + reqs.transfer_fill_hazard + h_conv + reqs.unfilled_exit_hazard)
        )
        hired = workforce.hire(book.roles(open_reqs[external]), dates[external])
        book.close(open_reqs[external], dates[external], "filled_external", hired)
        _transfers(config, workforce, book, open_reqs[transfer], dates[transfer], m, rng)
        _conversions(config, workforce, book, open_reqs[conversion], dates[conversion], rng)
        book.close(open_reqs[unfilled], dates[unfilled], "unfilled")
        workforce.snapshot(end)
        if promotions is not None:
            promotions.accrue()

    names = cost_center_names(config)
    months = [str(pd.Period(e, freq="M")) for e in month_ends[1:]]
    tops = level_one_names(config.org_level1_nodes)
    return {
        "movements": movements.manifest() if movements is not None else {},
        "attrition": {
            "cost_center_effect": dict(zip(names, cc_effect.round(6).tolist(), strict=True)),
            "turbulent_months": [mo for mo, on in zip(months, shocks.regime, strict=True) if on],
            "org_shocks": [
                {"start": months[m - 1], "org": tops[org]} for m, org in shocks.org_shocks
            ],
        },
        "requisitions": {
            "growth_rate": dict(zip(names, growth_rate.round(6).tolist(), strict=True))
        },
    }


def _transfers(
    config: WorldConfig,
    workforce: Workforce,
    book: RequisitionBook,
    requisitions: NDArray[np.int64],
    dates: Dates,
    month: int,
    rng: np.random.Generator,
) -> None:
    reqs = config.requisitions
    pool = TransferPool(workforce, reqs.transfer_level_band)
    for j, date in zip(requisitions.tolist(), dates, strict=True):
        role = book.roles(np.array([j]))
        mover = pool.pick(
            int(role["family"][0]), int(role["level"][0]), int(role["cost_center"][0]), rng
        )
        if mover is None:
            continue
        index = np.array([mover])
        old = (int(workforce.column("family")[mover]), int(workforce.column("level")[mover]))
        vacated = workforce.reassign(index, role, np.array([date]))
        pool.move(mover, old, (int(role["family"][0]), int(role["level"][0])))
        book.close(
            np.array([j]),
            np.array([date]),
            "filled_transfer",
            workforce.column("employee_id")[index],
        )
        if rng.random() < reqs.backfill_probability:
            book.open("backfill", vacated, np.array([date]), month + 1)


def _conversions(
    config: WorldConfig,
    workforce: Workforce,
    book: RequisitionBook,
    requisitions: NDArray[np.int64],
    dates: Dates,
    rng: np.random.Generator,
) -> None:
    """Fill requisitions by converting contingent workers of the same family, within the band,
    from any cost center. A requisition with no candidate stays open."""
    pool = TransferPool(
        workforce,
        config.movements.conversion_level_band,
        members=workforce.contingent_pool(),
        other_cost_center=False,
    )
    for j, date in zip(requisitions.tolist(), dates, strict=True):
        role = book.roles(np.array([j]))
        family, level = int(role["family"][0]), int(role["level"][0])
        worker = pool.pick(family, level, int(role["cost_center"][0]), rng)
        if worker is None:
            continue
        index = np.array([worker])
        old = (int(workforce.column("family")[worker]), int(workforce.column("level")[worker]))
        workforce.convert_in(index, role, np.array([date]))
        pool.move(worker, old, None)
        book.close(
            np.array([j]),
            np.array([date]),
            "filled_conversion",
            workforce.column("employee_id")[index],
        )
