"""The world's monthly loop: departures, requisitions, fills, and the month-end stock.

Each month, in order:

0. In a cycle month, a calibration rates everyone eligible (see `ratings`), and then a
   promotion cycle promotes everyone ready (see `promotions`). Then come reclassification and
   reorganization batches (see `movements`) and termination programs (see `programs`).
1. People who gave notice leave on schedule. Every other active employee may leave, with the
   true departure hazard; some give notice instead and leave a month or two later. Each
   vacancy may open a backfill requisition (see `Backfills`), on the day of notice for those
   who gave it.
2. Each cost center opens growth requisitions toward its latent plan (see `growth`).
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
from hcsim.synth.dates import MonthDays
from hcsim.synth.growth import LatentPlan
from hcsim.synth.movements import MovementProcess
from hcsim.synth.population import Roles, draw_roles
from hcsim.synth.programs import TerminationPrograms
from hcsim.synth.promotions import PromotionProcess
from hcsim.synth.ratings import RatingProcess
from hcsim.synth.requisitions import RequisitionBook
from hcsim.synth.structure import cost_center_names, level_one_names
from hcsim.synth.workforce import Workforce, tenure_months

Dates = NDArray[np.datetime64]


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


class Backfills:
    """Opens backfill requisitions for vacated roles (see `BackfillTruth`)."""

    def __init__(
        self, config: WorldConfig, org_of_cost_center: NDArray[np.int64], rng: np.random.Generator
    ) -> None:
        self.truth = config.backfill
        self.effect = rng.normal(0, self.truth.cost_center_sigma, config.cost_centers)
        self.shift = np.zeros(max(config.levels) + 1)
        for level, value in self.truth.level_shift.items():
            if level < len(self.shift):
                self.shift[level] = value
        base = float(np.clip(self.truth.base_probability, 1e-12, 1 - 1e-12))
        self.logit = np.log(base / (1 - base))
        self.org = org_of_cost_center
        self.siblings = {
            int(o): np.flatnonzero(org_of_cost_center == o) for o in np.unique(org_of_cost_center)
        }

    def probability(
        self, cost_centers: NDArray[np.integer], levels: NDArray[np.integer]
    ) -> NDArray[np.float64]:
        z = self.logit + self.effect[cost_centers] + self.shift[levels]
        p: NDArray[np.float64] = 1 / (1 + np.exp(-z))
        return p

    def open(
        self,
        book: RequisitionBook,
        vacated_by: NDArray[np.int64],
        roles: Roles,
        dates: Dates,
        month: int,
        rng: np.random.Generator,
    ) -> None:
        """Open a backfill for each vacated role with its probability. A share opens in a
        sibling cost center, one under the same level-1 org."""
        keep = rng.random(len(dates)) < self.probability(roles["cost_center"], roles["level"])
        kept = {name: values[keep].copy() for name, values in roles.items()}
        elsewhere = np.flatnonzero(rng.random(int(keep.sum())) < self.truth.other_cost_center_share)
        for i in elsewhere.tolist():
            own = int(kept["cost_center"][i])
            options = self.siblings[int(self.org[own])]
            options = options[options != own]
            if len(options):
                kept["cost_center"][i] = int(options[rng.integers(len(options))])
        book.open("backfill", kept, dates[keep], month, vacated_by[keep])


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
        active = workforce.available() if members is None else members
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
    programs: TerminationPrograms | None = None,
) -> dict[str, Any]:
    """Run the world over `month_ends`. The first month-end is the initial stock. Returns the
    random effects the run drew, for the truth manifest."""
    attrition, reqs = config.attrition, config.requisitions
    n = config.cost_centers
    cc_effect = np.exp(rng.normal(0, attrition.cost_center_sigma, n))
    growth_levels = np.array(list(reqs.growth_level_mix))
    growth_p = np.array(list(reqs.growth_level_mix.values()))
    orgs = (
        org_of_cost_center
        if org_of_cost_center is not None
        else np.arange(n, dtype=np.int64) % config.org_level1_nodes
    )
    shocks = Shocks(config, orgs)
    backfills = Backfills(config, orgs, rng)
    notice = config.backfill
    workforce.snapshot(np.datetime64(month_ends[0].date(), "D"))
    start = np.bincount(workforce.column("cost_center")[workforce.active()], minlength=n)
    plan = LatentPlan(
        config.growth, start.astype(np.int64), [pd.Period(e, freq="M") for e in month_ends], rng
    )

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
        # 0d. Termination programs, known events, come before sampled departures.
        if programs is not None:
            programs.run(month)
        # 1. Departures. People who gave notice leave on schedule; then the hazard picks new
        # leavers. Some give notice instead of leaving at once, and their backfill opens now.
        days = MonthDays(month, config.dates)
        ids = workforce.column("employee_id")
        active = workforce.active()
        exit_month = workforce.column("exit_month")
        scheduled = active[exit_month[active] == m]
        workforce.terminate(scheduled, workforce.column("exit_date")[scheduled])
        candidates = active[exit_month[active] < 0]
        multiplier = shocks.step(m, rng)
        cost_center = workforce.column("cost_center")[candidates]
        h = attrition.hazards(
            workforce.column("level")[candidates],
            tenure_months(workforce.column("hire_date")[candidates], end),
            month_end.month,
            cc_effect[cost_center],
            workforce.column("rating")[candidates],
            multiplier[cost_center],
        )
        leavers = candidates[rng.random(len(candidates)) < h]
        gives_notice = rng.random(len(leavers)) < notice.notice_share
        now, noticed = leavers[~gives_notice], leavers[gives_notice]
        dates = days.sample("exit", len(now), rng)
        workforce.terminate(now, dates)
        backfills.open(book, ids[now], workforce.roles(now), dates, m, rng)
        lag = rng.integers(notice.notice_months[0], notice.notice_months[1] + 1, len(noticed))
        notice_dates = days.sample("business", len(noticed), rng)
        exit_dates = np.empty(len(noticed), dtype="datetime64[D]")
        for k in np.unique(lag).tolist():
            later = MonthDays(month + k, config.dates)
            exit_dates[lag == k] = later.sample("exit", int((lag == k).sum()), rng)
        workforce.give_notice(noticed, notice_dates, exit_dates, m + lag)
        backfills.open(book, ids[noticed], workforce.roles(noticed), notice_dates, m, rng)

        # 1b. Conversions out, and the contingent pool's own ends and starts.
        if movements is not None:
            movements.contingent(month)

        # 2. Growth requisitions, closing part of each cost center's gap to its latent plan.
        committed = np.bincount(
            workforce.column("cost_center")[workforce.active()], minlength=n
        ) + np.bincount(book.column("cost_center")[book.open_requisitions()], minlength=n)
        cost_centers = np.repeat(np.arange(n), plan.openings(m, committed, rng))
        levels = rng.choice(growth_levels, size=len(cost_centers), p=growth_p)
        roles = draw_roles(config, cost_centers, levels, rng)
        book.open("growth", roles, days.sample("business", len(cost_centers), rng), m)

        # 3. Fills and unfilled exits. Backfills opened by transfers wait a month.
        open_reqs = book.open_requisitions()
        h_ext = reqs.external_hazards(m - book.column("opened_month")[open_reqs])
        u = rng.random(len(open_reqs))
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
        # External hires start on the start weekday; other outcomes fall on business days.
        opened = book.column("opened_date")[open_reqs]
        dates = days.sample("business", len(open_reqs), rng, opened)
        dates[external] = days.sample("start", int(external.sum()), rng, opened[external])
        hired = workforce.hire(book.roles(open_reqs[external]), dates[external])
        book.close(open_reqs[external], dates[external], "filled_external", hired)
        _transfers(config, workforce, book, open_reqs[transfer], dates[transfer], m, rng, backfills)
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
        "programs": programs.manifest() if programs is not None else [],
        "backfill": {
            "cost_center_effect": dict(zip(names, backfills.effect.round(6).tolist(), strict=True))
        },
        "attrition": {
            "cost_center_effect": dict(zip(names, cc_effect.round(6).tolist(), strict=True)),
            "turbulent_months": [mo for mo, on in zip(months, shocks.regime, strict=True) if on],
            "org_shocks": [
                {"start": months[m - 1], "org": tops[org]} for m, org in shocks.org_shocks
            ],
        },
        "plan": plan,
    }


def _transfers(
    config: WorldConfig,
    workforce: Workforce,
    book: RequisitionBook,
    requisitions: NDArray[np.int64],
    dates: Dates,
    month: int,
    rng: np.random.Generator,
    backfills: Backfills,
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
        backfills.open(
            book, workforce.column("employee_id")[index], vacated, np.array([date]), month + 1, rng
        )


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
