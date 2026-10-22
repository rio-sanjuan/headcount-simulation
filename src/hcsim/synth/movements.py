"""Moves that aren't hires or departures: reclassifications, reorganizations, and the
contingent pool with its conversions out. Conversions in fill requisitions, so they live with
the fills in `process`.

Each month, in order: a possible reclassification batch on the first day, a possible
reorganization batch on one day, then conversions out, contingent ends, and contingent starts
on days spread over the month. Batches are recorded for the truth manifest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.config import WorldConfig
from hcsim.synth.dates import MonthDays
from hcsim.synth.population import Roles, draw_roles
from hcsim.synth.structure import cost_center_names, level_one_names
from hcsim.synth.workforce import Workforce


class MovementProcess:
    def __init__(
        self,
        config: WorldConfig,
        workforce: Workforce,
        org_of_cost_center: NDArray[np.int64],
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.truth = config.movements
        self.workforce = workforce
        self.org = org_of_cost_center
        self.rng = rng
        self.contingent_account = [g for _, g in config.accounts].index("contingent")
        self.reorgs: list[dict[str, Any]] = []
        self.reclasses: list[dict[str, Any]] = []
        levels = list(config.requisitions.growth_level_mix)
        self._levels = np.array(levels)
        self._level_p = np.array(list(config.requisitions.growth_level_mix.values()))

    def _contingent_roles(self, n: int) -> Roles:
        cost_centers = self.rng.integers(self.config.cost_centers, size=n)
        levels = self.rng.choice(self._levels, size=n, p=self._level_p)
        roles = draw_roles(self.config, cost_centers, levels, self.rng)
        roles["account"] = np.full(n, self.contingent_account, dtype=np.int16)
        roles["is_manager"] = np.zeros(n, dtype=bool)
        return roles

    def seed(self, first_month: pd.Period) -> None:
        """The contingent pool in post when the world starts."""
        n = round(self.truth.contingent_share * len(self.workforce.active()))
        start = np.datetime64(first_month.start_time.date(), "D")
        tenure = self.rng.exponential(12 * 30.4, n).astype(np.int64)
        self.workforce.enroll_contingent(
            self._contingent_roles(n), start - (tenure + 1).astype("timedelta64[D]")
        )

    def reclassify(self, month: pd.Period) -> None:
        """With `reclass_probability`, move a share of one family in one level-1 org to
        another family, in place, on the first day of the month."""
        families = len(self.config.families)
        if families < 2 or self.rng.random() >= self.truth.reclass_probability:
            return
        wf = self.workforce
        active = wf.active()
        org = int(self.rng.integers(self.org.max() + 1))
        here = active[self.org[wf.column("cost_center")[active]] == org]
        if not len(here):
            return
        source = int(self.rng.choice(np.unique(wf.column("family")[here])))
        target = (source + 1 + int(self.rng.integers(families - 1))) % families
        members = here[wf.column("family")[here] == source]
        moved = members[self.rng.random(len(members)) < self.truth.reclass_share]
        day = np.datetime64(month.start_time.date(), "D")
        wf.reclassify(moved, target, np.full(len(moved), day))
        self.reclasses.append(
            {
                "month": str(month),
                "org": level_one_names(int(self.org.max()) + 1)[org],
                "from": self.config.families[source],
                "to": self.config.families[target],
                "people": len(moved),
            }
        )

    def reorganize(self, month: pd.Period) -> None:
        """With `reorg_probability`, move a share of one cost center's people to another cost
        center in the same level-1 org, all on one day."""
        if self.rng.random() >= self.truth.reorg_probability:
            return
        orgs = [o for o in np.unique(self.org) if (self.org == o).sum() >= 2]
        if not orgs:
            return
        org = int(self.rng.choice(orgs))
        source, target = self.rng.choice(np.flatnonzero(self.org == org), size=2, replace=False)
        wf = self.workforce
        active = wf.active()
        members = active[wf.column("cost_center")[active] == source]
        share = self.rng.uniform(*self.truth.reorg_share)
        moved = members[self.rng.random(len(members)) < share]
        day = MonthDays(month, self.config.dates).sample("business", 1, self.rng)[0]
        wf.reorganize(moved, int(target), np.full(len(moved), day))
        names = cost_center_names(self.config)
        self.reorgs.append(
            {"month": str(month), "from": names[source], "to": names[target], "people": len(moved)}
        )

    def contingent(self, month: pd.Period) -> None:
        """Conversions out, contingent ends, and contingent starts, on days in the month."""
        t, wf = self.truth, self.workforce
        days = MonthDays(month, self.config.dates)
        active = wf.active()
        out = active[self.rng.random(len(active)) < t.conversion_out_hazard]
        wf.convert_out(out, self.contingent_account, days.sample("business", len(out), self.rng))
        pool = wf.contingent_pool()
        ending = pool[self.rng.random(len(pool)) < t.contingent_end_hazard]
        wf.end_contingent(ending, days.sample("exit", len(ending), self.rng))
        starts = int(self.rng.poisson(t.contingent_start_rate * len(wf.active())))
        wf.start_contingent(self._contingent_roles(starts), days.sample("start", starts, self.rng))

    def manifest(self) -> dict[str, Any]:
        return {"reorganizations": self.reorgs, "reclassifications": self.reclasses}
