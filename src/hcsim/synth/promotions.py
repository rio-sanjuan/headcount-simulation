"""Promotions: a latent readiness that accrues every month, and promotion cycles.

Readiness grows each month for everyone active, by a non-negative draw whose mean depends on
the rating. A promotion cycle takes effect on the first day of each cycle month, after any
rating calibration and before the month's other events. Everyone ready (readiness at or above
the threshold, and below the world's top level) is promoted one level with their level's
chance. A promotion may also change the family, and one into a manager-eligible level may make
a manager. Promotion resets readiness to zero, and each one is recorded as a `PROMO` row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcsim.synth.config import WorldConfig
from hcsim.synth.workforce import Workforce


class PromotionProcess:
    def __init__(self, config: WorldConfig, workforce: Workforce, rng: np.random.Generator) -> None:
        self.config = config
        self.truth = config.promotions
        self.workforce = workforce
        self.rng = rng
        self._drift = np.asarray(self.truth.readiness_drift, dtype=np.float64)
        levels = np.asarray(config.levels)
        self._next_level = np.zeros(levels.max() + 1, dtype=np.int16)
        self._next_level[levels[:-1]] = levels[1:]
        self._chance = np.zeros(levels.max() + 1)
        for level, p in self.truth.promotion_probability.items():
            if level <= levels.max():
                self._chance[level] = p
        self._chance[levels[-1]] = 0.0  # the top level is never promoted

    def seed(self) -> None:
        """Incumbents start part of the way to their next promotion."""
        active = self.workforce.active()
        start = self.rng.uniform(0, self.truth.readiness_threshold, len(active))
        self.workforce.column("readiness")[active] = start

    def accrue(self) -> None:
        active = self.workforce.active()
        rating = self.workforce.column("rating")[active]
        mean = np.where(rating >= 0, self._drift[np.maximum(rating, 0)], self.truth.unrated_drift)
        step = np.maximum(0.0, self.rng.normal(mean, self.truth.readiness_sigma))
        self.workforce.column("readiness")[active] += step.astype(np.float32)

    def is_cycle(self, month: pd.Period) -> bool:
        return month.month in self.truth.cycle_months

    def promote(self, month: pd.Period) -> int:
        """Run a promotion cycle effective on the first day of `month`. Returns the count."""
        wf, cfg = self.workforce, self.config
        active = wf.active()
        level = wf.column("level")[active]
        ready = wf.column("readiness")[active] >= self.truth.readiness_threshold
        chosen = active[ready & (self.rng.random(len(active)) < self._chance[level])]
        roles = wf.roles(chosen)
        roles["level"] = self._next_level[roles["level"]]
        families = len(cfg.families)
        change = (self.rng.random(len(chosen)) < self.truth.family_change_probability) & (
            families > 1
        )
        shift = 1 + self.rng.integers(0, max(families - 1, 1), len(chosen))
        roles["family"] = np.where(change, (roles["family"] + shift) % families, roles["family"])
        roles["family"] = roles["family"].astype(np.int16)
        eligible = (roles["level"] >= cfg.manager_min_level) & ~roles["is_manager"]
        roles["is_manager"] = roles["is_manager"] | (
            eligible & (self.rng.random(len(chosen)) < cfg.manager_share)
        )
        wf.promote(chosen, roles, np.full(len(chosen), np.datetime64(month.start_time.date(), "D")))
        wf.column("readiness")[chosen] = 0.0
        return len(chosen)
