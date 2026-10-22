"""Growth requisitions, driven by a latent plan per cost center.

The plan starts from each cost center's headcount when the world starts. Each cost center
grows at its own annual rate, drawn once and revised each January by a random-walk step, and
the target compounds monthly. Each month a cost center opens growth requisitions with a
negative-binomial count: a gamma-mixed Poisson whose mean is a share of its gap to plan, times
a seasonal factor. The plan is the truth that the plan vintages (M2.6) are derived from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.config import GrowthTruth


class LatentPlan:
    def __init__(
        self,
        truth: GrowthTruth,
        start: NDArray[np.int64],
        months: list[pd.Period],
        rng: np.random.Generator,
    ) -> None:
        """Targets for every month in `months`, from the starting headcount `start`."""
        self.truth = truth
        self.months = months
        n = len(start)
        rate = rng.normal(truth.plan_growth_mean, truth.plan_growth_sigma, n)
        self.target = np.zeros((len(months), n))
        self.annual_growth = np.zeros((len(months), n))
        level = start.astype(np.float64)
        for k, month in enumerate(months):
            if k > 0 and month.month == 1:
                rate = rate + rng.normal(0, truth.plan_revision_sigma, n)
            if k > 0:
                level = level * (1 + np.maximum(rate, -0.9)) ** (1 / 12)
            self.target[k] = level
            self.annual_growth[k] = rate

    def openings(
        self, k: int, committed: NDArray[np.int64], rng: np.random.Generator
    ) -> NDArray[np.int64]:
        """Growth requisitions per cost center in month `k`, given committed strength."""
        t = self.truth
        month = self.months[k]
        season = 1 + t.seasonal_amplitude * np.cos(
            2 * np.pi * (month.month - t.seasonal_peak_month) / 12
        )
        mean = np.maximum(0.0, self.target[k] - committed) * t.gap_closing_rate * season
        rate = rng.gamma(t.dispersion, np.maximum(mean, 1e-12) / t.dispersion)
        counts: NDArray[np.int64] = rng.poisson(np.where(mean > 0, rate, 0.0)).astype(np.int64)
        return counts

    def frame(self, cost_centers: list[str]) -> pd.DataFrame:
        """The plan as a long table: one row per cost center and month-end."""
        months = len(self.months)
        return pd.DataFrame(
            {
                "cost_center": np.tile(np.array(cost_centers), months),
                "month_end": np.repeat(
                    [m.end_time.normalize().date() for m in self.months], len(cost_centers)
                ),
                "target_headcount": self.target.ravel().round(4),
                "annual_growth": self.annual_growth.ravel().round(6),
            }
        )
