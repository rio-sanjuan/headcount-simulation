"""Plan vintages: published headcount plans, derived from the latent plan (see `PlanTruth`).

Each vintage is published at the start of a publication month and runs for the horizon. A
cost center's regime in a vintage is drawn afresh, so successive vintages disagree. `planned`
cells are the latent plan times the vintage's noise; `flat` cells hold the regular headcount
at the month-end before publication; `decaying` cells start from the latent plan and decay
toward that headcount.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.config import WorldConfig
from hcsim.synth.growth import LatentPlan

REGIMES: Final = ("planned", "flat", "decaying")


def publications(config: WorldConfig) -> list[pd.Period]:
    """Every vintage's publication month, from the world's second year on."""
    first = pd.Period(config.start_month, freq="M")
    months = [first + k for k in range(12, config.months)]
    return [m for m in months if m.month in config.plans.publication_months]


def vintages(
    config: WorldConfig,
    plan: LatentPlan,
    headcount: NDArray[np.int64],
    cost_centers: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """`plan_headcount` at cost-center grain: one row per vintage, cost center, and month."""
    t = config.plans
    first = pd.Period(config.start_month, freq="M")
    n = len(cost_centers)
    frames = []
    for published in publications(config):
        k0 = (published - first).n
        span = np.arange(k0, k0 + t.horizon_months)
        regime = rng.choice(len(REGIMES), size=n, p=list(t.regime_shares))
        noise = np.exp(rng.normal(0, t.vintage_noise_sigma, n))
        latent = plan.target[span]  # (horizon, cost centers)
        base = headcount[k0 - 1].astype(np.float64)  # the month-end before publication
        decay = np.exp(-np.arange(t.horizon_months) / t.decay_months)[:, None]
        target = np.where(
            regime == 0,
            latent * noise,
            np.where(regime == 1, base, base + (latent * noise - base) * decay),
        )
        months = [first + int(k) for k in span]
        frames.append(
            pd.DataFrame(
                {
                    "vintage": str(published),
                    "published_date": published.start_time.date(),
                    "cost_center": np.tile(np.array(cost_centers), len(span)),
                    "month_end": np.repeat([m.end_time.normalize().date() for m in months], n),
                    "regime": np.tile(np.array(REGIMES)[regime], len(span)),
                    "target_headcount": target.ravel().round(4),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
