"""The `truth/` namespace: what the world really is, kept apart from the source tables.

The source tables are what a warehouse would hold. The truth is what only a simulator knows:
every parameter, the random effects and shocks a run drew, the latent growth plan, and each
plan vintage's regime. Tests and recovery reports read it; the engine and the guidance
pipeline never do.

The manifest holds every parameter and every recorded draw. The tables hold the same truth
in long form, ready to compare with what a model fitted:

- `latent_plan`: the latent growth plan, per cost center and month-end.
- `plan_regime`: each real cost center's regime in each plan vintage.
- `attrition_hazard`: the monthly hazard by level, tenure, and rating, before the season, the
  cost center's effect, and shocks. The full hazard is the product of all four, capped at 1.
- `attrition_season`: the seasonal factor per month of the year.
- `attrition_shock`: the regime and org shock multiplier, for every month and cost center where
  it isn't 1.
- `cost_center_effect`: each cost center's random effect on attrition (a multiplier) and on
  backfill (a shift on the logit scale).
- `requisition_hazard`: the monthly hazard of each channel by a requisition's age in months.
  The last age listed holds for every older one.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth.config import WorldConfig

#: The tenures the hazard surface covers: every month up to this one.
MAX_TENURE_MONTHS: Final = 240
UNRATED: Final = "unrated"


def attrition_hazard(config: WorldConfig) -> pd.DataFrame:
    truth = config.attrition
    ratings = [*config.ratings.scale, UNRATED]
    codes = np.r_[np.arange(len(config.ratings.scale)), -1]
    tenure = np.arange(MAX_TENURE_MONTHS + 1)
    levels = np.array(config.levels)
    grid = np.array(np.meshgrid(levels, tenure, np.arange(len(codes)), indexing="ij"))
    level, months, rating = (axis.ravel() for axis in grid)
    return pd.DataFrame(
        {
            "job_level": level,
            "tenure_month": months,
            "rating": np.array(ratings)[rating],
            "monthly_hazard": truth.baseline(level, months.astype(np.float64), codes[rating]),
        }
    )


def attrition_season(config: WorldConfig) -> pd.DataFrame:
    months = list(range(1, 13))
    return pd.DataFrame(
        {
            "month_of_year": months,
            "factor": [config.attrition.seasonal_factor(m) for m in months],
        }
    )


def attrition_shock(
    shock: NDArray[np.float64], month_ends: list[pd.Timestamp], cost_centers: list[str]
) -> pd.DataFrame:
    """`shock` has a row per month after the first, as the process steps it."""
    month, cc = np.nonzero(shock != 1.0)
    return pd.DataFrame(
        {
            "month_end": np.array([e.date() for e in month_ends[1:]], dtype="datetime64[D]")[month],
            "cost_center": np.array(cost_centers, dtype=object)[cc],
            "factor": shock[month, cc],
        }
    )


def cost_center_effect(processes: dict[str, Any]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "cost_center": list(effects),
                "process": process,
                "scale": scale,
                "effect": list(effects.values()),
            }
        )
        for process, scale in (("attrition", "multiplier"), ("backfill", "logit"))
        for effects in [processes[process]["cost_center_effect"]]
    ]
    return pd.concat(frames, ignore_index=True)


def requisition_hazard(config: WorldConfig) -> pd.DataFrame:
    reqs = config.requisitions
    channels = {
        "external": reqs.external_fill_hazard,
        "transfer": reqs.transfer_fill_hazard,
        "conversion": config.movements.conversion_fill_hazard,
        "unfilled": reqs.unfilled_exit_hazard,
    }
    return pd.DataFrame(
        [
            {"channel": channel, "age_month": age, "monthly_hazard": hazard}
            for channel, table in channels.items()
            for age, hazard in enumerate(table)
        ]
    )


def plan_regime(vintages: pd.DataFrame) -> pd.DataFrame:
    real = vintages[~vintages["is_hedge"]]
    return real[["vintage", "cost_center", "regime"]].drop_duplicates(ignore_index=True)
