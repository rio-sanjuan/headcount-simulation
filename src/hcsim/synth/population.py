"""Roles and the initial population.

A role is a cost center, a job family, a level, a location, a company, an account, and a
manager flag, held as integer codes: an index into the configuration's tuple for each
categorical, and the level's own value. Roles travel as dicts of equal-length arrays.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from hcsim.synth.config import WorldConfig

Roles = dict[str, np.ndarray]

ROLE_DTYPES: Final[dict[str, type]] = {
    "cost_center": np.int32,
    "family": np.int16,
    "level": np.int16,
    "location": np.int16,
    "company": np.int16,
    "account": np.int16,
    "is_manager": np.bool_,
}
#: The one account every role books to, until the account catalog arrives (M2.1).
ACCOUNTS: Final = ("ACCT-100",)


def draw_roles(
    config: WorldConfig,
    cost_centers: NDArray[np.integer],
    levels: NDArray[np.integer],
    rng: np.random.Generator,
) -> Roles:
    """Roles in the given cost centers and levels. Each cost center leans toward one family:
    its dominant family with `dominant_family_share`, and otherwise any other family."""
    n = len(cost_centers)
    families = len(config.families)
    dominant = np.asarray(cost_centers) % families
    keep = rng.random(n) < config.dominant_family_share
    other = (dominant + 1 + rng.integers(0, max(families - 1, 1), n)) % families
    family = np.where(keep | (families == 1), dominant, other)
    manager = (np.asarray(levels) >= config.manager_min_level) & (
        rng.random(n) < config.manager_share
    )
    return {
        "cost_center": np.asarray(cost_centers, dtype=np.int32),
        "family": family.astype(np.int16),
        "level": np.asarray(levels, dtype=np.int16),
        "location": rng.integers(len(config.locations), size=n).astype(np.int16),
        "company": rng.integers(len(config.companies), size=n).astype(np.int16),
        "account": np.zeros(n, dtype=np.int16),
        "is_manager": manager,
    }


def initial_population(
    config: WorldConfig, rng: np.random.Generator
) -> tuple[NDArray[np.int64], Roles, NDArray[np.datetime64]]:
    """Employees active at the end of the first month: ids, roles, and hire dates before the
    world's start."""
    n = config.initial_employees
    first_day = np.datetime64(f"{config.start_month}-01", "D")
    cost_centers = rng.integers(config.cost_centers, size=n)
    levels = rng.choice(np.asarray(config.levels), size=n, p=list(config.level_mix))
    tenure_days = rng.exponential(config.mean_initial_tenure_months * 30.4, n).astype(np.int64)
    roles = draw_roles(config, cost_centers, levels, rng)
    ids = config.first_employee_id + np.arange(n, dtype=np.int64)
    return ids, roles, first_day - (tenure_days + 1).astype("timedelta64[D]")
