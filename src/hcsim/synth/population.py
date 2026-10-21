"""The initial population: who is employed at the world's first month-end."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from hcsim.synth.config import WorldConfig
from hcsim.synth.structure import cost_center_names, dominant_family, job_code

EMPLOYEE_COLUMNS = [
    "employee_id",
    "cost_center",
    "job_code",
    "job_family",
    "job_level",
    "location",
    "company",
    "account",
    "is_manager",
    "hire_date",
]


def draw_role(
    config: WorldConfig, cost_center_index: int, level: int, rng: np.random.Generator
) -> dict[str, object]:
    """A role in a cost center: its family leans to the cost center's dominant one."""
    dominant = dominant_family(config, cost_center_index)
    if rng.random() < config.dominant_family_share:
        family = dominant
    else:
        others = [f for f in config.families if f != dominant]
        family = others[int(rng.integers(len(others)))]
    is_manager = level >= config.manager_min_level and rng.random() < config.manager_share
    return {
        "cost_center": cost_center_names(config)[cost_center_index],
        "job_code": job_code(family, level),
        "job_family": family,
        "job_level": level,
        "location": config.locations[int(rng.integers(len(config.locations)))],
        "company": config.companies[int(rng.integers(len(config.companies)))],
        "account": "ACCT-100",
        "is_manager": bool(is_manager),
    }


def initial_population(config: WorldConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Employees active at the end of the first month, with tenure before the start."""
    first_day = dt.date.fromisoformat(f"{config.start_month}-01")
    rows = []
    for i in range(config.initial_employees):
        cc = int(rng.integers(config.cost_centers))
        level = int(rng.choice(config.levels, p=list(config.level_mix)))
        tenure_days = int(rng.exponential(config.mean_initial_tenure_months * 30.4))
        role = draw_role(config, cc, level, rng)
        rows.append(
            {
                "employee_id": config.first_employee_id + i,
                **role,
                "hire_date": first_day - dt.timedelta(days=tenure_days + 1),
            }
        )
    return pd.DataFrame(rows, columns=EMPLOYEE_COLUMNS)
