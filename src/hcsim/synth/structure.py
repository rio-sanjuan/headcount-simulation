"""The world's reference structures: org, jobs, locations, companies, and accounts.

These become the dimension tables of the data contract. Deep org levels repeat the leaf as
padding, so every cost center has a value at every level.
"""

from __future__ import annotations

import pandas as pd

from hcsim.synth.config import FAMILIES, WorldConfig


def cost_center_names(config: WorldConfig) -> list[str]:
    return [f"CC-{i:02d}" for i in range(1, config.cost_centers + 1)]


def org_dimension(config: WorldConfig) -> pd.DataFrame:
    """One row per leaf cost center. Leaves are dealt round-robin to the level-1 nodes."""
    nodes = [f"ORG-{chr(ord('A') + i)}" for i in range(config.org_level1_nodes)]
    leaves = cost_center_names(config)
    return pd.DataFrame(
        {
            "cost_center": leaves,
            "org_level1": [nodes[i % len(nodes)] for i in range(len(leaves))],
            "org_level2": leaves,  # the leaf, as padding at the deepest level
            "cost_type": "fixed",
            "field_type": "corporate",
        }
    )


def job_code(family: str, level: int) -> str:
    return f"{family[:3].upper()}-{level}"


def job_codes(config: WorldConfig) -> list[str]:
    """Every job code, family by family and level by level: the job dimension's order."""
    return [job_code(family, level) for family in config.families for level in config.levels]


def job_dimension(config: WorldConfig) -> pd.DataFrame:
    rows = [
        {
            "job_code": job_code(family, level),
            "job_family": family,
            "job_level": level,
            "tech_indicator": FAMILIES[family],
        }
        for family in config.families
        for level in config.levels
    ]
    return pd.DataFrame(rows)


def location_dimension(config: WorldConfig) -> pd.DataFrame:
    return pd.DataFrame({"location": list(config.locations), "region": "REGION-1"})


def company_dimension(config: WorldConfig) -> pd.DataFrame:
    return pd.DataFrame({"company": list(config.companies), "planning_company": "PCO-1"})


def account_dimension() -> pd.DataFrame:
    return pd.DataFrame({"account": ["ACCT-100"], "account_group": ["regular"]})
