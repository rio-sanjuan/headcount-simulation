"""The world's reference structures: org, jobs, locations, companies, and accounts.

These become the dimension tables of the data contract. The org hierarchy has uneven depth,
and deep levels repeat the leaf as padding, so every cost center has a value at every level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcsim.synth.config import FAMILIES, WorldConfig


def cost_center_names(config: WorldConfig) -> list[str]:
    return [f"CC-{i:02d}" for i in range(1, config.cost_centers + 1)]


def level_one_names(n: int) -> list[str]:
    """`ORG-A` to `ORG-Z`, then `ORG-AA` onward."""
    names = []
    for i in range(n):
        letters, k = "", i
        while True:
            letters = chr(ord("A") + k % 26) + letters
            k = k // 26 - 1
            if k < 0:
                break
        names.append(f"ORG-{letters}")
    return names


def org_levels(config: WorldConfig) -> list[str]:
    return [f"org_level{k}" for k in range(1, config.org_levels + 1)]


def org_dimension(config: WorldConfig, rng: np.random.Generator) -> pd.DataFrame:
    """One row per leaf cost center, with its org node at every level.

    Leaves are dealt round-robin to the level-1 orgs. Below level 1, a node either holds its
    cost centers directly, with `org_stop_probability`, or splits them among up to
    `org_branching` children. A node at the second-deepest level always holds them directly.
    A cost center's own label pads every level below its depth. Node labels extend their
    parent's, such as `ORG-A-2-1`.
    """
    depth = config.org_levels
    leaves = cost_center_names(config)
    paths: dict[int, list[str]] = {}

    def place(members: list[int], path: list[str]) -> None:
        level = len(path) + 1  # the level the members' children would occupy
        if level == depth or len(members) == 1 or rng.random() < config.org_stop_probability:
            for i in members:
                paths[i] = path + [leaves[i]] * (depth - len(path))
            return
        groups = np.array_split(np.array(members), min(config.org_branching, len(members)))
        for j, group in enumerate(groups, start=1):
            place(group.tolist(), [*path, f"{path[-1]}-{j}"])

    tops = level_one_names(config.org_level1_nodes)
    for t, top in enumerate(tops):
        if members := list(range(t, len(leaves), len(tops))):
            place(members, [top])
    field_tops = set(np.array(tops)[rng.random(len(tops)) < config.field_org_share])
    frame = pd.DataFrame([paths[i] for i in range(len(leaves))], columns=org_levels(config)).assign(
        cost_center=leaves
    )
    return frame[["cost_center", *org_levels(config)]].assign(
        cost_type=np.where(
            rng.random(len(leaves)) < config.variable_cost_share, "variable", "fixed"
        ),
        field_type=np.where(frame["org_level1"].isin(field_tops), "field", "corporate"),
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
