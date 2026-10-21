"""The world's reference structures: org, jobs, locations, companies, and accounts.

These become the dimension tables of the data contract. The org hierarchy has uneven depth,
and deep levels repeat the leaf as padding, so every cost center has a value at every level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hcsim.synth.config import FAMILIES, FAMILY_CODES, WorldConfig


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


ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")


def job_code(family: str, level: int, manager: bool = False) -> str:
    """`ENG-4` for an individual role, and `ENG-4M` for its manager title."""
    return f"{FAMILY_CODES[family]}-{level}{'M' if manager else ''}"


def jobs(config: WorldConfig) -> list[tuple[str, int, bool]]:
    """Every job as (family, level, manager), in the job dimension's order. Manager titles
    exist from `manager_min_level` up."""
    return [
        (family, level, manager)
        for family in config.families
        for level in config.levels
        for manager in ((False, True) if level >= config.manager_min_level else (False,))
    ]


def job_codes(config: WorldConfig) -> list[str]:
    return [job_code(*job) for job in jobs(config)]


def job_dimension(config: WorldConfig) -> pd.DataFrame:
    """The job catalog: one row per job, with an ordered level vocabulary (`job_level`, with
    `level_name` and the executive band), the family's tech indicator, and a title that marks
    manager roles."""
    rows = [
        {
            "job_code": job_code(family, level, manager),
            "job_title": f"{family} {'Manager ' if manager else ''}{ROMAN[level - 1]}",
            "job_family": family,
            "job_level": level,
            "level_name": f"L{level}",
            "is_executive": level >= config.executive_min_level,
            "manager_title": manager,
            "tech_indicator": FAMILIES[family],
        }
        for family, level, manager in jobs(config)
    ]
    return pd.DataFrame(rows)


def location_dimension(config: WorldConfig) -> pd.DataFrame:
    """Locations with their region tree: locations dealt to countries, countries to regions."""
    index = np.arange(len(config.locations))
    country = index // config.locations_per_country
    region = country // config.countries_per_region
    return pd.DataFrame(
        {
            "location": list(config.locations),
            "country": [f"CTRY-{c + 1}" for c in country],
            "region": [f"REGION-{r + 1}" for r in region],
        }
    )


def company_dimension(config: WorldConfig) -> pd.DataFrame:
    """Companies and their planning rollup. The primary company and the online subsidiaries
    roll up to `PCO-1`. Offline subsidiaries, the last ones, roll up to `PCO-2`."""
    n = len(config.companies)
    offline = np.arange(n) >= n - config.offline_subsidiaries
    return pd.DataFrame(
        {
            "company": list(config.companies),
            "planning_company": np.where(offline, "PCO-2", "PCO-1"),
            "is_primary": np.arange(n) == 0,
            "is_offline": offline,
        }
    )


def account_dimension(config: WorldConfig) -> pd.DataFrame:
    return pd.DataFrame(list(config.accounts), columns=["account", "account_group"])
