"""World configuration. Every value here is invented, round, and synthetic.

The world is a seeded company with its own ground-truth processes. Its parameters are recorded
in the truth manifest, so models and studies can report how well they recover them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Families and their tech indicator. Synthetic names, chosen to read as generic functions.
FAMILIES: dict[str, str] = {
    "Engineering": "software_engineering",
    "Finance": "non_technical",
    "Operations": "other_technical",
    "Sales": "non_technical",
}


@dataclass(frozen=True)
class WorldConfig:
    """Everything that shapes a synthetic world. A preset fixes the scale."""

    seed: int = 0
    preset: str = "tiny"
    start_month: str = "2021-01"
    months: int = 60
    initial_employees: int = 200
    org_level1_nodes: int = 2
    cost_centers: int = 10
    families: tuple[str, ...] = tuple(FAMILIES)
    levels: tuple[int, ...] = (1, 2, 3, 4, 5)
    #: Share of the initial population at each level, in `levels` order.
    level_mix: tuple[float, ...] = (0.30, 0.30, 0.20, 0.15, 0.05)
    #: Share of a cost center's people in its dominant family.
    dominant_family_share: float = 0.70
    locations: tuple[str, ...] = ("LOC-1",)
    companies: tuple[str, ...] = ("CO-1",)
    #: Mean tenure of the initial population, in months.
    mean_initial_tenure_months: float = 36.0
    #: Chance that someone at the manager-eligible levels is a manager.
    manager_share: float = 0.30
    manager_min_level: int = 4
    #: First employee id. Ids are integers, so they fit the float64 buffers exactly.
    first_employee_id: int = 100_001
    processes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problems = []
        if len(self.level_mix) != len(self.levels):
            problems.append("level_mix needs one share per level")
        if abs(sum(self.level_mix) - 1.0) > 1e-9:
            problems.append("level_mix must sum to 1")
        if self.cost_centers < self.org_level1_nodes:
            problems.append("need at least one cost center per org_level1 node")
        if not 0 <= self.dominant_family_share <= 1:
            problems.append("dominant_family_share must lie in [0, 1]")
        if self.months < 1 or self.initial_employees < 1:
            problems.append("months and initial_employees must be positive")
        if problems:
            raise ValueError("invalid WorldConfig: " + "; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {},
}


def preset(name: str, seed: int = 0) -> WorldConfig:
    try:
        overrides = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; known: {', '.join(PRESETS)}") from None
    return WorldConfig(seed=seed, preset=name, **overrides)
