"""World configuration: every parameter that shapes a synthetic world, in one frozen object.

Every value here is invented, round, and synthetic. The truth manifest records the whole
configuration, so models and studies can report how well they recover it. A preset fixes the
scale; the processes' parameters are the same at every scale unless a preset overrides them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

import numpy as np

# Families and their tech indicator. Synthetic names, chosen to read as generic functions.
FAMILIES: dict[str, str] = {
    "Engineering": "software_engineering",
    "Finance": "non_technical",
    "Operations": "other_technical",
    "Sales": "non_technical",
}


# --- process truths ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttritionTruth:
    """The true monthly departure hazard. Invented, round values."""

    base_monthly_hazard: float = 0.008
    level_factor: dict[int, float] = field(
        default_factory=lambda: {1: 1.25, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.7}
    )
    first_year_factor: float = 1.5
    seasonal_amplitude: float = 0.2
    seasonal_peak_month: int = 3
    cost_center_sigma: float = 0.2

    def hazard(
        self, level: int, tenure_months: float, month_of_year: int, cost_center_effect: float
    ) -> float:
        season = 1 + self.seasonal_amplitude * np.cos(
            2 * np.pi * (month_of_year - self.seasonal_peak_month) / 12
        )
        tenure = self.first_year_factor if tenure_months < 12 else 1.0
        h = self.base_monthly_hazard * self.level_factor[level] * tenure * season
        return float(min(1.0, h * cost_center_effect))


@dataclass(frozen=True)
class RequisitionTruth:
    backfill_probability: float = 0.6
    growth_requisitions_per_month: float = 0.25
    growth_sigma: float = 0.3
    growth_level_mix: dict[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.3, 3: 0.2})
    #: Monthly external-fill hazard by age in months: 0, 1, 2, then 3 and older.
    external_fill_hazard: tuple[float, ...] = (0.15, 0.30, 0.30, 0.25)
    transfer_fill_hazard: float = 0.06
    unfilled_exit_hazard: float = 0.04
    #: A transfer candidate shares the family and sits within this many levels.
    transfer_level_band: int = 1

    def external_hazard(self, age_months: int) -> float:
        return self.external_fill_hazard[min(age_months, len(self.external_fill_hazard) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompensationTruth:
    base_midpoint_by_level: dict[int, float] = field(
        default_factory=lambda: {1: 60_000, 2: 80_000, 3: 110_000, 4: 150_000, 5: 200_000}
    )
    family_factor: dict[str, float] = field(
        default_factory=lambda: {
            "Engineering": 1.2,
            "Finance": 1.0,
            "Operations": 0.9,
            "Sales": 1.0,
        }
    )
    #: Equity's share of target total compensation, on top of base, by level.
    equity_share_by_level: dict[int, float] = field(
        default_factory=lambda: {1: 0.10, 2: 0.20, 3: 0.35, 4: 0.50, 5: 0.80}
    )
    range_half_width: float = 0.20
    base_adjustment: float = 1.03


@dataclass(frozen=True)
class PriceTruth:
    start_price: float = 100.0
    monthly_drift: float = 0.008
    monthly_volatility: float = 0.07
    #: Grant-date fair value = month-end close times a uniform factor in this band.
    fair_value_band: tuple[float, float] = (0.97, 1.03)


@dataclass(frozen=True)
class GrantTruth:
    new_hire_value_by_level: dict[int, float] = field(
        default_factory=lambda: {1: 5_000, 2: 15_000, 3: 40_000, 4: 90_000, 5: 200_000}
    )
    value_sigma: float = 0.15
    award_lag_days: tuple[int, int] = (30, 60)
    #: Vest schedule: months after the award, and the share of the award vesting at each.
    vest_months: tuple[int, ...] = (12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48)
    vest_shares: tuple[float, ...] = (0.25,) + (0.0625,) * 12


# --- the world --------------------------------------------------------------------------------


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
    attrition: AttritionTruth = field(default_factory=AttritionTruth)
    requisitions: RequisitionTruth = field(default_factory=RequisitionTruth)
    compensation: CompensationTruth = field(default_factory=CompensationTruth)
    prices: PriceTruth = field(default_factory=PriceTruth)
    grants: GrantTruth = field(default_factory=GrantTruth)

    def __post_init__(self) -> None:
        problems = []
        if len(self.level_mix) != len(self.levels):
            problems.append("level_mix needs one share per level")
        if abs(sum(self.level_mix) - 1.0) > 1e-9:
            problems.append("level_mix must sum to 1")
        if self.cost_centers < self.org_level1_nodes:
            problems.append("need at least one cost center per org_level1 node")
        if self.org_level1_nodes < 1:
            problems.append("org_level1_nodes must be at least 1")
        if not 0 <= self.dominant_family_share <= 1:
            problems.append("dominant_family_share must lie in [0, 1]")
        if self.months < 1 or self.initial_employees < 1:
            problems.append("months and initial_employees must be positive")
        if not self.locations or not self.companies:
            problems.append("a world needs at least one location and one company")
        unknown = [f for f in self.families if f not in FAMILIES]
        if unknown:
            problems.append(f"families without a tech indicator: {unknown}")
        levels = set(self.levels)
        for name, by_level in (
            ("attrition.level_factor", self.attrition.level_factor),
            ("compensation.base_midpoint_by_level", self.compensation.base_midpoint_by_level),
            ("compensation.equity_share_by_level", self.compensation.equity_share_by_level),
            ("grants.new_hire_value_by_level", self.grants.new_hire_value_by_level),
        ):
            if missing := sorted(levels - set(by_level)):
                problems.append(f"{name} has no value for levels {missing}")
        if no_factor := sorted(set(self.families) - set(self.compensation.family_factor)):
            problems.append(f"compensation.family_factor has no value for {no_factor}")
        if extra := sorted(set(self.requisitions.growth_level_mix) - levels):
            problems.append(f"requisitions.growth_level_mix names unknown levels {extra}")
        if problems:
            raise ValueError("invalid WorldConfig: " + "; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _labels(prefix: str, n: int) -> tuple[str, ...]:
    return tuple(f"{prefix}-{i}" for i in range(1, n + 1))


#: Scale overrides per preset. About 20 employees per cost center at every scale.
PRESETS: Final[dict[str, dict[str, Any]]] = {
    "tiny": {},
    "small": {
        "initial_employees": 5_000,
        "cost_centers": 250,
        "org_level1_nodes": 5,
        "locations": _labels("LOC", 4),
        "companies": _labels("CO", 2),
    },
    "medium": {
        "initial_employees": 50_000,
        "cost_centers": 2_500,
        "org_level1_nodes": 10,
        "locations": _labels("LOC", 10),
        "companies": _labels("CO", 3),
    },
    "large": {
        "initial_employees": 500_000,
        "cost_centers": 25_000,
        "org_level1_nodes": 20,
        "locations": _labels("LOC", 20),
        "companies": _labels("CO", 5),
    },
}


def preset(name: str, seed: int = 0) -> WorldConfig:
    try:
        overrides = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; known: {', '.join(PRESETS)}") from None
    return WorldConfig(seed=seed, preset=name, **overrides)
