"""World configuration: every parameter that shapes a synthetic world, in one frozen object.

Every value here is invented, round, and synthetic. The truth manifest records the whole
configuration, so models and studies can report how well they recover it. A preset fixes the
scale; the processes' parameters are the same at every scale unless a preset overrides them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# Families and their tech indicator. Synthetic names, chosen to read as generic functions. A
# world uses the first `n` families its preset asks for.
FAMILIES: dict[str, str] = {
    "Engineering": "software_engineering",
    "Finance": "non_technical",
    "Operations": "other_technical",
    "Sales": "non_technical",
    "Research": "other_technical",
    "Marketing": "non_technical",
    "Data": "software_engineering",
    "Support": "other_technical",
    "Design": "other_technical",
    "Legal": "non_technical",
    "People": "non_technical",
    "Security": "software_engineering",
    "Procurement": "non_technical",
    "Facilities": "other_technical",
    "Logistics": "other_technical",
    "Product": "software_engineering",
}
#: Each family's job-code prefix. Unique, so job codes never collide.
FAMILY_CODES: dict[str, str] = {
    "Engineering": "ENG",
    "Finance": "FIN",
    "Operations": "OPE",
    "Sales": "SAL",
    "Research": "RES",
    "Marketing": "MAR",
    "Data": "DAT",
    "Support": "SUP",
    "Design": "DES",
    "Legal": "LEG",
    "People": "PEO",
    "Security": "SEC",
    "Procurement": "PRC",
    "Facilities": "FAC",
    "Logistics": "LOG",
    "Product": "PRD",
}
#: A synthetic pay factor per family, relative to 1.0.
FAMILY_PAY: dict[str, float] = {
    "Engineering": 1.2,
    "Finance": 1.0,
    "Operations": 0.9,
    "Sales": 1.0,
    "Research": 1.15,
    "Marketing": 0.95,
    "Data": 1.15,
    "Support": 0.85,
    "Design": 1.0,
    "Legal": 1.1,
    "People": 0.9,
    "Security": 1.15,
    "Procurement": 0.9,
    "Facilities": 0.8,
    "Logistics": 0.85,
    "Product": 1.1,
}


# --- process truths ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttritionTruth:
    """The true monthly departure hazard. Invented, round values.

    The hazard multiplies a base rate by factors for level, tenure, rating, and season, by a
    random effect per cost center, and by two kinds of shock: a company-wide regime that
    switches between calm and turbulent, and short shocks to single level-1 orgs.

    Tenure enters through a smooth curve: risk rises from hire to a peak at
    `tenure_peak_months`, then fades toward `long_tenure_floor` for long-tenured people.
    """

    base_monthly_hazard: float = 0.006
    level_factor: dict[int, float] = field(
        default_factory=lambda: {1: 1.25, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.7, 6: 0.65, 7: 0.6, 8: 0.55}
    )
    tenure_peak_months: float = 18.0
    tenure_peak_factor: float = 1.8
    long_tenure_floor: float = 0.6
    long_tenure_scale_months: float = 120.0
    #: One factor per rating, worst to best, and one for the unrated.
    rating_factor: tuple[float, ...] = (3.0, 1.6, 1.0, 0.75, 0.6)
    unrated_factor: float = 1.1
    seasonal_amplitude: float = 0.2
    seasonal_peak_month: int = 3
    cost_center_sigma: float = 0.2
    #: The company-wide regime: a two-state Markov chain stepped every month.
    turbulent_factor: float = 1.5
    calm_to_turbulent: float = 0.04
    turbulent_to_calm: float = 0.25
    #: Org shocks: each level-1 org starts one with this monthly chance, and it lasts a while.
    org_shock_probability: float = 0.01
    org_shock_factor: float = 2.0
    org_shock_months: int = 3

    def tenure_factor(self, tenure_months: NDArray[np.floating]) -> NDArray[np.float64]:
        t = np.maximum(np.asarray(tenure_months, dtype=np.float64), 0.0) / self.tenure_peak_months
        bump = 1 + (self.tenure_peak_factor - 1) * t * np.exp(1 - t)
        fade = self.long_tenure_floor + (1 - self.long_tenure_floor) * np.exp(
            -np.asarray(tenure_months, dtype=np.float64) / self.long_tenure_scale_months
        )
        result: NDArray[np.float64] = bump * fade
        return result

    def hazards(
        self,
        levels: NDArray[np.integer],
        tenure_months: NDArray[np.floating],
        month_of_year: int,
        cost_center_effect: NDArray[np.floating],
        ratings: NDArray[np.integer] | None = None,
        shock: NDArray[np.floating] | float = 1.0,
    ) -> NDArray[np.float64]:
        """The monthly departure hazard for each person. `ratings` are codes into the rating
        scale, with -1 for the unrated; `shock` is the regime and org shock multiplier."""
        season = 1 + self.seasonal_amplitude * np.cos(
            2 * np.pi * (month_of_year - self.seasonal_peak_month) / 12
        )
        level = np.zeros(max(self.level_factor) + 1)
        level[list(self.level_factor)] = list(self.level_factor.values())
        rating = np.ones(len(np.asarray(levels)))
        if ratings is not None:
            codes = np.asarray(ratings)
            table = np.asarray(self.rating_factor, dtype=np.float64)
            rating = np.where(codes >= 0, table[np.maximum(codes, 0)], self.unrated_factor)
        h = (
            self.base_monthly_hazard
            * level[np.asarray(levels)]
            * self.tenure_factor(tenure_months)
            * rating
            * season
            * np.asarray(cost_center_effect)
            * np.asarray(shock)
        )
        return np.minimum(1.0, h)

    def hazard(
        self,
        level: int,
        tenure_months: float,
        month_of_year: int,
        cost_center_effect: float,
        rating: int | None = None,
        shock: float = 1.0,
    ) -> float:
        return float(
            self.hazards(
                np.array([level]),
                np.array([tenure_months]),
                month_of_year,
                np.array([cost_center_effect]),
                None if rating is None else np.array([rating]),
                shock,
            )[0]
        )

    def problems(self, rating_levels: int) -> list[str]:
        found = []
        if len(self.rating_factor) != rating_levels:
            found.append("attrition.rating_factor needs one factor per rating")
        for name in ("calm_to_turbulent", "turbulent_to_calm", "org_shock_probability"):
            if not 0 <= getattr(self, name) <= 1:
                found.append(f"attrition.{name} must lie in [0, 1]")
        if self.tenure_peak_months <= 0 or self.long_tenure_scale_months <= 0:
            found.append("attrition tenure scales must be positive")
        if self.org_shock_months < 1:
            found.append("attrition.org_shock_months must be at least 1")
        return found


@dataclass(frozen=True)
class RequisitionTruth:
    backfill_probability: float = 0.6
    growth_level_mix: dict[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.3, 3: 0.2})
    #: Monthly external-fill hazard by age in months: 0, 1, 2, then 3 and older.
    external_fill_hazard: tuple[float, ...] = (0.15, 0.30, 0.30, 0.25)
    transfer_fill_hazard: float = 0.06
    unfilled_exit_hazard: float = 0.04
    #: A transfer candidate shares the family and sits within this many levels.
    transfer_level_band: int = 1

    def external_hazard(self, age_months: int) -> float:
        return self.external_fill_hazard[min(age_months, len(self.external_fill_hazard) - 1)]

    def external_hazards(self, age_months: NDArray[np.integer]) -> NDArray[np.float64]:
        table = np.asarray(self.external_fill_hazard, dtype=np.float64)
        return table[np.minimum(np.asarray(age_months), len(table) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompensationTruth:
    base_midpoint_by_level: dict[int, float] = field(
        default_factory=lambda: {
            1: 60_000,
            2: 80_000,
            3: 110_000,
            4: 150_000,
            5: 200_000,
            6: 270_000,
            7: 360_000,
            8: 480_000,
        }
    )
    family_factor: dict[str, float] = field(default_factory=lambda: dict(FAMILY_PAY))
    #: Equity's share of target total compensation, on top of base, by level.
    equity_share_by_level: dict[int, float] = field(
        default_factory=lambda: {
            1: 0.10,
            2: 0.20,
            3: 0.35,
            4: 0.50,
            5: 0.80,
            6: 1.20,
            7: 1.60,
            8: 2.00,
        }
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
        default_factory=lambda: {
            1: 5_000,
            2: 15_000,
            3: 40_000,
            4: 90_000,
            5: 200_000,
            6: 400_000,
            7: 800_000,
            8: 1_500_000,
        }
    )
    value_sigma: float = 0.15
    award_lag_days: tuple[int, int] = (30, 60)
    #: Vest schedule: months after the award, and the share of the award vesting at each.
    vest_months: tuple[int, ...] = (12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48)
    vest_shares: tuple[float, ...] = (0.25,) + (0.0625,) * 12


@dataclass(frozen=True)
class RatingTruth:
    """Performance ratings: calibration cycles, a Markov chain between cycles, and a separate
    distribution for first ratings. Ratings are plain ordered labels, worst to best."""

    scale: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")
    #: Calendar months whose month-end closes a calibration cycle.
    cycle_months: tuple[int, ...] = (3,)
    #: Employees are first rated at the first cycle after this much tenure.
    min_tenure_months: float = 6.0
    #: First ratings, in scale order. New people cluster in the middle.
    entry_distribution: tuple[float, ...] = (0.04, 0.16, 0.60, 0.16, 0.04)
    #: transitions[i][j]: the chance a rating of scale[i] becomes scale[j] at the next cycle.
    transitions: tuple[tuple[float, ...], ...] = (
        (0.30, 0.40, 0.25, 0.05, 0.00),
        (0.08, 0.40, 0.42, 0.08, 0.02),
        (0.02, 0.10, 0.70, 0.15, 0.03),
        (0.01, 0.05, 0.34, 0.45, 0.15),
        (0.00, 0.03, 0.22, 0.35, 0.40),
    )

    def stationary(self) -> NDArray[np.float64]:
        """The chain's long-run distribution: the incumbents' ratings when the world starts."""
        matrix = np.asarray(self.transitions, dtype=np.float64)
        values, vectors = np.linalg.eig(matrix.T)
        vector = np.real(vectors[:, np.argmin(np.abs(values - 1))])
        return vector / vector.sum()

    def problems(self) -> list[str]:
        k = len(self.scale)
        found = []
        if len(set(self.scale)) != k or k < 2:
            found.append("ratings.scale needs at least two distinct labels")
        if not self.cycle_months or any(m not in range(1, 13) for m in self.cycle_months):
            found.append("ratings.cycle_months must be calendar months 1-12")
        matrix = np.asarray(self.transitions, dtype=np.float64)
        entry = np.asarray(self.entry_distribution, dtype=np.float64)
        if matrix.shape != (k, k) or (matrix < 0).any() or not np.allclose(matrix.sum(axis=1), 1):
            found.append("ratings.transitions must be a square stochastic matrix over the scale")
        if entry.shape != (k,) or (entry < 0).any() or not np.isclose(entry.sum(), 1):
            found.append("ratings.entry_distribution must be a distribution over the scale")
        return found


@dataclass(frozen=True)
class PromotionTruth:
    """Promotions: a latent readiness that accrues every month, and promotion cycles.

    Each month an employee's readiness grows by a non-negative draw whose mean depends on their
    rating. At a cycle, everyone whose readiness has reached `readiness_threshold` is promoted
    one level with their level's chance, and promotion resets readiness to zero. Some
    promotions also move the employee to another family. Invented, round values.
    """

    cycle_months: tuple[int, ...] = (3, 9)
    readiness_threshold: float = 36.0
    #: Mean monthly readiness by rating, worst to best, and for the unrated.
    readiness_drift: tuple[float, ...] = (0.2, 0.6, 1.0, 1.4, 1.9)
    unrated_drift: float = 0.8
    readiness_sigma: float = 0.5
    #: The chance a ready employee is promoted at a cycle, by current level. The top level of
    #: the world is never promoted.
    promotion_probability: dict[int, float] = field(
        default_factory=lambda: {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.12, 6: 0.1, 7: 0.08}
    )
    family_change_probability: float = 0.10

    def problems(self, rating_levels: int, levels: tuple[int, ...]) -> list[str]:
        found = []
        if not self.cycle_months or any(m not in range(1, 13) for m in self.cycle_months):
            found.append("promotions.cycle_months must be calendar months 1-12")
        if len(self.readiness_drift) != rating_levels:
            found.append("promotions.readiness_drift needs one value per rating")
        if missing := sorted(set(levels[:-1]) - set(self.promotion_probability)):
            found.append(f"promotions.promotion_probability has no value for levels {missing}")
        if not 0 <= self.family_change_probability <= 1:
            found.append("promotions.family_change_probability must lie in [0, 1]")
        if self.readiness_threshold <= 0:
            found.append("promotions.readiness_threshold must be positive")
        return found


@dataclass(frozen=True)
class MovementTruth:
    """Moves that aren't hires or departures: reclassifications, reorganizations, and the
    contingent pool with its conversions. Invented, round values.

    - A reclassification batch changes one family to another, in place, for a share of that
      family's people in one level-1 org.
    - A reorganization batch moves a share of one cost center's people to another cost center
      in the same level-1 org.
    - Contingent workers book to a contingent account and sit outside regular headcount. They
      start and end at their own rates. A regular employee may convert out to the pool, and an
      open requisition may fill by converting a contingent worker in.
    """

    reclass_probability: float = 0.03
    reclass_share: float = 0.30
    reorg_probability: float = 0.10
    reorg_share: tuple[float, float] = (0.3, 0.7)
    #: The pool's starting size, and its monthly starts, as shares of regular headcount.
    contingent_share: float = 0.08
    contingent_start_rate: float = 0.005
    contingent_end_hazard: float = 0.06
    conversion_out_hazard: float = 0.0008
    #: The monthly chance an open requisition fills by converting a contingent worker.
    conversion_fill_hazard: float = 0.03
    #: A converting worker shares the requisition's family and sits within this many levels.
    conversion_level_band: int = 1

    def problems(self) -> list[str]:
        found = []
        for name in (
            "reclass_probability",
            "reclass_share",
            "reorg_probability",
            "contingent_share",
            "contingent_start_rate",
            "contingent_end_hazard",
            "conversion_out_hazard",
            "conversion_fill_hazard",
        ):
            if not 0 <= getattr(self, name) <= 1:
                found.append(f"movements.{name} must lie in [0, 1]")
        low, high = self.reorg_share
        if not 0 < low <= high <= 1:
            found.append("movements.reorg_share must be a range inside (0, 1]")
        return found


@dataclass(frozen=True)
class ProgramTruth:
    """Termination programs: planned reductions in one level-1 org over a multi-month window.

    A program starts in a month with `start_probability`, in an org that isn't already in
    one. It removes a share of the org's regular headcount at the start, spread evenly over
    its window, choosing people with weights by rating. Program terminations are their own
    transaction (`RIF`), run before the month's sampled departures, and are never backfilled.
    Invented, round values.
    """

    start_probability: float = 0.025
    window_months: tuple[int, int] = (2, 4)
    share: tuple[float, float] = (0.05, 0.15)
    #: Selection weights by rating, worst to best, and for the unrated.
    rating_weight: tuple[float, ...] = (4.0, 2.0, 1.0, 0.6, 0.3)
    unrated_weight: float = 1.0

    def problems(self, rating_levels: int) -> list[str]:
        found = []
        if not 0 <= self.start_probability <= 1:
            found.append("programs.start_probability must lie in [0, 1]")
        low, high = self.window_months
        if not 1 <= low <= high:
            found.append("programs.window_months must be a range of at least one month")
        low_s, high_s = self.share
        if not 0 < low_s <= high_s <= 1:
            found.append("programs.share must be a range inside (0, 1]")
        if len(self.rating_weight) != rating_levels or min(self.rating_weight) <= 0:
            found.append("programs.rating_weight needs one positive weight per rating")
        return found


@dataclass(frozen=True)
class DateTruth:
    """How events fall within their month. Invented, round values.

    Hires start on `start_weekday` (Monday is 0). Departures fall on business days, with the
    month's last business day `exit_month_end_weight` times as likely as any other. Every
    other event falls on any business day. When no day of the profile is late enough, an
    event falls on any day from its earliest one.
    """

    start_weekday: int = 0
    exit_month_end_weight: float = 8.0

    def problems(self) -> list[str]:
        found = []
        if self.start_weekday not in range(5):
            found.append("dates.start_weekday must be a business day, 0 (Monday) to 4")
        if self.exit_month_end_weight <= 0:
            found.append("dates.exit_month_end_weight must be positive")
        return found


@dataclass(frozen=True)
class GrowthTruth:
    """Growth requisitions, driven by a latent plan per cost center. Invented, round values.

    Each cost center plans a headcount path from its starting headcount. It grows at an annual
    rate drawn per cost center, revised each January by a random-walk step. Each month a cost
    center opens growth requisitions with a negative-binomial count whose mean is a share of
    its gap to plan (the target less committed strength, headcount plus open openings), times
    a seasonal factor. A cost center at or above plan opens none.
    """

    plan_growth_mean: float = 0.05
    plan_growth_sigma: float = 0.08
    plan_revision_sigma: float = 0.04
    gap_closing_rate: float = 0.15
    #: Negative-binomial shape: the count's variance is mean + mean^2 / dispersion.
    dispersion: float = 2.0
    seasonal_amplitude: float = 0.3
    seasonal_peak_month: int = 1

    def problems(self) -> list[str]:
        found = []
        if not 0 < self.gap_closing_rate <= 1:
            found.append("growth.gap_closing_rate must lie in (0, 1]")
        if self.dispersion <= 0:
            found.append("growth.dispersion must be positive")
        if not 0 <= self.seasonal_amplitude < 1:
            found.append("growth.seasonal_amplitude must lie in [0, 1)")
        if self.seasonal_peak_month not in range(1, 13):
            found.append("growth.seasonal_peak_month must be a calendar month 1-12")
        if min(self.plan_growth_sigma, self.plan_revision_sigma) < 0:
            found.append("growth plan spreads must be non-negative")
        return found


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
    #: Depth of the org hierarchy, counting the leaf. A cost center may sit shallower, and then
    #: its own label pads every level below it.
    org_levels: int = 2
    #: Children per internal org node, and the chance that a node below level 1 holds cost
    #: centers directly instead of splitting further: the source of uneven depth.
    org_branching: int = 3
    org_stop_probability: float = 0.25
    #: Share of level-1 orgs that are field orgs, and of cost centers with variable cost.
    field_org_share: float = 0.30
    variable_cost_share: float = 0.20
    families: tuple[str, ...] = tuple(FAMILIES)[:4]
    levels: tuple[int, ...] = (1, 2, 3, 4, 5)
    #: Share of the initial population at each level, in `levels` order.
    level_mix: tuple[float, ...] = (0.30, 0.30, 0.20, 0.15, 0.05)
    #: Share of a cost center's people in its dominant family.
    dominant_family_share: float = 0.70
    #: Levels at and above this one are executive.
    executive_min_level: int = 7
    locations: tuple[str, ...] = ("LOC-1",)
    #: The location tree: locations dealt to countries, and countries to regions.
    locations_per_country: int = 3
    countries_per_region: int = 3
    #: The first company is the primary one. The last `offline_subsidiaries` companies are
    #: offline subsidiaries, which roll up to their own planning company.
    companies: tuple[str, ...] = ("CO-1",)
    offline_subsidiaries: int = 0
    #: Share of roles in the primary company. The rest spread evenly over the subsidiaries.
    primary_company_share: float = 0.80
    #: Accounts and their groups. Employees book to the first `regular` account.
    accounts: tuple[tuple[str, str], ...] = (
        ("ACCT-100", "regular"),
        ("ACCT-200", "contingent"),
        ("ACCT-300", "temporary"),
    )
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
    ratings: RatingTruth = field(default_factory=RatingTruth)
    promotions: PromotionTruth = field(default_factory=PromotionTruth)
    movements: MovementTruth = field(default_factory=MovementTruth)
    programs: ProgramTruth = field(default_factory=ProgramTruth)
    dates: DateTruth = field(default_factory=DateTruth)
    growth: GrowthTruth = field(default_factory=GrowthTruth)

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
        if self.org_levels < 2:
            problems.append("org_levels must be at least 2: a level-1 org and the leaf")
        if self.org_branching < 2:
            problems.append("org_branching must be at least 2")
        for name in (
            "org_stop_probability",
            "field_org_share",
            "variable_cost_share",
            "primary_company_share",
        ):
            if not 0 <= getattr(self, name) <= 1:
                problems.append(f"{name} must lie in [0, 1]")
        if not 0 <= self.dominant_family_share <= 1:
            problems.append("dominant_family_share must lie in [0, 1]")
        if self.months < 1 or self.initial_employees < 1:
            problems.append("months and initial_employees must be positive")
        if not self.locations or not self.companies:
            problems.append("a world needs at least one location and one company")
        if not 0 <= self.offline_subsidiaries < len(self.companies):
            problems.append("offline_subsidiaries must leave the primary company online")
        if self.locations_per_country < 1 or self.countries_per_region < 1:
            problems.append("locations_per_country and countries_per_region must be positive")
        groups = [group for _, group in self.accounts]
        if "regular" not in groups or not set(groups) <= {"regular", "contingent", "temporary"}:
            problems.append(
                "accounts need a regular account, and groups regular, contingent, or temporary"
            )
        if len({name for name, _ in self.accounts}) != len(self.accounts):
            problems.append("account names must be unique")
        unknown = [f for f in self.families if f not in FAMILIES or f not in FAMILY_CODES]
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
        problems += self.ratings.problems()
        problems += self.attrition.problems(len(self.ratings.scale))
        problems += self.promotions.problems(len(self.ratings.scale), self.levels)
        problems += self.movements.problems()
        problems += self.programs.problems(len(self.ratings.scale))
        problems += self.dates.problems()
        problems += self.growth.problems()
        if "contingent" not in [group for _, group in self.accounts]:
            problems.append("accounts need a contingent account for the contingent pool")
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
        "org_levels": 4,
        "families": tuple(FAMILIES)[:8],
        "levels": (1, 2, 3, 4, 5, 6, 7),
        "level_mix": (0.25, 0.27, 0.22, 0.14, 0.08, 0.03, 0.01),
        "locations": _labels("LOC", 4),
        "companies": _labels("CO", 3),
        "offline_subsidiaries": 1,
    },
    "medium": {
        "initial_employees": 50_000,
        "cost_centers": 2_500,
        "org_level1_nodes": 10,
        "org_levels": 5,
        "families": tuple(FAMILIES)[:12],
        "levels": (1, 2, 3, 4, 5, 6, 7, 8),
        "level_mix": (0.24, 0.26, 0.22, 0.14, 0.08, 0.04, 0.015, 0.005),
        "locations": _labels("LOC", 10),
        "companies": _labels("CO", 4),
        "offline_subsidiaries": 1,
    },
    "large": {
        "initial_employees": 500_000,
        "cost_centers": 25_000,
        "org_level1_nodes": 20,
        "org_levels": 6,
        "families": tuple(FAMILIES),
        "levels": (1, 2, 3, 4, 5, 6, 7, 8),
        "level_mix": (0.24, 0.26, 0.22, 0.14, 0.08, 0.04, 0.015, 0.005),
        "locations": _labels("LOC", 20),
        "companies": _labels("CO", 6),
        "offline_subsidiaries": 2,
    },
}


def preset(name: str, seed: int = 0) -> WorldConfig:
    try:
        overrides = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; known: {', '.join(PRESETS)}") from None
    return WorldConfig(seed=seed, preset=name, **overrides)
