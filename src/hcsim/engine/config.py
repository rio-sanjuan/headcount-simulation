"""The simulation configuration: one frozen dataclass that holds every parameter of a run.

Validation collects every problem before raising, so one run shows everything to fix. The
configuration is never changed during a run: it's frozen, and live objects inside it (the
schema, the guidance, the models) are treated as read-only.

**Settings and objects.** A field is either a *setting*, a scalar that loads in layers (see
`hcsim.engine.settings`), or an *object* passed in code: the schema, the models, the guidance,
and the award settings. Every setting belongs to one *field group*, listed in `GROUPS`, and
records it in its metadata. A setting's name is its group's name, an underscore, and its key,
so `promotion_cycle_months` is the key `cycle_months` of the group `promotion`. The `run`
group's settings go unprefixed. The configuration stays one flat record (ADR 0007), so the
whole tunable surface can be listed, and a group with no settings yet is where a later domain
puts its own.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

from hcsim.financial.expense import ExpenseConfig
from hcsim.financial.prices import StockPriceModel
from hcsim.financial.sizing import NewHireGapFill
from hcsim.model.core.aligner import GroupAligner
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.core.schema import COST_CENTER, SchemaError, SimulationSchema
from hcsim.model.interface import EntityModel, GroupRateModel
from hcsim.model.ml.generative import EmpiricalAttributeGenerator


class ConfigError(ValueError):
    """A configuration with one or more problems, all listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid simulation configuration:\n  - " + "\n  - ".join(problems))


class ForecastScope(StrEnum):
    HEADCOUNT = "hcf"
    HEADCOUNT_AND_SBC = "hcf+sbc"


class RunMode(StrEnum):
    """The run-mode ladder, from most pinned to least."""

    REPLAY = "replay"
    GUIDED_ACTUALS = "guided_actuals"
    GUIDED_FORECAST = "guided_forecast"
    UNGUIDED = "unguided"


class BackfillSource(StrEnum):
    """Where backfill's probability per departure comes from."""

    RATE = "rate"
    CLASSIFIER = "classifier"


#: Every field group, in the order a configuration lists them.
GROUPS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "run": "the run's start, length, size, seed, scope, and mode",
        "seeding": "how the seed state is read and how large its buffers are",
        "known_events": "anticipated movements, declared events, and their sanitizing",
        "gates": "registration gates and stochastic step gates",
        "performance": "rating cycles",
        "attrition": "departures",
        "backfill": "requisitions opened to replace departures and transfers out",
        "promotion": "promotion cycles",
        "pending_starts": "accepted offers that start during the run",
        "fills": "fill channels and their hazards",
        "transfers": "matching employees to transfer-assigned openings",
        "campus": "campus openings and starts",
        "cancellation": "requisitions that exit unfilled",
        "requisitions": "growth requisition generation",
        "replay": "replaying realized history",
        "results": "what a run records",
    }
)


def setting(group: str, default: Any, doc: str) -> Any:
    """A field that loads in layers, in `group`."""
    if group not in GROUPS:
        raise ValueError(f"no field group {group!r}")
    return field(default=default, metadata={"group": group, "doc": doc})


@dataclass(frozen=True)
class DomainModels:
    """The models behind the skeleton's domains."""

    attrition: EntityModel  # scores employee rows
    external_fill: EntityModel  # scores opening rows: a monthly fill hazard
    cancellation: EntityModel  # scores opening rows: a monthly unfilled-exit hazard
    backfill: GroupRateModel  # a rate per departure, by cost center code
    growth: GroupRateModel  # mean growth requisitions per month, by cost center code
    attributes: EmpiricalAttributeGenerator  # attributes of generated requisitions


@dataclass(frozen=True)
class AwardSettings:
    """Everything SBC needs. Required at `hcf+sbc` scope, refused at `hcf`."""

    new_hire: NewHireGapFill
    price_model: StockPriceModel
    expense: ExpenseConfig = field(default_factory=ExpenseConfig)


@dataclass(frozen=True)
class SimulationConfig:
    # run: its settings go unprefixed
    run_date: dt.date = field(
        metadata={"group": "run", "doc": "the first day of the run's first month"}
    )
    # objects, passed in code
    schema: SimulationSchema
    models: DomainModels
    # run
    simulation_months: int = setting("run", 12, "how many months the run simulates")
    num_trajectories: int = setting("run", 1, "how many independent trajectories it runs")
    seed: int | None = setting("run", None, "the root seed; none draws fresh entropy")
    forecast_scope: ForecastScope = setting(
        "run", ForecastScope.HEADCOUNT_AND_SBC, "headcount alone, or headcount and SBC"
    )
    run_mode: RunMode | None = setting(
        "run", None, "the run-mode rung; none infers guided_forecast from guidance, else unguided"
    )
    # seeding
    seeding_include_opening_buffer: bool = setting(
        "seeding", True, "seed open requisitions; without them a run opens every requisition"
    )
    seeding_max_openings: int = setting(
        "seeding", 10, "the most openings a seeded requisition keeps; more are clipped"
    )
    # performance
    performance_cycle_month: int = setting(
        "performance", 3, "the calendar month a rating cycle takes effect"
    )
    performance_first_rating_min_tenure_months: int = setting(
        "performance", 6, "the tenure an employee needs at a cycle to receive a first rating"
    )
    # backfill
    backfill_source: BackfillSource = setting(
        "backfill", BackfillSource.RATE, "a rate per cost center, or a classifier per departure"
    )
    # promotion
    promotion_cycle_months: tuple[int, ...] = setting(
        "promotion", (3, 9), "the calendar months promotions take effect"
    )
    # pending starts
    pending_starts_staleness_days: int = setting(
        "pending_starts", 90, "how long past its start date an unstarted offer is still expected"
    )
    # transfers
    transfers_availability_factor: float = setting(
        "transfers", 1.0, "the share of each employee cell available to transfer"
    )
    # replay
    replay_boundary_fill: bool = setting(
        "replay", True, "inject boundary fills for hires replay can't trace to a requisition"
    )
    # results
    results_snapshot_steps: tuple[int, ...] = setting(
        "results", (), "the steps whose buffers a run keeps as snapshots"
    )
    results_streaming: bool = setting(
        "results", False, "stream each trajectory's results out instead of holding them"
    )
    # objects, passed in code
    guidance: GuidanceForecast | None = None
    awards: AwardSettings | None = None
    aligner: GroupAligner | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.run_date.day != 1:
            problems.append(f"run_date must be the first of a month, got {self.run_date}")
        if self.simulation_months < 1:
            problems.append("simulation_months must be at least 1")
        if self.num_trajectories < 1:
            problems.append("num_trajectories must be at least 1")
        scope = ForecastScope(self.forecast_scope)
        if scope is ForecastScope.HEADCOUNT_AND_SBC and self.awards is None:
            problems.append("forecast_scope 'hcf+sbc' needs awards settings")
        if scope is ForecastScope.HEADCOUNT and self.awards is not None:
            problems.append("forecast_scope 'hcf' computes no SBC, so it refuses awards settings")
        if self.guidance is not None:
            problems += self._check_guidance(self.guidance)
        if problems:
            raise ConfigError(problems)
        object.__setattr__(self, "forecast_scope", scope)
        mode = self.run_mode
        if mode is None:
            mode = RunMode.UNGUIDED if self.guidance is None else RunMode.GUIDED_FORECAST
        object.__setattr__(self, "run_mode", RunMode(mode))

    def _check_guidance(self, guidance: GuidanceForecast) -> list[str]:
        problems = []
        if guidance.horizon < self.simulation_months:
            problems.append(
                f"guidance covers {guidance.horizon} months; the run needs {self.simulation_months}"
            )
        expected_origin = (self.run_date - dt.timedelta(days=1)).isoformat()
        if guidance.snapshot_date != expected_origin:
            problems.append(
                f"guidance was fitted through {guidance.snapshot_date}; a run from "
                f"{self.run_date} needs a snapshot of {expected_origin}"
            )
        if (
            GuidanceCategory.REQUISITIONS_OPENED in guidance.categories
            and guidance.grouping_column != COST_CENTER
        ):
            problems.append("requisition generation reads guidance only at cost-center grain")
        try:
            object.__setattr__(self, "aligner", guidance.aligner(self.schema))
        except SchemaError as error:
            problems.append(f"guidance doesn't fit the schema: {error}")
        return problems

    @property
    def computes_sbc(self) -> bool:
        return self.forecast_scope is ForecastScope.HEADCOUNT_AND_SBC
