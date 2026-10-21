"""The simulation configuration: one frozen dataclass that holds every parameter of a run.

Validation collects every problem before raising, so one run shows everything to fix. The
configuration is never changed during a run: it's frozen, and live objects inside it (the
schema, the guidance, the models) are treated as read-only.

This is the walking-skeleton configuration. Layered loading from YAML and the environment, and
serialization, arrive in M4.1.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum

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
    run_date: dt.date
    schema: SimulationSchema
    models: DomainModels
    simulation_months: int = 12
    num_trajectories: int = 1
    seed: int | None = None
    forecast_scope: ForecastScope = ForecastScope.HEADCOUNT_AND_SBC
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
