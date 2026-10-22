"""The guidance configuration: what a pipeline run forecasts, and with which components.

A configuration is read from YAML, or built in code, and frozen. Unknown keys are refused, so a
misspelled field fails loudly instead of running its default. Loading validates the whole
configuration, collecting every problem before it refuses:

- each field's type and range;
- every component it names, against its registry (see `registry`): the models, the combination
  method, and the reconciliation method. An unknown name lists the known ones, and a name from
  an uninstalled extra gives the command that installs it;
- the grouping, against the column catalog;
- the categories: each once, and the headcount anchor among them when reconciling.

A configuration that loads names only components the pipeline can build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hcsim.guidance.evaluation import EvaluationConfig, space_of
from hcsim.guidance.ingest import DEFAULT_CATEGORIES, grouping_problems
from hcsim.guidance.models import MODELS
from hcsim.guidance.reconcile import RECONCILERS
from hcsim.guidance.selection import COMBINERS, SelectionConfig
from hcsim.guidance.tuning import TuningSettings
from hcsim.model.core.categories import ANCHOR, GuidanceCategory


class GuidanceConfigError(ValueError):
    """A configuration that can't be read or doesn't validate."""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Evaluation(_Strict):
    """Rolling-origin backtesting: how many folds, how long, how far apart, and the metric a
    model's score averages (see `evaluation`)."""

    n_splits: int = Field(default=5, ge=1)
    test_months: int = Field(default=6, ge=1)
    step_months: int = Field(default=2, ge=1)
    min_train_months: int = Field(default=36, ge=1)
    metric: Literal["wape", "mape"] = "wape"


class Tuning(_Strict):
    """Nested TPE tuning of the models that declare a search space (see `tuning`). Off by
    default, which leaves every model its defaults."""

    enabled: bool = False
    trials: int = Field(default=20, ge=1)
    seed: int = Field(default=0, ge=0)
    inner_splits: int = Field(default=2, ge=1)
    inner_min_train_months: int = Field(default=24, ge=1)


class Combination(_Strict):
    """How the scored models combine into one forecast."""

    method: str = "inverse_error"
    top_k: int = Field(default=2, ge=1)


class Reconciliation(_Strict):
    """How each group's flows are reconciled to its anchor. No method skips the step."""

    method: str | None = "greedy"
    tolerance: float = Field(default=1.0, ge=0.0)


class Validation(_Strict):
    """How the series are settled before fitting (see `validation`)."""

    #: The last month of the history, as `YYYY-MM`. None applies the snapshot rule.
    snapshot: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    #: What a month with no rows at all inside the history is: refused, or zeros.
    gaps: Literal["refuse", "zero"] = "refuse"
    #: Whether a series that's zero throughout is dropped rather than forecast.
    drop_all_zero: bool = True


class GuidanceConfig(_Strict):
    horizon: int = Field(default=12, ge=1)
    grouping: str = "cost_center"
    #: The population's stamp, `name@digest`, or a built-in population's name.
    population: str | None = None
    #: The categories to forecast. An inner fill-tree node is composed from its leaves.
    categories: tuple[GuidanceCategory, ...] = Field(default=DEFAULT_CATEGORIES, min_length=1)
    models: tuple[str, ...] = Field(
        default=("native.SeasonalNaive", "native.HistoricalAverage"), min_length=1
    )
    combination: Combination = Combination()
    evaluation: Evaluation = Evaluation()
    validation: Validation = Validation()
    tuning: Tuning = Tuning()
    reconciliation: Reconciliation = Reconciliation()

    @model_validator(mode="after")
    def _registered(self) -> GuidanceConfig:
        problems = self.problems()
        if problems:
            raise ValueError("\n".join(problems))
        return self

    def problems(self) -> list[str]:
        """Every component the configuration names that can't be built, and the grouping's
        problems."""
        found = [f"models: {p}" for p in MODELS.problems(self.models)]
        repeated = sorted({m for m in self.models if self.models.count(m) > 1})
        if repeated:
            found.append(f"models: each model is named once; {repeated} repeat")
        found += [f"combination.method: {p}" for p in COMBINERS.problems([self.combination.method])]
        if self.reconciliation.method is not None:
            method = [self.reconciliation.method]
            found += [f"reconciliation.method: {p}" for p in RECONCILERS.problems(method)]
        found += [f"grouping: {p}" for p in grouping_problems(self.grouping)]
        repeated = sorted({c.value for c in self.categories if self.categories.count(c) > 1})
        if repeated:
            found.append(f"categories: each category is named once; {repeated} repeat")
        if self.tuning.enabled and not any(
            MODELS.problems([m]) == [] and space_of(m) for m in self.models
        ):
            found.append(
                "tuning: tuning is on, but no model in the suite has a search space; models "
                "that do include native.PoissonGLM and statsforecast.WindowAverage"
            )
        if self.reconcile and ANCHOR not in self.categories:
            found.append(
                f"categories: reconciliation holds the {ANCHOR.value!r} anchor fixed, so it "
                "needs it among the categories; add it, or set reconciliation.method to null"
            )
        return found

    @property
    def reconcile(self) -> bool:
        return self.reconciliation.method is not None

    @property
    def tolerance(self) -> float:
        return self.reconciliation.tolerance

    @property
    def selection(self) -> SelectionConfig:
        """The selection settings the forecasting step reads."""
        e = self.evaluation
        t = self.tuning
        return SelectionConfig(
            suite=self.models,
            top_k=self.combination.top_k,
            combination=self.combination.method,
            evaluation=EvaluationConfig(
                e.n_splits, e.test_months, e.step_months, e.min_train_months, e.metric
            ),
            tuning=(
                TuningSettings(t.trials, t.seed, t.inner_splits, t.inner_min_train_months)
                if t.enabled
                else None
            ),
        )

    @classmethod
    def from_mapping(cls, data: Any) -> GuidanceConfig:
        try:
            return cls.model_validate(data if data is not None else {})
        except ValidationError as error:
            problems = []
            for e in error.errors():
                where = ".".join(str(p) for p in e["loc"])
                message = str(e["msg"]).removeprefix("Value error, ")
                if where:
                    problems.append(f"{where}: {message}")
                else:  # the registry check, whose problems name their own fields
                    problems += message.split("\n")
            raise GuidanceConfigError(
                "invalid guidance configuration:\n  - " + "\n  - ".join(problems)
            ) from None

    @classmethod
    def load(cls, path: Path) -> GuidanceConfig:
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as error:
            raise GuidanceConfigError(f"can't read the configuration at {path}: {error}") from None
        return cls.from_mapping(data)
