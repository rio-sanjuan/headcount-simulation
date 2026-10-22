"""Training: a temporal split at a cutoff, and one driver that trains every arm.

**The split.** `TemporalSplit(start, cutoff, end)` trains on the outcomes in (start, cutoff]
and holds out those in (cutoff, end], cut as `Extract.before` and `Extract.after` cut them, so
no outcome falls on both sides. `TemporalSplit.holdout` holds out a window's last months.

**The design.** A classifier reads one float matrix of an extract's features, built by a
`Design` fitted on the training side:

- a column with a schema vocabulary is encoded through it, an unknown label as -1. It's
  categorical, except job level, which is ordinal;
- the calendar month ahead is categorical, as codes 0 to 11;
- any other text column is categorical through a vocabulary of the training side's labels;
- a flag is 0 or 1, and a number stays as it is.

Arms that one-hot encode leave out categorical columns wider than `LOGISTIC_MAX_LEVELS`, and
boosted arms those wider than a booster can hold, so a large cost-center vocabulary stays with
the arms that pool it.

**The driver.** `train` takes each domain of the arm matrix in turn:

1. extract it over the split's window, and cut the extract at the cutoff;
2. fit every arm on the training side;
3. score every arm on the held-out side, as per-row contributions to the family's primary
   metric and as the family's report metrics;
4. choose the champion with `select`;
5. with a root, save every arm with `save_model`, and write `training_run.json` there, naming
   each domain's arms, versions, decisions, and champion.

A generator is scored by the probability it gives each held-out outcome, and a count model by
its mean for each held-out cost-center month. Held-out rows whose outcome the categories don't
carry are scored by no arm, so every arm is scored on the same rows.
"""

from __future__ import annotations

import datetime as dt
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, cast

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.model.core.encoding import encode_frame, vocabulary_for
from hcsim.model.core.schema import (
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    UNATTRIBUTED,
    SchemaError,
    SimulationSchema,
)
from hcsim.model.data.extracts import ROLE, Extract, Grain, extracts
from hcsim.model.data.population import PopulationSpec
from hcsim.model.interface import EntityModel
from hcsim.model.ml.arms import DEFAULT_MATRIX, DEFAULT_Z, Arm, ArmMatrix, Selection, select
from hcsim.model.ml.arms import contributions as contributions_of
from hcsim.model.ml.classifiers import (
    MAX_CATEGORIES,
    BoostedModel,
    FeatureMask,
    LevelOnlyModel,
    LogisticModel,
)
from hcsim.model.ml.entity import ConstantModel, NaiveRateModel
from hcsim.model.ml.generative import ConditionalCategorical, EmpiricalAttributeGenerator
from hcsim.model.ml.group import (
    BackfillRateModel,
    NegativeBinomialGrowthModel,
    PoissonGrowthModel,
)
from hcsim.model.ml.loader import Probe
from hcsim.model.ml.markov import MarkovModel
from hcsim.model.ml.metrics import FAMILY_METRICS, METRICS, MetricError
from hcsim.model.ml.persistence import save_model

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

RUN_FILE: Final = "training_run.json"
#: The widest categorical column an arm that one-hot encodes reads.
LOGISTIC_MAX_LEVELS: Final = 50
#: Columns that hold numbers but read as categories, with their first value.
_CATEGORICAL_NUMBERS: Final = {"next_month": 1}
#: Generator defaults: thin cells borrow from above. Tuning can move them.
GENERATOR_MIN_COUNT: Final = 20
GENERATOR_SMOOTHING: Final = 1.0
GROWTH_MIN_MONTHS: Final = 6
#: Held-out rows a domain's probe carries.
PROBE_ROWS: Final = 20


class TrainingError(ValueError):
    """A split or a domain that can't be trained."""


@dataclass(frozen=True)
class TemporalSplit:
    start: dt.date
    cutoff: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        if not self.start < self.cutoff < self.end:
            raise TrainingError(
                f"a split needs start < cutoff < end, not {self.start}, {self.cutoff}, {self.end}"
            )

    @classmethod
    def holdout(cls, start: dt.date, end: dt.date, months: int = 12) -> TemporalSplit:
        """Hold out the window's last `months`."""
        if months < 1:
            raise TrainingError("a holdout needs at least one month")
        cutoff = (pd.Timestamp(end) - pd.offsets.MonthEnd(months)).date()
        return cls(start, cutoff, end)

    def cut(self, extract: Extract) -> tuple[Extract, Extract]:
        return extract.before(self.cutoff), extract.after(self.cutoff)

    def to_dict(self) -> dict[str, str]:
        return {k: getattr(self, k).isoformat() for k in ("start", "cutoff", "end")}


# --- the design --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """How an extract's features become one float matrix, fitted on the training side."""

    columns: tuple[str, ...]
    #: Each schema-encoded column's vocabulary.
    vocabularies: Mapping[str, str]
    #: Each other text column's labels, in code order.
    local: Mapping[str, tuple[str, ...]]
    #: Each categorical column's number of codes.
    levels: Mapping[str, int]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str], schema: SimulationSchema) -> Design:
        vocabularies, local, levels = {}, {}, {}
        for column in columns:
            values = frame[column]
            try:
                vocabulary = vocabulary_for(schema, column)
            except SchemaError:
                vocabulary = None
            if vocabulary is not None:
                vocabularies[column] = vocabulary
                if vocabulary != JOB_LEVEL:
                    levels[column] = len(schema.vocabulary(vocabulary))
            elif column in _CATEGORICAL_NUMBERS:
                levels[column] = 12
            elif values.dtype == object or isinstance(values.dtype, pd.StringDtype):
                labels = tuple(sorted({str(v) for v in values.dropna()}))
                local[column] = labels
                levels[column] = len(labels)
        return cls(tuple(columns), vocabularies, local, levels)

    @property
    def categorical(self) -> tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.columns) if c in self.levels)

    def mask(self, max_levels: int) -> FeatureMask:
        """Every column but the categorical ones with more than `max_levels` codes."""
        kept = [
            i
            for i, c in enumerate(self.columns)
            if c not in self.levels or self.levels[c] <= max_levels
        ]
        return FeatureMask(tuple(kept), tuple(i for i in self.categorical if i in kept))

    def matrix(self, frame: pd.DataFrame, schema: SimulationSchema) -> Floats:
        """The features as floats. A label a vocabulary lacks becomes -1, and the schema's
        gaps are warned of."""
        encoded = encode_frame(schema, frame, list(self.vocabularies)).codes
        out = np.empty((len(frame), len(self.columns)), dtype=np.float64)
        for i, column in enumerate(self.columns):
            values = frame[column]
            if column in self.vocabularies:
                out[:, i] = encoded[column]
            elif column in _CATEGORICAL_NUMBERS:
                out[:, i] = values.to_numpy(dtype=np.float64) - _CATEGORICAL_NUMBERS[column]
            elif column in self.local:
                code = {label: k for k, label in enumerate(self.local[column])}
                out[:, i] = [code.get(str(v), UNATTRIBUTED) for v in values]
            else:
                out[:, i] = values.to_numpy(dtype=np.float64)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "categorical": [self.columns[i] for i in self.categorical],
            "local": {k: list(v) for k, v in self.local.items()},
        }


# --- tasks: one domain's training and held-out data, in a family's shape -----------------------


@dataclass
class _Fitted:
    model: object
    parameters: dict[str, Any]


class _Task(ABC):
    """One domain's training and held-out data, in its family's shape. `FITS` maps each
    algorithm the family allows to the method that fits it."""

    family: ClassVar[str]
    FITS: ClassVar[Mapping[str, str]]
    train_rows: int
    test_rows: int

    def fit(self, arm: Arm) -> _Fitted:
        fitted: _Fitted = getattr(self, self.FITS[arm.algorithm])(arm)
        return fitted

    @abstractmethod
    def contributions(self, model: object) -> Floats:
        """Each held-out row's part of the family's primary metric."""

    @abstractmethod
    def metrics(self, model: object) -> dict[str, float | None]:
        """The family's report metrics on the held-out side."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """What the task reads, for the artifact's config."""

    @abstractmethod
    def probe(self) -> Probe:
        """A few held-out inputs, for the loader to score a saved model on."""


def _safely(compute: Callable[[], float]) -> float | None:
    try:
        value = float(compute())
    except MetricError:
        return None
    return value if np.isfinite(value) else None


class _ClassifierTask(_Task):
    family = "classifier"

    def __init__(self, train: Extract, test: Extract, schema: SimulationSchema) -> None:
        self.design = Design.fit(train.frame, train.features, schema)
        self.x = self.design.matrix(train.frame, schema)
        self.y = train.labels.astype(np.float64)
        self.test_x = self.design.matrix(test.frame, schema)
        self.test_y = test.labels.astype(np.float64)
        self.train_rows, self.test_rows = len(self.y), len(self.test_y)
        self.n_cost_centers = len(schema.vocabulary(COST_CENTER))
        if self.train_rows == 0 or self.test_rows == 0:
            raise TrainingError(f"{train.domain} needs rows on both sides of the cutoff")

    FITS = MappingProxyType(
        {
            "constant": "_constant",
            "level_only": "_level_only",
            "naive_rate": "_naive_rate",
            "backfill_rate": "_backfill_rate",
            "logistic": "_logistic",
            "boosted": "_boosted",
        }
    )

    def _constant(self, arm: Arm) -> _Fitted:
        return _Fitted(ConstantModel(float(self.y.mean())), {})

    def _level_only(self, arm: Arm) -> _Fitted:
        level = self.design.columns.index(JOB_LEVEL)
        return _Fitted(LevelOnlyModel(level).fit(self.x, self.y), {"level_column": level})

    def _naive_rate(self, arm: Arm) -> _Fitted:
        keys = [self.design.columns.index(c) for c in (JOB_LEVEL, COST_CENTER)]
        return _Fitted(NaiveRateModel(keys).fit(self.x, self.y), {"key_columns": keys})

    def _backfill_rate(self, arm: Arm) -> _Fitted:
        group = self.design.columns.index(COST_CENTER)
        model = BackfillRateModel(self.n_cost_centers, group_column=group)
        model.fit(self.x[:, group].astype(np.int64), self.y.astype(bool))
        return _Fitted(model, {"group_column": group})

    def _logistic(self, arm: Arm) -> _Fitted:
        mask = self.design.mask(LOGISTIC_MAX_LEVELS)
        return _Fitted(LogisticModel(mask).fit(self.x, self.y), {"columns": list(mask.columns)})

    def _boosted(self, arm: Arm) -> _Fitted:
        mask = self.design.mask(MAX_CATEGORIES - 1)
        booster = BoostedModel(mask).fit(self.x, self.y)
        return _Fitted(booster, {"columns": list(mask.columns), **booster.params})

    def _scores(self, model: object) -> Floats:
        result: Floats = np.asarray(
            cast(EntityModel, model).compute_probability(self.test_x), dtype=np.float64
        )
        return result

    def contributions(self, model: object) -> Floats:
        return contributions_of("classifier", labels=self.test_y, probabilities=self._scores(model))

    def metrics(self, model: object) -> dict[str, float | None]:
        p = self._scores(model)
        return {
            name: _safely(partial(METRICS[name].compute, self.test_y, p))
            for name in FAMILY_METRICS["classifier"]
        }

    def describe(self) -> dict[str, Any]:
        return {"features": self.design.to_dict()}

    def probe(self) -> Probe:
        return Probe(features=self.test_x[:PROBE_ROWS])


class _GeneratorTask(_Task):
    """Outcomes given contexts, scored by each held-out outcome's probability."""

    family = "generator"

    def __init__(
        self,
        n_categories: int,
        contexts: tuple[Ints, Ints],
        outcomes: tuple[Ints, Ints],
        *,
        context_columns: Sequence[str],
        roles: tuple[Ints, Ints] | None = None,
        ordered: bool = False,
    ) -> None:
        self.n_categories = n_categories
        self.ordered = ordered
        keep = [o >= 0 for o in outcomes]
        self.contexts = [c[k] for c, k in zip(contexts, keep, strict=True)]
        self.outcomes = [o[k] for o, k in zip(outcomes, keep, strict=True)]
        self.roles = None if roles is None else [r[k] for r, k in zip(roles, keep, strict=True)]
        self.context_columns = tuple(context_columns)
        self.train_rows, self.test_rows = (len(o) for o in self.outcomes)
        if self.train_rows == 0 or self.test_rows == 0:
            raise TrainingError("a generator needs outcomes on both sides of the cutoff")

    FITS = MappingProxyType(
        {
            "empirical_attributes": "_empirical",
            "conditional_categorical": "_conditional",
            "markov": "_markov",
        }
    )

    def _empirical(self, arm: Arm) -> _Fitted:
        if self.roles is None:
            raise TrainingError(f"{arm.name} draws whole roles, which this domain doesn't have")
        context, roles = self.contexts[0], self.roles[0]
        complete = (roles >= 0).all(axis=1) & (context[:, 0] >= 0)
        model = EmpiricalAttributeGenerator().fit(context[complete, 0], roles[complete])
        return _Fitted(model, {"min_rows": model.min_rows})

    def _conditional(self, arm: Arm) -> _Fitted:
        """A null arm reads no context: the overall distribution."""
        columns = [] if arm.null else list(self.context_columns)
        levels = self.contexts[0][:, : len(columns)]
        model = ConditionalCategorical(
            self.n_categories, min_count=GENERATOR_MIN_COUNT, smoothing=GENERATOR_SMOOTHING
        ).fit(levels, self.outcomes[0])
        parameters = {
            "min_count": GENERATOR_MIN_COUNT,
            "smoothing": GENERATOR_SMOOTHING,
            "context": columns,
        }
        return _Fitted(model, parameters)

    def _markov(self, arm: Arm) -> _Fitted:
        previous, outcome = self.contexts[0][:, 0], self.outcomes[0]
        moved = previous >= 0
        model = MarkovModel(self.n_categories, smoothing=GENERATOR_SMOOTHING).fit(
            previous[moved], outcome[moved], outcome[~moved]
        )
        return _Fitted(model, {"smoothing": GENERATOR_SMOOTHING})

    def probabilities(self, model: object) -> Floats:
        """(held-out rows, categories)."""
        context = self.contexts[1]
        if isinstance(model, EmpiricalAttributeGenerator):
            column = ROLE.index(JOB_FAMILY)
            return (
                np.stack(
                    [model.distribution(int(g), column, self.n_categories) for g in context[:, 0]]
                )
                if len(context)
                else np.zeros((0, self.n_categories))
            )
        if isinstance(model, MarkovModel):
            previous = context[:, 0]
            out: Floats = np.where(
                (previous >= 0)[:, None],
                model.transitions[np.maximum(previous, 0)],
                model.entry[None, :],
            )
            return out
        categorical = cast(ConditionalCategorical, model)
        levels = context[:, : categorical.n_levels]
        unique, inverse = np.unique(levels, axis=0, return_inverse=True)
        table = np.stack([categorical.distribution(row) for row in unique])
        result: Floats = table[inverse.reshape(-1)]
        return result

    def contributions(self, model: object) -> Floats:
        return contributions_of(
            "generator", probabilities=self.probabilities(model), outcomes=self.outcomes[1]
        )

    def metrics(self, model: object) -> dict[str, float | None]:
        p = self.probabilities(model)
        observed = np.bincount(self.outcomes[1], minlength=self.n_categories) / self.test_rows
        expected = p.mean(axis=0)
        expected = expected / expected.sum()
        metrics = {
            "log_likelihood": _safely(
                partial(METRICS["log_likelihood"].compute, p, self.outcomes[1])
            ),
            "jensen_shannon": _safely(
                partial(METRICS["jensen_shannon"].compute, expected, observed)
            ),
        }
        if self.ordered:  # an earth-mover distance needs categories in order
            metrics["wasserstein_1"] = _safely(
                partial(METRICS["wasserstein_1"].compute, expected, observed)
            )
        return metrics

    def describe(self) -> dict[str, Any]:
        return {"context": list(self.context_columns), "categories": self.n_categories}

    def probe(self) -> Probe:
        return Probe(contexts=self.contexts[1][:PROBE_ROWS], n_categories=self.n_categories)


class _CountTask(_Task):
    """Counts per group and month, scored by each held-out cell's mean."""

    family = "count"

    def __init__(self, train: Extract, test: Extract, schema: SimulationSchema) -> None:
        self.n_groups = len(schema.vocabulary(COST_CENTER))
        self.train_counts, self.train_seen = self._grid(train, schema)
        test_counts, test_seen = self._grid(test, schema)
        self.actual = test_counts[test_seen]
        self.cells = np.nonzero(test_seen)[1]  # each held-out cell's group
        self.train_rows, self.test_rows = int(self.train_seen.sum()), len(self.actual)
        if self.train_rows == 0 or self.test_rows == 0:
            raise TrainingError(f"{train.domain} needs observed months on both sides of the cutoff")

    def _grid(self, extract: Extract, schema: SimulationSchema) -> tuple[Floats, NDArray[np.bool_]]:
        frame = extract.frame
        groups = schema.encode(COST_CENTER, frame["cost_center"], unknown="unattributed")
        months = pd.to_datetime(frame["month_end"]).dt.to_period("M")
        first = pd.Period(extract.start, freq="M") + 1
        span = (pd.Period(extract.end, freq="M") - first).n + 1
        index = np.array([(m - first).n for m in months], dtype=np.int64)
        counts = np.zeros((span, self.n_groups))
        seen = np.zeros((span, self.n_groups), dtype=bool)
        known = groups >= 0
        counts[index[known], groups[known]] = frame[extract.label].to_numpy()[known]
        seen[index[known], groups[known]] = True
        return counts, seen

    FITS = MappingProxyType({"poisson_growth": "_poisson", "negative_binomial_growth": "_nb"})

    def _poisson(self, arm: Arm) -> _Fitted:
        model = PoissonGrowthModel(self.n_groups, min_months=GROWTH_MIN_MONTHS)
        return _Fitted(
            model.fit(self.train_counts, self.train_seen), {"min_months": GROWTH_MIN_MONTHS}
        )

    def _nb(self, arm: Arm) -> _Fitted:
        model = NegativeBinomialGrowthModel(self.n_groups, min_months=GROWTH_MIN_MONTHS)
        return _Fitted(
            model.fit(self.train_counts, self.train_seen), {"min_months": GROWTH_MIN_MONTHS}
        )

    def _forecast(self, model: object) -> Floats:
        return cast(PoissonGrowthModel, model).rates(self.cells)

    def contributions(self, model: object) -> Floats:
        return contributions_of("count", actual=self.actual, forecast=self._forecast(model))

    def metrics(self, model: object) -> dict[str, float | None]:
        forecast = self._forecast(model)
        return {"wape": _safely(lambda: METRICS["wape"].compute(self.actual, forecast))}

    def describe(self) -> dict[str, Any]:
        return {"groups": COST_CENTER}

    def probe(self) -> Probe:
        return Probe(groups=self.cells[:PROBE_ROWS])


def _role_task(train: Extract, test: Extract, schema: SimulationSchema) -> _GeneratorTask:
    def codes(extract: Extract, columns: Sequence[str]) -> Ints:
        frame = extract.frame
        out = np.empty((len(frame), len(columns)), dtype=np.int64)
        for i, column in enumerate(columns):
            if column == "is_manager":
                out[:, i] = frame[column].to_numpy(dtype=np.int64)
            else:
                out[:, i] = schema.encode(
                    vocabulary_for(schema, column), frame[column], unknown="unattributed"
                )
        return out

    context = train.features
    return _GeneratorTask(
        len(schema.vocabulary(JOB_FAMILY)),
        (codes(train, context), codes(test, context)),
        (codes(train, [JOB_FAMILY])[:, 0], codes(test, [JOB_FAMILY])[:, 0]),
        context_columns=context,
        roles=(codes(train, ROLE), codes(test, ROLE)),
    )


def _rating_task(train: Extract, test: Extract) -> _GeneratorTask:
    """Rating states: each scale's ranks in turn, the original scale first."""
    frames = (train.frame, test.frame)
    scales = sorted(
        {s for f in frames for s in f["rating_scale"].dropna()},
        key=lambda s: (s != "original", s),
    )
    sizes = {
        s: max(
            int(f.loc[f["rating_scale"] == s, "rating_rank"].max())
            for f in frames
            if (f["rating_scale"] == s).any()
        )
        for s in scales
    }
    offsets = dict(zip(scales, np.cumsum([0] + [sizes[s] for s in scales])[:-1], strict=True))

    def states(scale: pd.Series, rank: pd.Series) -> Ints:
        known = scale.isin(list(offsets))
        out = np.full(len(scale), UNATTRIBUTED, dtype=np.int64)
        out[known.to_numpy()] = (
            scale[known].map(offsets).to_numpy(dtype=np.int64)
            + rank[known].to_numpy(dtype=np.int64)
            - 1
        )
        return out

    def sides(columns: tuple[str, str]) -> tuple[Ints, Ints]:
        return (
            states(train.frame[columns[0]], train.frame[columns[1]]),
            states(test.frame[columns[0]], test.frame[columns[1]]),
        )

    previous = sides(("previous_scale", "previous_rank"))
    return _GeneratorTask(
        sum(sizes.values()),
        (previous[0][:, None], previous[1][:, None]),
        sides(("rating_scale", "rating_rank")),
        context_columns=("previous_state",),
        ordered=True,
    )


def _task(train: Extract, test: Extract, family: str, schema: SimulationSchema) -> _Task:
    if family == "classifier":
        return _ClassifierTask(train, test, schema)
    if family == "count":
        return _CountTask(train, test, schema)
    if train.grain is Grain.REQUISITION:
        return _role_task(train, test, schema)
    if train.grain is Grain.RATING:
        return _rating_task(train, test)
    raise TrainingError(f"no {family} task for a {train.grain.value} extract")


# --- the driver ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmResult:
    arm: Arm
    model: object
    parameters: Mapping[str, Any]
    #: The mean per-row contribution to the family's primary metric.
    primary: float
    metrics: Mapping[str, float | None]
    version: str | None = None


@dataclass(frozen=True)
class DomainResult:
    domain: str
    family: str
    train_rows: int
    test_rows: int
    arms: Mapping[str, ArmResult]
    selection: Selection
    description: Mapping[str, Any]
    probe: Probe

    @property
    def champion(self) -> ArmResult:
        return self.arms[self.selection.champion]


@dataclass(frozen=True)
class TrainingRun:
    split: TemporalSplit
    population: str
    schema: str
    domains: Mapping[str, DomainResult] = field(default_factory=dict)

    @property
    def probes(self) -> dict[str, Probe]:
        """Each domain's probe, for `load_models`."""
        return {name: result.probe for name, result in self.domains.items()}

    @property
    def chosen(self) -> dict[str, str]:
        """Each domain's champion version, for `load_models`."""
        return {
            name: result.champion.version
            for name, result in self.domains.items()
            if result.champion.version is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split.to_dict(),
            "population": self.population,
            "schema": self.schema,
            "domains": {
                name: {
                    "family": result.family,
                    "train_rows": result.train_rows,
                    "test_rows": result.test_rows,
                    "champion": result.selection.champion,
                    "arms": {
                        arm: {
                            "algorithm": r.arm.algorithm,
                            "version": r.version,
                            "primary": r.primary,
                            "metrics": dict(r.metrics),
                        }
                        for arm, r in result.arms.items()
                    },
                    "decisions": [
                        {
                            "arm": d.arm,
                            "against": d.against,
                            "mean_gain": d.mean_gain,
                            "standard_error": d.standard_error,
                            "cleared": d.cleared,
                        }
                        for d in result.selection.decisions
                    ],
                    **result.description,
                }
                for name, result in self.domains.items()
            },
        }


def train(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    population: PopulationSpec,
    split: TemporalSplit,
    *,
    matrix: ArmMatrix = DEFAULT_MATRIX,
    domains: Sequence[str] | None = None,
    root: Path | None = None,
    z: float = DEFAULT_Z,
) -> TrainingRun:
    """Train, score, and select every arm of `domains` (all of the matrix's by default)."""
    chosen = list(matrix.domains if domains is None else domains)
    for name in chosen:
        matrix.domain(name)  # refuses a domain the matrix lacks
    data = extracts(con, population, split.start, split.end, chosen)
    run = TrainingRun(split, str(population.stamp), schema.fingerprint())
    results = {}
    for name in chosen:
        arms = matrix.domain(name)
        family = arms[0].family
        train_side, test_side = split.cut(data[name])
        task = _task(train_side, test_side, family, schema)
        fitted = {arm.name: task.fit(arm) for arm in arms}
        scored = {arm.name: task.contributions(fitted[arm.name].model) for arm in arms}
        selection = select(matrix, name, scored, z=z)
        arm_results = {}
        for arm in arms:
            fit = fitted[arm.name]
            metrics = task.metrics(fit.model)
            version = None
            if root is not None:
                config = {
                    "arm": arm.name,
                    "family": family,
                    "parameters": fit.parameters,
                    "population": run.population,
                    "schema": run.schema,
                    "split": split.to_dict(),
                    **task.describe(),
                }
                metadata = {
                    "train_rows": task.train_rows,
                    "test_rows": task.test_rows,
                    "metrics": metrics,
                    "champion": arm.name == selection.champion,
                }
                version = save_model(fit.model, root, name, config, metadata).version
            arm_results[arm.name] = ArmResult(
                arm,
                fit.model,
                fit.parameters,
                float(scored[arm.name].mean()),
                metrics,
                version,
            )
        results[name] = DomainResult(
            name,
            family,
            task.train_rows,
            task.test_rows,
            arm_results,
            selection,
            task.describe(),
            task.probe(),
        )
    run = TrainingRun(split, run.population, run.schema, results)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        (root / RUN_FILE).write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n")
    return run
