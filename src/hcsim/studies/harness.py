"""The harness: from a profile, a world, and a guidance artifact to each arm's evaluated results.

**Preflight** (`preflight`) checks everything a study needs before anything simulates, and
reports every problem at once rather than stopping at the first:

- the world, and a guidance artifact where the mode needs one and none where it doesn't;
- that the world can serve every input kind the study needs (`resolve.probe`);
- the profile's subset, lowered through the world's schema;
- the plan, loaded or built;
- the fitted models and the seed;
- and every arm's configuration, its overrides and alignment method applied, each refused with
  the arm's name where it doesn't validate.

**Inputs resolved once.** Every arm reads one set of inputs (`Inputs`): one schema, one run date,
one set of fitted models, one seed, lowered to the subset, one guidance artifact, and one plan.
Each arm simulates its own copy of the seed, so arms differ only by what they override.

`run_study` runs preflight, refuses the study with every problem it found, and then runs each
arm in order: the declared and expanded arms of a profile with arms, or the profile itself.
The provenance it records names every input, so a scorecard row can be traced to the exact
world, guidance, schema, and inputs behind it.

**Window closure.** A study's months are wholly open, partially closed, or wholly closed by
whether the world holds their actuals (`window_closure`), up to its last snapshot.

**Deferred variants and recorded skips.** An arm whose mode reads the window's actuals can't run
until the window is wholly closed; it's **deferred**, recorded with when it can run, rather than
failing the study. An evaluator with nothing to score, such as plan alignment for a study with
no plan, is **skipped**, with its reason recorded.

**Pass A and Pass B.** With a screen, pass A runs every compared arm at the screen's
trajectories and ranks them by its metric, lower first; the arms it doesn't keep are skipped,
each with its rank. Pass B runs the null arm, the kept arms, and the reference arms at the
study's full size. Without a screen, pass B runs every arm.

**Metadata, persistence, and the timing ledger.** A study's run records its specification's
digest, closure, passes, skips, deferrals, and provenance (`StudyRun.metadata`), and a timing
ledger of how long preflight and each arm's simulation and evaluation took. `save` writes the
metadata and every arm's metrics to a directory.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import duckdb
import pandas as pd

from hcsim.engine.alignment import AlignmentMethod
from hcsim.engine.config import AwardSettings, ConfigError, ForecastScope, SimulationConfig
from hcsim.engine.loop import SimulationResult, simulate
from hcsim.engine.modes import cell, rung
from hcsim.engine.seeding import (
    SeedingError,
    SeedReport,
    opening_price,
    range_table_from_world,
    schema_from_world,
    seed_for,
    snapshot_before,
)
from hcsim.engine.settings import settings
from hcsim.engine.state import SimulationState
from hcsim.financial.prices import AnnualGrowthPrice
from hcsim.financial.sizing import NewHireGapFill
from hcsim.model.core.alignment import AlignmentError, AlignmentTarget
from hcsim.model.core.contract import role_column
from hcsim.model.core.guidance import GuidanceArtifactError, GuidanceForecast
from hcsim.model.core.schema import COST_CENTER, SimulationSchema
from hcsim.model.data.population import REGULAR
from hcsim.model.data.subset import LoweredSubset, SubsetError, SubsetSpec
from hcsim.studies.evaluators import EVALUATORS, Metric
from hcsim.studies.inputs import FittedModels, InputError, fit_models
from hcsim.studies.profile import Arm, StudyProfile
from hcsim.studies.resolve import probe, required_kinds

#: Every setting, by its `group.key` path.
SETTINGS = settings()
#: The arm a profile with no arms runs as.
ITSELF = Arm(name="study")


class StudyError(ValueError):
    """A study whose inputs don't fit its profile, every problem listed."""

    def __init__(self, problems: list[str] | str) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class Inputs:
    """What every arm reads, resolved once."""

    schema: SimulationSchema
    run_date: dt.date
    fitted: FittedModels
    state: SimulationState
    seed: SeedReport
    artifact: GuidanceForecast | None
    plan: AlignmentTarget | None
    subset: LoweredSubset | None
    #: The input kinds the study reads.
    kinds: tuple[str, ...]
    world: Mapping[str, str]
    #: The month-end the world holds actuals through: its last snapshot.
    actuals_through: dt.date


@dataclass(frozen=True)
class Preflight:
    """What preflight found: every problem, and when there are none, the resolved inputs and
    each arm's configuration."""

    problems: tuple[str, ...]
    inputs: Inputs | None = None
    configs: Mapping[str, SimulationConfig] = dataclasses.field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


class Closure(StrEnum):
    """Whether a study's months have actuals to score."""

    OPEN = "wholly_open"
    PARTIAL = "partially_closed"
    CLOSED = "wholly_closed"


def window_closure(run_date: dt.date, months: int, actuals_through: dt.date) -> Closure:
    """The closure of the `months` from `run_date`'s, against actuals through a month-end."""
    first = pd.Period(run_date, freq="M")
    last_actual = pd.Period(actuals_through, freq="M")
    if first > last_actual:
        return Closure.OPEN
    if first + months - 1 <= last_actual:
        return Closure.CLOSED
    return Closure.PARTIAL


@dataclass(frozen=True)
class Skip:
    """An arm or evaluator that didn't run, and why."""

    what: str
    name: str
    reason: str


@dataclass(frozen=True)
class Deferred:
    """An arm that can't run yet, and when it can."""

    arm: str
    reason: str


def deferral(arm: Arm, mode: str, closure: Closure) -> Deferred | None:
    """Why `arm` can't run yet under `mode` in a window of `closure`, or None if it can."""
    if rung(mode).needs_actuals and closure is not Closure.CLOSED:
        return Deferred(
            arm.name,
            f"its mode, {mode}, reads the window's actuals, which is {closure.value}; it runs once "
            "the window is wholly closed",
        )
    return None


#: What each evaluator needs from a study, and the reason it's skipped without it.
NEEDS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {"plan_alignment": ("plan", "the study has no plan to score against")}
)


@dataclass(frozen=True)
class Timing:
    """One phase of the run, and how long it took."""

    phase: str
    arm: str
    seconds: float


@dataclass(frozen=True)
class ArmRun:
    """One arm's run: its configuration, results, and metrics, and the pass it ran in."""

    arm: Arm
    config: SimulationConfig
    result: SimulationResult
    metrics: tuple[Metric, ...]
    pass_name: str = "B"

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics)


@dataclass(frozen=True)
class StudyRun:
    profile: StudyProfile
    config: SimulationConfig
    seed: SeedReport
    fitted: FittedModels
    result: SimulationResult
    metrics: tuple[Metric, ...]
    provenance: dict[str, str]
    #: Every arm's run, in order; the study's own run first is its null arm, or the profile.
    arms: tuple[ArmRun, ...] = ()
    closure: Closure = Closure.OPEN
    #: Pass A's runs, ranked, best first; none without a screen.
    screened: tuple[ArmRun, ...] = ()
    skips: tuple[Skip, ...] = ()
    deferred: tuple[Deferred, ...] = ()
    timing: tuple[Timing, ...] = ()

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics) and all(a.passed for a in self.arms)

    def metadata(self) -> dict[str, Any]:
        """Everything about the run but its metrics, as plain data."""
        return {
            "profile": self.profile.name,
            "spec": self.profile.spec().digest(),
            "closure": self.closure.value,
            "passed": self.passed,
            "arms": [
                {"arm": a.arm.name, "pass": a.pass_name, "passed": a.passed}
                for a in (*self.screened, *self.arms)
            ],
            "skips": [dataclasses.asdict(s) for s in self.skips],
            "deferred": [dataclasses.asdict(d) for d in self.deferred],
            "timing": [dataclasses.asdict(t) for t in self.timing],
            "provenance": dict(self.provenance),
        }


def _world_identity(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    (manifest,) = con.execute("SELECT manifest FROM truth.manifest").fetchone() or ("{}",)
    truth = json.loads(manifest)
    config = truth.get("config", {})
    return {
        "source": "synthetic_world",
        "world_generator": str(truth.get("generator_version", "")),
        "world_preset": str(config.get("preset", "")),
        "world_seed": str(config.get("seed", "")),
    }


def _arms(profile: StudyProfile) -> tuple[Arm, ...]:
    """The arms a study runs: its own when it declares them, else the profile as one arm. The
    null arm runs first."""
    arms = profile.expanded_arms()
    if not arms:
        return (ITSELF,)
    return (*(a for a in arms if a.null), *(a for a in arms if not a.null))


def _guidance(profile: StudyProfile, guidance: Path | None, problems: list[str]) -> Any:
    reads = rung(profile.mode.value).reads_guidance or any(
        a.mode is not None and rung(a.mode.value).reads_guidance for a in profile.expanded_arms()
    )
    if reads and guidance is None:
        problems.append(f"mode {profile.mode.value!r} needs a guidance artifact")
    if not reads and guidance is not None:
        problems.append(f"mode {profile.mode.value!r} reads no guidance; drop the artifact")
    if guidance is None:
        return None
    try:
        return GuidanceForecast.load(guidance)
    except (OSError, GuidanceArtifactError) as error:
        problems.append(f"can't load the guidance artifact at {guidance}: {error}")
        return None


def preflight(profile: StudyProfile, world: Path, guidance: Path | None = None) -> Preflight:
    """Everything the study needs, checked before anything simulates: every problem found, or
    the inputs every arm reads and each arm's configuration."""
    problems: list[str] = []
    artifact = _guidance(profile, guidance, problems)
    if not world.is_file():
        return Preflight((*problems, f"no world at {world}"))
    with duckdb.connect(str(world), read_only=True) as con:
        schema = schema_from_world(con)
        if profile.run_date is not None:
            run_date = profile.run_date
        elif artifact is not None:
            run_date = dt.date.fromisoformat(artifact.snapshot_date) + dt.timedelta(days=1)
        else:
            run_date = snapshot_before(con, dt.date.max) + dt.timedelta(days=1)
        kinds = required_kinds(
            profile.mode.value,
            profile.scope,
            plan_path=profile.plan is not None and profile.plan.path is not None,
        )
        problems += probe(con, kinds, REGULAR, role_column(con, "req_interval"))
        subset, lowered = None, None
        if profile.subset:
            try:
                subset = SubsetSpec({c: frozenset(v) for c, v in profile.subset.items()})
                lowered = subset.lower(schema)
            except SubsetError as error:
                problems.append(f"subset: {error}")
        if problems:
            return Preflight(tuple(problems))
        try:
            fitted = fit_models(con, schema, run_date)
        except InputError as error:
            return Preflight((f"can't fit the models: {error}",))
        awards = (
            AwardSettings(
                NewHireGapFill(range_table_from_world(con, schema)),
                AnnualGrowthPrice(opening_price(con, run_date)),
            )
            if profile.scope is ForecastScope.HEADCOUNT_AND_SBC
            else None
        )
        # The plan may be built from the seed, so the run is configured without it, seeded,
        # and then given the plan and the method that drives it.
        try:
            base = SimulationConfig(
                run_date=run_date,
                schema=schema,
                models=fitted.models,
                simulation_months=profile.months,
                num_trajectories=profile.trajectories,
                seed=profile.seed,
                forecast_scope=profile.scope,
                guidance=artifact if rung(profile.mode.value).reads_guidance else None,
                awards=awards,
                capacity_caps=profile.capacity_caps,
                fill_shares=profile.fill_share_targets(),
                tilts=profile.probability_tilts(),
                alignment_method=AlignmentMethod.NONE,
                **{SETTINGS[path].name: value for path, value in profile.settings.items()},
            )
            state, seed = seed_for(con, base, subset=subset)
        except (ConfigError, SeedingError) as error:
            return Preflight((f"the study's run: {error}",))
        plan = None
        if profile.plan is not None:
            try:
                labels = schema.vocabulary(COST_CENTER).labels
                heads = {labels[c]: float(n) for c, n in enumerate(state.headcount) if n > 0}
                plan = profile.plan.build(heads, run_date, profile.months)
            except (OSError, AlignmentError) as error:
                return Preflight((f"can't build the plan: {error}",))
        world_identity = _world_identity(con)
        actuals_through = snapshot_before(con, dt.date.max)
    configs = {}
    for arm in _arms(profile):
        try:
            configs[arm.name] = _arm_config(base, profile, arm, plan, artifact)
        except ConfigError as error:
            problems += [f"arm {arm.name}: {p}" for p in error.problems]
    if problems:
        return Preflight(tuple(problems))
    inputs = Inputs(
        schema,
        run_date,
        fitted,
        state,
        seed,
        artifact,
        plan,
        lowered,
        kinds,
        world_identity,
        actuals_through,
    )
    return Preflight((), inputs, MappingProxyType(configs))


def _arm_config(
    base: SimulationConfig,
    profile: StudyProfile,
    arm: Arm,
    plan: AlignmentTarget | None,
    artifact: GuidanceForecast | None,
) -> SimulationConfig:
    """The arm's configuration: the study's, with the plan, its alignment method, its own
    mode, and the guidance that mode reads, if it's a reference arm, and its overrides."""
    method = profile.alignment.choice().resolve(None if arm is ITSELF else arm.name)
    fields: dict[str, Any] = {"plan": plan, "alignment_method": method}
    if arm.mode is not None:
        fields["run_mode"] = arm.mode.value
        fields["guidance"] = artifact if rung(arm.mode.value).reads_guidance else None
    fields |= {SETTINGS[path].name: value for path, value in arm.overrides.items()}
    return dataclasses.replace(base, **fields)


def run_study(profile: StudyProfile, world: Path, guidance: Path | None = None) -> StudyRun:
    timing: list[Timing] = []
    started = time.perf_counter()
    checked = preflight(profile, world, guidance)
    timing.append(Timing("preflight", "", time.perf_counter() - started))
    if not checked.ok or checked.inputs is None:
        raise StudyError(list(checked.problems))
    inputs = checked.inputs
    closure = window_closure(inputs.run_date, profile.months, inputs.actuals_through)
    skips, deferred, runnable = [], [], []
    for arm in _arms(profile):
        mode = (arm.mode or profile.mode).value
        waiting = deferral(arm, mode, closure)
        if waiting is None:
            runnable.append(arm)
        else:
            deferred.append(waiting)
    if not runnable:
        raise StudyError([f"nothing can run yet: {d.arm} waits: {d.reason}" for d in deferred])
    evaluators = []
    for name in profile.evaluators:
        need, reason = NEEDS.get(name, ("", ""))
        if need == "plan" and inputs.plan is None:
            skips.append(Skip("evaluator", name, reason))
        else:
            evaluators.append(name)

    def run(arm: Arm, pass_name: str, trajectories: int | None = None) -> ArmRun:
        config = checked.configs[arm.name]
        if trajectories is not None:
            config = dataclasses.replace(config, num_trajectories=trajectories)
        began = time.perf_counter()
        result = simulate(config, inputs.state)
        simulated = time.perf_counter()
        metrics = tuple(m for name in evaluators for m in EVALUATORS[name](result))
        timing.append(Timing(f"simulate {pass_name}", arm.name, simulated - began))
        timing.append(Timing(f"evaluate {pass_name}", arm.name, time.perf_counter() - simulated))
        return ArmRun(arm, config, result, metrics, pass_name)

    screened: list[ArmRun] = []
    keep = runnable
    if profile.screen is not None:
        screen = profile.screen
        compared = [a for a in runnable if not (a.null or a.reference)]
        screened = sorted(
            (run(a, "A", screen.trajectories) for a in compared),
            key=lambda r: _ranked(r, screen.rank_by),
        )
        kept = {r.arm.name for r in screened[: screen.keep]}
        n = len(screened)
        for rank, r in enumerate(screened, start=1):
            if r.arm.name not in kept:
                out = f"screened out in pass A: ranked {rank} of {n} by {screen.rank_by}"
                skips.append(Skip("arm", r.arm.name, out))
        keep = [a for a in runnable if a.null or a.reference or a.name in kept]
    runs = [run(arm, "B") for arm in keep]
    first = runs[0]
    provenance = dict(inputs.world) | {
        "hcsim_version": version("hcsim"),
        "profile": profile.name,
        "mode": profile.mode.value,
        "scope": profile.scope.value,
        "isolates": cell(profile.mode.value, profile.scope).isolates,
        "seed": str(profile.seed),
        "run_date": inputs.run_date.isoformat(),
        "schema_fingerprint": inputs.schema.fingerprint(),
        "guidance_fingerprint": inputs.artifact.fingerprint() if inputs.artifact else "",
        "input_kinds": ",".join(inputs.kinds),
    }
    return StudyRun(
        profile,
        first.config,
        inputs.seed,
        inputs.fitted,
        first.result,
        first.metrics,
        provenance,
        tuple(runs) if profile.expanded_arms() else (),
        closure,
        tuple(screened),
        tuple(skips),
        tuple(deferred),
        tuple(timing),
    )


def _ranked(run: ArmRun, rank_by: str) -> float:
    evaluator, name = rank_by.split(".", 1)
    for m in run.metrics:
        if m.evaluator == evaluator and m.name == name:
            return m.value
    raise StudyError([f"arm {run.arm.name} has no metric {rank_by} to rank by"])


def save(run: StudyRun, directory: Path) -> Path:
    """Write the run's metadata, `study.json`, and every arm's metrics, `metrics.csv`, one row
    per pass, arm, evaluator, and metric. Returns the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "study.json").write_text(
        json.dumps(run.metadata(), indent=2, sort_keys=True, default=str) + "\n"
    )
    arms = (*run.screened, *(run.arms or (ArmRun(ITSELF, run.config, run.result, run.metrics),)))
    rows = [
        {
            "pass": a.pass_name,
            "arm": a.arm.name,
            "evaluator": m.evaluator,
            "metric": m.name,
            "value": m.value,
            "threshold": m.threshold,
            "passed": m.passed,
        }
        for a in arms
        for m in a.metrics
    ]
    columns = ["pass", "arm", "evaluator", "metric", "value", "threshold", "passed"]
    pd.DataFrame(rows, columns=columns).to_csv(directory / "metrics.csv", index=False)
    return directory
