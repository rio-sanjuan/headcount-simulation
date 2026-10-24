"""The harness: from a profile, a world, and a guidance artifact to each arm's evaluated results.

**Preflight** (`preflight`) checks everything a study needs before anything simulates, and
reports every problem at once rather than stopping at the first:

- the world, and a guidance artifact where a mode needs one and none where none does: a guided
  forecast reads the artifact it's given, and guided actuals the window's realized totals;
- that the study's own mode can run in its window: one that reads the window's actuals needs
  it wholly closed;
- that the world can serve every input kind the study needs (`resolve.probe`);
- the profile's subset, lowered through the world's schema;
- the plan, loaded or built;
- the fitted models, the seed, and each mode's inputs (`backends.mode_inputs`);
- and every arm's configuration, its overrides and alignment method applied, each refused with
  the arm's name where it doesn't validate.

**Inputs resolved once.** Every arm reads one set of inputs (`Inputs`): one schema, one run date,
one set of fitted models, one seed, lowered to the subset, one guidance artifact, one plan, and
one set of inputs for each mode its arms run under. Each arm simulates its own copy of the seed,
so arms differ only by their mode, if they're reference arms, and what they override.

**Backends.** On the in-simulation backend, each arm simulates under its mode; on the offline
backend, the study is scored from its inputs alone, with no simulation (see `backends`).

`run_study` runs preflight, refuses the study with every problem it found, and then runs each
arm in order: the declared and expanded arms of a profile with arms, or the profile itself.
The provenance it records names every input, so a scorecard row can be traced to the exact
world, guidance, schema, and inputs behind it.

**Window closure.** A study's months are wholly open, partially closed, or wholly closed by
whether the world holds their actuals (`window_closure`), up to its last snapshot.

**Deferred variants and recorded skips.** A reference arm whose mode reads the window's actuals
can't run until the window is wholly closed; it's **deferred**, recorded with when it can run,
rather than failing the study. A study whose own mode can't run yet is refused in preflight.
An evaluator with nothing to score, such as plan alignment for a study with no plan, is
**skipped**, with its reason recorded.

**Pass A and Pass B.** With a screen, pass A runs every compared arm at the screen's
trajectories and ranks them by its metric, lower first; the arms it doesn't keep are skipped,
each with its rank. Pass B runs the null arm, the kept arms, and the reference arms at the
study's full size. Without a screen, pass B runs every arm.

**Gates.** In pass B, the gates the study holds for its mode and closure (`gates.gate_set`)
apply to every arm but the reference arms, each as it finishes. A failed stop gate stops the
study, the arms after it skipped, and the run's verdict says whether it passed, is provisional,
or stopped.

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
from typing import Any

import duckdb
import pandas as pd

from hcsim.engine.alignment import AlignmentMethod
from hcsim.engine.config import AwardSettings, ConfigError, ForecastScope, SimulationConfig
from hcsim.engine.loop import SimulationResult
from hcsim.engine.modes import RunMode, cell, rung
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
from hcsim.studies.actuals import Actuals, resolve
from hcsim.studies.backends import ModeInputs, mode_inputs, offline_metrics, simulate_arm
from hcsim.studies.evaluators import EVALUATORS, OFFLINE_EVALUATORS, Evaluator, Metric, Scoring
from hcsim.studies.gates import Action, GateResult, Verdict, apply, gate_set, verdict
from hcsim.studies.inputs import FittedModels, InputError, fit_models
from hcsim.studies.profile import Arm, Backend, StudyProfile
from hcsim.studies.resolve import probe, provisional, required_kinds

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
    #: What each mode the study's runnable arms run under reads, by mode.
    modes: Mapping[str, ModeInputs] = dataclasses.field(default_factory=dict)
    #: The window's actuals, the parts the study's evaluators read; none for an open window.
    actuals: Actuals | None = None


@dataclass(frozen=True)
class Preflight:
    """What preflight found: every problem, and when there are none, the resolved inputs, each
    runnable arm's configuration, the window's closure, and the arms deferred until it closes."""

    problems: tuple[str, ...]
    inputs: Inputs | None = None
    configs: Mapping[str, SimulationConfig] = dataclasses.field(default_factory=dict)
    closure: Closure | None = None
    deferred: tuple[Deferred, ...] = ()
    #: The evaluators the study runs, and those it skips.
    evaluators: tuple[str, ...] = ()
    skips: tuple[Skip, ...] = ()

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


def choose_evaluators(
    profile: StudyProfile, closure: Closure
) -> tuple[list[Evaluator], list[Skip]]:
    """The profile's evaluators the study runs, and those it skips, each with its reason."""
    registry = OFFLINE_EVALUATORS if profile.backend is Backend.OFFLINE else EVALUATORS
    chosen, skips = [], []
    for name in profile.evaluators:
        evaluator = registry[name]
        reason = evaluator.skipped(
            plan=profile.plan is not None,
            sbc=cell(profile.mode.value, profile.scope).computes_sbc,
            trajectories=profile.trajectories,
            closed=closure is Closure.CLOSED,
            subset=bool(profile.subset),
        )
        if reason is None:
            chosen.append(evaluator)
        else:
            skips.append(Skip("evaluator", name, reason))
    return chosen, skips


@dataclass(frozen=True)
class Timing:
    """One phase of the run, and how long it took."""

    phase: str
    arm: str
    seconds: float


@dataclass(frozen=True)
class ArmRun:
    """One arm's run: its configuration, results, and metrics, and the pass it ran in. On the
    offline backend, an arm has no simulation's results, only its metrics."""

    arm: Arm
    config: SimulationConfig
    result: SimulationResult | None
    metrics: tuple[Metric, ...]
    pass_name: str = "B"

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics)

    @property
    def simulated(self) -> SimulationResult:
        """The arm's simulated results; an arm run offline has none."""
        if self.result is None:
            raise StudyError(f"arm {self.arm.name} ran offline, so has no simulated results")
        return self.result


@dataclass(frozen=True)
class StudyRun:
    profile: StudyProfile
    config: SimulationConfig
    seed: SeedReport
    fitted: FittedModels
    result: SimulationResult | None
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
    #: Every gate on every gated arm, in the order they ran, and the study's verdict.
    gates: tuple[GateResult, ...] = ()
    verdict: Verdict = Verdict.PASSED

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics) and all(a.passed for a in self.arms)

    @property
    def simulated(self) -> SimulationResult:
        """The study's own simulated results; a study run offline has none."""
        if self.result is None:
            raise StudyError(f"study {self.profile.name} ran offline, so has no simulated results")
        return self.result

    def metadata(self) -> dict[str, Any]:
        """Everything about the run but its metrics, as plain data."""
        return {
            "profile": self.profile.name,
            "spec": self.profile.spec().digest(),
            "closure": self.closure.value,
            "passed": self.passed,
            "verdict": self.verdict.value,
            "arms": [
                {"arm": a.arm.name, "pass": a.pass_name, "passed": a.passed}
                for a in (*self.screened, *self.arms)
            ],
            "skips": [dataclasses.asdict(s) for s in self.skips],
            "deferred": [dataclasses.asdict(d) for d in self.deferred],
            "timing": [dataclasses.asdict(t) for t in self.timing],
            "gates": [dataclasses.asdict(g) for g in self.gates],
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


def _modes(profile: StudyProfile) -> list[str]:
    """Every mode the study's arms run under, its own first."""
    modes = [profile.mode.value]
    modes += [a.mode.value for a in profile.expanded_arms() if a.mode is not None]
    return list(dict.fromkeys(modes))


def _reads_artifact(mode: str) -> bool:
    """Whether `mode` reads the guidance artifact a study is given, not the window's actuals."""
    return rung(mode).reads_guidance and not rung(mode).realized_guidance


def _guidance(profile: StudyProfile, guidance: Path | None, problems: list[str]) -> Any:
    reading = [m for m in _modes(profile) if _reads_artifact(m)]
    if reading and guidance is None:
        problems.append(f"mode {reading[0]!r} needs a guidance artifact")
    if not reading and guidance is not None:
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
        actuals_through = snapshot_before(con, dt.date.max)
        closure = window_closure(run_date, profile.months, actuals_through)
        own = deferral(ITSELF, profile.mode.value, closure)
        if own is not None:
            problems.append(f"nothing can run yet: {own.reason}")
        deferred = [
            d
            for a in _arms(profile)
            if a.mode is not None and (d := deferral(a, a.mode.value, closure)) is not None
        ]
        runnable = [a for a in _arms(profile) if a.name not in {d.arm for d in deferred}]
        chosen, skips = choose_evaluators(profile, closure)
        kinds = required_kinds(
            profile.mode.value,
            profile.scope,
            plan_path=profile.plan is not None and profile.plan.path is not None,
            evaluator_kinds=[k for e in chosen for k in e.kinds],
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
                guidance=artifact if _reads_artifact(profile.mode.value) else None,
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
        modes = {}
        for mode in dict.fromkeys((a.mode or profile.mode).value for a in runnable):
            try:
                modes[mode] = mode_inputs(
                    con,
                    mode,
                    schema,
                    state,
                    run_date,
                    profile.months,
                    base.awards,
                    profile.replay_lag,
                )
            except (ValueError, duckdb.Error) as error:
                problems.append(f"can't resolve the inputs {mode} reads: {error}")
        parts = list(dict.fromkeys(p for e in chosen for p in e.actuals))
        actuals = None
        if parts:
            try:
                actuals = resolve(con, schema, run_date, profile.months, parts)
            except (ValueError, duckdb.Error) as error:
                problems.append(f"can't resolve the window's actuals: {error}")
    if problems:
        return Preflight(tuple(problems))
    configs = {}
    for arm in runnable:
        try:
            configs[arm.name] = _arm_config(base, profile, arm, plan, artifact, modes)
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
        MappingProxyType(modes),
        actuals,
    )
    return Preflight(
        (),
        inputs,
        MappingProxyType(configs),
        closure,
        tuple(deferred),
        tuple(e.name for e in chosen),
        tuple(skips),
    )


def _arm_config(
    base: SimulationConfig,
    profile: StudyProfile,
    arm: Arm,
    plan: AlignmentTarget | None,
    artifact: GuidanceForecast | None,
    modes: Mapping[str, ModeInputs],
) -> SimulationConfig:
    """The arm's configuration: the study's, with the plan, its alignment method, its mode (its
    own, if it's a reference arm), the guidance and awards that mode reads, and its
    overrides."""
    method = profile.alignment.choice().resolve(None if arm is ITSELF else arm.name)
    mode = (arm.mode or profile.mode).value
    reads = modes[mode]
    fields: dict[str, Any] = {
        "plan": plan,
        "alignment_method": method,
        "run_mode": RunMode(mode),
        "guidance": artifact if _reads_artifact(mode) else reads.guidance,
        "awards": reads.awards,
    }
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
    closure = checked.closure or Closure.OPEN
    runnable = [a for a in _arms(profile) if a.name in checked.configs]
    skips = list(checked.skips)
    evaluators = checked.evaluators

    def run(arm: Arm, pass_name: str, trajectories: int | None = None) -> ArmRun:
        config = checked.configs[arm.name]
        if trajectories is not None:
            config = dataclasses.replace(config, num_trajectories=trajectories)
        began = time.perf_counter()
        if profile.backend is Backend.OFFLINE:
            metrics = offline_metrics(evaluators, Scoring(None, inputs.fitted, inputs.actuals))
            timing.append(Timing(f"evaluate {pass_name}", arm.name, time.perf_counter() - began))
            return ArmRun(arm, config, None, metrics, pass_name)
        result = simulate_arm(config, inputs.state, inputs.modes[config.run_mode or ""])
        simulated = time.perf_counter()
        scoring = Scoring(result, inputs.fitted, inputs.actuals)
        metrics = tuple(m for name in evaluators for m in EVALUATORS[name].score(scoring))
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
    gates = gate_set(
        profile.mode.value, closure.value, profile.own_gates(), builtin=profile.builtin_gates
    )
    runs: list[ArmRun] = []
    checks: list[GateResult] = []
    for k, arm in enumerate(keep):
        runs.append(run(arm, "B"))
        if arm.reference:
            continue  # a reference arm is never gated
        checks += apply(gates, arm.name, runs[-1].metrics)
        stops = [g for g in checks if g.failed and g.action is Action.STOP]
        if stops:
            why = f"the study stopped: gate {stops[0].gate} failed on arm {stops[0].arm}"
            skips += [Skip("arm", a.name, why) for a in keep[k + 1 :]]
            break
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
        "backend": profile.backend.value,
        "guidance_fingerprint": first.config.guidance.fingerprint()
        if first.config.guidance
        else "",
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
        checked.deferred,
        tuple(timing),
        tuple(checks),
        verdict(checks, provisional(provenance["source"])),
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
