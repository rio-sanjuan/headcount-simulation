"""Study profiles: what a study runs, read from YAML.

A profile declares a study's forecast scope, run mode, seed, size, and window, any settings it
overrides for the whole study, the plan it aligns to and the alignment method that says which
valve the plan drives, with overrides per arm, the planner's capacity caps, fill-share targets,
and probability tilts, the evaluators that score it, and its **arms**: the alternatives it
compares.

**The plan** is a saved alignment target, by `path`, or a `synthetic` one: the synthetic
builder's shape and arguments (see `model.core.alignment_builders`), its base the seed's own
headcount per cost center, starting the run's first month and lasting the run's months unless
it says otherwise. Unknown keys are
refused everywhere, so a misspelled field fails loudly instead of running the default.

**Arms.** An arm is a name and the settings it overrides, by their `group.key` paths (see
`engine.settings`), each path one the configuration knows. Every arm is scored against the
**null arm**, the naive benchmark, so a profile with arms has exactly one. A **reference arm**
runs for comparison only: it alone may run under another mode, and it stays out of rankings and
gates. A profile with no arms runs once, as itself.

**Arm groups** expand into arms: a group varies settings over lists of values, and its arms are
every combination, in the order the settings are listed, each named for its values, as
`gain[requisitions.supply_gain=0.5]`. Settings a group fixes apply to all of its arms. Arm names,
declared and expanded, are unique.

**A frozen specification.** `spec` resolves a profile into a `StudySpec`: the profile with its
arm groups expanded, immutable, and a digest over it, so a study's record can name exactly what
it ran. `to_mapping` writes a profile back as plain data, which `from_mapping` reads as the same
profile.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from hcsim.engine.alignment import AlignmentMethod, MethodChoice
from hcsim.engine.config import ForecastScope
from hcsim.engine.planner import FillShareTarget, PlannerError, PlannerScope, ProbabilityTilt
from hcsim.engine.settings import settings
from hcsim.model.core.alignment import AlignmentTarget
from hcsim.model.core.alignment_builders import Shape, synthetic
from hcsim.model.core.schema import COST_CENTER
from hcsim.model.data.subset import SubsetError, SubsetSpec
from hcsim.studies.evaluators import EVALUATORS

#: Every setting path an arm may override.
SETTING_PATHS: Final = frozenset(settings())
#: The alignment method's setting path.
METHOD: Final = "alignment.method"
#: The settings a profile sets in its own fields, never under `settings`.
MANAGED: Final = {
    "run.run_date": "run_date",
    "run.simulation_months": "months",
    "run.num_trajectories": "trajectories",
    "run.seed": "seed",
    "run.forecast_scope": "scope",
    "run.run_mode": "mode",
}


class ProfileError(ValueError):
    """A profile that can't be read or doesn't validate."""


class Mode(StrEnum):
    """The run modes the skeleton supports. `replay` and `guided_actuals` arrive in M8.1."""

    GUIDED_FORECAST = "guided_forecast"
    UNGUIDED = "unguided"


def _known_paths(paths: Any) -> None:
    unknown = sorted(set(paths) - SETTING_PATHS)
    if unknown:
        raise ValueError(f"no settings {unknown}; a setting is a group.key path")


class Arm(BaseModel):
    """One alternative a study compares: its name, and the settings it overrides."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    #: Settings by `group.key` path.
    overrides: dict[str, Any] = Field(default_factory=dict)
    #: The naive benchmark every other arm is scored against.
    null: bool = False
    #: Run for comparison only: another mode allowed, out of rankings and gates.
    reference: bool = False
    #: A reference arm's own mode; other arms run under the study's.
    mode: Mode | None = None

    @field_validator("overrides")
    @classmethod
    def _settings(cls, value: dict[str, Any]) -> dict[str, Any]:
        _known_paths(value)
        return value

    @model_validator(mode="after")
    def _kind(self) -> Arm:
        if self.null and self.reference:
            raise ValueError(f"arm {self.name!r} can't be both the null arm and a reference arm")
        if self.mode is not None and not self.reference:
            raise ValueError(f"arm {self.name!r} sets a mode, which only a reference arm may")
        return self


class ArmGroup(BaseModel):
    """Arms that vary settings over lists of values: one arm per combination."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    #: Settings by `group.key` path, and the values each arm takes.
    vary: dict[str, list[Any]] = Field(min_length=1)
    #: Settings every arm of the group takes.
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _settings(self) -> ArmGroup:
        _known_paths([*self.vary, *self.overrides])
        empty = sorted(k for k, values in self.vary.items() if not values)
        if empty:
            raise ValueError(f"group {self.name!r} varies {empty} over no values")
        both = sorted(set(self.vary) & set(self.overrides))
        if both:
            raise ValueError(f"group {self.name!r} both varies and fixes {both}")
        return self

    def expand(self) -> list[Arm]:
        """One arm per combination of the varied values, in the order they're listed."""
        keys = list(self.vary)
        arms = []
        for values in itertools.product(*(self.vary[k] for k in keys)):
            label = ",".join(f"{k}={json.dumps(v)}" for k, v in zip(keys, values, strict=True))
            varied = dict(zip(keys, values, strict=True))
            arms.append(Arm(name=f"{self.name}[{label}]", overrides=self.overrides | varied))
        return arms


class FillShare(BaseModel):
    """A fill-share target, as a profile declares it: a scope in raw labels, and each channel's
    share of the scope's fills (see `engine.planner.FillShareTarget`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: dict[str, list[str]] = Field(default_factory=dict)
    shares: dict[str, float]

    @model_validator(mode="after")
    def _target(self) -> FillShare:
        try:
            self.target()
        except PlannerError as error:
            raise ValueError("; ".join(error.problems)) from None
        return self

    def target(self) -> FillShareTarget:
        return FillShareTarget(PlannerScope.from_dict(self.scope), dict(self.shares))


#: The synthetic builder's arguments a profile may give; its base is the seed's headcount.
SYNTHETIC_KEYS: Final = frozenset(
    {"shape", "months", "planned_depth", "final", "at", "jump", "values", "factors", "dates"}
)


class PlanSource(BaseModel):
    """Where a study's plan comes from: a saved alignment target, or a synthetic one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str | None = None
    synthetic: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _one(self) -> PlanSource:
        if (self.path is None) == (self.synthetic is None):
            raise ValueError("a plan is a saved target's path or a synthetic one, exactly one")
        if self.synthetic is not None:
            unknown = sorted(set(self.synthetic) - SYNTHETIC_KEYS)
            if unknown:
                raise ValueError(f"a synthetic plan takes {sorted(SYNTHETIC_KEYS)}, not {unknown}")
            shape = self.synthetic.get("shape")
            if shape not in {s.value for s in Shape}:
                raise ValueError(f"a synthetic plan's shape is one of {[s.value for s in Shape]}")
        return self

    def build(
        self, headcount: Mapping[str, float], run_date: dt.date, months: int
    ) -> AlignmentTarget:
        """The plan: loaded from its path, or built over `headcount` per cost center from the
        run's first month."""
        if self.path is not None:
            return AlignmentTarget.load(Path(self.path))
        args = dict(self.synthetic or {})
        shape = args.pop("shape")
        args.setdefault("months", months)
        return synthetic(shape, COST_CENTER, headcount, start=pd.Period(run_date, freq="M"), **args)


class Alignment(BaseModel):
    """The alignment method, and the arms that run under another (see `engine.alignment`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: AlignmentMethod = AlignmentMethod.SUPPLY
    overrides: dict[str, AlignmentMethod] = Field(default_factory=dict)

    def choice(self) -> MethodChoice:
        return MethodChoice(self.method, dict(self.overrides))


class Tilt(BaseModel):
    """A probability tilt, as a profile declares it (see `engine.planner.ProbabilityTilt`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain: str
    scope: dict[str, list[str]] = Field(default_factory=dict)
    multiplier: float
    start: str | None = None
    end: str | None = None
    cap: float = 1.0
    reason: str = ""

    @model_validator(mode="after")
    def _tilt(self) -> Tilt:
        try:
            self.tilt()
        except PlannerError as error:
            raise ValueError("; ".join(error.problems)) from None
        return self

    def tilt(self) -> ProbabilityTilt:
        return ProbabilityTilt.from_dict(self.model_dump() | {"scope": self.scope})


class StudyProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    scope: ForecastScope = ForecastScope.HEADCOUNT_AND_SBC
    mode: Mode = Mode.GUIDED_FORECAST
    #: Studies are always seeded, so every study is reproducible.
    seed: int = Field(ge=0)
    trajectories: int = Field(default=20, ge=1)
    months: int = Field(default=12, ge=1)
    #: The first simulated month. Defaults to the month after the world's last snapshot.
    run_date: dt.date | None = None
    #: The part of the population the study runs on its own, as a subset's rules in raw labels
    #: (see `model.data.subset`); everyone, with none.
    subset: dict[str, list[str]] = Field(default_factory=dict)
    #: Settings the whole study overrides, by `group.key` path, as the supply rule.
    settings: dict[str, Any] = Field(default_factory=dict)
    #: The plan the study aligns to, and which valve it drives.
    plan: PlanSource | None = None
    alignment: Alignment = Field(default_factory=Alignment)
    #: Probability tilts, each on one domain over a scope and window.
    tilts: list[Tilt] = Field(default_factory=list)
    #: Capacity caps, `Cap(c)`, by cost-center label: heads of committed strength each may hold.
    capacity_caps: dict[str, int] = Field(default_factory=dict)
    #: Fill-share targets, each a scope and its channels' shares of the scope's fills.
    fill_shares: list[FillShare] = Field(default_factory=list)
    #: The evaluators that score the study, by name.
    evaluators: list[str] = Field(default_factory=lambda: ["invariants", "summary"])
    #: The alternatives the study compares, declared and in groups.
    arms: list[Arm] = Field(default_factory=list)
    arm_groups: list[ArmGroup] = Field(default_factory=list)

    @field_validator("subset")
    @classmethod
    def _subset(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        try:
            SubsetSpec({c: frozenset(v) for c, v in value.items()})
        except SubsetError as error:
            raise ValueError(str(error)) from None
        return value

    @field_validator("settings")
    @classmethod
    def _settings(cls, value: dict[str, Any]) -> dict[str, Any]:
        _known_paths(value)
        managed = sorted(f"{MANAGED[k]} for {k}" for k in value if k in MANAGED)
        if managed:
            raise ValueError(f"set the profile's own fields, {managed}, not settings")
        return value

    @field_validator("evaluators")
    @classmethod
    def _evaluators(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(EVALUATORS))
        if unknown:
            raise ValueError(f"no evaluators {unknown}; known: {', '.join(EVALUATORS)}")
        return value

    @model_validator(mode="after")
    def _arms(self) -> StudyProfile:
        arms = self.expanded_arms()
        names = [a.name for a in arms]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"arm names repeat: {repeated}")
        nulls = [a.name for a in arms if a.null]
        if arms and len(nulls) != 1:
            raise ValueError(
                f"a study with arms scores them against exactly one null arm, not {len(nulls)}"
            )
        strangers = sorted(set(self.alignment.overrides) - set(names))
        if strangers:
            raise ValueError(f"alignment overrides name no arms {strangers}")
        twice = sorted(
            a.name for a in arms if a.name in self.alignment.overrides and METHOD in a.overrides
        )
        if twice:
            raise ValueError(f"arms {twice} set {METHOD} and an alignment override both")
        if METHOD in self.settings:
            raise ValueError(f"the study's method is alignment.method, not settings.{METHOD}")
        return self

    @field_validator("capacity_caps")
    @classmethod
    def _whole_heads(cls, value: dict[str, int]) -> dict[str, int]:
        negative = sorted(c for c, cap in value.items() if cap < 0)
        if negative:
            raise ValueError(f"caps are non-negative heads, which {negative} aren't")
        return value

    @field_validator("run_date")
    @classmethod
    def _first_of_month(cls, value: dt.date | None) -> dt.date | None:
        if value is not None and value.day != 1:
            raise ValueError("run_date must be the first of a month")
        return value

    @property
    def guided(self) -> bool:
        return self.mode is Mode.GUIDED_FORECAST

    def probability_tilts(self) -> tuple[ProbabilityTilt, ...]:
        return tuple(t.tilt() for t in self.tilts)

    def fill_share_targets(self) -> tuple[FillShareTarget, ...]:
        return tuple(f.target() for f in self.fill_shares)

    def expanded_arms(self) -> tuple[Arm, ...]:
        """The declared arms, then each group's, expanded."""
        return (*self.arms, *(arm for g in self.arm_groups for arm in g.expand()))

    def to_mapping(self) -> dict[str, Any]:
        """The profile as plain data, which `from_mapping` reads back as the same profile."""
        data: dict[str, Any] = self.model_dump(mode="json")
        return data

    def spec(self) -> StudySpec:
        """The frozen specification: this profile with its arm groups expanded."""
        return StudySpec(self, self.expanded_arms())

    @classmethod
    def from_mapping(cls, data: Any) -> StudyProfile:
        try:
            return cls.model_validate(data)
        except ValidationError as error:
            problems = [
                f"{'.'.join(str(p) for p in e['loc']) or 'profile'}: {e['msg']}"
                for e in error.errors()
            ]
            raise ProfileError("invalid study profile:\n  - " + "\n  - ".join(problems)) from None

    @classmethod
    def load(cls, path: Path) -> StudyProfile:
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as error:
            raise ProfileError(f"can't read the profile at {path}: {error}") from None
        return cls.from_mapping(data)


@dataclass(frozen=True)
class StudySpec:
    """A profile resolved for running: its arms expanded, and a digest over all of it."""

    profile: StudyProfile
    arms: tuple[Arm, ...]

    @property
    def null_arm(self) -> Arm | None:
        return next((a for a in self.arms if a.null), None)

    @property
    def compared(self) -> tuple[Arm, ...]:
        """The arms that rank: neither the null arm nor a reference arm."""
        return tuple(a for a in self.arms if not (a.null or a.reference))

    def digest(self) -> str:
        """`sp1-` and 16 hex characters of a SHA-256 over the profile and its expanded arms."""
        payload = self.profile.to_mapping() | {
            "arms": [a.model_dump(mode="json") for a in self.arms],
            "arm_groups": [],
        }
        text = json.dumps(payload, sort_keys=True, default=str)
        return "sp1-" + hashlib.sha256(text.encode()).hexdigest()[:16]
