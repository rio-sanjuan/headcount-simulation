"""Study profiles: what a study runs, read from YAML.

This is the walking-skeleton profile: a forecast scope, a run mode, a seed, and the run's size.
Unknown keys are refused, so a misspelled field fails loudly instead of running the default.
The full schema, with arms and the rest of the run-mode ladder, arrives in M8.2.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hcsim.engine.config import ForecastScope


class ProfileError(ValueError):
    """A profile that can't be read or doesn't validate."""


class Mode(StrEnum):
    """The run modes the skeleton supports. `replay` and `guided_actuals` arrive in M8.1."""

    GUIDED_FORECAST = "guided_forecast"
    UNGUIDED = "unguided"


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

    @field_validator("run_date")
    @classmethod
    def _first_of_month(cls, value: dt.date | None) -> dt.date | None:
        if value is not None and value.day != 1:
            raise ValueError("run_date must be the first of a month")
        return value

    @property
    def guided(self) -> bool:
        return self.mode is Mode.GUIDED_FORECAST

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
