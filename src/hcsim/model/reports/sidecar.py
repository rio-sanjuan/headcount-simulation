"""A report's provenance and its metrics sidecar, both validated when written and when read.

`Provenance` names everything behind a training run: the world (its generator, preset, seed,
and data contract), the population, the schema, the split, the decision margin, the package
version, and each domain's champion version. `provenance(con, run)` reads it from the world
and the run.

`MetricsSidecar` holds every arm's metrics, the selection decisions, and the provenance, as
`metrics.json` beside the report. A sidecar is refused if it doesn't hold together:

- a domain's family is known, and its primary metric is the family's;
- every metric belongs to its family, and every value is finite or null;
- the champion and every decision name the domain's own arms, and a decision that cleared
  gained something;
- a version reads as a model version, and each provenance version is its domain's champion's.

A sidecar may also hold recovery rows, each a quantity's truth, fit, and held-out value, but only
for a synthetic world, which has a truth to recover. A sidecar is plain data, so tools other
than this package can read it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal, Self

import duckdb
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hcsim.model.data.population import PopulationStamp
from hcsim.model.ml.metrics import FAMILY_METRICS
from hcsim.model.ml.training import TrainingRun

SIDECAR_FILE: Final = "metrics.json"
SIDECAR_VERSION: Final = 1
_MODEL_VERSION: Final = re.compile(r"v1-[0-9a-f]{12}")
_SCHEMA_FINGERPRINT: Final = re.compile(r"sc2-[0-9a-f]+")


class SidecarError(ValueError):
    """A sidecar or provenance that doesn't hold together."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Provenance(_Strict):
    source: Literal["synthetic_world", "warehouse"]
    world_generator: str = ""
    world_preset: str = ""
    world_seed: str = ""
    contract_version: str = ""
    population: str
    schema_fingerprint: str
    split_start: dt.date
    split_cutoff: dt.date
    split_end: dt.date
    decision_z: float = Field(gt=0)
    hcsim_version: str = Field(min_length=1)
    #: Each domain's champion version, where the run saved its models.
    models: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _holds(self) -> Self:
        problems = []
        try:
            PopulationStamp.parse(self.population)
        except ValueError as error:
            problems.append(str(error))
        if not _SCHEMA_FINGERPRINT.fullmatch(self.schema_fingerprint):
            problems.append(f"{self.schema_fingerprint!r} isn't a schema fingerprint")
        if not self.split_start < self.split_cutoff < self.split_end:
            problems.append("the split's dates must run start < cutoff < end")
        bad = sorted(d for d, v in self.models.items() if not _MODEL_VERSION.fullmatch(v))
        if bad:
            problems.append(f"versions that aren't model versions, for {bad}")
        if self.source == "synthetic_world" and not (self.world_preset and self.world_seed):
            problems.append("a synthetic world's provenance names its preset and seed")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class ArmMetrics(_Strict):
    algorithm: str
    version: str | None = None
    #: The mean per-row contribution to the family's primary metric.
    primary: float
    metrics: dict[str, float | None]


class Decision(_Strict):
    arm: str
    against: str
    mean_gain: float
    standard_error: float = Field(ge=0)
    cleared: bool


class DomainMetrics(_Strict):
    family: str
    primary_metric: str
    champion: str
    train_rows: int = Field(gt=0)
    test_rows: int = Field(gt=0)
    arms: dict[str, ArmMetrics]
    decisions: list[Decision]

    @model_validator(mode="after")
    def _holds(self) -> Self:
        if self.family not in FAMILY_METRICS:
            raise ValueError(f"no model family {self.family!r}")
        problems = []
        if self.primary_metric != FAMILY_METRICS[self.family][0]:
            problems.append(f"a {self.family}'s primary metric is {FAMILY_METRICS[self.family][0]}")
        if self.champion not in self.arms:
            problems.append(f"the champion {self.champion!r} isn't among the arms")
        allowed = set(FAMILY_METRICS[self.family])
        for name, arm in self.arms.items():
            stray = sorted(set(arm.metrics) - allowed)
            if stray:
                problems.append(f"{name} has metrics outside its family: {stray}")
            if arm.version is not None and not _MODEL_VERSION.fullmatch(arm.version):
                problems.append(f"{name}'s version {arm.version!r} isn't a model version")
        for d in self.decisions:
            if {d.arm, d.against} - set(self.arms):
                problems.append(f"a decision between {d.arm} and {d.against} names other arms")
            if d.cleared and d.mean_gain <= 0:
                problems.append(f"{d.arm} cleared the margin without gaining")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class RecoveryRow(_Strict):
    """One quantity's truth, its fit, and what the held-out data showed."""

    quantity: str = Field(min_length=1)
    truth: float
    fitted: float
    observed: float | None = None


class MetricsSidecar(_Strict):
    sidecar_version: Literal[1] = SIDECAR_VERSION
    provenance: Provenance
    domains: dict[str, DomainMetrics]
    #: Each recovery's rows, from a synthetic world's truth.
    recovery: dict[str, list[RecoveryRow]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _holds(self) -> Self:
        problems = []
        if self.recovery and self.provenance.source != "synthetic_world":
            problems.append("recovery needs a synthetic world's truth")
        for domain, model in self.provenance.models.items():
            if domain not in self.domains:
                problems.append(f"the provenance names a model for {domain}, which has no metrics")
            elif (d := self.domains[domain]).arms[d.champion].version != model:
                problems.append(f"{domain}'s provenance version isn't its champion's")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @classmethod
    def of(
        cls,
        run: TrainingRun,
        provenance: Provenance,
        recovery: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> MetricsSidecar:
        data = run.to_dict()
        domains = {}
        for name, result in data["domains"].items():
            domains[name] = {
                "family": result["family"],
                "primary_metric": FAMILY_METRICS[result["family"]][0],
                "champion": result["champion"],
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "arms": result["arms"],
                "decisions": result["decisions"],
            }
        return cls.validated(
            {
                "provenance": provenance.model_dump(),
                "domains": domains,
                "recovery": {k: list(v) for k, v in (recovery or {}).items()},
            }
        )

    @classmethod
    def validated(cls, data: object) -> MetricsSidecar:
        """The sidecar `data` describes, or every reason it doesn't hold, at once."""
        try:
            return cls.model_validate(data)
        except ValidationError as error:
            problems = "; ".join(_reason(e) for e in error.errors())
            raise SidecarError(f"invalid metrics sidecar: {problems}") from None

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / SIDECAR_FILE
        text = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)
        path.write_text(text + "\n")
        return path

    @classmethod
    def load(cls, directory: Path) -> MetricsSidecar:
        path = directory / SIDECAR_FILE
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SidecarError(f"can't read a metrics sidecar at {path}: {error}") from None
        return cls.validated(data)


def _reason(error: object) -> str:
    detail = dict(error)  # type: ignore[call-overload]
    where = ".".join(str(p) for p in detail.get("loc", ()))
    message = str(detail.get("msg", "")).removeprefix("Value error, ")
    return f"{where}: {message}" if where else message


def provenance(con: duckdb.DuckDBPyConnection, run: TrainingRun) -> Provenance:
    """The run's provenance, with the world's identity where it records one."""
    (present,) = con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'truth' AND table_name = 'manifest'"
    ).fetchone() or (0,)
    world: dict[str, str] = {"source": "warehouse"}
    if present:
        (text,) = con.execute("SELECT manifest FROM truth.manifest").fetchone() or ("{}",)
        truth = json.loads(text)
        config = truth.get("config", {})
        world = {
            "source": "synthetic_world",
            "world_generator": str(truth.get("generator_version", "")),
            "world_preset": str(config.get("preset", "")),
            "world_seed": str(config.get("seed", "")),
            "contract_version": str(truth.get("contract_version", "")),
        }
    try:
        return Provenance(
            **world,
            population=run.population,
            schema_fingerprint=run.schema,
            split_start=run.split.start,
            split_cutoff=run.split.cutoff,
            split_end=run.split.end,
            decision_z=run.z,
            hcsim_version=version("hcsim"),
            models=run.chosen,
        )
    except ValidationError as error:
        problems = "; ".join(_reason(e) for e in error.errors())
        raise SidecarError(f"invalid provenance: {problems}") from None
