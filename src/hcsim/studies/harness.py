"""The one-arm harness: from a profile, a world, and a guidance artifact to evaluated results.

It reads every input once, before simulating: the schema, the seed state, the fitted models,
the comp ranges and opening price when the run computes SBC, and the guidance artifact when the
mode is guided. The provenance it records names every input, so a scorecard row can be traced
to the exact world, guidance, and schema behind it.

Arms, preflight, subsets, and persistence arrive in M8.4.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import duckdb

from hcsim.engine.config import AwardSettings, ForecastScope, SimulationConfig
from hcsim.engine.loop import SimulationResult, simulate
from hcsim.engine.seeding import (
    SeedReport,
    opening_price,
    range_table_from_world,
    schema_from_world,
    seed_state,
    snapshot_before,
)
from hcsim.financial.prices import AnnualGrowthPrice
from hcsim.financial.sizing import NewHireGapFill
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.studies.evaluators import Metric, invariants, summary
from hcsim.studies.inputs import FittedModels, fit_models
from hcsim.studies.profile import StudyProfile


class StudyError(ValueError):
    """A study whose inputs don't fit its profile."""


@dataclass(frozen=True)
class StudyRun:
    profile: StudyProfile
    config: SimulationConfig
    seed: SeedReport
    fitted: FittedModels
    result: SimulationResult
    metrics: tuple[Metric, ...]
    provenance: dict[str, str]

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.metrics)


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


def run_study(profile: StudyProfile, world: Path, guidance: Path | None = None) -> StudyRun:
    if profile.guided and guidance is None:
        raise StudyError(f"mode {profile.mode.value!r} needs a guidance artifact")
    if not profile.guided and guidance is not None:
        raise StudyError(f"mode {profile.mode.value!r} reads no guidance; drop the artifact")
    if not world.is_file():
        raise StudyError(f"no world at {world}")
    artifact = GuidanceForecast.load(guidance) if guidance is not None else None

    with duckdb.connect(str(world), read_only=True) as con:
        schema = schema_from_world(con)
        if profile.run_date is not None:
            run_date = profile.run_date
        elif artifact is not None:
            run_date = dt.date.fromisoformat(artifact.snapshot_date) + dt.timedelta(days=1)
        else:
            run_date = snapshot_before(con, dt.date.max) + dt.timedelta(days=1)
        sbc = profile.scope is ForecastScope.HEADCOUNT_AND_SBC
        state, seed = seed_state(con, schema, run_date, with_ledger=sbc)
        fitted = fit_models(con, schema, run_date)
        awards = (
            AwardSettings(
                NewHireGapFill(range_table_from_world(con, schema)),
                AnnualGrowthPrice(opening_price(con, run_date)),
            )
            if sbc
            else None
        )
        provenance = _world_identity(con)

    config = SimulationConfig(
        run_date=run_date,
        schema=schema,
        models=fitted.models,
        simulation_months=profile.months,
        num_trajectories=profile.trajectories,
        seed=profile.seed,
        forecast_scope=profile.scope,
        guidance=artifact,
        awards=awards,
    )
    result = simulate(config, state)
    provenance |= {
        "hcsim_version": version("hcsim"),
        "profile": profile.name,
        "mode": profile.mode.value,
        "scope": profile.scope.value,
        "seed": str(profile.seed),
        "run_date": run_date.isoformat(),
        "schema_fingerprint": schema.fingerprint(),
        "guidance_fingerprint": artifact.fingerprint() if artifact is not None else "",
    }
    metrics = (*invariants(result), *summary(result))
    return StudyRun(profile, config, seed, fitted, result, metrics, provenance)
