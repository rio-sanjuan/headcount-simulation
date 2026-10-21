"""Generate a synthetic world and write it to DuckDB.

`generate_world` is pure: the same configuration and seed give identical tables. The tables
follow the data contract that hcsim's extraction SQL reads. Only `synth` writes them, and
`synth` imports nothing else from hcsim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from hcsim.synth import population, structure, workforce
from hcsim.synth.config import WorldConfig

GENERATOR_VERSION = 1

TXN_COLUMNS = [
    "employee_id",
    "reporting_date",
    "event_date",
    "trans_type",
    "amount",
    "cost_center",
    "cost_center_previous",
    "job_code",
    "job_code_previous",
    "location",
    "company",
    "account",
    "is_manager",
    "hire_date",
]


def month_ends(config: WorldConfig) -> list[pd.Timestamp]:
    first = pd.Period(config.start_month, freq="M")
    return [(first + i).end_time.normalize() for i in range(config.months)]


def active_rows(employees: pd.DataFrame, month_end: pd.Timestamp) -> pd.DataFrame:
    """One ACTIVE stock row per employee at a month-end."""
    rows = employees.assign(
        reporting_date=month_end.date(),
        event_date=month_end.date(),
        trans_type="ACTIVE",
        amount=1,
        cost_center_previous=employees["cost_center"],
        job_code_previous=employees["job_code"],
    )
    return rows[TXN_COLUMNS]


@dataclass
class World:
    config: WorldConfig
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    truth: dict[str, Any] = field(default_factory=dict)


def generate_world(config: WorldConfig) -> World:
    rng = np.random.default_rng(np.random.SeedSequence(config.seed))
    world = World(config)
    world.tables["dim_org"] = structure.org_dimension(config)
    world.tables["dim_job"] = structure.job_dimension(config)
    world.tables["dim_location"] = structure.location_dimension(config)
    world.tables["dim_company"] = structure.company_dimension(config)
    world.tables["dim_account"] = structure.account_dimension()
    people = population.initial_population(config, rng)
    txn, process_truth = workforce.simulate_workforce(
        people,
        month_ends(config),
        structure.cost_center_names(config),
        config.first_employee_id + len(people),
        rng,
        lambda cc, level, r: population.draw_role(config, cc, level, r),
    )
    world.tables["workforce_txn"] = txn[TXN_COLUMNS].reset_index(drop=True)
    world.truth = {
        "generator_version": GENERATOR_VERSION,
        "config": config.to_dict(),
        "processes": process_truth,
    }
    return world


def write_world(world: World, path: Path) -> None:
    """Write every table, and the truth manifest, to a fresh DuckDB file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with duckdb.connect(str(path)) as con:
        for name, frame in world.tables.items():
            con.register("frame", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
            con.unregister("frame")
        con.execute("CREATE TABLE truth_manifest (manifest JSON)")
        con.execute("INSERT INTO truth_manifest VALUES (?)", [json.dumps(world.truth)])
