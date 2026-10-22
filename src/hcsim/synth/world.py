"""Generate a synthetic world and write it to DuckDB.

`generate_world` is pure: the same configuration and seed give identical tables. The tables
follow the data contract that hcsim's extraction SQL reads. Only `synth` writes them, and
`synth` imports nothing else from hcsim.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth import equity, population, process, structure
from hcsim.synth.config import WorldConfig
from hcsim.synth.growth import LatentPlan
from hcsim.synth.movements import MovementProcess
from hcsim.synth.offers import PENDING_DTYPES, Offers
from hcsim.synth.programs import TerminationPrograms
from hcsim.synth.promotions import PromotionProcess
from hcsim.synth.ratings import RATING_DTYPES, RatingProcess
from hcsim.synth.requisitions import (
    CLOSE_REASONS,
    INTERVAL_DTYPES,
    OPEN,
    REQ_TYPES,
    RequisitionBook,
)
from hcsim.synth.workforce import TRANS_TYPES, TXN_DTYPES, Workforce

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
REQ_COLUMNS = [
    "req_id",
    "req_type",
    "cost_center",
    "job_code",
    "job_family",
    "job_level",
    "location",
    "company",
    "is_manager",
    "openings",
    "opened_date",
    "interval_start",
    "closed_date",
    "close_reason",
    "filled_by",
    "backfill_of",
]


def month_ends(config: WorldConfig) -> list[pd.Timestamp]:
    first = pd.Period(config.start_month, freq="M")
    return [(first + i).end_time.normalize() for i in range(config.months)]


@dataclass
class World:
    config: WorldConfig
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    truth: dict[str, Any] = field(default_factory=dict)


class _Labels:
    """Turns the world's integer codes into the data contract's labels."""

    def __init__(self, config: WorldConfig) -> None:
        self.config = config
        self.cost_centers = structure.cost_center_names(config)
        self.jobs = structure.job_codes(config)
        self.level_index = np.zeros(max(config.levels) + 1, dtype=np.int64)
        self.level_index[list(config.levels)] = np.arange(len(config.levels))
        # Position of each (family, level, manager) job in the catalog, or -1 for none.
        self.job_position = np.full((len(config.families), len(config.levels), 2), -1)
        for position, (family, level, manager) in enumerate(structure.jobs(config)):
            index = (config.families.index(family), config.levels.index(level), int(manager))
            self.job_position[index] = position

    @staticmethod
    def of(codes: np.ndarray, labels: Sequence[str]) -> pd.Categorical:
        # The stubs want a Sequence; an integer array is the fast path pandas takes anyway.
        return pd.Categorical.from_codes(
            cast("Sequence[int]", codes.astype(np.int64)), categories=pd.Index(list(labels))
        )

    def job(self, family: np.ndarray, level: np.ndarray, manager: np.ndarray) -> pd.Categorical:
        position = self.job_position[family, self.level_index[level], manager.astype(np.int64)]
        return self.of(position, self.jobs)

    def dates(self, values: np.ndarray) -> np.ndarray:
        return values.astype("datetime64[s]")


def _transactions(workforce: Workforce, labels: _Labels) -> pd.DataFrame:
    c = workforce.log.collect(TXN_DTYPES)
    config = labels.config
    return pd.DataFrame(
        {
            "employee_id": c["employee_id"],
            "reporting_date": labels.dates(c["reporting_date"]),
            "event_date": labels.dates(c["event_date"]),
            "trans_type": labels.of(c["trans_type"], TRANS_TYPES),
            "amount": c["amount"].astype(np.int64),
            "cost_center": labels.of(c["cost_center"], labels.cost_centers),
            "cost_center_previous": labels.of(c["cost_center_previous"], labels.cost_centers),
            "job_code": labels.job(c["family"], c["level"], c["is_manager"]),
            "job_code_previous": labels.job(
                c["family_previous"], c["level_previous"], c["is_manager_previous"]
            ),
            "location": labels.of(c["location"], config.locations),
            "company": labels.of(c["company"], config.companies),
            "account": labels.of(c["account"], [name for name, _ in config.accounts]),
            "is_manager": c["is_manager"],
            "hire_date": labels.dates(c["hire_date"]),
        },
        columns=TXN_COLUMNS,
    )


def _requisitions(book: RequisitionBook, labels: _Labels) -> pd.DataFrame:
    """One row per requisition and state interval. `openings` are those still to fill during
    the interval. `closed_date` and `close_reason` say how the interval ended, and both are
    empty for a requisition's current interval while it stays open."""
    config = labels.config
    col = book.column
    ended = book.intervals.collect(INTERVAL_DTYPES)
    current = book.open_requisitions()
    req = np.concatenate([ended["req"], current])
    start = np.concatenate([ended["interval_start"], col("interval_start")[current]])
    end = np.concatenate([ended["interval_end"], np.full(len(current), np.datetime64("NaT", "D"))])
    openings = np.concatenate([ended["openings"], col("remaining")[current]]).astype(np.int64)
    reason = np.concatenate([ended["end_reason"], np.full(len(current), OPEN)])
    filler = np.concatenate([ended["filled_by"], np.full(len(current), -1)])
    filled_by = pd.array(filler, dtype="Int64")
    filled_by[filler < 0] = pd.NA
    backfill_of = pd.array(col("backfill_of")[req], dtype="Int64")
    backfill_of[col("backfill_of")[req] < 0] = pd.NA
    frame = pd.DataFrame(
        {
            "req_id": col("req_id")[req],
            "req_type": labels.of(col("req_type")[req], REQ_TYPES),
            "cost_center": labels.of(col("cost_center")[req], labels.cost_centers),
            "job_code": labels.job(col("family")[req], col("level")[req], col("is_manager")[req]),
            "job_family": labels.of(col("family")[req], config.families),
            "job_level": col("level")[req].astype(np.int64),
            "location": labels.of(col("location")[req], config.locations),
            "company": labels.of(col("company")[req], config.companies),
            "is_manager": col("is_manager")[req],
            "openings": openings,
            "opened_date": labels.dates(col("opened_date")[req]),
            "interval_start": labels.dates(start),
            "closed_date": labels.dates(end),
            "close_reason": labels.of(reason, CLOSE_REASONS),
            "filled_by": filled_by,
            "backfill_of": backfill_of,
        },
        columns=REQ_COLUMNS,
    )
    return frame.sort_values(
        ["req_id", "interval_start", "closed_date"], na_position="last", ignore_index=True
    )


def _pending_starts(offers: Offers, book: RequisitionBook, labels: _Labels) -> pd.DataFrame:
    """One row per accepted offer and month-end between the offer and the start."""
    c = offers.pending.collect(PENDING_DTYPES)
    o, config = offers.book.column, labels.config
    k = c["offer"]
    return pd.DataFrame(
        {
            "snapshot_date": labels.dates(c["snapshot_date"]),
            "employee_id": o("employee_id")[k],
            "req_id": book.column("req_id")[o("req")[k]],
            "cost_center": labels.of(o("cost_center")[k], labels.cost_centers),
            "job_code": labels.job(o("family")[k], o("level")[k], o("is_manager")[k]),
            "location": labels.of(o("location")[k], config.locations),
            "company": labels.of(o("company")[k], config.companies),
            "offer_date": labels.dates(o("offer_date")[k]),
            "start_date": labels.dates(o("start_date")[k]),
        }
    )


def _ratings(ratings: RatingProcess, config: WorldConfig) -> pd.DataFrame:
    """One row per employee and calibration cycle, with the rating's label and rank."""
    c = ratings.log.collect(RATING_DTYPES)
    return pd.DataFrame(
        {
            "employee_id": c["employee_id"],
            "cycle_date": c["cycle_date"].astype("datetime64[s]"),
            "rating": _Labels.of(c["rating"], config.ratings.scale),
            "rating_rank": c["rating"].astype(np.int64) + 1,
            "is_first": c["is_first"],
        }
    )


def _as_dates(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Plain `datetime.date` values, for the equity world's per-award arithmetic."""
    return frame.assign(**{c: frame[c].dt.date for c in columns})


def generate_world(config: WorldConfig) -> World:
    # One spawned stream per subsystem, so adding a subsystem never perturbs the others.
    # A stream's draws depend only on its position, so new streams go at the end.
    streams = np.random.SeedSequence(config.seed).spawn(10)
    (
        population_rng,
        process_rng,
        price_rng,
        grant_rng,
        structure_rng,
        rating_rng,
        promotion_rng,
        movement_rng,
        program_rng,
        offer_rng,
    ) = (np.random.default_rng(s) for s in streams)
    world = World(config)
    world.tables["dim_org"] = structure.org_dimension(config, structure_rng)
    world.tables["dim_job"] = structure.job_dimension(config)
    world.tables["dim_location"] = structure.location_dimension(config)
    world.tables["dim_company"] = structure.company_dimension(config)
    world.tables["dim_account"] = structure.account_dimension(config)
    ids, roles, hire_dates = population.initial_population(config, population_rng)
    workforce = Workforce(ids, roles, hire_dates)
    book = RequisitionBook()
    ends = month_ends(config)
    ratings = RatingProcess(config.ratings, workforce, rating_rng)
    org = world.tables["dim_org"]
    tops = structure.level_one_names(config.org_level1_nodes)
    org_of_cost_center: NDArray[np.int64] = np.asarray(
        org["org_level1"].map({t: i for i, t in enumerate(tops)}), dtype=np.int64
    )
    promotions = PromotionProcess(config, workforce, promotion_rng)
    movements = MovementProcess(config, workforce, org_of_cost_center, movement_rng)
    movements.seed(pd.Period(ends[0], freq="M"))
    offers = Offers(config, offer_rng)
    process_truth = process.simulate(
        config,
        workforce,
        book,
        ends,
        process_rng,
        ratings,
        org_of_cost_center,
        promotions,
        movements,
        TerminationPrograms(config, workforce, org_of_cost_center, program_rng),
        offers,
    )
    plan: LatentPlan = process_truth.pop("plan")
    world.tables["truth_latent_plan"] = plan.frame(structure.cost_center_names(config))
    labels = _Labels(config)
    world.tables["rating"] = _ratings(ratings, config)
    txn = _transactions(workforce, labels)
    world.tables["workforce_txn"] = txn
    world.tables["req_interval"] = _requisitions(book, labels)
    world.tables["pending_start"] = _pending_starts(offers, book, labels)
    world.tables["calendar"] = pd.DataFrame({"month_end": [d.date() for d in ends]})
    comp, price, grant = config.compensation, config.prices, config.grants
    world.tables["comp_range"] = equity.compensation_ranges(
        config.families, config.levels, config.locations, comp
    )
    world.tables["stock_price"] = equity.stock_prices(ends, price, price_rng)
    hires = _as_dates(txn[txn["trans_type"] == "HIRE"].astype({"job_code": str}), "event_date")
    terms = _as_dates(txn[txn["trans_type"] == "TERM"], "event_date")
    awards, vests = equity.new_hire_awards(
        hires,
        terms,
        world.tables["dim_job"],
        world.tables["stock_price"],
        grant,
        price,
        ends[-1].date(),
        grant_rng,
    )
    vests = equity.vest_status(vests, awards, terms, ends[-1].date())
    world.tables["equity_award"] = awards
    world.tables["equity_vest"] = vests
    world.tables["equity_expense"] = equity.reference_ledger(awards, vests, ends)
    world.truth = {
        "generator_version": GENERATOR_VERSION,
        "config": config.to_dict(),
        "processes": process_truth | {"equity": equity.truth_dict(comp, price, grant)},
    }
    return world


def _select_list(frame: pd.DataFrame) -> str:
    """Columns as the warehouse stores them: categoricals as text, timestamps as dates."""
    parts = []
    for name, dtype in frame.dtypes.items():
        quoted = f'"{name}"'
        if isinstance(dtype, pd.CategoricalDtype):
            parts.append(f"CAST({quoted} AS VARCHAR) AS {quoted}")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            parts.append(f"CAST({quoted} AS DATE) AS {quoted}")
        else:
            parts.append(quoted)
    return ", ".join(parts)


def write_world(world: World, path: Path) -> None:
    """Write every table, and the truth manifest, to a fresh DuckDB file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with duckdb.connect(str(path)) as con:
        for name, frame in world.tables.items():
            con.register("frame", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT {_select_list(frame)} FROM frame")
            con.unregister("frame")
        # Month-end snapshots of every open requisition, with the openings still to fill then.
        # A requisition opened and closed inside one month never appears, as with real
        # snapshot data.
        con.execute(
            """
            CREATE VIEW req_snapshot AS
            SELECT c.month_end AS snapshot_date, r.req_id, r.req_type, r.cost_center,
                   r.job_code, r.location, r.company, r.is_manager, r.openings,
                   'open' AS state, r.opened_date
            FROM calendar c
            JOIN req_interval r
              ON r.interval_start <= c.month_end
             AND (r.closed_date IS NULL OR r.closed_date > c.month_end)
            """
        )
        con.execute("CREATE TABLE truth_manifest (manifest JSON)")
        con.execute("INSERT INTO truth_manifest VALUES (?)", [json.dumps(world.truth)])
