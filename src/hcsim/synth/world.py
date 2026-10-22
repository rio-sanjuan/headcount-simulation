"""Generate a synthetic world and write it to DuckDB.

`generate_world` is pure: the same configuration and seed give identical tables. The tables
follow the data contract that hcsim's extraction SQL reads. Only `synth` writes them, and
`synth` imports nothing else from hcsim.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.synth import (
    awards,
    codes,
    equity,
    plans,
    population,
    process,
    quirks,
    structure,
    truth,
)
from hcsim.synth.columns import categorical
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
#: The version of `docs/data-contract.md` the tables follow.
CONTRACT_VERSION = "1.0"
#: DuckDB's working memory while a world is written.
WRITE_MEMORY_LIMIT = "1GB"

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
    "location_previous",
    "company",
    "company_previous",
    "account",
    "account_previous",
    "is_manager",
    "is_manager_previous",
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
    #: The source tables, as a warehouse would hold them.
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: The `truth/` manifest: every parameter and every draw the run recorded.
    truth: dict[str, Any] = field(default_factory=dict)
    #: The `truth/` tables (see `hcsim.synth.truth`).
    truth_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Source tables too big to hold whole at scale, produced in chunks when they're written
    #: or asked for. A table of the same name in `tables` takes their place.
    deferred: dict[str, Callable[[], Iterator[pd.DataFrame]]] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        """A source table, assembled whole if it's deferred."""
        if name in self.tables:
            return self.tables[name]
        return pd.concat(list(self.deferred[name]()), ignore_index=True)

    def chunks(self, name: str) -> Iterator[pd.DataFrame]:
        """A source table, a chunk at a time."""
        if name in self.tables:
            yield self.tables[name]
        else:
            yield from self.deferred[name]()

    @property
    def table_names(self) -> list[str]:
        return sorted(set(self.tables) | set(self.deferred))


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
        return categorical(codes, labels)

    def job(self, family: np.ndarray, level: np.ndarray, manager: np.ndarray) -> pd.Categorical:
        position = self.job_position[family, self.level_index[level], manager.astype(np.int64)]
        return self.of(position, self.jobs)

    def dates(self, values: np.ndarray) -> np.ndarray:
        return values.astype("datetime64[s]")


def _reporting_dates(
    c: dict[str, np.ndarray], config: WorldConfig, rng: np.random.Generator
) -> np.ndarray:
    """When each row reached the warehouse. A row recorded with its own reporting date, such as
    a notice, keeps it; stocks report on their month-end; every other event reports on the day
    or some days later."""
    reported = c["reporting_date"].copy()
    lagged = (c["reporting_date"] == c["event_date"]) & (
        c["trans_type"] != TRANS_TYPES.index("ACTIVE")
    )
    t = config.dates
    late = rng.random(len(reported)) >= t.same_day_reporting_share
    lag = np.where(late, rng.integers(1, t.reporting_lag_days + 1, len(reported)), 0)
    reported[lagged] = reported[lagged] + lag[lagged].astype("timedelta64[D]")
    return reported


def _transactions(workforce: Workforce, labels: _Labels, rng: np.random.Generator) -> pd.DataFrame:
    c = workforce.log.collect(TXN_DTYPES, release=True)  # nothing reads the log after this
    config = labels.config
    accounts = [name for name, _ in config.accounts]
    # Each raw column is released as soon as it's labelled, so the raw log and the finished
    # table are never held whole at once.
    reporting = labels.dates(_reporting_dates(c, config, rng))
    columns: dict[str, Any] = {}
    columns["employee_id"] = c.pop("employee_id")
    columns["reporting_date"] = reporting
    c.pop("reporting_date")
    columns["event_date"] = labels.dates(c.pop("event_date"))
    columns["trans_type"] = labels.of(c.pop("trans_type"), TRANS_TYPES)
    columns["amount"] = c.pop("amount").astype(np.int64)
    columns["cost_center"] = labels.of(c.pop("cost_center"), labels.cost_centers)
    columns["cost_center_previous"] = labels.of(c.pop("cost_center_previous"), labels.cost_centers)
    columns["job_code"] = labels.job(c.pop("family"), c.pop("level"), c["is_manager"])
    columns["job_code_previous"] = labels.job(
        c.pop("family_previous"), c.pop("level_previous"), c["is_manager_previous"]
    )
    for name, names in (("location", config.locations), ("company", config.companies)):
        columns[name] = labels.of(c.pop(name), names)
        columns[f"{name}_previous"] = labels.of(c.pop(f"{name}_previous"), names)
    columns["account"] = labels.of(c.pop("account"), accounts)
    columns["account_previous"] = labels.of(c.pop("account_previous"), accounts)
    columns["is_manager"] = c.pop("is_manager")
    columns["is_manager_previous"] = c.pop("is_manager_previous")
    columns["hire_date"] = labels.dates(c.pop("hire_date"))
    return pd.DataFrame({name: columns.pop(name) for name in TXN_COLUMNS}, copy=False)


def _requisitions(book: RequisitionBook, labels: _Labels) -> pd.DataFrame:
    """One row per requisition and state interval, numbered from 1 within the requisition.
    `openings` are those still to fill during the interval. `closed_date` and `close_reason`
    say how the interval ended, and both are empty for a requisition's current interval while
    it stays open."""
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
    # Ties keep the order the intervals ended in, which is the order of the state changes.
    frame = frame.sort_values(
        ["req_id", "interval_start", "closed_date"],
        na_position="last",
        ignore_index=True,
        kind="stable",
    )
    frame.insert(1, "interval_number", frame.groupby("req_id").cumcount().add(1))
    return frame


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
    """One row per employee and calibration cycle, with the rating's label, its rank on its
    scale, and the scale it's reported on. After a scale change, cycles report the new scale."""
    c = ratings.log.collect(RATING_DTYPES)
    truth = config.ratings
    code = c["rating"].astype(np.int64)
    labels = list(truth.scale)
    revised = np.zeros(len(code), dtype=bool)
    if truth.scale_change is not None:
        change = truth.scale_change
        year = c["cycle_date"].astype("datetime64[Y]").astype(np.int64) + 1970
        revised = year >= change.first_year
        code = np.where(revised, np.asarray(change.mapping)[code], code)
        labels = labels + list(change.scale)
    position = np.where(revised, code + len(truth.scale), code)
    return pd.DataFrame(
        {
            "employee_id": c["employee_id"],
            "cycle_date": c["cycle_date"].astype("datetime64[s]"),
            "rating": _Labels.of(position, labels),
            "rating_rank": code + 1,
            "rating_scale": np.where(revised, "revised", "original"),
            "is_first": c["is_first"],
        }
    )


AGG_KEYS = [
    "period",
    "cost_center",
    "job_family",
    "job_level",
    "location",
    "company",
    "account",
    "trans_type",
]


def _aggregate(txn: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    """`workforce_agg`: the transactions pre-summed by month, org and job attributes, account,
    and code. `headcount` counts rows and `amount` sums their signed amounts. A month is the
    event's month, and `period` is its month-end.

    Groups are found by sorting one packed integer key per row, in the keys' order, rather
    than by a frame-wide group-by that would copy the table several times over at scale."""
    job = jobs.set_index("job_code").reindex(txn["job_code"].cat.categories)
    families = list(dict.fromkeys(jobs["job_family"]))
    family_of = pd.Categorical(job["job_family"], categories=families).codes.astype(np.int64)
    level_of = job["job_level"].to_numpy(np.int64)
    levels = np.unique(level_of)
    rank_of = np.searchsorted(levels, level_of)
    months = txn["event_date"].to_numpy().astype("datetime64[M]")
    first = months.min() if len(months) else np.datetime64("2000-01", "M")
    job_code = txn["job_code"].cat.codes.to_numpy(np.int64)
    coded = {
        "period": (months - first).astype(np.int64),
        "cost_center": txn["cost_center"].cat.codes.to_numpy(np.int64),
        "job_family": np.where(job_code >= 0, family_of[job_code], -1),
        "job_level": np.where(job_code >= 0, rank_of[job_code], -1),
        **{
            name: txn[name].cat.codes.to_numpy(np.int64)
            for name in ("location", "company", "account", "trans_type")
        },
    }
    del months, job_code
    sizes = {
        "period": int(coded["period"].max(initial=0)) + 1,
        "cost_center": len(txn["cost_center"].cat.categories),
        "job_family": len(families),
        "job_level": len(levels),
        **{n: len(txn[n].cat.categories) for n in ("location", "company", "account", "trans_type")},
    }
    assert float(np.prod([float(v) for v in sizes.values()])) < 2.0**62, "aggregate key overflow"
    # A row with any key missing belongs to no group, as in a group-by that drops them.
    keep = np.ones(len(txn), dtype=bool)
    key = np.zeros(len(txn), dtype=np.int64)
    for name in AGG_KEYS:
        keep &= coded[name] >= 0
        key = key * sizes[name] + coded.pop(name)
    rows = np.flatnonzero(keep)
    order = rows[np.argsort(key[rows], kind="stable")]
    del rows, keep
    ordered = key[order]
    del key
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]][: len(ordered)])
    packed = ordered[starts]
    del ordered
    amount = (
        np.add.reduceat(txn["amount"].to_numpy(np.int64)[order], starts)
        if len(starts)
        else np.zeros(0, np.int64)
    )
    headcount = np.diff(np.r_[starts, len(order)]).astype(np.int64)
    del order
    decoded: dict[str, NDArray[np.int64]] = {}
    for name in reversed(AGG_KEYS):
        packed, decoded[name] = np.divmod(packed, sizes[name])
    month = first + decoded["period"].astype("timedelta64[M]")
    period_end = (month + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    columns: dict[str, Any] = {
        "period": period_end.astype("datetime64[ns]"),
        "cost_center": categorical(decoded["cost_center"], list(txn["cost_center"].cat.categories)),
        "job_family": categorical(decoded["job_family"], families),
        "job_level": levels[decoded["job_level"]],
        **{
            name: categorical(decoded[name], list(txn[name].cat.categories))
            for name in ("location", "company", "account", "trans_type")
        },
        "headcount": headcount,
        "amount": amount,
    }
    return pd.DataFrame(columns, copy=False)


def _rating_dimension(config: WorldConfig) -> pd.DataFrame:
    """Every rating label with its rank on its scale, and the scale it belongs to."""
    rows = [
        {"rating": label, "rating_rank": k + 1, "rating_scale": "original"}
        for k, label in enumerate(config.ratings.scale)
    ]
    if config.ratings.scale_change is not None:
        rows += [
            {"rating": label, "rating_rank": k + 1, "rating_scale": "revised"}
            for k, label in enumerate(config.ratings.scale_change.scale)
        ]
    return pd.DataFrame(rows)


def generate_world(config: WorldConfig) -> World:
    # One spawned stream per subsystem, so adding a subsystem never perturbs the others.
    # A stream's draws depend only on its position, so new streams go at the end.
    streams = np.random.SeedSequence(config.seed).spawn(13)
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
        reporting_rng,
        plan_rng,
        quirk_rng,
    ) = (np.random.default_rng(s) for s in streams)
    world = World(config)
    world.tables["dim_org"] = structure.org_dimension(config, structure_rng)
    world.tables["dim_job"] = structure.job_dimension(config)
    world.tables["dim_location"] = structure.location_dimension(config)
    world.tables["dim_company"] = structure.company_dimension(config)
    world.tables["dim_account"] = structure.account_dimension(config)
    world.tables["dim_trans_type"] = codes.catalog_frame()
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
    headcount = process_truth.pop("headcount")
    shock = process_truth.pop("shock")
    names = structure.cost_center_names(config)
    vintages = plans.vintages(
        config,
        plan,
        headcount,
        process_truth.pop("family_headcount"),
        names,
        org_of_cost_center,
        tops,
        plan_rng,
    )
    # A cell's regime is the truth: the warehouse sees only the targets.
    world.tables["plan_headcount"] = vintages.drop(columns="regime")
    labels = _Labels(config)
    world.tables["rating"] = _ratings(ratings, config)
    # The world's true transactions drive the equity world; the warehouse holds them as
    # recorded, quirks and all, and its aggregate is built from what it holds.
    clean = _transactions(workforce, labels, reporting_rng)
    txn = quirks.apply(config, clean, offers, quirk_rng)
    world.tables["workforce_txn"] = txn
    world.tables["dim_rating"] = _rating_dimension(config)
    world.tables["req_interval"] = _requisitions(book, labels)
    world.tables["pending_start"] = _pending_starts(offers, book, labels)
    world.tables["calendar"] = pd.DataFrame({"month_end": [d.date() for d in ends]})
    comp, price, grant = config.compensation, config.prices, config.grants
    world.tables["comp_range"] = equity.compensation_ranges(
        config.families,
        config.levels,
        config.locations,
        comp,
        np.random.default_rng([config.seed, 257]),
    )
    daily = equity.daily_prices(ends, price, price_rng)
    world.tables["stock_price_daily"] = daily
    world.tables["stock_price"] = equity.month_end_closes(daily, ends)
    logged = ratings.log.collect(RATING_DTYPES)
    rating_codes = pd.DataFrame(
        {
            "employee_id": logged["employee_id"],
            "cycle_date": logged["cycle_date"],
            "rating": logged["rating"].astype(np.int64),
        }
    )
    equity_tables = awards.make_awards(
        config,
        clean,
        rating_codes,
        world.tables["dim_job"],
        daily,
        world.tables["stock_price"],
        grant_rng,
    )
    world.tables.update(equity_tables)
    estimates = awards.forfeiture_estimates(
        config, equity_tables["equity_award"], equity_tables["equity_vest"]
    )
    world.tables["equity_forfeiture_estimate"] = estimates
    # The ledger is the biggest table by far, and nothing else reads it: it's built in chunks
    # as it's written.
    world.deferred["equity_expense"] = functools.partial(
        awards.ledger_frames,
        config,
        equity_tables["equity_award"],
        equity_tables["equity_vest"],
        estimates,
        ends,
    )
    # The aggregate comes last: nothing above reads it, so it isn't held while the equity world
    # is built. The quirk stream draws in the same order as ever: rows, aggregate, snapshots.
    world.tables["workforce_agg"] = quirks.aggregate_only(
        config, _aggregate(txn, world.tables["dim_job"]), quirk_rng
    )
    del clean, txn
    world.tables["contract_columns"] = quirks.cost_center_columns(config, world.tables)
    quirks.rename_cost_centers(config, world.tables)
    gaps = quirks.snapshot_gaps(config, ends, quirk_rng)
    world.truth = {
        "quirks": {"snapshot_gaps": gaps},
        "generator_version": GENERATOR_VERSION,
        "contract_version": CONTRACT_VERSION,
        "config": config.to_dict(),
        "processes": process_truth | {"equity": equity.truth_dict(comp, price, grant)},
    }
    world.truth_tables = {
        "latent_plan": plan.frame(names),
        "plan_regime": truth.plan_regime(vintages),
        "attrition_hazard": truth.attrition_hazard(config),
        "attrition_season": truth.attrition_season(config),
        "attrition_shock": truth.attrition_shock(shock, ends, names),
        "cost_center_effect": truth.cost_center_effect(process_truth),
        "requisition_hazard": truth.requisition_hazard(config),
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


def digest(world: World) -> str:
    """A fingerprint of everything the world holds: every table's columns, types, and values,
    the truth tables, and the manifest. Equal digests mean identical worlds."""
    h = hashlib.sha256()

    def add(kind: str, name: str, frames: Iterator[pd.DataFrame]) -> None:
        for k, frame in enumerate(frames):
            if k == 0:  # a deferred table hashes as if it were whole
                header = f"{kind}:{name}:{list(frame.columns)}:{list(map(str, frame.dtypes))}"
                h.update(header.encode())
            h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())

    for name in world.table_names:
        add("table", name, world.chunks(name))
    for name in sorted(world.truth_tables):
        add("truth", name, iter([world.truth_tables[name]]))
    h.update(json.dumps(world.truth, sort_keys=True).encode())
    return h.hexdigest()


def write_world(world: World, path: Path) -> None:
    """Write every table, and the truth manifest, to a fresh DuckDB file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with duckdb.connect(str(path)) as con:
        # Writing needs little working memory; capped, DuckDB flushes as it goes rather than
        # buffering up to most of the machine's memory.
        con.execute(f"SET memory_limit = '{WRITE_MEMORY_LIMIT}'")
        for name in [*world.tables, *(n for n in world.deferred if n not in world.tables)]:
            for k, frame in enumerate(world.chunks(name)):
                con.register("frame", frame)
                select = f"SELECT {_select_list(frame)} FROM frame"
                con.execute(
                    f"CREATE TABLE {name} AS {select}" if k == 0 else f"INSERT INTO {name} {select}"
                )
                con.unregister("frame")
        # Month-end snapshots of every open requisition, with the openings still to fill then.
        # A requisition opened and closed inside one month never appears, as with real
        # snapshot data. Quirks may drop whole month-ends, and spell the state differently
        # from a seam month on.
        recorded = world.truth.get("quirks", {})
        gaps = ", ".join(f"DATE '{d}'" for d in recorded.get("snapshot_gaps", []))
        seam = world.config.quirks.state_seam_month
        # The snapshot feed names the cost center its own way: the canonical name.
        req_cost_center = (
            quirks.COST_CENTER_ALIASES["req_interval"]
            if world.config.quirks.cost_center_aliases
            else "cost_center"
        )
        state = (
            f"CASE WHEN c.month_end >= DATE '{pd.Period(seam, freq='M').start_time.date()}' "
            "THEN 'OPEN' ELSE 'open' END"
            if seam
            else "'open'"
        )
        con.execute(
            f"""
            CREATE VIEW req_snapshot AS
            SELECT c.month_end AS snapshot_date, r.req_id, r.req_type,
                   r.{req_cost_center} AS cost_center,
                   r.job_code, r.location, r.company, r.is_manager, r.openings,
                   {state} AS state, r.opened_date
            FROM calendar c
            JOIN req_interval r
              ON r.interval_start <= c.month_end
             AND (r.closed_date IS NULL OR r.closed_date > c.month_end)
            {f"WHERE c.month_end NOT IN ({gaps})" if gaps else ""}
            """
        )
        # The truth lives in its own schema, apart from the source tables.
        con.execute("CREATE SCHEMA truth")
        con.execute("CREATE TABLE truth.manifest (manifest JSON)")
        con.execute("INSERT INTO truth.manifest VALUES (?)", [json.dumps(world.truth)])
        for name, frame in world.truth_tables.items():
            con.register("frame", frame)
            con.execute(f"CREATE TABLE truth.{name} AS SELECT {_select_list(frame)} FROM frame")
            con.unregister("frame")
