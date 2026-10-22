"""The engine's acceptance checks, on synthetic worlds.

- Replay reproduces each world's month-end headcount per cost center, exactly, every month,
  and its open openings too, once the openings it reserves for accepted offers are counted.
- A `guided_actuals` run, on the realized-total producer's guidance, honors every quota, and
  its flows show it.
- An unguided run's counts land within the `stat_check` bound of what its models expect, given
  the exposure its stocks and event cells record.
- A run is reproducible: the same seed gives the same digests, fewer trajectories a prefix of
  them, and a streamed run the same digests, bit for bit.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.config import AwardSettings, ForecastScope, RunMode, SimulationConfig
from hcsim.engine.known_events import KnownEvents, Source, collect, from_replay, known_events
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import ResolvedPlan, jobs_from_world, resolve_plan
from hcsim.engine.loop import DOMAIN_INDEX, SimulationResult, TrajectoryResult, simulate
from hcsim.engine.replay import RESERVED_OPENINGS, replayed_requisitions
from hcsim.engine.seeding import (
    opening_price,
    range_table_from_world,
    schema_from_world,
    seed_for,
    seed_state,
)
from hcsim.engine.state import SimulationState
from hcsim.financial.prices import AnnualGrowthPrice
from hcsim.financial.sizing import NewHireGapFill
from hcsim.guidance.realized import realized as realized_totals
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.contract import role_column
from hcsim.model.core.guidance import GuidanceForecast
from hcsim.model.core.layouts import EmployeeColumn, OpeningColumn
from hcsim.model.core.measures import CATEGORY_MEASURES
from hcsim.model.core.schema import COST_CENTER, SimulationSchema
from hcsim.model.ml.entity import ConstantModel
from hcsim.studies.evaluators import invariants
from hcsim.studies.inputs import FittedModels, fit_models
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world
from tests.support.stat_check import bernoulli_sum_se, stat_check

pytestmark = pytest.mark.acceptance

RUN: Final = dt.date(2025, 1, 1)
_C = GuidanceCategory
#: Every category a guided run draws against.
GUIDED: Final = (
    _C.ATTRITION,
    _C.PROMOTIONS,
    _C.EXTERNAL_HIRES,
    _C.CONVERSION_HIRES,
    _C.INTERNAL_STARTS,
    _C.REQUISITIONS_OPENED,
    _C.REQUISITIONS_CLOSED_UNFILLED,
)


def connect(path: Path, name: str, seed: int) -> duckdb.DuckDBPyConnection:
    write_world(generate_world(preset(name, seed=seed)), path)
    return duckdb.connect(str(path), read_only=True)


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    with connect(tmp_path_factory.mktemp("world") / "small.duckdb", "small", 6) as connection:
        yield connection


def realized(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    category: GuidanceCategory,
    start: pd.Period,
    months: int,
) -> np.ndarray:
    """(months, cost centers): the world's realized `category` from the month `start`, counted
    by the category's own measure."""
    measure = CATEGORY_MEASURES[category]
    frame = con.execute(
        f"SELECT month, cost_center, sum(value)::DOUBLE AS value "
        f"FROM ({measure.sql(role_column(con, measure.table))}) GROUP BY ALL"
    ).df()
    steps = (pd.to_datetime(frame["month"]).dt.to_period("M") - start).map(lambda d: d.n)
    codes = schema.encode(COST_CENTER, frame[COST_CENTER], unknown="unattributed")
    keep = ((steps >= 0) & (steps < months)).to_numpy() & (codes >= 0)
    out = np.zeros((months, len(schema.vocabulary(COST_CENTER))))
    np.add.at(out, (steps.to_numpy()[keep], codes[keep]), frame["value"].to_numpy()[keep])
    return out


def plan_for(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    state: SimulationState,
    events: KnownEvents,
) -> ResolvedPlan:
    employees = state.live_employees[:, EmployeeColumn.EMPLOYEE_ID].astype(int).tolist()
    reqs = state.live_openings[:, OpeningColumn.REQ_ID].astype(int).tolist()
    return resolve_plan(sanitize(events, employees, reqs), schema, jobs_from_world(con))


def configure(
    con: duckdb.DuckDBPyConnection, fitted: FittedModels | None = None, **overrides: Any
) -> SimulationConfig:
    schema = schema_from_world(con)
    args: dict[str, Any] = {
        "run_date": RUN,
        "schema": schema,
        "models": (fitted or fit_models(con, schema, RUN)).models,
        "simulation_months": 6,
        "num_trajectories": 4,
        "seed": 11,
        "forecast_scope": ForecastScope.HEADCOUNT,
    } | overrides
    return SimulationConfig(**args)


# --- replay fidelity ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Replayed:
    result: SimulationResult
    #: (months + 1, cost centers): the realized month-end before each step, then the last's.
    headcount: np.ndarray
    open_openings: np.ndarray


@pytest.fixture(
    scope="module",
    params=[("small", 6, dt.date(2025, 1, 1), 12), ("small", 11, dt.date(2024, 7, 1), 18)],
    ids=lambda p: f"{p[0]}-{p[1]}",
)
def replayed(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Replayed:
    name, seed, run, months = request.param
    with connect(tmp_path_factory.mktemp("world") / "world.duckdb", name, seed) as con:
        schema = schema_from_world(con)
        config = SimulationConfig(
            run_date=run,
            schema=schema,
            models=fit_models(con, schema, run).models,
            simulation_months=months,
            num_trajectories=2,
            seed=3,
            forecast_scope=ForecastScope.HEADCOUNT,
            run_mode=RunMode.REPLAY,
        )
        state, _ = seed_for(con, config)
        events = collect(run, months, [(Source.REPLAY, from_replay(con, run))])
        plan = plan_for(con, schema, state, events)
        replay = replayed_requisitions(con, schema, run, months)
        result = simulate(config, state, known=plan, replay=replay, validate=True)
        # Row k is the month-end before step k: the seed's, then each simulated month's.
        start = pd.Period(run, freq="M")
        headcount = realized(con, schema, _C.HEADCOUNT, start - 1, months + 1)
        # The stock a run starting a month later would seed.
        openings = [
            seed_state(con, schema, (start + k).start_time.date())[0].open_openings
            for k in range(months + 1)
        ]
    return Replayed(result, headcount.astype(np.int64), np.stack(openings))


def test_replay_reproduces_the_month_end_headcount_per_cost_center(replayed: Replayed) -> None:
    assert replayed.headcount.sum(axis=1).min() > 1_000
    for t in replayed.result.trajectories:
        assert np.array_equal(t.headcount, replayed.headcount)
        assert t.known_fired["termination"] > 0
        assert int(t.flows["F_pend"].sum() + t.flows["F_bdy"].sum()) > 0


def test_replay_reproduces_the_month_end_open_openings_per_cost_center(
    replayed: Replayed,
) -> None:
    """History closes an opening when its offer is accepted, and the run when its person
    starts, so the run's stock is history's plus the openings it holds reserved for them."""
    truth = replayed.open_openings
    assert truth.sum(axis=1).min() > 100
    for t in replayed.result.trajectories:
        reserved = t.diagnostics[RESERVED_OPENINGS]
        assert int(reserved.sum()) > 0
        assert np.array_equal(t.open_openings[0], truth[0])
        assert np.array_equal(t.open_openings[1:] - reserved, truth[1:])
    card = {m.name: m.value for m in invariants(replayed.result)}
    assert card["headcount_identity_max_gap"] == 0
    assert card["opening_identity_max_gap"] == 0


# --- guided_actuals -----------------------------------------------------------------------------


def actuals(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema, months: int
) -> GuidanceForecast:
    """Guidance carrying the world's realized totals for every guided category, as the
    realized-total producer counts them, on the whole grid of months and cost centers, zeros
    included, so every cost center is guided."""
    labels = schema.vocabulary(COST_CENTER).labels
    start = pd.Period(RUN, freq="M")
    return realized_totals(con, start, months, COST_CENTER, GUIDED, groups=labels).artifact


@pytest.fixture(scope="module")
def guided_actuals(con: duckdb.DuckDBPyConnection) -> SimulationResult:
    schema = schema_from_world(con)
    fitted = fit_models(con, schema, RUN)
    models = dataclasses.replace(
        fitted.models,
        conversion_fill=ConstantModel(0.05),
        transfer_fill=ConstantModel(0.1),
        promotion=ConstantModel(0.02),
    )
    config = configure(
        con,
        dataclasses.replace(fitted, models=models),
        run_mode=RunMode.GUIDED_ACTUALS,
        guidance=actuals(con, schema, 6),
        promotion_cycle_months=tuple(range(1, 13)),  # the world promotes in every month
    )
    state, _ = seed_for(con, config)
    plan = plan_for(con, schema, state, known_events(con, RUN, config.simulation_months))
    return simulate(config, state, known=plan, validate=True)


def by_group(t: TrajectoryResult, domain: str, category: GuidanceCategory, months: int) -> Any:
    """(months, groups): each step's realized count for `category` from `domain`'s draws."""
    rows = [q for q in t.quotas if q.domain == domain and q.category == category.value]
    assert sorted(q.step for q in rows) == list(range(months))
    return np.stack([q.record.realized for q in sorted(rows, key=lambda q: q.step)])


def test_guided_actuals_honors_every_quota(guided_actuals: SimulationResult) -> None:
    result = guided_actuals
    months = result.config.simulation_months
    for t in result.trajectories:
        drawn = {(q.step, q.category) for q in t.quotas}
        assert drawn == {(s, c.value) for s in range(months) for c in GUIDED}
        for q in t.quotas:
            assert q.record.honored.all(), (t.trajectory, q.step, q.domain, q.category)
    card = {m.name: m.value for m in invariants(result)}
    assert card["quota_violations"] == 0
    assert card["headcount_identity_max_gap"] == 0
    assert card["opening_identity_max_gap"] == 0


def test_guided_actuals_flows_are_what_the_quotas_realized(
    guided_actuals: SimulationResult,
) -> None:
    result = guided_actuals
    months = result.config.simulation_months
    assert result.config.guidance is not None
    groups = result.config.guidance.groups
    assert list(groups) == list(result.config.schema.vocabulary(COST_CENTER).labels)
    for t in result.trajectories:
        f = t.flows
        checks = {
            "A": (by_group(t, "attrition", _C.ATTRITION, months), f["A"]),
            "Pi": (by_group(t, "promotion", _C.PROMOTIONS, months), f["Pi"]),
            "U": (by_group(t, "cancellation", _C.REQUISITIONS_CLOSED_UNFILLED, months), f["U"]),
            "F_tr": (by_group(t, "transfers", _C.INTERNAL_STARTS, months), f["F_tr"]),
            "G + B": (
                by_group(t, "growth", _C.REQUISITIONS_OPENED, months),
                f["G"] + f["B_att"] + f["B_tr"],
            ),
            # A pending start is an external hire, matched to its requisition or not.
            "F_ext + F_pend + F_bdy": (
                by_group(t, "fills", _C.EXTERNAL_HIRES, months),
                f["F_ext"] + f["F_pend"] + f["F_bdy"],
            ),
            "F_conv": (by_group(t, "fills", _C.CONVERSION_HIRES, months), f["F_conv"]),
        }
        for name, (quota_realized, flow) in checks.items():
            assert np.array_equal(quota_realized, flow), name


# --- unguided counts ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unguided(con: duckdb.DuckDBPyConnection) -> tuple[SimulationResult, FittedModels]:
    schema = schema_from_world(con)
    fitted = fit_models(con, schema, RUN)
    config = configure(con, fitted, simulation_months=12, num_trajectories=10)
    state, _ = seed_for(con, config)
    return simulate(config, state), fitted


def opening_rows_before(result: SimulationResult, domain: str, seeded: int) -> np.ndarray:
    """(trajectories, months): requisition rows open when `domain` runs, from the seeded rows
    and every earlier change the event cells record."""
    cells = result.event_cells()
    cells = cells[cells["buffer"] == "opening"]
    sign = cells["change"].map({"append": 1, "drop": -1}).fillna(0).to_numpy()
    order = cells["domain"].map(DOMAIN_INDEX).to_numpy()
    months = result.config.simulation_months
    net = np.zeros((len(result.trajectories), months, len(DOMAIN_INDEX)), dtype=np.int64)
    index = (cells["trajectory"].to_numpy(), cells["step"].to_numpy(), order)
    np.add.at(net, index, sign * cells["rows"].to_numpy())
    flat = net.reshape(len(result.trajectories), -1)
    before = seeded + np.cumsum(flat, axis=1) - flat  # every change before each slot
    result_rows: np.ndarray = before.reshape(net.shape)[:, :, DOMAIN_INDEX[domain]]
    return result_rows


def test_unguided_counts_land_within_the_bound_of_their_models(
    unguided: tuple[SimulationResult, FittedModels], con: duckdb.DuckDBPyConnection
) -> None:
    result, fitted = unguided
    rates = fitted.rates
    h, o = result.headcount, result.open_openings

    # Attrition: everyone at the start of a month is at risk at the model's rate.
    at_risk = int(h[:, :-1].sum())
    departures = int(result.flow("A").sum())
    p = rates["attrition"]
    stat_check(departures, p * at_risk, bernoulli_sum_se([p] * at_risk), label="attrition")

    # Backfill: each departure opens one at the pooled rate.
    p = rates["backfill"]
    backfills = int(result.flow("B_att").sum())
    stat_check(backfills, p * departures, bernoulli_sum_se([p] * departures), label="backfill")

    # External fills: every opening open when fills run, the month's start plus its backfills,
    # fills at the model's hazard.
    exposed = int(o[:, :-1].sum() + result.flow("B_att").sum())
    p = rates["external_fill"]
    fills = int(result.flow("F_ext").sum())
    stat_check(fills, p * exposed, bernoulli_sum_se([p] * exposed), label="external fills")

    # Cancellation: every requisition open when cancellation runs exits at the model's hazard.
    seeded = int(seed_for(con, result.config)[0].n_openings)
    exposed = int(opening_rows_before(result, "cancellation", seeded).sum())
    cells = result.event_cells()
    exits = int(
        cells.loc[(cells["domain"] == "cancellation") & (cells["change"] == "drop"), "rows"].sum()
    )
    p = rates["cancellation"]
    stat_check(exits, p * exposed, bernoulli_sum_se([p] * exposed), label="cancellation")

    # Growth: Poisson counts per cost center at the model's rates.
    model = result.config.models.growth
    n = len(result.config.schema.vocabulary(COST_CENTER))
    mean = sum(model.rate_for(c) for c in range(n)) * h.shape[0] * result.config.simulation_months
    stat_check(int(result.flow("G").sum()), mean, float(np.sqrt(mean)), label="growth")


# --- reproducibility ----------------------------------------------------------------------------


def test_a_run_is_reproducible_bit_for_bit(con: duckdb.DuckDBPyConnection) -> None:
    schema = schema_from_world(con)
    awards = AwardSettings(
        NewHireGapFill(range_table_from_world(con, schema)),
        AnnualGrowthPrice(opening_price(con, RUN)),
    )
    config = configure(
        con,
        run_mode=RunMode.GUIDED_ACTUALS,
        guidance=actuals(con, schema, 6),
        forecast_scope=ForecastScope.HEADCOUNT_AND_SBC,
        awards=awards,
        results_snapshot_steps=(0, 6),
    )
    state, _ = seed_for(con, config)
    plan = plan_for(con, schema, state, known_events(con, RUN, 6))

    def run(**overrides: Any) -> SimulationResult:
        sink: list[TrajectoryResult] = []
        cfg = dataclasses.replace(config, **overrides)
        if cfg.results_streaming:
            return simulate(cfg, state, known=plan, sink=sink.append)
        return simulate(cfg, state, known=plan)

    first = run()
    assert all(t.sbc is not None and len(t.sbc) for t in first.trajectories)
    digests = first.digests()
    assert len(set(digests)) == len(digests)  # trajectories differ
    assert run().digests() == digests
    assert run(num_trajectories=2).digests() == digests[:2]
    assert run(results_streaming=True).streamed == digests
    assert not set(run(seed=12).digests()) & set(digests)
