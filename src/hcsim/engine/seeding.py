"""Seed a run's initial state from the synthetic warehouse.

The employee buffer comes from the population's active stock at the last month-end before the
run date. The opening buffer comes from the requisitions open at the same month-end, read from
their state intervals, with their openings clipped to 1..10. Both are read through the
population compiler, with a subset's rules pushed down where the warehouse can apply them.

Every row then passes the same fixed order, and each row dropped is counted under the first
cause that drops it:

1. **encoding**: every label becomes its code through the schema, or -1 if the schema lacks it;
2. **subset filtering**: a row outside the run's subset is dropped (`outside_subset`). Rules the
   warehouse couldn't apply, such as the rating, take effect here;
3. **pruning**: a row still carrying a -1 is dropped, under the first column that has one
   (`unknown_<column>`).

Beyond the labels, an employee carries a manager flag, read from the stock with a missing flag
as false, and their rating as of the run date: the rank of their latest rating from a cycle
before it, or 0 if they've never been rated. A requisition is seeded when one of its state
intervals is open at the snapshot. Its type is mapped from the warehouse's label, with case and
spacing ignored, to one of the engine's three, and an unmapped type is pruned as unknown. Its
openings are clipped to 1 through the configured maximum.

Tenure and age are seeded as negative steps: `HIRE_STEP` and `CREATION_STEP` are months relative
to the run's first month, so `step - HIRE_STEP` is real tenure. Simulated hires and
requisitions get ids from a high base, so they never collide with the world's ids.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from hcsim.engine.state import SimulationState
from hcsim.financial.ranges import RangeTable
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.contract import role_column
from hcsim.model.core.layouts import (
    EMPLOYEE,
    MAX_OPENINGS_PER_REQUISITION,
    OPENING,
    BufferLayout,
    EmployeeColumn,
    OpeningColumn,
)
from hcsim.model.core.schema import (
    AWARD_TYPE,
    COMPANY,
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    LOCATION,
    REQ_TYPE,
    TECH_INDICATOR,
    SimulationSchema,
    org_levels_from_frame,
)
from hcsim.model.data.population import REGULAR as REGULAR_POPULATION
from hcsim.model.data.population import PopulationSpec
from hcsim.model.data.subset import LoweredSubset, SubsetSpec
from hcsim.model.sql.extract import seed_employees, seed_openings

REQ_TYPES: Final = ("backfill", "campus", "growth")
#: The warehouse's requisition types, as the engine's.
REQ_TYPE_MAP: Final[Mapping[str, str]] = MappingProxyType({t: t for t in REQ_TYPES})
AWARD_TYPES: Final = ("new_hire",)
SIMULATED_ID_BASE: Final = 1_000_000_000
#: Headcount is regular employees: contingent and temporary workers sit outside it.
REGULAR: Final = "SELECT account FROM dim_account WHERE account_group = 'regular'"

_EMPLOYEE_CODES = {
    COST_CENTER: EmployeeColumn.COST_CENTER,
    JOB_FAMILY: EmployeeColumn.JOB_FAMILY,
    JOB_LEVEL: EmployeeColumn.LEVEL,
    TECH_INDICATOR: EmployeeColumn.TECH_INDICATOR,
    LOCATION: EmployeeColumn.LOCATION,
    COMPANY: EmployeeColumn.COMPANY,
}
_OPENING_CODES = {
    COST_CENTER: OpeningColumn.COST_CENTER,
    JOB_FAMILY: OpeningColumn.JOB_FAMILY,
    JOB_LEVEL: OpeningColumn.LEVEL,
    TECH_INDICATOR: OpeningColumn.TECH_INDICATOR,
    LOCATION: OpeningColumn.LOCATION,
    COMPANY: OpeningColumn.COMPANY,
    REQ_TYPE: OpeningColumn.REQ_TYPE,
}


class SeedingError(ValueError):
    """A warehouse that can't seed a run."""


@dataclass(frozen=True)
class SeedReport:
    snapshot_date: dt.date
    employees_seeded: int
    employees_pruned: int
    openings_seeded: int
    openings_pruned: int
    openings_clipped: int
    employee_capacity: int
    opening_capacity: int
    #: Rows dropped per cause, in the order they were checked.
    employee_causes: Mapping[str, int] = field(default_factory=dict)
    opening_causes: Mapping[str, int] = field(default_factory=dict)


def schema_from_world(con: duckdb.DuckDBPyConnection) -> SimulationSchema:
    """Every vocabulary the run encodes through, from the world's dimension tables."""
    org = con.execute("SELECT * FROM dim_org").df()
    jobs = con.execute("SELECT job_family, job_level, tech_indicator FROM dim_job").df()
    return SimulationSchema.build(
        {
            COST_CENTER: org["cost_center"],
            JOB_FAMILY: jobs["job_family"],
            JOB_LEVEL: jobs["job_level"],
            TECH_INDICATOR: jobs["tech_indicator"],
            LOCATION: con.execute("SELECT location FROM dim_location").df()["location"],
            COMPANY: con.execute("SELECT company FROM dim_company").df()["company"],
            REQ_TYPE: REQ_TYPES,
            AWARD_TYPE: AWARD_TYPES,
        },
        org_levels=org_levels_from_frame(org),
    )


def check_transaction_codes(con: duckdb.DuckDBPyConnection) -> None:
    """Refuse transactions whose code the warehouse's catalog doesn't define. An unknown code
    has no disposition, so counting it, or skipping it, would be a guess."""
    unknown = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT trans_type FROM workforce_txn "
            "WHERE trans_type NOT IN (SELECT trans_type FROM dim_trans_type) ORDER BY 1"
        ).fetchall()
    ]
    if unknown:
        raise SeedingError(f"transaction codes missing from dim_trans_type: {unknown}")


def snapshot_before(con: duckdb.DuckDBPyConnection, run_date: dt.date) -> dt.date:
    """The last month-end with an active stock before `run_date`."""
    (snapshot,) = con.execute(
        "SELECT max(event_date) FROM workforce_txn WHERE trans_type = 'ACTIVE' AND event_date < ?",
        [run_date],
    ).fetchone() or (None,)
    if snapshot is None:
        raise SeedingError(f"no active stock before {run_date}")
    return pd.Timestamp(snapshot).date()


def _steps_from(dates: pd.Series, run_date: dt.date) -> np.ndarray:
    """Months from the run's first month to each date's month: negative for the past."""
    run = pd.Period(run_date, freq="M")
    months = pd.to_datetime(dates).dt.to_period("M")
    return np.array([(m - run).n for m in months], dtype=np.float64)


def _pipeline(
    frame: pd.DataFrame,
    schema: SimulationSchema,
    columns: Mapping[str, int],
    layout: BufferLayout,
    subset: LoweredSubset | None,
    extras: Callable[[pd.DataFrame, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Encode `frame` into a buffer, with `extras` writing the columns that aren't labels,
    then filter it to the subset, then prune what's left of unknown labels. Returns the
    buffer, the rows kept, and the rows dropped per cause."""
    rows = layout.empty(len(frame))
    for vocabulary, column in columns.items():
        rows[:, column] = schema.encode(vocabulary, frame[vocabulary], unknown="unattributed")
    if extras is not None:
        extras(frame, rows)
    keep = np.ones(len(frame), dtype=bool)
    causes: dict[str, int] = {}
    if subset is not None:
        outside = ~subset.mask(rows, layout)
        causes["outside_subset"] = int(outside.sum())
        keep &= ~outside
    for vocabulary, column in columns.items():
        unknown = keep & (rows[:, column] < 0)
        causes[f"unknown_{vocabulary}"] = int(unknown.sum())
        keep &= ~unknown
    return rows, keep, {cause: n for cause, n in causes.items() if n}


def ratings_as_of(
    con: duckdb.DuckDBPyConnection, run_date: dt.date, employees: pd.Series
) -> np.ndarray:
    """Each employee's rating rank from their latest cycle before `run_date`, or 0."""
    latest = con.execute(
        "SELECT employee_id, arg_max(rating_rank, cycle_date) AS rank FROM rating "
        "WHERE cycle_date < ? GROUP BY employee_id",
        [run_date],
    ).df()
    rank = dict(zip(latest["employee_id"], latest["rank"], strict=True))
    return np.array([rank.get(e, 0) for e in employees], dtype=np.float64)


def _flag(values: pd.Series) -> np.ndarray:
    """A flag column as 0 or 1, a missing flag as 0."""
    result: np.ndarray = values.astype("boolean").fillna(False).to_numpy(dtype=np.float64)
    return result


def _employee_extras(people: pd.DataFrame, rows: np.ndarray) -> None:
    rows[:, EmployeeColumn.EMPLOYEE_ID] = people["employee_id"].to_numpy(dtype=np.float64)
    rows[:, EmployeeColumn.IS_MANAGER] = _flag(people["is_manager"])
    rows[:, EmployeeColumn.PERF_RATING] = people["perf_rating"].to_numpy(dtype=np.float64)


def _opening_extras(run_date: dt.date) -> Callable[[pd.DataFrame, np.ndarray], None]:
    def write(reqs: pd.DataFrame, rows: np.ndarray) -> None:
        rows[:, OpeningColumn.REQ_ID] = reqs["req_id"].to_numpy(dtype=np.float64)
        rows[:, OpeningColumn.IS_MANAGER] = _flag(reqs["is_manager"])
        if len(reqs):
            rows[:, OpeningColumn.CREATION_STEP] = _steps_from(reqs["opened_date"], run_date)

    return write


def _mapped_types(labels: pd.Series, mapping: Mapping[str, str]) -> pd.Series:
    """Warehouse requisition types as the engine's, case and spacing ignored. An unmapped
    type becomes missing, which encodes as unknown."""
    folded = {str(k).strip().lower(): v for k, v in mapping.items()}
    return labels.map(lambda t: folded.get(str(t).strip().lower()))


def seed_state(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run_date: dt.date,
    *,
    population: PopulationSpec = REGULAR_POPULATION,
    subset: SubsetSpec | None = None,
    include_openings: bool = True,
    max_openings: int = MAX_OPENINGS_PER_REQUISITION,
    req_types: Mapping[str, str] = REQ_TYPE_MAP,
    opening_ceiling: int | None = None,
    with_ledger: bool = False,
) -> tuple[SimulationState, SeedReport]:
    check_transaction_codes(con)
    snapshot = snapshot_before(con, run_date)
    lowered = None if subset is None else subset.lower(schema)
    extracted = population if lowered is None else lowered.pushdown(population)
    people = con.execute(seed_employees(extracted), {"snapshot": snapshot}).df()
    people["perf_rating"] = ratings_as_of(con, run_date, people["employee_id"])
    rows, keep, employee_causes = _pipeline(
        people, schema, _EMPLOYEE_CODES, EMPLOYEE, lowered, _employee_extras
    )
    rows[:, EmployeeColumn.HIRE_STEP] = _steps_from(people["hire_date"], run_date)
    employees = rows[keep]
    n_employees = len(employees)
    employees[:, EmployeeColumn.VEST_INDEX] = np.arange(n_employees)

    if max_openings < 1:
        raise SeedingError("a requisition keeps at least one opening")
    if include_openings:
        cost_center = role_column(con, "req_interval")
        reqs = con.execute(seed_openings(extracted, cost_center), {"snapshot": snapshot}).df()
    else:
        reqs = pd.DataFrame(
            columns=["req_id", *_OPENING_CODES, "is_manager", "openings", "opened_date"]
        )
    reqs[REQ_TYPE] = _mapped_types(reqs[REQ_TYPE], req_types)
    req_rows, req_keep, opening_causes = _pipeline(
        reqs, schema, _OPENING_CODES, OPENING, lowered, _opening_extras(run_date)
    )
    raw_openings = reqs["openings"].to_numpy(dtype=np.float64)
    clipped = np.clip(raw_openings, 1, max_openings)
    req_rows[:, OpeningColumn.NUM_OPENINGS] = clipped
    openings = req_rows[req_keep]

    # Capacity: the seed plus room for the horizon. Buffers never grow.
    employee_capacity = n_employees + max(1_000, n_employees // 10)
    ceiling = n_employees if opening_ceiling is None else opening_ceiling
    opening_capacity = len(openings) + ceiling + max(500, len(openings) // 5)
    employee_buffer = EMPLOYEE.empty(employee_capacity)
    employee_buffer[:n_employees] = employees
    opening_buffer = OPENING.empty(opening_capacity)
    opening_buffer[: len(openings)] = openings
    state = SimulationState(
        employee_buffer,
        n_employees,
        opening_buffer,
        len(openings),
        n_cost_centers=len(schema.vocabulary(COST_CENTER)),
        next_employee_id=SIMULATED_ID_BASE,
        next_requisition_id=SIMULATED_ID_BASE,
        next_vest_slot=n_employees,
        ledger=HolderLedger() if with_ledger else None,
    )
    report = SeedReport(
        snapshot_date=snapshot,
        employees_seeded=n_employees,
        employees_pruned=int((~keep).sum()),
        openings_seeded=len(openings),
        openings_pruned=int((~req_keep).sum()),
        openings_clipped=int((clipped[req_keep] != raw_openings[req_keep]).sum()),
        employee_capacity=employee_capacity,
        opening_capacity=opening_capacity,
        employee_causes=MappingProxyType(employee_causes),
        opening_causes=MappingProxyType(opening_causes),
    )
    return state, report


def range_table_from_world(con: duckdb.DuckDBPyConnection, schema: SimulationSchema) -> RangeTable:
    return RangeTable.from_frame(con.execute("SELECT * FROM comp_range").df(), schema)


def opening_price(con: duckdb.DuckDBPyConnection, run_date: dt.date) -> float:
    """The month-end close before the run starts."""
    (close,) = con.execute(
        "SELECT close FROM stock_price WHERE month_end < ? ORDER BY month_end DESC LIMIT 1",
        [run_date],
    ).fetchone() or (None,)
    if close is None:
        raise SeedingError(f"no stock price before {run_date}")
    return float(close)
