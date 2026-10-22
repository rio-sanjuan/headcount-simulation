"""Seed a run's initial state from the synthetic warehouse.

The employee buffer comes from the active-headcount stock at the last month-end before the run
date. The opening buffer comes from the requisition snapshot at the same month-end: open
requisitions, with their openings clipped to 1..10. Both seeds encode labels through the schema
and then prune rows carrying a label the schema lacks, counting each cause.

Tenure and age are seeded as negative steps: `HIRE_STEP` and `CREATION_STEP` are months relative
to the run's first month, so `step - HIRE_STEP` is real tenure. Simulated hires and
requisitions get ids from a high base, so they never collide with the world's ids.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from hcsim.engine.state import SimulationState
from hcsim.financial.ranges import RangeTable
from hcsim.financial.tracker import HolderLedger
from hcsim.model.core.layouts import (
    EMPLOYEE,
    MAX_OPENINGS_PER_REQUISITION,
    OPENING,
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

REQ_TYPES: Final = ("backfill", "campus", "growth")
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


def _encode(
    frame: pd.DataFrame, schema: SimulationSchema, columns: Mapping[str, int], buffer: np.ndarray
) -> np.ndarray:
    """Encode labels into `buffer`; return the mask of rows every label resolved for."""
    keep = np.ones(len(frame), dtype=bool)
    for vocabulary, column in columns.items():
        codes = schema.encode(vocabulary, frame[vocabulary], unknown="unattributed")
        buffer[: len(frame), column] = codes
        keep &= codes >= 0
    return keep


def seed_state(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run_date: dt.date,
    *,
    include_openings: bool = True,
    opening_ceiling: int | None = None,
    with_ledger: bool = False,
) -> tuple[SimulationState, SeedReport]:
    check_transaction_codes(con)
    snapshot = snapshot_before(con, run_date)
    people = con.execute(
        f"""
        SELECT t.employee_id, t.cost_center, j.job_family, j.job_level, j.tech_indicator,
               t.location, t.company, t.is_manager, t.hire_date
        FROM workforce_txn t JOIN dim_job j USING (job_code)
        WHERE t.trans_type = 'ACTIVE' AND t.event_date = ? AND t.account IN ({REGULAR})
        ORDER BY t.employee_id
        """,
        [snapshot],
    ).df()
    rows = EMPLOYEE.empty(len(people))
    keep = _encode(people, schema, _EMPLOYEE_CODES, rows)
    rows[:, EmployeeColumn.EMPLOYEE_ID] = people["employee_id"].to_numpy(dtype=np.float64)
    rows[:, EmployeeColumn.IS_MANAGER] = people["is_manager"].to_numpy(dtype=np.float64)
    rows[:, EmployeeColumn.HIRE_STEP] = _steps_from(people["hire_date"], run_date)
    employees = rows[keep]
    n_employees = len(employees)
    employees[:, EmployeeColumn.VEST_INDEX] = np.arange(n_employees)

    if include_openings:
        reqs = con.execute(
            """
            SELECT s.req_id, s.req_type, s.cost_center, j.job_family, j.job_level,
                   j.tech_indicator, s.location, s.company, s.is_manager, s.openings,
                   s.opened_date
            FROM req_snapshot s JOIN dim_job j USING (job_code)
            WHERE s.snapshot_date = ?
            ORDER BY s.req_id
            """,
            [snapshot],
        ).df()
    else:
        reqs = pd.DataFrame(
            columns=["req_id", *_OPENING_CODES, "is_manager", "openings", "opened_date"]
        )
    req_rows = OPENING.empty(len(reqs))
    req_keep = _encode(reqs, schema, _OPENING_CODES, req_rows) if len(reqs) else np.zeros(0, bool)
    raw_openings = reqs["openings"].to_numpy(dtype=np.float64)
    clipped = np.clip(raw_openings, 1, MAX_OPENINGS_PER_REQUISITION)
    req_rows[:, OpeningColumn.REQ_ID] = reqs["req_id"].to_numpy(dtype=np.float64)
    req_rows[:, OpeningColumn.IS_MANAGER] = reqs["is_manager"].to_numpy(dtype=np.float64)
    req_rows[:, OpeningColumn.NUM_OPENINGS] = clipped
    if len(reqs):
        req_rows[:, OpeningColumn.CREATION_STEP] = _steps_from(reqs["opened_date"], run_date)
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
        openings_clipped=int((clipped != raw_openings).sum()),
        employee_capacity=employee_capacity,
        opening_capacity=opening_capacity,
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
