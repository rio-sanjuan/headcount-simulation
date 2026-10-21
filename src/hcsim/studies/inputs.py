"""Inputs: fit the walking skeleton's domain models from the synthetic warehouse.

Every rate is fitted over the twelve months that end at the seed snapshot, so a run never reads
the future. The skeleton's models are deliberately simple: constant monthly hazards for
attrition, external fill, and unfilled exit; one pooled backfill rate; a Poisson growth count
per cost center; and empirical requisition roles. Per-domain extracts and trained models
arrive in M3 and M8.3.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from hcsim.engine.config import DomainModels
from hcsim.engine.seeding import snapshot_before
from hcsim.model.core.layouts import OPENING
from hcsim.model.core.schema import COST_CENTER, SimulationSchema
from hcsim.model.ml.entity import ConstantModel
from hcsim.model.ml.generative import ATTRIBUTES, EmpiricalAttributeGenerator
from hcsim.model.ml.group import BackfillRateModel, PoissonGrowthModel

WINDOW_MONTHS: Final = 12


class InputError(ValueError):
    """A warehouse that can't supply the skeleton's inputs."""


@dataclass(frozen=True)
class FittedModels:
    models: DomainModels
    #: The fitted pooled rates, for the run record.
    rates: dict[str, float]
    window: tuple[dt.date, dt.date]


def _scalar(con: duckdb.DuckDBPyConnection, sql: str, params: list[object]) -> float:
    (value,) = con.execute(sql, params).fetchone() or (None,)
    return float(value or 0.0)


def _ratio(events: float, exposure: float, what: str) -> float:
    if exposure <= 0:
        raise InputError(f"no exposure to fit the {what} rate on")
    return min(1.0, events / exposure)


def _role_codes(frame: pd.DataFrame, schema: SimulationSchema) -> np.ndarray:
    """Requisition roles as (N, 6) codes in ATTRIBUTES order."""
    columns = []
    for column in ATTRIBUTES:
        vocabulary = OPENING.vocabularies.get(column)
        if vocabulary is None:
            columns.append(frame["is_manager"].to_numpy(dtype=np.int64))
        else:
            columns.append(schema.encode(vocabulary, frame[vocabulary], unknown="unattributed"))
    return np.column_stack(columns).astype(np.int64)


def fit_models(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema, run_date: dt.date
) -> FittedModels:
    end = snapshot_before(con, run_date)
    start = (pd.Period(end, freq="M") - WINDOW_MONTHS).end_time.date()
    window: list[object] = [start, end]
    n = len(schema.vocabulary(COST_CENTER))

    # Attrition: departures in the window over the stock at risk at each prior month-end.
    departures = _scalar(
        con,
        "SELECT count(*) FROM workforce_txn WHERE trans_type = 'TERM' "
        "AND event_date > ? AND event_date <= ?",
        window,
    )
    at_risk = _scalar(
        con,
        "SELECT count(*) FROM workforce_txn WHERE trans_type = 'ACTIVE' "
        "AND event_date >= ? AND event_date < ?",
        window,
    )
    # Fills and exits: closures over the openings exposed each month, open at the prior
    # month-end or opened during the month, as the engine lets a new requisition fill at once.
    exposed = _scalar(
        con,
        "SELECT sum(openings) FROM req_snapshot WHERE snapshot_date >= ? AND snapshot_date < ?",
        window,
    ) + _scalar(
        con,
        "SELECT sum(openings) FROM req_interval WHERE opened_date > ? AND opened_date <= ?",
        window,
    )

    def closed(reason: str) -> float:
        return _scalar(
            con,
            "SELECT sum(openings) FROM req_interval WHERE close_reason = ? "
            "AND closed_date > ? AND closed_date <= ?",
            [reason, *window],
        )

    vacancies = _scalar(
        con,
        "SELECT count(*) FROM workforce_txn WHERE trans_type IN ('TERM', 'XFER_OUT') "
        "AND event_date > ? AND event_date <= ?",
        window,
    )
    backfills = _scalar(
        con,
        "SELECT count(*) FROM req_interval WHERE req_type = 'backfill' "
        "AND opened_date > ? AND opened_date <= ?",
        window,
    )
    rates = {
        "attrition": _ratio(departures, at_risk, "attrition"),
        "external_fill": _ratio(closed("filled_external"), exposed, "external fill"),
        "cancellation": _ratio(closed("unfilled"), exposed, "unfilled exit"),
        "backfill": _ratio(backfills, vacancies, "backfill"),
    }

    # Growth: requisitions opened per cost center and month.
    growth = con.execute(
        """
        SELECT date_trunc('month', CAST(opened_date AS DATE)) AS month, cost_center,
               count(*) AS n
        FROM req_interval WHERE req_type = 'growth' AND opened_date > ? AND opened_date <= ?
        GROUP BY ALL
        """,
        window,
    ).df()
    counts = np.zeros((WINDOW_MONTHS, n))
    if len(growth):
        months = pd.to_datetime(growth["month"]).dt.to_period("M")
        offset = np.array([(m - pd.Period(start, freq="M")).n - 1 for m in months])
        codes = schema.encode(COST_CENTER, growth["cost_center"], unknown="unattributed")
        known = codes >= 0
        np.add.at(counts, (offset[known], codes[known]), growth["n"].to_numpy()[known])
    growth_model = PoissonGrowthModel(n, min_months=6).fit(counts)
    rates["growth"] = growth_model.pooled

    # Roles: every growth requisition up to the snapshot, or every requisition if none.
    roles = con.execute(
        """
        SELECT r.req_type, r.cost_center, r.location, r.company, r.is_manager,
               j.job_family, j.job_level, j.tech_indicator
        FROM req_interval r JOIN dim_job j USING (job_code)
        WHERE r.opened_date <= ?
        """,
        [end],
    ).df()
    # The exposure check above guarantees at least one requisition.
    if roles["req_type"].eq("growth").any():
        roles = roles[roles["req_type"] == "growth"]
    codes = _role_codes(roles, schema)
    cost_centers = schema.encode(COST_CENTER, roles["cost_center"], unknown="unattributed")
    keep = (codes >= 0).all(axis=1) & (cost_centers >= 0)
    attributes = EmpiricalAttributeGenerator().fit(cost_centers[keep], codes[keep])

    models = DomainModels(
        attrition=ConstantModel(rates["attrition"]),
        external_fill=ConstantModel(rates["external_fill"]),
        cancellation=ConstantModel(rates["cancellation"]),
        backfill=BackfillRateModel.constant(rates["backfill"], n),
        growth=growth_model,
        attributes=attributes,
    )
    return FittedModels(models, rates, (start, end))
