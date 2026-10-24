"""Actuals: what history recorded over a study's window, for the evaluators that score against it.

Only a wholly closed window has actuals. A study resolves them once, and only the parts its
evaluators read (`resolve`):

- `facts`: history's event facts over the window (see `facts`);
- `headcount`: each month-end's headcount per cost center, as a (months, cost centers) array,
  row k the end of the window's month k;
- `sbc`: the reference ledger's SBC over the window, by part and the SBC keys
  (`seed_equity.actual_sbc`);
- `awards`: the awards granted over the window, as recorded (`replay.recorded_awards`);
- `refit`: the domain models refitted over the window itself, by the same estimators the study
  fitted over the year before it (`inputs.fit_models`), so each fitted rate can be scored out of
  time.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.replay import recorded_awards
from hcsim.engine.seed_equity import actual_sbc
from hcsim.financial.routing import RecordedAwards
from hcsim.guidance.ingest import CATEGORY, GROUP, MONTH, VALUE
from hcsim.guidance.realized import realized
from hcsim.model.core.categories import GuidanceCategory
from hcsim.model.core.schema import COST_CENTER, SimulationSchema
from hcsim.studies.facts import actual_facts
from hcsim.studies.inputs import FittedModels, fit_models

#: Every part of a window's actuals, in the order `resolve` reads them.
PARTS: Final = ("facts", "headcount", "sbc", "awards", "refit")


class ActualsError(ValueError):
    """Actuals a study can't have."""


@dataclass(frozen=True)
class Actuals:
    """History over the `months` from `run_date`: the parts a study's evaluators read, and None
    for the rest."""

    run_date: dt.date
    months: int
    facts: pd.DataFrame | None = None
    headcount: NDArray[np.int64] | None = None
    sbc: pd.DataFrame | None = None
    awards: RecordedAwards | None = None
    refit: FittedModels | None = None

    def need(self, part: str) -> object:
        """The part, refusing one that wasn't resolved."""
        value = getattr(self, part)
        if value is None:
            raise ActualsError(f"the study's actuals don't hold {part}")
        return value


def month_end_headcount(
    con: duckdb.DuckDBPyConnection, schema: SimulationSchema, run_date: dt.date, months: int
) -> NDArray[np.int64]:
    """(months, cost centers): the headcount at each month-end of the window, as the
    realized-total producer counts the stock."""
    labels = schema.vocabulary(COST_CENTER).labels
    start = pd.Period(run_date, freq="M")
    frame = realized(con, start, months, COST_CENTER, [GuidanceCategory.HEADCOUNT], groups=labels)
    stock = frame.frame[frame.frame[CATEGORY] == GuidanceCategory.HEADCOUNT.value]
    grid = stock.pivot_table(index=MONTH, columns=GROUP, values=VALUE, aggfunc="sum")
    grid = grid.reindex(index=pd.period_range(start, periods=months, freq="M"), columns=labels)
    return np.rint(grid.fillna(0.0).to_numpy()).astype(np.int64)


def resolve(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run_date: dt.date,
    months: int,
    parts: Iterable[str],
) -> Actuals:
    """The window's actuals: each of `parts`, read from `con`."""
    wanted = set(parts)
    unknown = sorted(wanted - set(PARTS))
    if unknown:
        raise ActualsError(f"no parts of the actuals {unknown}; the parts are {', '.join(PARTS)}")
    after = (pd.Period(run_date, freq="M") + months).start_time.date()
    return Actuals(
        run_date,
        months,
        facts=actual_facts(con, schema, run_date, months) if "facts" in wanted else None,
        headcount=(
            month_end_headcount(con, schema, run_date, months) if "headcount" in wanted else None
        ),
        sbc=actual_sbc(con, schema, run_date, months) if "sbc" in wanted else None,
        awards=recorded_awards(con, run_date, months) if "awards" in wanted else None,
        refit=fit_models(con, schema, after, window_months=months) if "refit" in wanted else None,
    )
