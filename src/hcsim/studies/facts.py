"""Event facts: what happened, as sparse long-form rows, so a run and history score alike.

An **event fact** counts one event in one step and cost center, for one trajectory of a run
(`run_facts`) or for history (`actual_facts`, whose trajectory is `ACTUAL`). Each event
(`EVENTS`) is one both sides can count: a run from its flows, and history from guidance
categories, as the realized-total producer counts them (`guidance.realized`):

| Event | A run's flows | History's categories |
|---|---|---|
| `attrition` | `A` | attrition |
| `hires` | `F_ext + F_conv + F_camp + F_pend + F_bdy` | external and conversion hires |
| `internal_starts` | `F_tr` | internal starts |
| `conversions_out` | `V_conv` | conversions out |
| `promotions` | `Π` | promotions |
| `requisitions_opened` | `G + G_camp + B_att + B_tr` | requisitions opened |
| `requisitions_closed_unfilled` | `U` | requisitions closed unfilled |

Hires are pooled because a start whose offer was accepted before it, a pending start or a
boundary fill, doesn't record whether it's an external hire or a conversion, and history
counts it as one or the other in the month it starts. Requisitions opened count campus
requisitions and transfer backfill as opened, as history does. History counts requisitions
from month-end snapshots, which can't see one opened and closed between two of them, so its
requisition counts can differ from a replay's, which reads every interval.

Facts are **sparse**: a zero has no row, so a run's facts grow with what happened, not with
its months, cost centers, and categories. And they're **long-form**: one row per trajectory,
step, cost center, and category, so a run's and history's join on their keys and aggregate by
any of them. A cost center is its schema code, and one history names that the schema doesn't
carry is `UNATTRIBUTED`.

Since a zero has no row, a statistic over trajectories reads the dense grid (`dense`), and a
mean divides by the run's trajectories, not its rows (`mean`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.engine.loop import FLOW_SYMBOLS, SimulationResult
from hcsim.guidance.ingest import CATEGORY, GROUP, MONTH, VALUE
from hcsim.guidance.realized import realized
from hcsim.model.core.categories import CATEGORIES, GuidanceCategory
from hcsim.model.core.schema import COST_CENTER, SimulationSchema


@dataclass(frozen=True)
class Event:
    """An event both a run and history count: the run from `flows`, history from
    `categories`."""

    name: str
    flows: tuple[str, ...]
    categories: tuple[GuidanceCategory, ...]


_C = GuidanceCategory
#: Every event, in order.
EVENTS: Final = (
    Event("attrition", ("A",), (_C.ATTRITION,)),
    Event(
        "hires",
        ("F_ext", "F_conv", "F_camp", "F_pend", "F_bdy"),
        (_C.EXTERNAL_HIRES, _C.CONVERSION_HIRES),
    ),
    Event("internal_starts", ("F_tr",), (_C.INTERNAL_STARTS,)),
    Event("conversions_out", ("V_conv",), (_C.CONVERSIONS_OUT,)),
    Event("promotions", ("Pi",), (_C.PROMOTIONS,)),
    Event("requisitions_opened", ("G", "G_camp", "B_att", "B_tr"), (_C.REQUISITIONS_OPENED,)),
    Event("requisitions_closed_unfilled", ("U",), (_C.REQUISITIONS_CLOSED_UNFILLED,)),
)
#: Each event by name.
BY_NAME: Final = MappingProxyType({e.name: e for e in EVENTS})
#: The trajectory of history's facts.
ACTUAL: Final = -1
TRAJECTORY, STEP, EVENT, COUNT = "trajectory", "step", "event", "count"
#: The columns of a fact frame.
FACT_COLUMNS: Final = (TRAJECTORY, STEP, COST_CENTER, EVENT, COUNT)


class FactError(ValueError):
    """Facts that can't be counted."""


def check_events(events: Sequence[Event]) -> list[str]:
    """Every way `events` fail to be one vocabulary: a flow a run doesn't record, a flow or a
    category two events count, or a category that isn't a flow of events."""
    problems = []
    flows = [f for e in events for f in e.flows]
    categories = [str(c) for e in events for c in e.categories]
    unknown = sorted(set(flows) - set(FLOW_SYMBOLS))
    if unknown:
        problems.append(f"no run records the flows {unknown}")
    for name, items in (("flows", flows), ("categories", categories)):
        twice = sorted({x for x in items if items.count(x) > 1})
        if twice:
            problems.append(f"two events count the {name} {twice}")
    stocks = sorted(c for c in categories if CATEGORIES[GuidanceCategory(c)].coherency_sign == 0)
    if stocks:
        problems.append(f"the categories {stocks} are stocks, not events")
    return problems


if _problems := check_events(EVENTS):  # pragma: no cover - the vocabulary is fixed
    raise RuntimeError("; ".join(_problems))


def _names() -> str:
    return ", ".join(e.name for e in EVENTS)


def _frame(
    trajectory: NDArray[np.int64],
    step: NDArray[np.int64],
    cost_center: NDArray[np.int64],
    event: Sequence[str],
    count: NDArray[np.int64],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            TRAJECTORY: trajectory.astype(np.int64),
            STEP: step.astype(np.int64),
            COST_CENTER: cost_center.astype(np.int64),
            EVENT: pd.Categorical(list(event), categories=[e.name for e in EVENTS]),
            COUNT: count.astype(np.int64),
        }
    )
    return frame.sort_values([TRAJECTORY, STEP, COST_CENTER, EVENT], ignore_index=True)


def empty_ints() -> NDArray[np.int64]:
    return np.zeros(0, dtype=np.int64)


def empty() -> pd.DataFrame:
    """A fact frame with no facts."""
    none = empty_ints()
    return _frame(none, none, none, [], none)


def run_facts(result: SimulationResult) -> pd.DataFrame:
    """Every trajectory's events, from its flows."""
    if not result.trajectories and result.streamed:
        raise FactError("a streamed run keeps no trajectories to count events from")
    trajectory, steps, cost_centers, counts = [empty_ints()], [empty_ints()], [empty_ints()], []
    names: list[str] = []
    for t in result.trajectories:
        for event in EVENTS:
            total = np.array(t.flows[event.flows[0]], dtype=np.int64)
            for f in event.flows[1:]:
                total = total + t.flows[f]
            step, cost_center = np.nonzero(total)
            trajectory.append(np.full(len(step), t.trajectory, dtype=np.int64))
            steps.append(step.astype(np.int64))
            cost_centers.append(cost_center.astype(np.int64))
            counts.append(total[step, cost_center])
            names += [event.name] * len(step)
    return _frame(
        np.concatenate(trajectory),
        np.concatenate(steps),
        np.concatenate(cost_centers),
        names,
        np.concatenate([empty_ints(), *counts]),
    )


def actual_facts(
    con: duckdb.DuckDBPyConnection,
    schema: SimulationSchema,
    run_date: dt.date,
    months: int,
    events: Sequence[str] = tuple(BY_NAME),
) -> pd.DataFrame:
    """History's `events` in the `months` from `run_date`, as the realized-total producer
    counts them."""
    unknown = sorted(set(events) - set(BY_NAME))
    if unknown:
        raise FactError(f"no events {unknown}; the events are {_names()}")
    of = {c.value: e for e in events for c in BY_NAME[e].categories}
    start = pd.Period(run_date, freq="M")
    vocabulary = schema.vocabulary(COST_CENTER)
    categories = [GuidanceCategory(c) for c in of]
    counted = realized(con, start, months, COST_CENTER, categories, groups=vocabulary.labels)
    frame = counted.frame[counted.frame[CATEGORY].isin(list(of))].copy()
    frame[EVENT] = frame[CATEGORY].map(of)
    frame = frame.groupby([MONTH, GROUP, EVENT])[VALUE].sum().reset_index()
    frame = frame[frame[VALUE] != 0]
    values = frame[VALUE].to_numpy(dtype=np.float64)
    if not np.array_equal(values, np.rint(values)):
        raise FactError("history's counts must be whole")
    steps = np.fromiter(((m - start).n for m in frame[MONTH]), dtype=np.int64, count=len(frame))
    return _frame(
        np.full(len(frame), ACTUAL),
        steps,
        vocabulary.encode(frame[GROUP], unknown="unattributed"),
        frame[EVENT].astype(str).tolist(),
        np.rint(values).astype(np.int64),
    )


def dense(
    facts: pd.DataFrame, trajectories: Sequence[int], months: int, cost_centers: int
) -> dict[str, NDArray[np.int64]]:
    """Each event's facts as a dense (trajectories, months, cost centers) array, the
    trajectories in the order given, zeros where there's no fact."""
    place = {t: k for k, t in enumerate(trajectories)}
    shape = (len(trajectories), months, cost_centers)
    arrays = {e.name: np.zeros(shape, dtype=np.int64) for e in EVENTS}
    if facts.empty:
        return arrays
    if not set(facts[TRAJECTORY]) <= set(place):
        raise FactError("facts name trajectories outside those given")
    for event, rows in facts.groupby(EVENT, observed=True):
        k = rows[TRAJECTORY].map(place).to_numpy()
        cc = rows[COST_CENTER].to_numpy()
        if (cc < 0).any() or (cc >= cost_centers).any():
            raise FactError(f"{event} facts name cost centers outside the schema")
        np.add.at(arrays[str(event)], (k, rows[STEP].to_numpy(), cc), rows[COUNT].to_numpy())
    return arrays


def mean(facts: pd.DataFrame, trajectories: int, by: Sequence[str]) -> pd.Series:
    """The mean count per trajectory, by `by`: each key's sum over the run's `trajectories`,
    those with no fact counting zero."""
    if trajectories < 1:
        raise FactError("a mean needs at least one trajectory")
    return facts.groupby(list(by), observed=True)[COUNT].sum() / trajectories
