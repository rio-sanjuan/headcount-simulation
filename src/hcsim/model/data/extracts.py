"""Training extracts: each domain's rows, outcome, and features, read from the warehouse.

An extract reads one population over a window given by two month-ends, `start` and `end`, and
takes the outcomes that fall in (start, end], as realized flows do. A cutoff therefore splits
a window into two that share no outcome. By domain:

- `attrition` and `promotion`: person-months. Each month-end's stock in [start, end), with
  tenure, the latest rating, and the calendar month ahead, labelled by a net departure, or a
  promotion, the next month;
- `external_fill`, `conversion_fill`, `transfer_fill`, and `cancellation`: one shared frame of
  requisition-months, from the month each requisition was approved, so one filled in its first
  month still has a row. A month's channel is its earliest fill's: a fill domain's label is
  whether that's its channel (an external fill is an external or campus hire), and
  cancellation's is an exit unfilled. The frame also carries the openings left at the month's
  start and how many filled, for a hazard per opening;
- `backfill`: terminations and transfers out, labelled by a backfill being opened for them;
- `growth`: growth requisitions opened per cost center and month, over the cost centers the
  population staffs that month, so a row is an observed month;
- `job_family`: growth requisitions, with the role each was opened for and its org context,
  finest first;
- `performance`: ratings, each with the one before it, so pairs train transitions and first
  ratings train entries.

Frames keep labels, not codes: `encode_frame` encodes them through the run's schema, with any
gap reported. Requisitions read their cost center through the data contract's role column, and
account rules don't constrain them, since a requisition books to no account.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import duckdb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hcsim.model.core.contract import role_column
from hcsim.model.core.measures import NET_DEPARTURES, PROMOTIONS_IN_PLACE
from hcsim.model.data.population import PopulationSpec, PopulationStamp
from hcsim.model.sql import extract as sql

_ORG_LEVEL: Final = re.compile(r"org_level(\d+)")

#: A person-month's features, in the employee buffer's order where it has the column.
EMPLOYEE_FEATURES: Final = (
    "job_level",
    "location",
    "job_family",
    "tech_indicator",
    "is_manager",
    "rating_rank",
    "cost_center",
    "tenure_months",
    "company",
    "next_month",
)
#: A requisition-month's features, in the opening buffer's order.
OPENING_FEATURES: Final = (
    "job_level",
    "location",
    "job_family",
    "tech_indicator",
    "is_manager",
    "cost_center",
    "age_months",
    "req_type",
    "company",
    "openings",
)
#: A role's attributes, in the generators' order.
ROLE: Final = ("job_level", "location", "job_family", "tech_indicator", "is_manager", "company")
#: The fill channels each fill domain's label counts.
CHANNELS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "external_fill": frozenset({"external", "campus"}),
        "conversion_fill": frozenset({"conversion"}),
        "transfer_fill": frozenset({"transfer"}),
    }
)


class ExtractError(ValueError):
    """An extract that can't be built, or doesn't hold what it claims."""


class Grain(StrEnum):
    PERSON_MONTH = "person_month"
    REQUISITION_MONTH = "requisition_month"
    DEPARTURE = "departure"
    COST_CENTER_MONTH = "cost_center_month"
    REQUISITION = "requisition"
    RATING = "rating"


@dataclass(frozen=True, eq=False)
class Extract:
    """One domain's training rows: `label` is the outcome column, and `features` the columns
    a model may read. `time` is the column that places a row in its window."""

    domain: str
    grain: Grain
    frame: pd.DataFrame
    label: str
    features: tuple[str, ...]
    time: str
    population: PopulationStamp
    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        missing = sorted({self.label, self.time, *self.features} - set(self.frame.columns))
        if missing:
            raise ExtractError(f"the {self.domain} extract has no columns {missing}")

    @property
    def rows(self) -> int:
        return len(self.frame)

    @property
    def labels(self) -> NDArray[np.generic]:
        result: NDArray[np.generic] = self.frame[self.label].to_numpy()
        return result

    def before(self, cutoff: dt.date) -> Extract:
        """The rows at or before `cutoff`, as an extract over (start, cutoff]."""
        return self._part(self._times() <= pd.Timestamp(cutoff), self.start, cutoff)

    def after(self, cutoff: dt.date) -> Extract:
        """The rows after `cutoff`, as an extract over (cutoff, end]."""
        return self._part(self._times() > pd.Timestamp(cutoff), cutoff, self.end)

    def _times(self) -> pd.Series:
        times = pd.to_datetime(self.frame[self.time])
        # A person-month sits at its month-end but its outcome falls the month after.
        if self.grain is Grain.PERSON_MONTH:
            times = times + pd.offsets.MonthEnd(1)
        return times

    def _part(self, keep: pd.Series, start: dt.date, end: dt.date) -> Extract:
        if not self.start <= start <= end <= self.end:
            raise ExtractError(f"a cutoff must lie in the window [{self.start}, {self.end}]")
        frame = self.frame[keep.to_numpy()].reset_index(drop=True)
        return Extract(
            self.domain,
            self.grain,
            frame,
            self.label,
            self.features,
            self.time,
            self.population,
            start,
            end,
        )


@dataclass(frozen=True)
class _Domain:
    grain: Grain
    query: str
    label: str
    features: Callable[[pd.DataFrame], tuple[str, ...]]
    time: str
    derive: Callable[[pd.DataFrame], pd.Series] | None = None


def _fixed(columns: tuple[str, ...]) -> Callable[[pd.DataFrame], tuple[str, ...]]:
    return lambda _: columns


def _org_context(frame: pd.DataFrame) -> tuple[str, ...]:
    """The cost center, then its org levels from finest to coarsest."""
    levels = sorted(
        (c for c in frame.columns if _ORG_LEVEL.fullmatch(c)),
        key=lambda c: -int(c.removeprefix("org_level")),
    )
    return ("cost_center", *(c for c in levels if not frame[c].equals(frame["cost_center"])))


def _channel(domain: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda frame: frame["fill_channel"].isin(CHANNELS[domain])


_RATING_FEATURES: Final = ("previous_scale", "previous_rank", "rating_scale", *ROLE, "cost_center")

DOMAINS: Final[Mapping[str, _Domain]] = MappingProxyType(
    {
        "attrition": _Domain(
            Grain.PERSON_MONTH, "attrition", "label", _fixed(EMPLOYEE_FEATURES), "month_end"
        ),
        "promotion": _Domain(
            Grain.PERSON_MONTH, "promotion", "label", _fixed(EMPLOYEE_FEATURES), "month_end"
        ),
        **{
            domain: _Domain(
                Grain.REQUISITION_MONTH,
                "requisition_months",
                "label",
                _fixed(OPENING_FEATURES),
                "month_end",
                _channel(domain),
            )
            for domain in CHANNELS
        },
        "cancellation": _Domain(
            Grain.REQUISITION_MONTH,
            "requisition_months",
            "label",
            _fixed(OPENING_FEATURES),
            "month_end",
            lambda frame: frame["closed_unfilled"],
        ),
        "backfill": _Domain(
            Grain.DEPARTURE,
            "departures",
            "label",
            _fixed((*ROLE, "cost_center", "departure")),
            "event_date",
        ),
        "growth": _Domain(
            Grain.COST_CENTER_MONTH,
            "growth_openings",
            "opened",
            _fixed(("cost_center",)),
            "month_end",
        ),
        "job_family": _Domain(
            Grain.REQUISITION, "requisition_roles", "job_family", _org_context, "opened_date"
        ),
        "performance": _Domain(
            Grain.RATING, "ratings", "rating_rank", _fixed(_RATING_FEATURES), "cycle_date"
        ),
    }
)


def _queries(
    con: duckdb.DuckDBPyConnection, spec: PopulationSpec
) -> Mapping[str, Callable[[], str]]:
    attributes = [c for c in EMPLOYEE_FEATURES if c in {*ROLE, "cost_center"}]

    def cost_center() -> str:
        return role_column(con, "req_interval")

    return {
        "attrition": lambda: sql.person_months(spec, NET_DEPARTURES, attributes),
        "promotion": lambda: sql.person_months(spec, PROMOTIONS_IN_PLACE, attributes),
        "requisition_months": lambda: sql.requisition_months(spec, cost_center()),
        "departures": lambda: sql.departures(spec),
        "growth_openings": lambda: sql.growth_openings(spec, cost_center()),
        "requisition_roles": lambda: sql.requisition_roles(spec, cost_center()),
        "ratings": lambda: sql.ratings(spec),
    }


def extracts(
    con: duckdb.DuckDBPyConnection,
    spec: PopulationSpec,
    start: dt.date,
    end: dt.date,
    domains: Iterable[str] | None = None,
) -> dict[str, Extract]:
    """Each domain's extract over (start, end]. Domains that share a query share one run."""
    chosen = list(DOMAINS) if domains is None else list(dict.fromkeys(domains))
    unknown = sorted(set(chosen) - set(DOMAINS))
    if unknown:
        raise ExtractError(f"no extract for domains {unknown}; known: {', '.join(DOMAINS)}")
    if not start < end:
        raise ExtractError(f"a window needs its start before its end, not {start} to {end}")
    queries = _queries(con, spec)
    frames: dict[str, pd.DataFrame] = {}
    result = {}
    for domain in chosen:
        definition = DOMAINS[domain]
        if definition.query not in frames:
            rendered = queries[definition.query]()
            frames[definition.query] = con.execute(rendered, {"start": start, "end": end}).df()
        frame = frames[definition.query]
        if definition.derive is not None:
            frame = frame.assign(**{definition.label: definition.derive(frame).to_numpy()})
        result[domain] = Extract(
            domain,
            definition.grain,
            frame,
            definition.label,
            definition.features(frame),
            definition.time,
            spec.stamp,
            start,
            end,
        )
    return result


def extract(
    con: duckdb.DuckDBPyConnection,
    domain: str,
    spec: PopulationSpec,
    start: dt.date,
    end: dt.date,
) -> Extract:
    return extracts(con, spec, start, end, [domain])[domain]
