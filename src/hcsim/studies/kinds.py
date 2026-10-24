"""Input kinds: every input a study can need, declared once.

A study's inputs come in three **families**:

- an **extract** is rows read from the warehouse by SQL, rendered for the study's population and
  the data contract's columns, over a window;
- an **artifact** is a file a study is given, such as a guidance forecast or a plan;
- a **derived** input is computed from others and from the warehouse by the engine, such as
  the seed state or the fitted models.

Each kind declares:
- **its SQL**, an extract's renderer;
- **its chunking**, the time column an extract can be split along, so a long window can be
  read and cached a piece at a time;
- **its cache-key fields**, what an input of the kind is keyed on, so two studies that need the
  same rows read them once;
- **its dependencies**, the kinds it's computed from;
- **its column contract**, each column an extract must hold and the kind of value in it;
- **its tables**, the warehouse tables a kind read outside SQL needs;
- **its fixture**, a small frame that honors its contract, for tests and runs with no world.

**Checks at import** (`check_kinds`): every dependency is a declared kind and none is circular.
Only extracts carry SQL, chunking, or a column contract, and every extract has all three. Every
cache-key field is one a study can supply, and every fixture honors its own kind's contract. A
registry that breaks any of these refuses to import.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

from hcsim.model.core.measures import NET_DEPARTURES, PROMOTIONS_IN_PLACE
from hcsim.model.data.population import PopulationSpec
from hcsim.model.sql import extract as sql


class Family(StrEnum):
    EXTRACT = "extract"
    ARTIFACT = "artifact"
    DERIVED = "derived"


class Column(StrEnum):
    """The kind of value a contract's column holds."""

    INTEGER = "integer"
    NUMBER = "number"
    TEXT = "text"
    FLAG = "flag"
    DATE = "date"


#: Every field a study can key an input on.
CACHE_FIELDS: Final = frozenset(
    {"population", "contract", "start", "end", "run_date", "months", "path", "fingerprint"}
)
#: The columns of a person-month extract.
_ROLE: Final = {
    "job_level": Column.INTEGER,
    "location": Column.TEXT,
    "job_family": Column.TEXT,
    "tech_indicator": Column.TEXT,
    "is_manager": Column.FLAG,
    "cost_center": Column.TEXT,
    "company": Column.TEXT,
}
_PERSON_MONTHS: Final = {
    "employee_id": Column.INTEGER,
    "month_end": Column.DATE,
    **_ROLE,
    "tenure_months": Column.INTEGER,
    "next_month": Column.INTEGER,
    "rating_rank": Column.INTEGER,
    "rating_scale": Column.TEXT,
    "label": Column.FLAG,
}

#: A renderer of an extract's SQL: the population, and the requisition cost-center column the
#: data contract names.
Render = Callable[[PopulationSpec, str], str]


class KindError(RuntimeError):
    """A registry of kinds that doesn't hold together: a defect, refused at import."""


@dataclass(frozen=True)
class InputKind:
    name: str
    family: Family
    doc: str
    sql: Render | None = None
    #: The time column an extract can be split along.
    chunk_by: str | None = None
    cache_key: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    #: Each column an extract must hold, and the kind of value in it. More may follow.
    columns: Mapping[str, Column] = field(default_factory=dict)
    #: The warehouse tables a kind read outside SQL needs.
    tables: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))

    def fixture(self, rows: int = 2) -> pd.DataFrame:
        """A small frame that honors the contract: `rows` rows of plain placeholder values."""
        values: dict[Column, Callable[[int], object]] = {
            Column.INTEGER: lambda n: np.arange(n, dtype=np.int64),
            Column.NUMBER: lambda n: np.linspace(0.0, 1.0, n),
            Column.TEXT: lambda n: [f"{self.name}-{k}" for k in range(n)],
            Column.FLAG: lambda n: np.arange(n) % 2 == 0,
            Column.DATE: lambda n: pd.date_range(dt.date(2026, 1, 31), periods=n, freq="ME"),
        }
        return pd.DataFrame({c: values[kind](rows) for c, kind in self.columns.items()})


def contract_problems(kind: InputKind, frame: pd.DataFrame) -> list[str]:
    """Every way `frame` breaks `kind`'s column contract: a column missing, or holding another
    kind of value."""
    problems = []
    checks: dict[Column, Callable[[pd.Series], bool]] = {
        Column.INTEGER: pd.api.types.is_integer_dtype,
        Column.NUMBER: pd.api.types.is_numeric_dtype,
        Column.TEXT: lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
        Column.FLAG: pd.api.types.is_bool_dtype,
        Column.DATE: pd.api.types.is_datetime64_any_dtype,
    }
    missing = [c for c in kind.columns if c not in frame.columns]
    if missing:
        problems.append(f"{kind.name} needs the columns {missing}")
    for column, value in kind.columns.items():
        if column in frame.columns and not checks[value](frame[column]):
            problems.append(f"{kind.name}.{column} holds {frame[column].dtype}, not {value}")
    return problems


def _extract(
    name: str, doc: str, render: Render, chunk_by: str, columns: Mapping[str, Column]
) -> InputKind:
    return InputKind(
        name,
        Family.EXTRACT,
        doc,
        render,
        chunk_by,
        ("population", "contract", "start", "end"),
        columns=columns,
    )


_ATTRIBUTES: Final = list(_ROLE)

KINDS: Final[Mapping[str, InputKind]] = MappingProxyType(
    {
        k.name: k
        for k in (
            _extract(
                "departure_months",
                "person-months labelled by a net departure the next month",
                lambda spec, _: sql.person_months(spec, NET_DEPARTURES, _ATTRIBUTES),
                "month_end",
                _PERSON_MONTHS,
            ),
            _extract(
                "promotion_months",
                "person-months labelled by a promotion the next month",
                lambda spec, _: sql.person_months(spec, PROMOTIONS_IN_PLACE, _ATTRIBUTES),
                "month_end",
                _PERSON_MONTHS,
            ),
            _extract(
                "requisition_months",
                "requisition-months from approval, with their fills and exits",
                sql.requisition_months,
                "month_end",
                {
                    "req_id": Column.INTEGER,
                    "month_end": Column.DATE,
                    "req_type": Column.TEXT,
                    **_ROLE,
                    "age_months": Column.INTEGER,
                    "openings": Column.INTEGER,
                    "fills": Column.INTEGER,
                    "fill_channel": Column.TEXT,
                    "closed_unfilled": Column.FLAG,
                },
            ),
            _extract(
                "departures",
                "terminations and transfers out, labelled by a backfill opened for them",
                lambda spec, _: sql.departures(spec),
                "event_date",
                {
                    "employee_id": Column.INTEGER,
                    "event_date": Column.DATE,
                    "departure": Column.TEXT,
                    **_ROLE,
                    "label": Column.FLAG,
                },
            ),
            _extract(
                "growth_openings",
                "growth requisitions opened per cost center and month",
                sql.growth_openings,
                "month_end",
                {"month_end": Column.DATE, "cost_center": Column.TEXT, "opened": Column.INTEGER},
            ),
            _extract(
                "requisition_roles",
                "growth requisitions, with the role and org context each was opened for",
                sql.requisition_roles,
                "opened_date",
                {"req_id": Column.INTEGER, "opened_date": Column.DATE, **_ROLE},
            ),
            _extract(
                "ratings",
                "ratings, each with the one before it",
                lambda spec, _: sql.ratings(spec),
                "cycle_date",
                {
                    "employee_id": Column.INTEGER,
                    "cycle_date": Column.DATE,
                    **_ROLE,
                    "previous_rank": Column.INTEGER,
                    "rating_rank": Column.INTEGER,
                    "is_first": Column.FLAG,
                },
            ),
            InputKind(
                "guidance_forecast",
                Family.ARTIFACT,
                "a guidance forecast's artifact, for a guided run",
                cache_key=("path", "fingerprint"),
            ),
            InputKind(
                "plan",
                Family.ARTIFACT,
                "an alignment target, saved, for a run that aligns to one",
                cache_key=("path", "fingerprint"),
            ),
            InputKind(
                "fitted_models",
                Family.DERIVED,
                "the domain models, fitted over the year before the run",
                cache_key=("population", "contract", "run_date"),
                tables=frozenset({"workforce_txn", "req_interval", "dim_job", "dim_account"}),
            ),
            InputKind(
                "seed_state",
                Family.DERIVED,
                "the workforce and open requisitions at the snapshot before the run",
                cache_key=("population", "contract", "run_date"),
                tables=frozenset({"workforce_agg", "req_snapshot", "rating", "dim_org", "dim_job"}),
            ),
            InputKind(
                "seed_equity",
                Family.DERIVED,
                "the seed's unvested equity, and the offline stream",
                cache_key=("population", "contract", "run_date"),
                depends_on=("seed_state",),
                tables=frozenset(
                    {"equity_award", "equity_vest", "equity_holder", "equity_valuation"}
                ),
            ),
            InputKind(
                "replay_history",
                Family.DERIVED,
                "realized movements as known events, realized requisitions, and recorded awards",
                cache_key=("population", "contract", "run_date", "months"),
                depends_on=("seed_state",),
                tables=frozenset(
                    {"workforce_txn", "req_interval", "pending_start", "equity_award"}
                ),
            ),
            InputKind(
                "realized_totals",
                Family.DERIVED,
                "the window's realized counts, the guidance a guided-actuals run reads",
                cache_key=("population", "contract", "run_date", "months"),
                tables=frozenset({"workforce_txn", "req_snapshot", "pending_start"}),
            ),
            InputKind(
                "actual_sbc",
                Family.DERIVED,
                "the reference ledger's SBC over the window, to score SBC against",
                cache_key=("population", "contract", "run_date", "months"),
                tables=frozenset({"equity_expense", "equity_award", "equity_holder"}),
            ),
        )
    }
)


def check_kinds(kinds: Mapping[str, InputKind]) -> list[str]:
    """Every way a registry of kinds fails to hold together."""
    problems = []
    for name, kind in kinds.items():
        if name != kind.name:
            problems.append(f"{name} is registered under another name, {kind.name}")
        unknown = sorted(set(kind.depends_on) - set(kinds))
        if unknown:
            problems.append(f"{name} depends on no kinds {unknown}")
        extract = kind.family is Family.EXTRACT
        declared = (kind.sql is not None, kind.chunk_by is not None, bool(kind.columns))
        if extract and not all(declared):
            problems.append(f"the extract {name} needs its SQL, chunking, and column contract")
        if not extract and any(declared):
            problems.append(f"{name} isn't an extract, so has no SQL, chunking, or contract")
        if extract and kind.chunk_by not in kind.columns:
            problems.append(f"{name} chunks by {kind.chunk_by}, which its contract lacks")
        strangers = sorted(set(kind.cache_key) - CACHE_FIELDS)
        if strangers:
            problems.append(f"{name} keys its cache on unknown fields {strangers}")
        if extract and contract_problems(kind, kind.fixture()):
            problems.append(f"{name}'s fixture breaks its own contract")
    problems += _cycles(kinds)
    return problems


def _cycles(kinds: Mapping[str, InputKind]) -> list[str]:
    found = []
    for start in kinds:
        stack: list[tuple[str, tuple[str, ...]]] = [(start, (start,))]
        while stack:
            name, path = stack.pop()
            for nxt in kinds[name].depends_on if name in kinds else ():
                if nxt == start:
                    found.append(" -> ".join((*path, nxt)))
                elif nxt not in path:
                    stack.append((nxt, (*path, nxt)))
    first = sorted(found)[:1]
    return [f"kinds depend on themselves: {c}" for c in first]


def closure(names: Sequence[str]) -> tuple[str, ...]:
    """`names` and every kind they depend on, dependencies first."""
    order: list[str] = []

    def visit(name: str) -> None:
        if name in order:
            return
        for dependency in KINDS[name].depends_on:
            visit(dependency)
        order.append(name)

    for name in names:
        visit(name)
    return tuple(order)


_problems = check_kinds(KINDS)
if _problems:  # pragma: no cover - a defect in the table above
    raise KindError("; ".join(_problems))
