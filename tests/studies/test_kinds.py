"""Tests for input kinds: every input a study can need, declared once, and checked."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from hcsim.model.core.contract import role_column
from hcsim.model.data.population import REGULAR
from hcsim.studies.kinds import (
    KINDS,
    Column,
    Family,
    InputKind,
    check_kinds,
    closure,
    contract_problems,
)


def test_every_extract_the_warehouse_gives_honors_its_contract(inputs: tuple[Path, Path]) -> None:
    world, _ = inputs
    window = {"start": dt.date(2024, 12, 31), "end": dt.date(2025, 12, 31)}
    with duckdb.connect(str(world), read_only=True) as con:
        cost_center = role_column(con, "req_interval")
        for kind in KINDS.values():
            if kind.family is not Family.EXTRACT:
                continue
            assert kind.sql is not None
            frame = con.execute(kind.sql(REGULAR, cost_center), window).df()
            assert len(frame), kind.name
            assert contract_problems(kind, frame) == [], kind.name


def test_the_registry_holds_together() -> None:
    assert check_kinds(KINDS) == []
    families = {k.family for k in KINDS.values()}
    assert families == set(Family)
    for kind in KINDS.values():
        if kind.family is Family.EXTRACT:
            assert contract_problems(kind, kind.fixture()) == []
            assert len(kind.fixture(3)) == 3
    assert closure(["seed_equity", "fitted_models", "replay_history"]) == (
        "seed_state",
        "seed_equity",
        "fitted_models",
        "replay_history",
    )


def test_a_contract_names_what_a_frame_lacks_or_holds_wrongly() -> None:
    kind = KINDS["growth_openings"]
    frame = pd.DataFrame({"month_end": ["2026-01-31"], "opened": [1.5]})
    assert contract_problems(kind, frame) == [
        "growth_openings needs the columns ['cost_center']",
        "growth_openings.month_end holds object, not date",
        "growth_openings.opened holds float64, not integer",
    ]


def test_a_registry_that_doesnt_hold_together_is_refused() -> None:
    good = KINDS["growth_openings"]
    broken = {
        "lost": InputKind("lost", Family.DERIVED, "", depends_on=("nowhere",)),
        "bare": dataclasses.replace(good, name="bare", sql=None),
        "loose": InputKind("loose", Family.ARTIFACT, "", chunk_by="month_end"),
        "odd": dataclasses.replace(good, name="odd", chunk_by="day"),
        "keyed": InputKind("keyed", Family.DERIVED, "", cache_key=("weather",)),
        "wrong": dataclasses.replace(good, name="right"),
        "a": InputKind("a", Family.DERIVED, "", depends_on=("b",)),
        "b": InputKind("b", Family.DERIVED, "", depends_on=("a",)),
        "bent": dataclasses.replace(good, name="bent", columns={"month_end": Column.FLAG}),
    }
    problems = check_kinds(broken)
    assert problems == [
        "lost depends on no kinds ['nowhere']",
        "the extract bare needs its SQL, chunking, and column contract",
        "loose isn't an extract, so has no SQL, chunking, or contract",
        "odd chunks by day, which its contract lacks",
        "keyed keys its cache on unknown fields ['weather']",
        "wrong is registered under another name, right",
        "kinds depend on themselves: a -> b -> a",
    ]
    assert all(p != "bent's fixture breaks its own contract" for p in problems)  # flags fit flags
    with pytest.raises(KeyError):
        closure(["nowhere"])
