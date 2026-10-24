"""Tests for capacity caps, `Cap(c)`: a vector field of finite whole caps, and what a subset run
does with caps it can't reach."""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from hcsim.engine.ceiling import Ceiling, CeilingError
from hcsim.engine.seeding import seed_for
from hcsim.model.data.subset import SubsetSpec
from tests.engine.test_loop import SCHEMA
from tests.engine.test_seed_equity import con, configured  # noqa: F401  # the tiny world

INF = float("inf")


def test_caps_are_a_vector_over_the_schemas_cost_centers() -> None:
    caps = Ceiling.build(SCHEMA, {"CC-01": 40, "CC-03": 12.0}).caps
    assert caps.tolist() == [40.0, INF, 12.0]  # uncapped where none is declared
    assert Ceiling.build(SCHEMA).caps.tolist() == [INF, INF, INF]


def test_a_cap_is_a_finite_whole_number_of_heads() -> None:
    with pytest.raises(CeilingError) as caught:
        Ceiling.build(
            SCHEMA,
            {"CC-01": float("inf"), "CC-02": 2.5, "CC-03": "lots"},
        )
    assert caught.value.problems == [
        "the cap on CC-01 must be finite, not inf: leave it out to leave CC-01 uncapped",
        "the cap on CC-02 must be a non-negative whole number, not 2.5",
        "the cap on CC-03 is a number of heads, not 'lots'",
    ]
    with pytest.raises(CeilingError, match="must be finite, not nan"):
        Ceiling.build(SCHEMA, {"CC-01": float("nan")})
    assert Ceiling.build(SCHEMA, {"CC-01": np.int64(7)}).caps[0] == 7.0


def test_a_subset_run_reports_the_caps_it_cant_reach(con: duckdb.DuckDBPyConnection) -> None:  # noqa: F811
    import dataclasses

    labels = list(configured(con).schema.vocabulary("cost_center").labels)
    inside, outside = labels[0], labels[-1]
    run = dataclasses.replace(configured(con), capacity_caps={inside: 500, outside: 500})
    subset = SubsetSpec({"cost_center": frozenset({inside})})
    _, report = seed_for(con, run, subset=subset)
    assert report.caps_outside_subset == (outside,)
    _, whole = seed_for(con, run)
    assert whole.caps_outside_subset == ()  # a run of everyone reaches every cap
