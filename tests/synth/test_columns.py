"""Tests for the world's columnar building blocks and its vectorized process pieces."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.synth.columns import EventLog, Table
from hcsim.synth.config import AttritionTruth, RequisitionTruth, WorldConfig, preset
from hcsim.synth.population import Roles, draw_roles
from hcsim.synth.process import TransferPool, random_days
from hcsim.synth.workforce import Workforce
from hcsim.synth.world import generate_world, write_world


def test_a_table_grows_and_fills_unnamed_columns_with_empties() -> None:
    table = Table({"id": np.int64, "when": "datetime64[D]", "flag": bool}, capacity=2)
    first = table.append(2, id=[1, 2], flag=True)
    later = table.append(3, id=7)  # past capacity, with a broadcast scalar
    assert first.tolist() == [0, 1]
    assert later.tolist() == [2, 3, 4]
    assert table.column("id").tolist() == [1, 2, 7, 7, 7]
    assert np.isnat(table.column("when")).all()
    assert table.column("flag").tolist() == [True, True, False, False, False]
    table.column("flag")[3] = True  # views write through
    assert table.column("flag")[3]
    with pytest.raises(KeyError, match="no columns"):
        table.append(1, nope=1)


def test_an_event_log_concatenates_chunks_and_checks_their_columns() -> None:
    log = EventLog(["a", "b"])
    log.add(2, a=[1, 2], b=0)
    log.add(0, a=[], b=[])  # an empty chunk is skipped
    log.add(1, a=[3], b=[9])
    assert len(log) == 3
    out = log.collect({"a": np.int64, "b": np.int8})
    assert out["a"].tolist() == [1, 2, 3]
    assert out["b"].dtype == np.int8
    assert out["b"].tolist() == [0, 0, 9]
    with pytest.raises(KeyError, match="missing \\['b'\\]"):
        log.add(1, a=[1])
    assert EventLog(["a"]).collect({"a": np.int64})["a"].size == 0


def test_vector_hazards_match_the_scalar_definitions() -> None:
    truth = AttritionTruth()
    levels, tenure, effect = np.array([1, 3, 5]), np.array([3.0, 30.0, 11.9]), np.array([1, 2, 0.5])
    vector = truth.hazards(levels, tenure, 7, effect)
    scalar = [
        truth.hazard(int(lv), float(t), 7, float(e))
        for lv, t, e in zip(levels, tenure, effect, strict=True)
    ]
    assert vector.tolist() == pytest.approx(scalar)
    reqs = RequisitionTruth()
    assert reqs.external_hazards(np.array([0, 2, 9])).tolist() == [
        reqs.external_hazard(0),
        reqs.external_hazard(2),
        reqs.external_hazard(9),
    ]


def test_roles_lean_to_the_dominant_family() -> None:
    config = WorldConfig()
    rng = np.random.default_rng(0)
    roles = draw_roles(config, np.full(20_000, 5), np.full(20_000, 4), rng)
    dominant = 5 % len(config.families)
    share = (roles["family"] == dominant).mean()
    assert share == pytest.approx(config.dominant_family_share, abs=0.02)
    assert roles["is_manager"].mean() == pytest.approx(config.manager_share, abs=0.02)
    junior = draw_roles(config, np.zeros(100, int), np.ones(100, int), rng)
    assert not junior["is_manager"].any()  # managers start at manager_min_level


def test_random_days_stay_in_the_month_and_after_the_earliest_day() -> None:
    month_end = pd.Timestamp("2024-02-29")
    rng = np.random.default_rng(1)
    earliest = np.full(500, np.datetime64("2024-02-20"))
    days = random_days(month_end, 500, rng, earliest)
    assert days.min() >= np.datetime64("2024-02-20")
    assert days.max() <= np.datetime64("2024-02-29")
    anywhere = random_days(month_end, 2_000, rng)
    assert anywhere.min() == np.datetime64("2024-02-01")


def _workforce(cost_centers: list[int], families: list[int], levels: list[int]) -> Workforce:
    n = len(cost_centers)
    roles: Roles = {
        "cost_center": np.array(cost_centers, dtype=np.int32),
        "family": np.array(families, dtype=np.int16),
        "level": np.array(levels, dtype=np.int16),
        "location": np.zeros(n, np.int16),
        "company": np.zeros(n, np.int16),
        "account": np.zeros(n, np.int16),
        "is_manager": np.zeros(n, bool),
    }
    return Workforce(np.arange(n, dtype=np.int64), roles, np.full(n, np.datetime64("2020-01-01")))


def test_the_transfer_pool_picks_only_valid_candidates_uniformly() -> None:
    # Candidates for family 0, level 2, from outside cost center 0: rows 2, 3, and 4.
    wf = _workforce([0, 0, 1, 2, 3, 1], [0, 0, 0, 0, 0, 1], [2, 3, 1, 3, 2, 2])
    pool = TransferPool(wf, band=1)
    rng = np.random.default_rng(3)
    picks = [pool.pick(0, 2, 0, rng) for _ in range(3_000)]
    counts = np.bincount(np.array(picks), minlength=6)
    assert set(np.flatnonzero(counts)) == {2, 3, 4}
    assert counts[[2, 3, 4]] / 3_000 == pytest.approx([1 / 3] * 3, abs=0.04)
    assert pool.pick(1, 5, 1, rng) is None  # nobody within the band
    assert pool.pick(1, 2, 1, rng) is None  # the only candidate works in the same cost center


def test_the_pool_falls_back_to_an_exact_draw_when_rejection_keeps_missing() -> None:
    # 200 in the requisition's own cost center and one elsewhere: rejection mostly misses.
    wf = _workforce([0] * 200 + [1], [0] * 201, [1] * 201)
    pool = TransferPool(wf, band=0)
    rng = np.random.default_rng(4)
    assert {pool.pick(0, 1, 0, rng) for _ in range(20)} == {200}


def test_a_moved_employee_changes_group() -> None:
    wf = _workforce([0, 1], [0, 0], [1, 1])
    pool = TransferPool(wf, band=0)
    pool.move(1, (0, 1), (0, 3))
    rng = np.random.default_rng(5)
    assert pool.pick(0, 1, 5, rng) == 0
    assert pool.pick(0, 3, 5, rng) == 1


def test_the_warehouse_stores_dates_and_text(tmp_path: Path) -> None:
    world = generate_world(preset("tiny", seed=2))
    assert isinstance(world.tables["workforce_txn"]["cost_center"].dtype, pd.CategoricalDtype)
    path = tmp_path / "w.duckdb"
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        types = dict(con.execute("DESCRIBE workforce_txn").fetchall()[i][:2] for i in range(14))
        reqs = dict(r[:2] for r in con.execute("DESCRIBE req_interval").fetchall())
    assert (types["event_date"], types["hire_date"], types["cost_center"]) == (
        "DATE",
        "DATE",
        "VARCHAR",
    )
    assert (reqs["closed_date"], reqs["close_reason"], reqs["filled_by"]) == (
        "DATE",
        "VARCHAR",
        "BIGINT",
    )
