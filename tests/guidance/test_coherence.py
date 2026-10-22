"""Tests for the fill-tree coherence passes."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import numpy as np
import pytest

from hcsim.guidance.coherence import PASSES, cohere
from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.pipeline import run
from hcsim.model.core.categories import GuidanceCategory
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

C = GuidanceCategory


def tree(fills: float, hires: float, ext: float, conv: float, internal: float) -> dict:
    values = {C.FILLS: fills, C.HIRES: hires, C.EXTERNAL_HIRES: ext}
    values |= {C.CONVERSION_HIRES: conv, C.INTERNAL_STARTS: internal}
    return {("G", c): np.array([v]) for c, v in values.items()}


def coherent(result: dict) -> bool:
    hires = result["G", C.EXTERNAL_HIRES] + result["G", C.CONVERSION_HIRES]
    fills = result["G", C.HIRES] + result["G", C.INTERNAL_STARTS]
    return bool(
        np.allclose(result["G", C.HIRES], hires) and np.allclose(result["G", C.FILLS], fills)
    )


def test_bottom_up_carries_the_leaves_up_deepest_first() -> None:
    result, closed = cohere(tree(20, 9, 6, 2, 5), "bottom_up")
    assert coherent(result)
    assert result["G", C.HIRES].tolist() == [8.0]
    assert result["G", C.FILLS].tolist() == [13.0]  # from the rebuilt hires, not the old 9
    assert closed == {"hires": 1.0, "fills": 7.0}  # against the rebuilt hires


def test_top_down_scales_the_children_to_their_parent_shallowest_first() -> None:
    result, _ = cohere(tree(20, 9, 6, 2, 5), "top_down")
    assert coherent(result)
    assert result["G", C.FILLS].tolist() == [20.0]
    # fills splits 9 : 5 between hires and internal starts, then hires 6 : 2 between its own.
    assert result["G", C.HIRES][0] == pytest.approx(20 * 9 / 14)
    assert result["G", C.EXTERNAL_HIRES][0] == pytest.approx(20 * 9 / 14 * 6 / 8)
    zeros, _ = cohere(tree(10, 4, 0, 0, 6), "top_down")
    assert zeros["G", C.EXTERNAL_HIRES][0] == zeros["G", C.CONVERSION_HIRES][0]  # equal shares


def test_ols_makes_the_least_change_to_every_node_at_once() -> None:
    result, _ = cohere(tree(20, 9, 6, 2, 5), "ols")
    assert coherent(result)
    assert all((v >= 0).all() for v in result.values())
    # Neither the leaves' nor the root's forecast wins outright: each moves part of the way.
    assert 13 < result["G", C.FILLS][0] < 20
    one, _ = cohere(
        {
            ("G", C.HIRES): np.array([9.0]),
            ("G", C.EXTERNAL_HIRES): np.array([6.0]),
            ("G", C.CONVERSION_HIRES): np.array([2.0]),
        },
        "ols",
    )
    # One parent, two children, a gap of 1: the parent falls 1/3 and each child rises 1/3.
    assert one["G", C.HIRES][0] == pytest.approx(9 - 1 / 3)
    assert one["G", C.EXTERNAL_HIRES][0] == pytest.approx(6 + 1 / 3)


def test_a_parent_without_all_its_children_is_left_alone() -> None:
    partial = {("G", C.HIRES): np.array([9.0]), ("G", C.EXTERNAL_HIRES): np.array([6.0])}
    for method in ("bottom_up", "top_down", "ols"):
        result, closed = cohere(partial, method)
        assert result["G", C.HIRES].tolist() == [9.0]
        assert closed == {}
    with pytest.raises(ValueError, match="unknown coherence method"):
        cohere(partial, "median")  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


TREE = [
    "headcount",
    "attrition",
    "fills",
    "hires",
    "external_hires",
    "conversion_hires",
    "internal_starts",
]


@pytest.mark.parametrize("method", ["bottom_up", "top_down", "ols"])
def test_a_run_carrying_the_whole_tree_writes_a_coherent_artifact(
    con: duckdb.DuckDBPyConnection, method: str
) -> None:
    config = GuidanceConfig.from_mapping(
        {
            "horizon": 6,
            "grouping": "TOPLINE",
            "categories": TREE,
            "coherence": {"method": method},
        }
    )
    result = run(con, config)  # the artifact refuses an incoherent tree, so this is the check
    coherence = result.report["coherence"]
    assert coherence["method"] == method
    assert set(coherence["closed"]) == {"generated", "reconciled", "final"}
    assert coherence["closed"]["generated"]["fills"] > 0  # independent forecasts disagreed
    assert set(PASSES) >= set(coherence["closed"])
    # Reconciliation adjusts the leaves, the identity's terms, and the passes after it rebuild
    # the parents from them.
    terms = set(result.report["reconciliation"]["terms"])
    assert terms == {"external_hires", "conversion_hires", "internal_starts", "attrition"}
