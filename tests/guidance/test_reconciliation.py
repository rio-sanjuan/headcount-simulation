"""Tests for the reconciliation methods: greedy, sign-aware greedy, OLS, WLS, and MinT."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import numpy as np
import pytest

from hcsim.guidance.config import GuidanceConfig
from hcsim.guidance.pipeline import run
from hcsim.guidance.reconcile import (
    ANCHOR_KEY,
    Problem,
    _shrunk,
    gaps,
    greedy,
    mint,
    ols,
    wls,
)
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

SIGNS = {"hires": 1, "conversions": 1, "attrition": -1}


def problem(anchor: list[float], **kwargs: object) -> Problem:
    flows = {
        "hires": np.array([3.0, 3.0]),
        "conversions": np.array([1.0, 1.0]),
        "attrition": np.array([2.0, 2.0]),
    }
    return Problem(np.array(anchor), 100.0, flows, SIGNS, 0.0, **kwargs)  # type: ignore[arg-type]


def test_a_gap_is_the_anchor_s_change_less_the_signed_flows() -> None:
    p = problem([104.0, 104.0])
    assert gaps(p.anchor, p.last_actual, p.flows, p.signs).tolist() == [2.0, -2.0]


def test_sign_aware_greedy_raises_the_flows_moving_with_the_gap() -> None:
    # Month 1: +6 against +2 implied, so a gap of +4; month 2: -8 against +2, a gap of -10.
    result = greedy(problem([106.0, 98.0], sign_aware=True))
    # The rise lifts the inflows in proportion, 3 to 1, and leaves attrition alone.
    assert result.flows["hires"].tolist() == pytest.approx([6.0, 3.0])
    assert result.flows["conversions"].tolist() == pytest.approx([2.0, 1.0])
    # The fall lifts attrition, the only outflow, and cuts no inflow.
    assert result.flows["attrition"].tolist() == pytest.approx([2.0, 12.0])
    assert np.allclose(result.residual, 0.0)
    plain = greedy(problem([106.0, 98.0]))
    assert plain.flows["conversions"][0] == pytest.approx(5.0)  # the smallest flow takes it all


@pytest.mark.parametrize("method", [ols, wls, mint])
def test_a_projection_satisfies_the_identity_exactly(method: object) -> None:
    errors = np.random.default_rng(0).normal(0, 1, (3, 12))
    residuals = dict(zip(SIGNS, errors, strict=True))
    result = method(problem([104.0, 106.0], residuals=residuals))  # type: ignore[operator]
    assert result.anchor is None  # held fixed
    assert np.allclose(result.residual, 0.0, atol=1e-6)


BASE = {"hires": 3.0, "conversions": 1.0, "attrition": 2.0}


def test_ols_spreads_the_gap_equally_and_wls_by_each_series_error_variance() -> None:
    # Month 1: +4 against +2 implied; month 2: +2 against +2.
    even = ols(problem([104.0, 106.0]))
    # Each of three flows moves 2/3 in the identity's direction.
    assert even.flows["hires"].tolist() == pytest.approx([3 + 2 / 3, 3.0])
    assert even.flows["attrition"].tolist() == pytest.approx([2 - 2 / 3, 2.0])
    noisy = {
        "hires": np.r_[-9.0, 9.0],
        "conversions": np.r_[-1.0, 1.0],
        "attrition": np.r_[-1.0, 1.0],
    }
    weighted = wls(problem([104.0, 106.0], residuals=noisy))
    assert weighted.method == "wls"
    moves = {name: abs(weighted.flows[name][0] - BASE[name]) for name in BASE}
    # Variances 81 : 1 : 1, so hires takes 81/83 of the gap.
    assert moves["hires"] == pytest.approx(2 * 81 / 83)
    assert moves["conversions"] == pytest.approx(2 / 83)


def test_mint_shrinks_the_covariance_and_falls_back_without_errors() -> None:
    rng = np.random.default_rng(1)
    common = rng.normal(0, 1, 24)
    errors = np.vstack([common + rng.normal(0, 0.1, 24) for _ in range(3)])
    _, intensity = _shrunk(errors, None)
    assert 0.0 <= intensity <= 1.0
    full, _ = _shrunk(errors, 1.0)
    assert np.allclose(full, np.diag(np.diag(full)))  # full shrinkage is the diagonal
    residuals = dict(zip(SIGNS, errors, strict=True))
    assert mint(problem([106.0, 104.0], residuals=residuals)).method == "mint"
    diagonal = mint(problem([106.0, 104.0], residuals=residuals, shrinkage=1.0))
    weighted = wls(problem([106.0, 104.0], residuals=residuals))
    for name in SIGNS:
        assert np.allclose(diagonal.flows[name], weighted.flows[name], atol=1e-4)
    assert mint(problem([106.0, 104.0])).method == "ols"  # no errors: WLS can't, so OLS
    assert wls(problem([106.0, 104.0])).method == "ols"


def test_a_projection_can_move_the_anchor_too() -> None:
    free = ols(problem([106.0, 104.0], anchor_fixed=False))
    assert free.anchor is not None
    assert not np.allclose(free.anchor, [106.0, 104.0])
    assert np.allclose(gaps(free.anchor, 100.0, free.flows, SIGNS), 0.0, atol=1e-6)
    errors = dict.fromkeys([ANCHOR_KEY, *SIGNS], np.r_[-1.0, 1.0])
    assert wls(problem([106.0, 104.0], anchor_fixed=False, residuals=errors)).method == "wls"


def test_no_flow_goes_negative_and_the_rest_absorb_the_gap() -> None:
    # A fall of 22 against +2 implied: the inflows can't fall below zero, so attrition rises.
    result = ols(problem([80.0, 82.0]))
    assert all((v >= 0).all() for v in result.flows.values())
    assert result.flows["hires"][0] == pytest.approx(0.0)
    assert result.flows["conversions"][0] == pytest.approx(0.0)
    assert result.flows["attrition"][0] == pytest.approx(20.0)
    assert np.allclose(result.residual, 0.0, atol=1e-6)
    # With only inflows to move and a fall they can't make, the gap is what's left.
    inflows = Problem(np.array([90.0]), 100.0, {"hires": np.array([3.0])}, {"hires": 1}, 0.0)
    stuck = ols(inflows)
    assert stuck.flows["hires"].tolist() == [0.0]
    assert stuck.residual.tolist() == pytest.approx([-10.0])


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


@pytest.mark.parametrize("method", ["greedy", "ols", "wls", "mint"])
def test_a_run_reconciles_by_each_method(con: duckdb.DuckDBPyConnection, method: str) -> None:
    config = GuidanceConfig.from_mapping(
        {
            "horizon": 6,
            "grouping": "TOPLINE",
            "reconciliation": {"method": method, "tolerance": 0.0},
        }
    )
    report = run(con, config).report["reconciliation"]
    assert sum(report["methods"].values()) == 1  # one group
    assert report["max_abs_residual"] < 1e-6
