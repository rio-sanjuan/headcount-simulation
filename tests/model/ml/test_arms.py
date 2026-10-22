"""Tests for the arm matrix, the selection rule, and lift tables."""

from __future__ import annotations

import numpy as np
import pytest

from hcsim.model.ml.arms import (
    DEFAULT_MATRIX,
    DEFAULT_Z,
    Arm,
    ArmError,
    ArmMatrix,
    contributions,
    lift_table,
    select,
)
from hcsim.model.ml.metrics import log_loss, wape


def test_the_authored_matrix_holds() -> None:
    assert "attrition" in DEFAULT_MATRIX.domains
    arms = DEFAULT_MATRIX.domain("attrition")
    assert [a.algorithm for a in arms] == ["constant", "level_only", "logistic", "boosted"]
    assert DEFAULT_MATRIX.null("growth").algorithm == "poisson_growth"
    for domain in DEFAULT_MATRIX.domains:
        assert DEFAULT_MATRIX.null(domain).null
    with pytest.raises(ArmError, match="no arms for domain 'reorg'"):
        DEFAULT_MATRIX.domain("reorg")


def test_a_bad_matrix_reports_every_problem() -> None:
    arms = (
        Arm("a.null", "a", "classifier", "constant", 1, null=True),
        Arm("a.rich", "a", "classifier", "boosted", 0),
        Arm("a.rich", "a", "count", "boosted", 0),
        Arm("b.one", "b", "classifier", "markov", 0),
        Arm("c.x", "c", "tree", "x", 0, null=True),
    )
    with pytest.raises(ArmError, match=r"arm names repeat: \[.a\.rich.\]") as error:
        ArmMatrix(arms)
    message = str(error.value)
    for problem in (
        "a mixes families ['classifier', 'count']",
        "a has arms of equal simplicity",
        "a's null arm isn't its simplest",
        "b needs exactly one null arm, not 0",
        "b.one: 'markov' isn't a classifier algorithm",
        "c.x has family 'tree'",
    ):
        assert problem in message, problem


def rows(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0.01, 0.3, n)
    return (rng.random(n) < risk).astype(float), risk


def matrix() -> ArmMatrix:
    return ArmMatrix.from_mapping(
        {
            "attrition": [
                {"name": "null", "family": "classifier", "algorithm": "constant", "null": True},
                {"name": "useful", "family": "classifier", "algorithm": "logistic"},
                {"name": "noise", "family": "classifier", "algorithm": "boosted"},
            ]
        }
    )


def test_a_richer_arm_wins_only_by_clearing_the_margin() -> None:
    y, risk = rows()
    rng = np.random.default_rng(1)
    scored = {
        "null": contributions("classifier", labels=y, probabilities=np.full(len(y), y.mean())),
        "useful": contributions("classifier", labels=y, probabilities=risk),
        # The truth plus noise: no better than "useful", so it doesn't clear the margin.
        "noise": contributions(
            "classifier",
            labels=y,
            probabilities=np.clip(risk + rng.normal(0, 0.01, len(y)), 0.001, 0.999),
        ),
    }
    assert scored["useful"].mean() == pytest.approx(log_loss(y, risk))
    picked = select(matrix(), "attrition", scored)
    assert picked.champion == "useful"
    first, second = picked.decisions
    assert (first.arm, first.against, first.cleared) == ("useful", "null", True)
    assert first.mean_gain > DEFAULT_Z * first.standard_error
    assert (second.arm, second.against, second.cleared) == ("noise", "useful", False)
    # An impossible margin keeps the null arm.
    assert select(matrix(), "attrition", scored, z=1e6).champion == "null"


def test_generators_and_counts_select_in_their_own_direction() -> None:
    gen = ArmMatrix.from_mapping(
        {
            "family": [
                {
                    "name": "flat",
                    "family": "generator",
                    "algorithm": "empirical_attributes",
                    "null": True,
                },
                {"name": "sharp", "family": "generator", "algorithm": "conditional_categorical"},
            ]
        }
    )
    outcomes = np.array([0, 0, 1, 0] * 50)
    flat = np.tile([0.5, 0.5], (200, 1))
    sharp = np.where(outcomes[:, None] == np.arange(2), 0.9, 0.1)
    scored = {
        "flat": contributions("generator", probabilities=flat, outcomes=outcomes),
        "sharp": contributions("generator", probabilities=sharp, outcomes=outcomes),
    }
    assert select(gen, "family", scored).champion == "sharp"
    counts = ArmMatrix.from_mapping(
        {
            "growth": [
                {"name": "poisson", "family": "count", "algorithm": "poisson_growth", "null": True},
                {"name": "nb", "family": "count", "algorithm": "negative_binomial_growth"},
            ]
        }
    )
    actual = np.array([3.0, 5.0, 2.0, 8.0] * 10)
    good = contributions("count", actual=actual, forecast=actual + 0.5)
    bad = contributions("count", actual=actual, forecast=actual * 2)
    assert good.sum() == pytest.approx(wape(actual, actual + 0.5))
    assert select(counts, "growth", {"poisson": good, "nb": bad}).champion == "poisson"


def test_selection_refuses_scores_that_don_t_line_up() -> None:
    _, risk = rows(10)
    with pytest.raises(ArmError, match="without scores: \\['noise'\\]"):
        select(matrix(), "attrition", {"null": risk, "useful": risk})
    with pytest.raises(ArmError, match="the same rows"):
        select(matrix(), "attrition", {"null": risk, "useful": risk[:5], "noise": risk})
    with pytest.raises(ArmError, match="no model family 'tree'"):
        contributions("tree")
    with pytest.raises(ArmError, match="non-zero actuals"):
        contributions("count", actual=np.zeros(3), forecast=np.ones(3))


def test_a_lift_table_ranks_the_highest_scores_first() -> None:
    y, risk = rows()
    table = lift_table(y, risk, bands=5)
    assert len(table) == 5
    assert table["rows"].sum() == len(y)
    assert table["lift"].iloc[0] > 1 > table["lift"].iloc[-1]
    assert table["event_rate"].is_monotonic_decreasing
    assert table["captured"].iloc[-1] == pytest.approx(1.0)
    assert np.isnan(lift_table(np.zeros(4), np.ones(4) / 2, bands=2)["lift"]).all()
    with pytest.raises(ArmError, match="one value each per row"):
        lift_table(y, risk[:3])
    with pytest.raises(ArmError, match="at least one band"):
        lift_table(y, risk, bands=0)
