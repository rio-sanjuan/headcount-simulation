"""Tests for alignment targets: marginals, interpolation, bands, the window, and depth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.model.core.alignment import AlignmentError, AlignmentTarget, Interpolation, marginal

M = pd.Period


def plan(interpolation: Interpolation = Interpolation.HOLD_LAST) -> AlignmentTarget:
    """CC-01 aims for 100 in March and 130 in June; CC-02 only for 40 in June, give or take 5."""
    cc = marginal(
        "cost_center",
        ["CC-01", "CC-02"],
        ["2026-03", "2026-06"],
        [[100.0, 130.0], [np.nan, 40.0]],
        bands=[[0.0, np.nan], [np.nan, 5.0]],
    )
    return AlignmentTarget(
        (cc,), M("2026-01", "M"), months=12, planned_depth=6, interpolation=interpolation
    )


def at(target: AlignmentTarget, month: str) -> list[float]:
    return target.target("cost_center", M(month, "M")).tolist()


def test_hold_last_keeps_the_latest_target() -> None:
    t = plan(Interpolation.HOLD_LAST)
    assert at(t, "2026-02") == pytest.approx([np.nan, np.nan], nan_ok=True)  # before any target
    assert at(t, "2026-04") == pytest.approx([100.0, np.nan], nan_ok=True)
    assert at(t, "2026-06") == [130.0, 40.0]
    assert at(t, "2026-12") == [130.0, 40.0]


def test_linear_draws_a_line_between_dates_and_holds_after() -> None:
    t = plan(Interpolation.LINEAR)
    assert at(t, "2026-04") == pytest.approx([110.0, np.nan], nan_ok=True)
    assert at(t, "2026-05") == pytest.approx([120.0, np.nan], nan_ok=True)
    assert at(t, "2026-09") == [130.0, 40.0]


def test_none_gives_targets_only_on_their_dates() -> None:
    t = plan(Interpolation.NONE)
    assert at(t, "2026-03") == pytest.approx([100.0, np.nan], nan_ok=True)
    assert at(t, "2026-04") == pytest.approx([np.nan, np.nan], nan_ok=True)
    assert at(t, "2026-09") == pytest.approx([np.nan, np.nan], nan_ok=True)
    assert np.isnan(t.path("cost_center")).sum() == 2 * 12 - 3


def test_bands_default_to_exact_on_target_dates() -> None:
    m = plan().marginal("cost_center")
    assert m.band_at(M("2026-06", "M")).tolist() == [0.0, 5.0]
    assert m.band_at(M("2026-03", "M")).tolist() == pytest.approx([0.0, np.nan], nan_ok=True)
    assert np.isnan(m.band_at(M("2026-04", "M"))).all()
    bandless = marginal("org_level1", ["ORG-A"], ["2026-02"], [[10.0]])
    assert bandless.band_at(M("2026-02", "M")).tolist() == [0.0]


def test_the_window_and_planned_depth_place_the_evaluation_month() -> None:
    t = plan()
    assert t.window == (M("2026-01", "M"), M("2026-12", "M"))
    assert t.evaluation_month == M("2026-06", "M")
    assert t.is_planned(M("2026-06", "M"))
    assert not t.is_planned(M("2026-07", "M"))
    assert not t.is_planned(M("2025-12", "M"))
    with pytest.raises(AlignmentError, match="outside the window"):
        t.target("cost_center", M("2027-01", "M"))
    with pytest.raises(AlignmentError, match="no marginal for 'org_level1'; known: cost_center"):
        t.target("org_level1", M("2026-03", "M"))


def test_bad_marginals_report_every_problem() -> None:
    with pytest.raises(AlignmentError, match="groups repeat") as error:
        marginal("cost_center", ["A", "A"], ["2026-03", "2026-02"], [[1.0, 2.0], [1.0, 2.0]])
    assert "distinct months in order" in str(error.value)
    with pytest.raises(AlignmentError, match="values are \\(1, 1\\), not \\(1, 2\\)"):
        marginal("cost_center", ["A"], ["2026-01", "2026-02"], [[1.0]])
    with pytest.raises(AlignmentError, match="finite and non-negative"):
        marginal("cost_center", ["A"], ["2026-01"], [[-1.0]])
    with pytest.raises(AlignmentError, match="bands need the values' shape"):
        marginal("cost_center", ["A"], ["2026-01"], [[1.0]], bands=[[1.0, 2.0]])
    with pytest.raises(AlignmentError, match="bands must be finite"):
        marginal("cost_center", ["A"], ["2026-01"], [[1.0]], bands=[[-1.0]])
    with pytest.raises(AlignmentError, match="a band needs a target"):
        marginal("cost_center", ["A"], ["2026-01"], [[np.nan]], bands=[[1.0]])


def test_bad_targets_report_every_problem() -> None:
    cc = marginal("cost_center", ["A"], ["2027-03"], [[1.0]])
    with pytest.raises(AlignmentError, match=r"planned depth must lie in 1\.\.12") as error:
        AlignmentTarget((cc, cc), M("2026-01", "M"), months=12, planned_depth=13)
    message = str(error.value)
    assert "repeat a grouping" in message
    assert "dates outside the window: ['2027-03']" in message
    with pytest.raises(AlignmentError, match="at least one month") as error:
        AlignmentTarget((), M("2026-01", "M"), months=0, planned_depth=1)
    assert "at least one marginal" in str(error.value)
