"""Tests for alignment targets: marginals, interpolation, bands, the window, and depth."""

from __future__ import annotations

import json
from pathlib import Path

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


# --- the manifest, files, and digest -------------------------------------------------------------


def test_a_saved_target_loads_back_with_its_digest(tmp_path: Path) -> None:
    target = AlignmentTarget(
        plan().marginals, M("2026-01", "M"), 12, 6, normalization={"duplicates_summed": 3}
    )
    target.save(tmp_path / "plan")
    manifest = json.loads((tmp_path / "plan" / "manifest.json").read_text())
    assert manifest["digest"] == target.digest()
    assert manifest["normalization"] == {"duplicates_summed": 3}
    assert manifest["marginals"] == [
        {"grouping": "cost_center", "file": "targets_cost_center.csv", "rows": 3}
    ]
    rows = (tmp_path / "plan" / "targets_cost_center.csv").read_text().splitlines()
    assert rows[0] == "group,month,target,band"
    assert "CC-02,2026-06,40,5" in rows
    loaded = AlignmentTarget.load(tmp_path / "plan")
    assert loaded.digest() == target.digest()
    assert dict(loaded.normalization) == {"duplicates_summed": 3}
    np.testing.assert_array_equal(loaded.path("cost_center"), target.path("cost_center"))


def test_the_digest_is_the_content_in_any_group_order() -> None:
    a = marginal("cost_center", ["CC-01", "CC-02"], ["2026-03"], [[1.0], [2.0]])
    b = marginal("cost_center", ["CC-02", "CC-01"], ["2026-03"], [[2.0], [1.0]])
    one = AlignmentTarget((a,), M("2026-01", "M"), 6, 3)
    two = AlignmentTarget((b,), M("2026-01", "M"), 6, 3, normalization={"excluded": 1})
    assert one.digest() == two.digest()  # how it was built doesn't change what it is
    assert one.digest().startswith("at1-")
    other = AlignmentTarget((a,), M("2026-01", "M"), 6, 4)
    assert other.digest() != one.digest()


def test_loading_refuses_a_target_that_doesn_t_hold(tmp_path: Path) -> None:
    with pytest.raises(AlignmentError, match="no alignment target at"):
        AlignmentTarget.load(tmp_path / "none")
    plan().save(tmp_path / "plan")
    manifest_path = tmp_path / "plan" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(manifest | {"format_version": 9}))
    with pytest.raises(AlignmentError, match="unsupported format_version 9"):
        AlignmentTarget.load(tmp_path / "plan")
    manifest_path.write_text(json.dumps(manifest | {"digest": "at1-0"}))
    with pytest.raises(AlignmentError, match="doesn't match its recorded digest"):
        AlignmentTarget.load(tmp_path / "plan")
    manifest_path.write_text(json.dumps(manifest))
    csv = tmp_path / "plan" / "targets_cost_center.csv"
    lines = csv.read_text().splitlines()
    csv.write_text("\n".join([*lines, lines[1]]) + "\n")
    with pytest.raises(AlignmentError, match="repeats 1 cells"):
        AlignmentTarget.load(tmp_path / "plan")
    csv.write_text("group,month\nCC-01,2026-03\n")
    with pytest.raises(AlignmentError, match="lacks columns \\['target'\\]"):
        AlignmentTarget.load(tmp_path / "plan")


def test_a_file_without_bands_reads_as_exact(tmp_path: Path) -> None:
    exact = AlignmentTarget(
        (marginal("org_level1", ["ORG-A"], ["2026-02"], [[10.0]]),), M("2026-01", "M"), 3, 2
    )
    exact.save(tmp_path / "x")
    loaded = AlignmentTarget.load(tmp_path / "x")
    assert loaded.marginal("org_level1").bands is None
    assert loaded.marginal("org_level1").band_at(M("2026-02", "M")).tolist() == [0.0]
