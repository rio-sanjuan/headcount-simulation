"""Tests for the alignment-target builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.model.core.alignment import AlignmentError, Interpolation
from hcsim.model.core.alignment_builders import Aggregation, FrameSpec, from_frame, synthetic
from hcsim.model.data.population import PopulationSpec
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world

M = pd.Period


@pytest.fixture(scope="module")
def plan() -> pd.DataFrame:
    vintages = generate_world(preset("tiny", seed=2)).tables["plan_headcount"]
    return vintages[vintages["vintage"] == "2025-07"].astype(
        {"vintage": str, "cost_center": str, "job_family": object}
    )


def test_a_plan_table_sums_its_families_and_drops_its_hedges(plan: pd.DataFrame) -> None:
    spec = FrameSpec("cost_center", exclusions={"is_hedge": [True]})
    target = from_frame(plan, spec, start="2025-07", months=18, planned_depth=6)
    real = plan[~plan["is_hedge"]]
    expected = real.groupby(["cost_center", "month_end"])["target_headcount"].sum()
    m = target.marginal("cost_center")
    assert m.groups == tuple(sorted(real["cost_center"].unique()))
    g, d = m.groups.index("CC-01"), m.dates.index(M("2025-12", "M"))
    assert m.values[g, d] == pytest.approx(expected["CC-01", pd.Timestamp("2025-12-31")])
    assert target.normalization["excluded_by_is_hedge"] == int(plan["is_hedge"].sum())
    assert target.normalization["duplicates_aggregated"] == len(real) - len(expected)
    assert target.evaluation_month == M("2025-12", "M")


def test_a_wide_table_reads_its_month_headers() -> None:
    wide = pd.DataFrame(
        {"cost_center": ["CC-01", "CC-02"], "2026-01": [10.0, 5.0], "2026-03": [12.0, np.nan]}
    )
    target = from_frame(
        wide,
        FrameSpec("cost_center", layout="wide", time_format="%Y-%m"),
        start="2026-01",
        months=6,
        planned_depth=3,
    )
    m = target.marginal("cost_center")
    assert m.dates == (M("2026-01", "M"), M("2026-03", "M"))
    np.testing.assert_allclose(m.values, [[10.0, 12.0], [5.0, np.nan]])


def long(**columns: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "cost_center": ["CC-01", "CC-01", "CC-02", "CC-03", "CC-02"],
        "month_end": ["2026-02-28", "2026-02-28", "2026-02-28", "2026-02-28", "2027-06-30"],
        "target_headcount": [4.0, 6.0, -2.0, 9.0, 3.0],
    }
    data.update(columns)
    return pd.DataFrame(data)


def test_filters_anchors_clipping_and_counts() -> None:
    spec = FrameSpec(
        "cost_center",
        aggregate=Aggregation.MEAN,
        population=PopulationSpec.of(cost_center=["CC-01", "CC-02"]),
        anchors={"CC-01": 7.0},
    )
    target = from_frame(
        long(), spec, start="2026-01", months=6, planned_depth=6, interpolation=Interpolation.LINEAR
    )
    m = target.marginal("cost_center")
    assert m.groups == ("CC-01", "CC-02")
    assert target.target("cost_center", M("2026-01", "M")).tolist() == pytest.approx(
        [7.0, np.nan], nan_ok=True
    )
    assert target.target("cost_center", M("2026-02", "M")).tolist() == [5.0, 0.0]  # mean, clipped
    assert dict(target.normalization) == {
        "outside_population": 1,
        "outside_window": 1,
        "duplicates_aggregated": 1,
        "anchored": 1,
        "negatives_clipped": 1,
    }
    last = from_frame(
        long(),
        FrameSpec("cost_center", aggregate=Aggregation.LAST),
        start="2026-01",
        months=6,
        planned_depth=2,
    )
    assert last.target("cost_center", M("2026-02", "M"))[0] == 6.0


def test_a_band_column_is_carried() -> None:
    banded = long(band=[1.0, 1.0, 0.5, 2.0, 1.0])
    target = from_frame(
        banded, FrameSpec("cost_center", band="band"), start="2026-01", months=6, planned_depth=6
    )
    assert target.marginal("cost_center").band_at(M("2026-02", "M")).tolist() == [2.0, 0.5, 2.0]


@pytest.mark.parametrize(
    ("spec", "frame", "message"),
    [
        (FrameSpec("team"), long(), "no grouping column 'team'"),
        (FrameSpec("cost_center", exclusions={"is_hedge": [True]}), long(), "no column 'is_hedge'"),
        (FrameSpec("cost_center", value="plan"), long(), "a long table needs columns \\['plan'\\]"),
        (FrameSpec("cost_center"), long(month_end=["soon"] * 5), "time keys don't read as dates"),
        (
            FrameSpec("cost_center", exclusions={"cost_center": ["CC-01", "CC-02", "CC-03"]}),
            long(),
            "nothing of the table is left",
        ),
    ],
)
def test_bad_tables_are_refused(spec: FrameSpec, frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(AlignmentError, match=message):
        from_frame(frame, spec, start="2026-01", months=6, planned_depth=6)


BASE = {"CC-01": 10.0, "CC-02": 20.0}


def path(target: object) -> list[list[float]]:
    return target.path("cost_center").tolist()  # type: ignore[attr-defined]


def test_synthetic_shapes() -> None:
    assert path(synthetic("constant", "cost_center", BASE, start="2026-01", months=3)) == [
        [10.0] * 3,
        [20.0] * 3,
    ]
    line = synthetic(
        "linear",
        "cost_center",
        BASE,
        start="2026-01",
        months=5,
        final={"CC-01": 14.0, "CC-02": 20.0},
    )
    assert path(line)[0] == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert synthetic("linear", "cost_center", BASE, start="2026-01", months=1, final=3.0).path(
        "cost_center"
    ).tolist() == [[3.0], [3.0]]
    step = synthetic(
        "step", "cost_center", BASE, start="2026-01", months=4, at="2026-03", jump=-25.0
    )
    assert path(step) == [[10.0, 10.0, 0.0, 0.0], [20.0, 20.0, 0.0, 0.0]]  # clipped at zero
    dated = synthetic(
        "dates",
        "cost_center",
        BASE,
        start="2026-01",
        months=6,
        dates=["2026-03", "2026-06"],
        values={"CC-01": [11, 12], "CC-02": [21, 22]},
    )
    assert dated.target("cost_center", M("2026-04", "M")).tolist() == [11.0, 21.0]  # held
    relative = synthetic(
        "relative",
        "cost_center",
        BASE,
        start="2026-01",
        months=6,
        planned_depth=3,
        dates=["2026-06"],
        factors=[1.5],
    )
    assert relative.target("cost_center", M("2026-06", "M")).tolist() == [15.0, 30.0]
    assert relative.evaluation_month == M("2026-03", "M")


@pytest.mark.parametrize(
    ("shape", "kwargs", "message"),
    [
        ("linear", {}, "needs a final value"),
        ("step", {}, "needs the month it steps at"),
        ("dates", {}, "needs dates"),
        ("dates", {"dates": ["2026-02"], "values": {"CC-01": [1]}}, "values for every group"),
        (
            "dates",
            {"dates": ["2026-02"], "values": {"CC-01": [1, 2], "CC-02": [1, 2]}},
            "one value per date",
        ),
        ("relative", {"dates": ["2026-02"], "factors": [1.0, 2.0]}, "one factor per date"),
        ("wiggly", {}, "'wiggly' is not a valid Shape"),
    ],
)
def test_bad_synthetic_requests_are_refused(
    shape: str, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((AlignmentError, ValueError), match=message):
        synthetic(shape, "cost_center", BASE, start="2026-01", months=6, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(AlignmentError, match="at least one group"):
        synthetic("constant", "cost_center", {}, start="2026-01", months=3)
