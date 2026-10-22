"""Tests for disaggregation: the five strategies, the fallbacks, and the chunked basis."""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.guidance.config import GuidanceConfig, GuidanceConfigError
from hcsim.guidance.disaggregation import STRATEGIES, Basis, shares, split
from hcsim.guidance.ingest import IngestionError
from hcsim.guidance.pipeline import run
from hcsim.model.core.categories import GuidanceCategory
from hcsim.synth.config import preset
from hcsim.synth.world import generate_world, write_world

C = GuidanceCategory
SNAPSHOT = pd.Period("2025-02", "M")


def basis_of(rows: list[tuple[str, str, str, float]]) -> Basis:
    """A hand-made basis for group G: (month, leaf, category, value) rows."""
    frame = pd.DataFrame(rows, columns=["month", "leaf", "category", "value"]).assign(group="G")
    frame["month"] = frame["month"].map(lambda m: pd.Period(m, "M"))
    return Basis({"G": tuple(sorted(frame["leaf"].unique()))}, frame, 1)


HISTORY = basis_of(
    [
        ("2025-01", "A", "attrition", 1.0),
        ("2025-01", "B", "attrition", 1.0),
        ("2025-02", "A", "attrition", 0.0),
        ("2025-02", "B", "attrition", 98.0),
        ("2025-02", "A", "headcount", 30.0),
        ("2025-02", "B", "headcount", 10.0),
    ]
)
KEY = [("G", C.ATTRITION)]


@pytest.mark.parametrize(
    ("strategy", "a"),
    [
        ("average_proportions", 0.25),  # (1/2 + 0/98) / 2
        ("proportions_of_averages", 0.01),  # 1 / 100
        ("last_proportions", 0.0),  # 0 / 98
        ("exposure", 0.75),  # 30 / 40 heads
        ("equal", 0.5),
    ],
)
def test_each_strategy_splits_by_its_own_shares(strategy: str, a: float) -> None:
    found = shares(HISTORY, KEY, strategy, SNAPSHOT)  # type: ignore[arg-type]
    share = found.by_key["G", C.ATTRITION]
    assert share["A"] == pytest.approx(a)
    assert share.sum() == pytest.approx(1.0)
    assert found.fallbacks == {}


def test_a_group_with_no_basis_falls_back_to_exposure_then_equal() -> None:
    quiet = basis_of(
        [
            ("2025-02", "A", "attrition", 0.0),
            ("2025-02", "B", "attrition", 0.0),
            ("2025-02", "A", "headcount", 3.0),
            ("2025-02", "B", "headcount", 1.0),
        ]
    )
    found = shares(quiet, KEY, "proportions_of_averages", SNAPSHOT)
    assert found.by_key["G", C.ATTRITION]["A"] == pytest.approx(0.75)
    assert found.fallbacks == {"exposure": 1}
    empty = basis_of([("2025-02", "A", "attrition", 0.0), ("2025-02", "B", "attrition", 0.0)])
    found = shares(empty, KEY, "proportions_of_averages", SNAPSHOT)
    assert found.by_key["G", C.ATTRITION].tolist() == [0.5, 0.5]
    assert found.fallbacks == {"equal": 1}


def test_a_split_adds_up_to_its_group_and_scales_the_paths() -> None:
    found = shares(HISTORY, KEY, "exposure", SNAPSHOT)
    values = {("G", C.ATTRITION): np.array([8.0, 4.0])}
    paths = {("G", C.ATTRITION): np.array([[8.0, 4.0], [10.0, 2.0]])}
    leaves, leaf_paths, uncovered = split(values, paths, found)
    assert leaves["A", C.ATTRITION].tolist() == [6.0, 3.0]
    assert np.allclose(leaves["A", C.ATTRITION] + leaves["B", C.ATTRITION], [8.0, 4.0])
    assert leaf_paths["B", C.ATTRITION].tolist() == [[2.0, 1.0], [2.5, 0.5]]
    assert uncovered == []
    _, _, missing = split({("H", C.ATTRITION): np.ones(2)}, {}, found)
    assert missing == [("H", C.ATTRITION)]


# --- in a run ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    path = tmp_path_factory.mktemp("world") / "world.duckdb"
    write_world(generate_world(preset("tiny", seed=4)), path)
    with duckdb.connect(str(path), read_only=True) as connection:
        yield connection


def by_org(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> pd.Series:
    org = con.execute("SELECT cost_center, org_level1 FROM dim_org").df()
    rolled = frame.merge(org, on="cost_center")
    return rolled.groupby(["org_level1", "category", "forecast_month"])["value"].sum()


@pytest.fixture(scope="module")
def coarse(con: duckdb.DuckDBPyConnection) -> pd.Series:
    frame = run(con, GuidanceConfig(horizon=6, grouping="org_level1")).artifact.frame
    return frame.set_index(["org_level1", "category", "forecast_month"])["value"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_run_splits_onto_cost_centers_that_add_up_to_their_org(
    con: duckdb.DuckDBPyConnection, coarse: pd.Series, strategy: str
) -> None:
    config = GuidanceConfig.from_mapping(
        {
            "horizon": 6,
            "grouping": "org_level1",
            "disaggregation": {"strategy": strategy, "leaf": "cost_center"},
        }
    )
    result = run(con, config)
    assert result.artifact.grouping_column == "cost_center"
    assert result.report["grain"] == "cost_center"
    rolled = by_org(con, result.artifact.frame)
    pd.testing.assert_series_equal(
        rolled.sort_index(), coarse.sort_index(), check_names=False, atol=1e-9
    )


def test_the_basis_is_counted_in_chunks_with_the_same_result(
    con: duckdb.DuckDBPyConnection,
) -> None:
    def config(chunk: int) -> GuidanceConfig:
        return GuidanceConfig.from_mapping(
            {
                "horizon": 3,
                "grouping": "org_level1",
                "disaggregation": {"strategy": "average_proportions", "chunk_size": chunk},
            }
        )

    whole, chunked = run(con, config(500)), run(con, config(1))
    assert chunked.report["disaggregation"]["chunks"] > whole.report["disaggregation"]["chunks"]
    assert chunked.artifact.fingerprint() == whole.artifact.fingerprint()


def test_a_leaf_grain_that_doesn_t_nest_is_refused(con: duckdb.DuckDBPyConnection) -> None:
    from hcsim.guidance.disaggregation import basis

    with pytest.raises(IngestionError, match="job_family doesn't nest in org_level1"):
        basis(
            con,
            "org_level1",
            "job_family",
            ["ORG-A", "ORG-B"],
            [C.ATTRITION],
            pd.Period("2025-12", "M"),
            12,
        )


def test_the_disaggregation_settings_are_checked_at_load() -> None:
    with pytest.raises(GuidanceConfigError) as error:
        GuidanceConfig.from_mapping(
            {"grouping": "cost_center", "disaggregation": {"strategy": "equal"}}
        )
    assert "'cost_center' is no finer than the grouping 'cost_center'" in str(error.value)
    with pytest.raises(GuidanceConfigError, match="isn't in the column catalog"):
        GuidanceConfig.from_mapping(
            {"grouping": "TOPLINE", "disaggregation": {"strategy": "equal", "leaf": "team"}}
        )
    with pytest.raises(GuidanceConfigError, match=r"disaggregation\.strategy"):
        GuidanceConfig.from_mapping({"disaggregation": {"strategy": "median"}})
