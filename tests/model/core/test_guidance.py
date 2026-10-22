"""Tests for the guidance forecast artifact."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.model.core.categories import GuidanceCategory, UnknownCategoryError
from hcsim.model.core.guidance import GuidanceArtifactError, GuidanceForecast
from hcsim.model.core.schema import COST_CENTER, SchemaError, SimulationSchema
from hcsim.model.data.population import REGULAR, population
from hcsim.model.interface import ProbabilisticOutput

ATTRITION = GuidanceCategory.ATTRITION
HIRES = GuidanceCategory.HIRES


def frame(**overrides: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "forecast_month": [0, 0, 1, 0, 0],
        "category": ["attrition", "attrition", "attrition", "hires", "attrition"],
        "value": [2.0, 1.5, 3.0, 4.0, 0.5],
        "cost_center": ["CC-02", "CC-01", "CC-02", "CC-01", "CC-02"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def artifact(**overrides: object) -> GuidanceForecast:
    return GuidanceForecast(frame(**overrides), COST_CENTER, "2025-12-31", horizon=12)


def test_groups_are_sorted_and_quotas_follow_group_order() -> None:
    g = artifact()
    assert g.groups == ("CC-01", "CC-02")
    assert g.get_quota(0, ATTRITION).tolist() == [1.5, 2.5]  # CC-02's duplicate rows sum
    assert g.get_quota(1, ATTRITION).tolist() == [0.0, 3.0]  # absent cells read 0
    assert g.get_quota(0, HIRES).tolist() == [4.0, 0.0]


def test_coverage_tells_an_uncarried_group_from_a_zero() -> None:
    g = artifact()
    assert g.covers(HIRES)
    assert not g.covers(GuidanceCategory.REQUISITIONS_OPENED)
    assert g.group_coverage(HIRES).tolist() == [True, False]


def test_a_date_typed_month_is_refused() -> None:
    with pytest.raises(GuidanceArtifactError, match="integer step offsets"):
        artifact(forecast_month=pd.to_datetime(["2026-01-31"] * 5))


def test_a_measure_name_instead_of_a_category_is_refused() -> None:
    with pytest.raises(UnknownCategoryError, match="'terms'"):
        artifact(category=["terms"] * 5)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"value": [1.0, -0.5, 1.0, 1.0, 1.0]}, "negative cells"),
        ({"value": [1.0, float("nan"), 1.0, 1.0, 1.0]}, "finite"),
        ({"forecast_month": [0, 0, 12, 0, 0]}, "0..11"),
    ],
)
def test_invalid_values_and_steps_are_refused(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(GuidanceArtifactError, match=match):
        artifact(**overrides)


def test_missing_columns_are_named() -> None:
    with pytest.raises(GuidanceArtifactError, match="missing columns"):
        GuidanceForecast(frame().drop(columns="value"), COST_CENTER, "2025-12-31", 12)


def test_the_fingerprint_ignores_row_order_but_not_values() -> None:
    g = artifact()
    shuffled = GuidanceForecast(frame().iloc[::-1], COST_CENTER, "2025-12-31", 12)
    assert shuffled.fingerprint() == g.fingerprint()
    assert g.fingerprint().startswith("gf2-")
    assert artifact(value=[2.0, 1.5, 3.0, 4.0, 0.25]).fingerprint() != g.fingerprint()


def test_a_recorded_population_changes_the_fingerprint() -> None:
    stamped = GuidanceForecast(frame(), COST_CENTER, "2025-12-31", 12, population="regular")
    assert stamped.fingerprint() != artifact().fingerprint()


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    g = artifact()
    g.save(tmp_path / "g")
    loaded = GuidanceForecast.load(tmp_path / "g")
    assert loaded.fingerprint() == g.fingerprint()
    assert loaded.get_quota(0, ATTRITION).tolist() == [1.5, 2.5]


def test_a_tampered_frame_fails_its_fingerprint(tmp_path: Path) -> None:
    artifact().save(tmp_path / "g")
    artifact(value=[9.0] * 5).save(tmp_path / "other")
    (tmp_path / "other" / "forecast.parquet").replace(tmp_path / "g" / "forecast.parquet")
    with pytest.raises(GuidanceArtifactError, match="fingerprint"):
        GuidanceForecast.load(tmp_path / "g")


def test_the_aligner_validates_groups_against_the_schema() -> None:
    schema = SimulationSchema.build({COST_CENTER: ["CC-01", "CC-02", "CC-03"]})
    assert artifact().aligner(schema).n_groups == 2
    with pytest.raises(SchemaError, match="not in the 'cost_center' vocabulary"):
        artifact().aligner(SimulationSchema.build({COST_CENTER: ["CC-01"]}))


# --- probabilistic outputs ---------------------------------------------------------------------


def probabilistic(**overrides: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "forecast_month": [0, 0, 1],
        "category": ["attrition", "attrition", "attrition"],
        "value": [2.0, 1.0, 3.0],
        "cost_center": ["CC-01", "CC-02", "CC-01"],
        "q90": [4.0, 2.0, 5.0],
        "q10": [1.0, 0.0, 1.0],
        "q50": [2.0, 1.0, 3.0],
        "distribution": ["negative_binomial", "poisson", "normal"],
        "mean": [2.0, 1.0, 3.0],
        "dispersion": [5.0, np.nan, 1.5],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_quantiles_and_distributions_are_carried_and_read() -> None:
    g = GuidanceForecast(probabilistic(), COST_CENTER, "2025-12-31", 2)
    assert g.quantile_levels == (10, 50, 90)  # ordered by level, whatever the column order
    assert g.get_quantile(0, ATTRITION, 90).tolist() == [4.0, 2.0]
    assert g.get_quantile(1, ATTRITION, 10).tolist() == [1.0, 0.0]
    dist = g.get_distribution(0, ATTRITION)
    assert dist.loc["CC-01", "distribution"] == "negative_binomial"
    assert np.isnan(dist["dispersion"].to_numpy()[1])
    assert g.outputs == {ProbabilisticOutput.QUANTILES, ProbabilisticOutput.PARAMETRIC}
    assert g.metadata()["quantiles"] == [10, 50, 90]
    with pytest.raises(GuidanceArtifactError, match="no quantile 25; carried: \\[10, 50, 90\\]"):
        g.get_quantile(0, ATTRITION, 25)


def test_point_only_artifacts_declare_none() -> None:
    g = artifact()
    assert g.outputs == {ProbabilisticOutput.NONE}
    assert g.metadata()["outputs"] == ["none"]
    with pytest.raises(GuidanceArtifactError, match="no distributions"):
        g.get_distribution(0, ATTRITION)
    with pytest.raises(GuidanceArtifactError, match="no sample paths"):
        g.get_paths("CC-01", ATTRITION)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"q10": [5.0, 0.0, 1.0]}, "1 rows have quantiles that fall"),
        ({"q10": [-1.0, 0.0, 1.0]}, "1 negative quantiles, such as attrition at step 0 in CC-01"),
        ({"q50": [np.nan, 1.0, 3.0]}, "quantiles must be finite"),
        ({"distribution": ["gamma", "poisson", "normal"]}, "unknown distributions \\['gamma'\\]"),
        ({"mean": [-1.0, 1.0, 3.0]}, "means must be finite and non-negative"),
        ({"dispersion": [np.nan, np.nan, 1.5]}, "need a dispersion > 0"),
        ({"dispersion": [5.0, 2.0, 1.5]}, "poisson rows take no dispersion"),
    ],
)
def test_bad_probabilistic_columns_are_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(GuidanceArtifactError, match=message):
        GuidanceForecast(probabilistic(**overrides), COST_CENTER, "2025-12-31", 2)


def test_distribution_columns_come_together() -> None:
    with pytest.raises(GuidanceArtifactError, match="need a 'distribution' column"):
        GuidanceForecast(probabilistic().drop(columns="distribution"), COST_CENTER, "2025-12-31", 2)
    with pytest.raises(GuidanceArtifactError, match="needs a 'mean' column"):
        GuidanceForecast(probabilistic().drop(columns="mean"), COST_CENTER, "2025-12-31", 2)
    poisson = probabilistic(distribution=["poisson"] * 3).drop(columns="dispersion")
    g = GuidanceForecast(poisson, COST_CENTER, "2025-12-31", 2)
    assert np.isnan(g.get_distribution(0, ATTRITION)["dispersion"]).all()


def paths() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.poisson(3.0, size=(2, 50, 12)).astype(np.float64)


def test_sample_paths_round_trip_with_their_index(tmp_path: Path) -> None:
    index = (("CC-01", "attrition"), ("CC-02", "hires"))
    g = GuidanceForecast(
        frame(), COST_CENTER, "2025-12-31", 12, sample_paths=paths(), path_index=index
    )
    assert g.get_paths("CC-02", HIRES).shape == (50, 12)
    assert ProbabilisticOutput.SAMPLE_PATHS in g.outputs
    with pytest.raises(GuidanceArtifactError, match="no sample paths for 'CC-02', attrition"):
        g.get_paths("CC-02", ATTRITION)
    g.save(tmp_path / "a")
    loaded = GuidanceForecast.load(tmp_path / "a")
    np.testing.assert_array_equal(loaded.sample_paths, g.sample_paths)
    assert loaded.path_index == index
    assert loaded.fingerprint() == g.fingerprint()
    other = paths()
    other[0, 0, 0] += 1
    changed = GuidanceForecast(
        frame(), COST_CENTER, "2025-12-31", 12, sample_paths=other, path_index=index
    )
    assert changed.fingerprint() != g.fingerprint()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_paths": np.zeros((2, 5, 3))}, "must be \\(series, paths, 12\\)"),
        ({"sample_paths": np.zeros((1, 5, 12))}, "one distinct \\(group, category\\) per series"),
        ({"sample_paths": -np.ones((2, 5, 12))}, "negative path values"),
        ({"sample_paths": np.full((2, 5, 12), np.inf)}, "must be finite"),
        ({"sample_paths": None}, "a path index needs sample paths"),
    ],
)
def test_bad_sample_paths_are_refused(kwargs: dict[str, object], message: str) -> None:
    index = (("CC-01", "attrition"), ("CC-02", "hires"))
    with pytest.raises(GuidanceArtifactError, match=message):
        GuidanceForecast(frame(), COST_CENTER, "2025-12-31", 12, path_index=index, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(UnknownCategoryError):
        GuidanceForecast(
            frame(),
            COST_CENTER,
            "2025-12-31",
            12,
            sample_paths=np.zeros((1, 5, 12)),
            path_index=(("CC-01", "terms"),),
        )


# --- stamps, coherence, and combining --------------------------------------------------------


def test_populations_are_recorded_as_stamps() -> None:
    named = GuidanceForecast(frame(), COST_CENTER, "2025-12-31", 12, population="regular")
    assert named.population == str(REGULAR.stamp)
    assert named.population_stamp == REGULAR.stamp
    assert artifact().population_stamp is None
    explicit = GuidanceForecast(
        frame(), COST_CENTER, "2025-12-31", 12, population=str(REGULAR.stamp)
    )
    assert explicit.population == named.population
    named.check_population(REGULAR)
    with pytest.raises(GuidanceArtifactError, match="built for population regular@"):
        named.check_population(population("every_account"))
    for bad in ("staff", "regular@"):
        with pytest.raises(GuidanceArtifactError):
            GuidanceForecast(frame(), COST_CENTER, "2025-12-31", 12, population=bad)


def test_the_alignment_stamp_is_recorded_and_checked(tmp_path: Path) -> None:
    aligned = GuidanceForecast(frame(), COST_CENTER, "2025-12-31", 12, aligned_to="at1-0123")
    assert aligned.metadata()["aligned_to"] == "at1-0123"
    assert aligned.fingerprint() != artifact().fingerprint()
    aligned.save(tmp_path / "a")
    assert GuidanceForecast.load(tmp_path / "a").aligned_to == "at1-0123"
    for bad in ("", "a b"):
        with pytest.raises(GuidanceArtifactError, match="aligned_to must be a digest"):
            GuidanceForecast(frame(), COST_CENTER, "2025-12-31", 12, aligned_to=bad)


def tree(hires: list[float]) -> pd.DataFrame:
    """Hires and both its parts, for CC-01 and CC-02 at step 0."""
    return pd.DataFrame(
        {
            "forecast_month": [0] * 6,
            "category": [
                "hires",
                "hires",
                "external_hires",
                "external_hires",
                "conversion_hires",
                "conversion_hires",
            ],
            "value": [*hires, 3.0, 1.0, 1.0, 1.0],
            "cost_center": ["CC-01", "CC-02"] * 3,
        }
    )


def test_a_parent_must_be_the_sum_of_its_carried_parts() -> None:
    GuidanceForecast(tree([4.0, 2.0]), COST_CENTER, "2025-12-31", 12)  # coherent
    with pytest.raises(GuidanceArtifactError, match="2 cells where a fill-tree parent") as error:
        GuidanceForecast(tree([9.0, 2.5]), COST_CENTER, "2025-12-31", 12)
    message = str(error.value)
    assert message.index("CC-01: 9 against 4") < message.index("CC-02: 2.5 against 2")
    alone = tree([9.0, 2.5])
    alone = alone[alone["category"] != "conversion_hires"]  # a part missing: nothing to check
    GuidanceForecast(alone, COST_CENTER, "2025-12-31", 12)


def test_artifacts_covering_different_series_combine() -> None:
    hires = artifact(category=["hires"] * 5)
    attrition = artifact()
    combined = GuidanceForecast.combine(
        [
            GuidanceForecast(
                attrition.frame[attrition.frame["category"] == "attrition"],
                COST_CENTER,
                "2025-12-31",
                12,
            ),
            hires,
        ]
    )
    assert combined.categories == {ATTRITION, HIRES}
    assert len(combined.frame) == 4 + 5
    with_paths = GuidanceForecast(
        frame(),
        COST_CENTER,
        "2025-12-31",
        12,
        sample_paths=np.zeros((1, 3, 12)),
        path_index=(("CC-01", "attrition"),),
    )
    other = GuidanceForecast(
        frame(category=["promotions"] * 5),
        COST_CENTER,
        "2025-12-31",
        12,
        sample_paths=np.ones((1, 3, 12)),
        path_index=(("CC-01", "promotions"),),
    )
    joined = GuidanceForecast.combine([with_paths, other])
    assert joined.path_index == (("CC-01", "attrition"), ("CC-01", "promotions"))


def test_combining_refuses_artifacts_that_don_t_fit() -> None:
    base = artifact()
    with pytest.raises(GuidanceArtifactError, match="nothing to combine"):
        GuidanceForecast.combine([])
    with pytest.raises(GuidanceArtifactError, match="series carried twice"):
        GuidanceForecast.combine([base, base])
    later = GuidanceForecast(frame(category=["hires"] * 5), COST_CENTER, "2026-01-31", 12)
    with pytest.raises(GuidanceArtifactError, match="differs in grouping, snapshot"):
        GuidanceForecast.combine([base, later])
    quantiled = GuidanceForecast(
        frame(category=["hires"] * 5, q50=[1.0] * 5), COST_CENTER, "2025-12-31", 12
    )
    with pytest.raises(GuidanceArtifactError, match="carries other columns"):
        GuidanceForecast.combine([base, quantiled])
    pathed = GuidanceForecast(
        frame(category=["hires"] * 5),
        COST_CENTER,
        "2025-12-31",
        12,
        sample_paths=np.zeros((1, 3, 12)),
        path_index=(("CC-01", "hires"),),
    )
    with pytest.raises(GuidanceArtifactError, match="differs in carrying sample paths"):
        GuidanceForecast.combine([base, pathed])
    fewer = GuidanceForecast(
        frame(category=["promotions"] * 5),
        COST_CENTER,
        "2025-12-31",
        12,
        sample_paths=np.zeros((1, 2, 12)),
        path_index=(("CC-01", "promotions"),),
    )
    with pytest.raises(GuidanceArtifactError, match="sample paths differ in count"):
        GuidanceForecast.combine([pathed, fewer])


def test_negative_cells_are_named_most_negative_first() -> None:
    with pytest.raises(GuidanceArtifactError, match="2 negative cells, such as") as error:
        artifact(value=[2.0, -0.5, 3.0, -4.0, 0.5])
    message = str(error.value)
    assert message.index("hires at step 0 in CC-01 (-4)") < message.index(
        "attrition at step 0 in CC-01 (-0.5)"
    )
    assert "a negative cell means a defect upstream" in message


def test_the_loader_refuses_a_negative_cell_written_behind_its_back(tmp_path: Path) -> None:
    """Writers clamp, so a file with a negative cell was changed after it was written. The
    loader refuses it before the fingerprint is even checked."""
    artifact().save(tmp_path / "a")
    parquet = str(tmp_path / "a" / "forecast.parquet")
    tampered = duckdb.read_parquet(parquet).df()
    tampered.loc[0, "value"] = -3.0
    duckdb.from_df(tampered).write_parquet(parquet)
    with pytest.raises(
        GuidanceArtifactError, match="1 negative cells, such as attrition at step 0"
    ):
        GuidanceForecast.load(tmp_path / "a")
    paths = GuidanceForecast(
        frame(),
        COST_CENTER,
        "2025-12-31",
        12,
        sample_paths=np.zeros((1, 2, 12)),
        path_index=(("CC-01", "attrition"),),
    )
    paths.save(tmp_path / "p")
    np.save(tmp_path / "p" / "paths.npy", -np.ones((1, 2, 12)))
    with pytest.raises(GuidanceArtifactError, match="24 negative path values"):
        GuidanceForecast.load(tmp_path / "p")
