"""Tests for the guidance forecast artifact."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hcsim.model.core.categories import GuidanceCategory, UnknownCategoryError
from hcsim.model.core.guidance import GuidanceArtifactError, GuidanceForecast
from hcsim.model.core.schema import COST_CENTER, SchemaError, SimulationSchema

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
    assert g.fingerprint().startswith("gf1-")
    assert artifact(value=[2.0, 1.5, 3.0, 4.0, 0.25]).fingerprint() != g.fingerprint()


def test_a_recorded_population_changes_the_fingerprint() -> None:
    stamped = GuidanceForecast(frame(), COST_CENTER, "2025-12-31", 12, population="tiny")
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
