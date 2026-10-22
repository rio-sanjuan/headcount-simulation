"""Tests for the world's compensation ranges."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import duckdb
import pytest

from hcsim.engine.seeding import range_table_from_world, schema_from_world
from hcsim.financial.ranges import RangeError
from hcsim.synth.config import CompensationTruth, WorldConfig, preset
from hcsim.synth.world import World, generate_world, write_world


def test_ranges_cover_every_family_level_and_location(small_world: World) -> None:
    config = small_world.config
    ranges = small_world.tables["comp_range"]
    assert len(ranges) == len(config.families) * len(config.levels) * len(config.locations)
    assert not ranges.duplicated(["job_family", "job_level", "location"]).any()
    executive = small_world.tables["dim_job"].query("is_executive")["job_level"].unique()
    assert len(executive) > 0
    assert set(executive) <= set(ranges["job_level"])
    assert (ranges["min_base"] <= ranges["max_base"]).all()
    assert (ranges["min_target"] <= ranges["max_target"]).all()


def test_pay_differs_by_location_and_the_first_pays_one(small_world: World) -> None:
    ranges = small_world.tables["comp_range"]
    cell = ranges[(ranges["job_family"] == "Engineering") & (ranges["job_level"] == 3)]
    by_location = cell.set_index("location")["max_base"]
    assert by_location.nunique() > 1
    truth = CompensationTruth()
    assert truth.location_factor(0) == 1.0
    expected = truth.base_midpoint_by_level[3] * truth.family_factor["Engineering"] * 1.2
    assert by_location["LOC-1"] == round(expected)


def test_the_engine_refuses_ranges_with_missing_cells(tmp_path: Path) -> None:
    """With the quirk on, cells go missing, and seeding the engine's range table fails by name
    rather than imputing them."""
    config = preset("small", seed=2)
    quirk = dataclasses.replace(config.compensation, missing_range_share=0.1)
    world = generate_world(dataclasses.replace(config, compensation=quirk))
    full = len(config.families) * len(config.levels) * len(config.locations)
    assert len(world.tables["comp_range"]) < full
    path = tmp_path / "w.duckdb"
    write_world(world, path)
    with (
        duckdb.connect(str(path), read_only=True) as con,
        pytest.raises(RangeError, match="missing compensation ranges, such as"),
    ):
        range_table_from_world(con, schema_from_world(con))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("missing_range_share", 1.0, "missing_range_share"),
        ("location_pay_spread", -0.1, "location_pay_spread"),
    ],
)
def test_bad_range_settings_are_refused(field: str, value: float, message: str) -> None:
    compensation = dataclasses.replace(CompensationTruth(), **{field: value})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        WorldConfig(compensation=compensation)
