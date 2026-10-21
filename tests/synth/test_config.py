"""Tests for the world configuration and its presets."""

from __future__ import annotations

import dataclasses

import pytest

from hcsim.synth.config import (
    PRESETS,
    AttritionTruth,
    CompensationTruth,
    RequisitionTruth,
    WorldConfig,
    preset,
)
from hcsim.synth.world import generate_world


def test_every_preset_builds_and_scales_up() -> None:
    worlds = [preset(name) for name in ("tiny", "small", "medium", "large")]
    assert list(PRESETS) == ["tiny", "small", "medium", "large"]
    assert [w.initial_employees for w in worlds] == [200, 5_000, 50_000, 500_000]
    for w in worlds:
        assert w.initial_employees / w.cost_centers == 20
        assert w.preset in PRESETS
    assert [len(w.locations) for w in worlds] == sorted(len(w.locations) for w in worlds)


def test_the_tiny_preset_is_the_default_configuration() -> None:
    assert preset("tiny", seed=3) == WorldConfig(seed=3)


def test_an_unknown_preset_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="unknown preset 'huge'; known: tiny, small"):
        preset("huge")


def test_every_parameter_lives_in_the_configuration() -> None:
    record = WorldConfig().to_dict()
    assert {"attrition", "requisitions", "compensation", "prices", "grants"} <= set(record)
    assert record["attrition"]["base_monthly_hazard"] == AttritionTruth().base_monthly_hazard
    assert record["requisitions"]["backfill_probability"] == 0.6


def test_the_manifest_records_the_whole_configuration() -> None:
    config = WorldConfig(attrition=AttritionTruth(base_monthly_hazard=0.0))
    world = generate_world(config)
    assert world.truth["config"]["attrition"]["base_monthly_hazard"] == 0.0
    # The parameter reaches the process: with no hazard, nobody leaves.
    assert not (world.tables["workforce_txn"]["trans_type"] == "TERM").any()


def test_a_bad_configuration_lists_every_problem() -> None:
    with pytest.raises(ValueError, match="invalid WorldConfig") as error:
        WorldConfig(
            levels=(1, 2, 3, 4, 5, 6),
            level_mix=(0.2,) * 5 + (0.0,),
            families=("Engineering", "Astronomy"),
            locations=(),
            requisitions=RequisitionTruth(growth_level_mix={9: 1.0}),
            compensation=dataclasses.replace(CompensationTruth(), family_factor={}),
        )
    message = str(error.value)
    for expected in (
        "families without a tech indicator: ['Astronomy']",
        "at least one location",
        "attrition.level_factor has no value for levels [6]",
        "compensation.base_midpoint_by_level has no value for levels [6]",
        "grants.new_hire_value_by_level has no value for levels [6]",
        "compensation.family_factor has no value for ['Astronomy', 'Engineering']",
        "growth_level_mix names unknown levels [9]",
    ):
        assert expected in message


@pytest.mark.slow
def test_the_small_preset_generates_at_its_scale() -> None:
    world = generate_world(preset("small", seed=1))
    assert len(world.tables["dim_org"]) == 250
    assert world.tables["dim_org"]["org_level1"].nunique() == 5
    txn = world.tables["workforce_txn"]
    first = txn[(txn["trans_type"] == "ACTIVE") & (txn["event_date"] == txn["event_date"].min())]
    assert len(first) == 5_000
    assert len(world.tables["dim_location"]) == 4
    assert len(world.tables["comp_range"]) == 4 * 5 * 4  # families x levels x locations
