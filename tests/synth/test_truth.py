"""Tests for the `truth/` namespace: kept apart from the source tables, and true to the world."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import preset
from hcsim.synth.truth import MAX_TENURE_MONTHS, UNRATED
from hcsim.synth.world import World, generate_world, write_world


def test_the_truth_is_written_apart_from_the_source_tables(tmp_path: Path) -> None:
    world = generate_world(preset("tiny", seed=2))
    assert not any(name.startswith("truth") for name in world.tables)
    assert "regime" not in world.tables["plan_headcount"].columns
    path = tmp_path / "w.duckdb"
    write_world(world, path)
    with duckdb.connect(str(path), read_only=True) as con:
        schemas = con.execute(
            "SELECT table_schema, table_name, table_type FROM information_schema.tables"
        ).df()
        (manifest,) = con.execute("SELECT manifest FROM truth.manifest").fetchone() or ("",)
        (kind,) = con.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = 'truth' AND table_name = 'attrition_shock' "
            "AND column_name = 'month_end'"
        ).fetchone() or ("",)
    truth = schemas[schemas["table_schema"] == "truth"]
    assert set(truth["table_name"]) == {"manifest", *world.truth_tables}
    assert set(schemas.loc[schemas["table_schema"] == "main", "table_name"]) == {
        *world.table_names,
        "req_snapshot",
    }
    assert json.loads(manifest)["config"]["seed"] == 2
    assert kind == "DATE"


def test_the_hazard_tables_compose_into_the_process_s_hazard(small_world: World) -> None:
    """The surface, the season, the cost center's effect, and the shock multiply back into
    the hazard the process draws departures from."""
    config, truth = small_world.config, small_world.truth_tables
    rng = np.random.default_rng(0)
    n = 2_000
    scale = list(config.ratings.scale)
    level = rng.choice(np.array(config.levels), n)
    tenure = rng.integers(0, MAX_TENURE_MONTHS + 1, n)
    code = rng.integers(-1, len(scale), n)
    month = int(rng.integers(1, 13))
    effects = truth["cost_center_effect"]
    attrition = effects[effects["process"] == "attrition"].set_index("cost_center")["effect"]
    effect = attrition.reindex(rng.choice(attrition.index.to_numpy(), n)).to_numpy()
    factors = np.r_[1.0, truth["attrition_shock"]["factor"].unique()]
    shock = rng.choice(factors, n).astype(np.float64)

    expected = config.attrition.hazards(
        level, tenure.astype(np.float64), month, effect, code, shock
    )
    surface = truth["attrition_hazard"].set_index(["job_level", "tenure_month", "rating"])
    rating = np.array([*scale, UNRATED])[code]  # -1 picks the last label, the unrated
    keys = pd.MultiIndex.from_arrays([level, tenure, rating])
    base = surface["monthly_hazard"].reindex(keys).to_numpy()
    season = float(truth["attrition_season"].set_index("month_of_year")["factor"].loc[month])
    got = np.minimum(1.0, base * season * effect * shock)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_the_season_averages_to_one_and_peaks_when_configured(small_world: World) -> None:
    season = small_world.truth_tables["attrition_season"].set_index("month_of_year")["factor"]
    assert list(season.index) == list(range(1, 13))
    assert season.mean() == pytest.approx(1.0)
    assert season.idxmax() == small_world.config.attrition.seasonal_peak_month


def test_the_effects_and_channel_hazards_are_the_manifest_s(small_world: World) -> None:
    effects = small_world.truth_tables["cost_center_effect"]
    for process, scale in (("attrition", "multiplier"), ("backfill", "logit")):
        rows = effects[effects["process"] == process]
        recorded = small_world.truth["processes"][process]["cost_center_effect"]
        assert dict(zip(rows["cost_center"], rows["effect"], strict=True)) == recorded
        assert set(rows["scale"]) == {scale}
    hazards = small_world.truth_tables["requisition_hazard"]
    config = small_world.config
    for channel, table in (
        ("external", config.requisitions.external_fill_hazard),
        ("transfer", config.requisitions.transfer_fill_hazard),
        ("conversion", config.movements.conversion_fill_hazard),
        ("unfilled", config.requisitions.unfilled_exit_hazard),
    ):
        rows = hazards[hazards["channel"] == channel]
        assert tuple(rows["monthly_hazard"]) == table
        assert list(rows["age_month"]) == list(range(len(table)))


def test_every_real_cost_center_has_one_regime_per_vintage(small_world: World) -> None:
    regimes = small_world.truth_tables["plan_regime"]
    plan = small_world.tables["plan_headcount"]
    cells = plan.loc[~plan["is_hedge"], ["vintage", "cost_center"]].drop_duplicates()
    assert len(regimes) == len(cells)
    assert not regimes.duplicated(["vintage", "cost_center"]).any()
    assert set(regimes["regime"]) == {"planned", "flat", "decaying"}
