"""Tests for field groups and the layered loading of settings."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import pytest

from hcsim.engine.config import (
    GROUPS,
    BackfillSource,
    ConfigError,
    ForecastScope,
    RunMode,
    SimulationConfig,
    setting,
)
from hcsim.engine.settings import (
    ENV_PREFIX,
    LAYERS,
    coerce,
    load_config,
    load_settings,
    settings,
)
from tests.engine.test_config_state_seeds import SCHEMA, awards, guidance, models

RUN_DATE = {"run.run_date": "2026-01-01"}


def test_every_setting_belongs_to_a_group_and_is_documented() -> None:
    found = settings()
    assert found["run.simulation_months"].name == "simulation_months"
    assert found["promotion.cycle_months"].name == "promotion_cycle_months"
    for path, s in found.items():
        assert s.group in GROUPS
        assert path == f"{s.group}.{s.key}"
        assert s.name == (s.key if s.group == "run" else f"{s.group}_{s.key}")
        assert s.doc
        if not s.required:
            assert coerce(s.default, s.hint) == s.default  # every type has a reader
    assert [s.path for s in found.values() if s.required] == ["run.run_date"]
    # Objects aren't settings.
    objects = {f.name for f in dataclasses.fields(SimulationConfig)} - {
        s.name for s in found.values()
    }
    assert objects == {
        "schema",
        "models",
        "guidance",
        "awards",
        "eligibility",
        "gates",
        "capacity_caps",
        "plan",
        "fill_mix",
        "aligner",
        "resolved_eligibility",
        "ceiling",
    }
    with pytest.raises(ValueError, match="no field group 'weather'"):
        setting("weather", 1, "")


def test_defaults_alone_need_only_the_run_date() -> None:
    layered = load_settings(env={}, overrides=RUN_DATE)
    assert layered.values["run_date"] == dt.date(2026, 1, 1)
    assert layered.values["simulation_months"] == 12
    assert layered.sources["run_date"] == "override"
    assert set(layered.sources.values()) <= set(LAYERS)
    assert {k for k, v in layered.sources.items() if v != "default"} == {"run_date"}
    with pytest.raises(ConfigError, match=r"run\.run_date is required"):
        load_settings(env={})


def test_each_layer_overrides_the_one_before(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        "run:\n  run_date: 2026-01-01\n  simulation_months: 24\n  seed: 7\n"
        "promotion:\n  cycle_months: [3]\n"
        "transfers:\n  availability_factor: 0.5\n"
    )
    env = {
        f"{ENV_PREFIX}RUN__SIMULATION_MONTHS": "18",
        f"{ENV_PREFIX}BACKFILL__SOURCE": "classifier",
        f"{ENV_PREFIX}PROMOTION__CYCLE_MONTHS": "3, 9, 12",
        "HCSIM_DENYLIST": "/elsewhere",  # other HCSIM_ variables aren't settings
    }
    layered = load_settings(yaml_path=path, env=env, overrides={"run.seed": "11"})
    values, sources = layered.values, layered.sources
    assert (values["simulation_months"], sources["simulation_months"]) == (18, "environment")
    assert (values["seed"], sources["seed"]) == (11, "override")
    assert values["promotion_cycle_months"] == (3, 9, 12)
    assert values["backfill_source"] is BackfillSource.CLASSIFIER
    assert (values["transfers_availability_factor"], sources["run_date"]) == (0.5, "yaml")


def test_load_config_builds_a_configuration_from_layers(tmp_path: Path) -> None:
    config = load_config(
        schema=SCHEMA,
        models=models(),
        awards=awards(),
        env={f"{ENV_PREFIX}RUN__NUM_TRAJECTORIES": "3"},
        overrides=RUN_DATE | {"results.snapshot_steps": [0, 5]},
    )
    assert config.num_trajectories == 3
    assert config.results_snapshot_steps == (0, 5)
    assert config.run_mode is RunMode.UNGUIDED
    guided = load_config(
        schema=SCHEMA,
        models=models(),
        awards=awards(),
        guidance=guidance(),
        env={},
        overrides=RUN_DATE,
    )
    assert guided.run_mode is RunMode.GUIDED_FORECAST
    replay = load_config(
        schema=SCHEMA,
        models=models(),
        awards=awards(),
        env={},
        overrides=RUN_DATE | {"run.run_mode": "replay"},
    )
    assert replay.run_mode is RunMode.REPLAY
    # The configuration's own validation still runs.
    with pytest.raises(ConfigError, match="needs awards settings"):
        load_config(schema=SCHEMA, models=models(), env={}, overrides=RUN_DATE)


def test_every_problem_in_every_layer_is_reported_at_once(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "weather: {rain: 1}\npromotion: 3\nrun: {simulation_months: twelve, colour: red}\n"
    )
    env = {
        f"{ENV_PREFIX}RUN": "1",
        f"{ENV_PREFIX}SEEDING__INCLUDE_OPENING_BUFFER": "maybe",
        f"{ENV_PREFIX}GATES__STRICT": "true",
    }
    overrides = {"run.forecast_scope": "sbc-only", "run.run_date": "Jan 1", "run.seed": 1.5}
    with pytest.raises(ConfigError) as caught:
        load_settings(yaml_path=path, env=env, overrides=overrides)
    problems = caught.value.problems
    expected = [
        "yaml: no field group 'weather'",
        "yaml: group 'promotion' must be a mapping of keys",
        "yaml: run.simulation_months: expected an integer, got 'twelve'",
        "yaml: run.colour: group 'run' has no key",
        f"environment: {ENV_PREFIX}RUN must read",
        "environment: seeding.include_opening_buffer: expected true or false, got 'maybe'",
        "environment: gates.strict: group 'gates' has no key",
        "override: run.forecast_scope: expected one of hcf, hcf+sbc, got 'sbc-only'",
        "override: run.run_date: expected an ISO date, got 'Jan 1'",
        "override: run.seed: expected an integer, got 1.5",
        "run.run_date is required",
    ]
    for reason in expected:
        assert any(p.startswith(reason) for p in problems), reason
    assert len(problems) == len(expected)


def test_unreadable_or_odd_yaml_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="yaml: can't read"):
        load_settings(yaml_path=tmp_path / "missing.yaml", env={}, overrides=RUN_DATE)
    listed = tmp_path / "list.yaml"
    listed.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="must be a mapping of field groups"):
        load_settings(yaml_path=listed, env={}, overrides=RUN_DATE)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert load_settings(yaml_path=empty, env={}, overrides=RUN_DATE).values["seed"] is None


@pytest.mark.parametrize(
    ("value", "hint", "expected"),
    [
        ("none", int | None, None),
        ("", int | None, None),
        ("5", int | None, 5),
        ("YES", bool, True),
        ("0", bool, False),
        (False, bool, False),
        (3, float, 3.0),
        ("0.25", float, 0.25),
        (dt.date(2026, 1, 1), dt.date, dt.date(2026, 1, 1)),
        ("hcf", ForecastScope, ForecastScope.HEADCOUNT),
        ("", tuple[int, ...], ()),
        ((1, "2"), tuple[int, ...], (1, 2)),
        ("text", str, "text"),
    ],
)
def test_text_reads_as_each_type(value: object, hint: object, expected: object) -> None:
    assert coerce(value, hint) == expected


@pytest.mark.parametrize(
    ("value", "hint", "message"),
    [
        (True, int, "an integer"),
        ([1], int, "an integer"),
        (True, float, "a number"),
        ("x", float, "a number"),
        (None, float, "a number"),
        (5, tuple[int, ...], "a list"),
        (1, bool, "true or false"),
        (dt.datetime(2026, 1, 1, 9), dt.date, "an ISO date"),
        (5, str, "text"),
    ],
)
def test_values_that_don_t_read_are_refused(value: object, hint: object, message: str) -> None:
    with pytest.raises(ValueError, match=f"expected {message}"):
        coerce(value, hint)
