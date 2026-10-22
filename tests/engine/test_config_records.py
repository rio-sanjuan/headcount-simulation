"""Tests for collect-all validation of every setting, and configuration records."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from hcsim.engine.config import ConfigError, DomainModels, RunMode
from hcsim.engine.settings import RECORD_VERSION, from_record, to_record
from hcsim.model.ml.persistence import save_model
from tests.engine.test_config_state_seeds import SCHEMA, awards, config, guidance, models


def _problems(**kw: Any) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        config(**kw)
    return caught.value.problems


def test_every_setting_is_checked_and_every_problem_reported() -> None:
    problems = _problems(
        seed=-1,
        seeding_max_openings=0,
        performance_cycle_month=13,
        performance_first_rating_min_tenure_months=-1,
        promotion_cycle_months=(3, 3, 14),
        pending_starts_staleness_days=-5,
        transfers_availability_factor=1.5,
        results_snapshot_steps=(0, 12, 13),
    )
    assert problems == [
        "seed must be non-negative",
        "seeding_max_openings must be at least 1",
        "performance_cycle_month must be a calendar month, 1 to 12",
        "performance_first_rating_min_tenure_months must be non-negative",
        "promotion_cycle_months must lie in 1..12, got [3, 3, 14]",
        "promotion_cycle_months repeat: [3, 3, 14]",
        "pending_starts_staleness_days must be non-negative",
        "transfers_availability_factor must lie in [0, 1]",
        "results_snapshot_steps must lie in 0..12, got [0, 12, 13]",
    ]


def test_a_setting_of_the_wrong_type_is_refused_without_hiding_the_rest() -> None:
    problems = _problems(
        simulation_months="12",
        run_date=dt.datetime(2026, 1, 1, 9),
        seeding_include_opening_buffer=1,
        backfill_source="survey",
        promotion_cycle_months=("3",),
        num_trajectories=0,
    )
    assert problems == [
        "run_date: expected date, got datetime.datetime(2026, 1, 1, 9, 0)",
        "simulation_months: expected int, got '12'",
        "seeding_include_opening_buffer: expected bool, got 1",
        "backfill_source: expected one of rate, classifier, got 'survey'",
        "promotion_cycle_months: expected a list of integers, got ('3',)",
        "num_trajectories must be at least 1",
    ]


def test_values_are_read_as_their_types() -> None:
    c = config(
        backfill_source="classifier",
        promotion_cycle_months=[3],
        transfers_availability_factor=1,
        run_mode="unguided",
    )
    assert c.promotion_cycle_months == (3,)
    assert isinstance(c.transfers_availability_factor, float)
    assert c.run_mode is RunMode.UNGUIDED


def test_the_run_mode_must_agree_with_the_guidance() -> None:
    assert config(guidance=guidance()).run_mode is RunMode.GUIDED_FORECAST
    assert config(guidance=guidance(), run_mode="guided_actuals").run_mode is RunMode.GUIDED_ACTUALS
    assert _problems(run_mode="guided_actuals") == ["run_mode guided_actuals needs guidance"]
    for mode in ("unguided", "replay"):
        assert _problems(run_mode=mode, guidance=guidance()) == [
            f"run_mode {mode} reads no guidance; drop it"
        ]


# --- records ------------------------------------------------------------------------------------


def _saved(root: Path, schema: str | None = None) -> DomainModels:
    built = models()
    names = ("attrition", "external_fill", "cancellation", "backfill", "growth", "attributes")
    versions = {}
    for name in names:
        domain = "job_family" if name == "attributes" else name
        extra = {} if schema is None else {"schema": schema}
        versions[name] = save_model(getattr(built, name), root, domain, extra).version
    return DomainModels(**{n: getattr(built, n) for n in names}, versions=versions)


def _round_trip(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(json.dumps(record))
    return result


def test_a_record_round_trips_and_re_attaches_models_by_version(tmp_path: Path) -> None:
    original = config(
        models=_saved(tmp_path, SCHEMA.fingerprint()),
        guidance=guidance(),
        seed=7,
        promotion_cycle_months=(3,),
    )
    record = _round_trip(to_record(original))
    assert record["record_version"] == RECORD_VERSION
    assert record["settings"]["run.run_date"] == "2026-01-01"
    assert record["settings"]["run.run_mode"] == "guided_forecast"
    assert record["guidance"] == original.guidance.fingerprint()  # type: ignore[union-attr]
    rebuilt = from_record(
        record, schema=SCHEMA, models_root=tmp_path, guidance=guidance(), awards=awards()
    )
    assert to_record(rebuilt) == to_record(original)
    assert rebuilt.models.versions == original.models.versions
    assert type(rebuilt.models.attrition) is type(original.models.attrition)


def test_a_configuration_with_models_built_in_code_has_no_record() -> None:
    with pytest.raises(ConfigError, match="no version to record: attrition, external_fill"):
        to_record(config())
    built = models()
    with pytest.raises(ConfigError, match=r"doesn't hold: \['weather'\]"):
        DomainModels(
            built.attrition,
            built.external_fill,
            built.cancellation,
            built.backfill,
            built.growth,
            built.attributes,
            versions={"weather": "v1-0"},
        )


def test_a_record_that_doesn_t_match_its_inputs_is_refused_with_every_reason(
    tmp_path: Path,
) -> None:
    record = _round_trip(to_record(config(models=_saved(tmp_path, "sc2-elsewhere"))))
    record["schema"] = "sc2-other"
    record["settings"]["run.simulation_months"] = "many"
    record["models"]["growth"] = "v1-000000000000"
    del record["models"]["cancellation"]
    with pytest.raises(ConfigError) as caught:
        from_record(record, schema=SCHEMA, models_root=tmp_path, guidance=guidance())
    problems = caught.value.problems
    expected = [
        "the record was written for schema sc2-other",
        "the record names guidance None, but the call passes gf2-",
        "the record needs award settings, which the call doesn't match",
        "settings: override: run.simulation_months: expected an integer",
        "the record names no version for the cancellation model",
        "growth: ",
        "attrition v1-",
    ]
    for reason in expected:
        assert any(p.startswith(reason) for p in problems), reason


def test_a_model_that_can_t_serve_its_field_is_refused(tmp_path: Path) -> None:
    saved = _saved(tmp_path)
    record = _round_trip(to_record(config(models=saved, forecast_scope="hcf", awards=None)))
    # Put the attribute generator where the attrition classifier belongs.
    attributes = saved.versions["attributes"]
    (tmp_path / "attrition" / attributes).symlink_to(tmp_path / "job_family" / attributes)
    record["models"]["attrition"] = attributes
    with pytest.raises(ConfigError, match="holds a EmpiricalAttributeGenerator, which can't serve"):
        from_record(record, schema=SCHEMA, models_root=tmp_path)
    assert from_record(
        to_record(config(models=saved, forecast_scope="hcf", awards=None)),
        schema=SCHEMA,
        models_root=tmp_path,
    ).models.versions == dict(saved.versions)


def test_an_unknown_record_version_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unsupported configuration record 9"):
        from_record({"record_version": 9}, schema=SCHEMA, models_root=tmp_path)
