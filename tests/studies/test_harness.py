"""Tests for the study profile, the skeleton's inputs, the one-arm harness, and `invariants`."""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.config import ForecastScope
from hcsim.engine.results import QuotaRecord
from hcsim.engine.seeding import schema_from_world
from hcsim.financial.expense import ExpenseResult
from hcsim.studies.evaluators import invariants
from hcsim.studies.harness import StudyError, StudyRun, run_study
from hcsim.studies.inputs import InputError, fit_models
from hcsim.studies.profile import Mode, ProfileError, StudyProfile

RUN = dt.date(2026, 1, 1)


def profile(**fields: object) -> StudyProfile:
    return StudyProfile.from_mapping(
        {"name": "skeleton", "seed": 11, "trajectories": 3, "months": 4} | fields
    )


@pytest.fixture(scope="module")
def guided(inputs: tuple[Path, Path]) -> StudyRun:
    return run_study(profile(), *inputs)


# --- profiles ---------------------------------------------------------------------------------


def test_a_profile_loads_from_yaml_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("name: skeleton\nseed: 3\nmode: unguided\nrun_date: 2026-01-01\n")
    loaded = StudyProfile.load(path)
    assert (loaded.mode, loaded.scope, loaded.trajectories, loaded.months) == (
        Mode.UNGUIDED,
        ForecastScope.HEADCOUNT_AND_SBC,
        20,
        12,
    )
    assert loaded.run_date == RUN
    assert not loaded.guided


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"trajectoris": 5}, "trajectoris: Extra inputs are not permitted"),
        (
            {"mode": "replayed"},
            "mode: Input should be 'replay', 'guided_actuals', 'guided_forecast' or 'unguided'",
        ),
        ({"seed": -1}, "seed: Input should be greater than or equal to 0"),
        ({"run_date": "2026-01-15"}, "run_date must be the first of a month"),
        ({"scope": "sbc"}, "scope: Input should be 'hcf' or 'hcf+sbc'"),
    ],
)
def test_a_bad_profile_lists_its_problems(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ProfileError, match=re.escape(message)):
        profile(**fields)


def test_a_profile_needs_a_seed_and_is_frozen() -> None:
    with pytest.raises(ProfileError, match="seed: Field required"):
        StudyProfile.from_mapping({"name": "x"})
    with pytest.raises(ValueError, match="frozen"):
        profile().seed = 2  # type: ignore[misc]


def test_an_unreadable_profile_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="can't read"):
        StudyProfile.load(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ProfileError, match="can't read"):
        StudyProfile.load(bad)


# --- inputs -----------------------------------------------------------------------------------


def test_models_fit_from_the_window_before_the_run(inputs: tuple[Path, Path]) -> None:
    with duckdb.connect(str(inputs[0]), read_only=True) as con:
        fitted = fit_models(con, schema_from_world(con), RUN)
    assert fitted.window == (dt.date(2024, 12, 31), dt.date(2025, 12, 31))
    rates = fitted.rates
    assert set(rates) == {"attrition", "external_fill", "cancellation", "backfill", "growth"}
    assert all(0 < r <= 1 for r in rates.values())
    # The world's truth: a backfill probability of 0.6 and a growth rate near 0.25 a month.
    assert 0.3 < rates["backfill"] < 0.9
    assert 0.05 < rates["growth"] < 0.6
    rows = fitted.models.attributes.sample(0, 5, np.random.default_rng(0))
    assert rows.shape == (5, 6)
    assert (rows >= 0).all()


def test_fitting_needs_history(inputs: tuple[Path, Path]) -> None:
    with duckdb.connect(str(inputs[0]), read_only=True) as con:
        schema = schema_from_world(con)
        with pytest.raises(InputError, match="no exposure"):
            fit_models(con, schema, dt.date(2021, 2, 1))  # the first month-end has no prior stock


# --- the harness ------------------------------------------------------------------------------


def test_a_guided_study_passes_its_invariants(guided: StudyRun) -> None:
    assert guided.passed
    metrics = {m.name: m for m in guided.metrics}
    assert metrics["quota_violations"].value == 0
    assert "guided draws checked" in metrics["quota_violations"].detail
    assert metrics["headcount_identity_max_gap"].value == 0
    assert metrics["opening_identity_max_gap"].value == 0
    assert metrics["accounting_identity_max_residual"].value <= 0.01
    assert metrics["quota_shortfall"].threshold is None
    assert guided.config.run_date == RUN
    assert guided.simulated.headcount.shape == (3, 5, 10)


def test_the_provenance_names_every_input(guided: StudyRun, inputs: tuple[Path, Path]) -> None:
    p = guided.provenance
    assert p["source"] == "synthetic_world"
    assert (p["world_preset"], p["world_seed"]) == ("tiny", "5")
    assert p["guidance_fingerprint"].startswith("gf2-")
    assert p["schema_fingerprint"] == guided.config.schema.fingerprint()
    assert (p["mode"], p["scope"], p["seed"], p["run_date"]) == (
        "guided_forecast",
        "hcf+sbc",
        "11",
        "2026-01-01",
    )
    assert p["isolates"] == "the guidance forecast"  # its cell of rung by scope


def test_a_profile_can_declare_capacity_caps(inputs: tuple[Path, Path]) -> None:
    with pytest.raises(ProfileError, match="caps are non-negative heads"):
        profile(capacity_caps={"CC-01": -1})
    run = run_study(profile(mode="unguided", scope="hcf", trajectories=1), inputs[0])
    label = run.config.schema.vocabulary("cost_center").labels[0]
    capped = run_study(
        profile(mode="unguided", scope="hcf", trajectories=1, capacity_caps={label: 0}), inputs[0]
    )
    assert dict(capped.config.capacity_caps) == {label: 0}
    shaped = run_study(
        profile(
            mode="unguided",
            scope="hcf",
            trajectories=1,
            fill_shares=[{"scope": {"cost_center": [label]}, "shares": {"external": 1.0}}],
        ),
        inputs[0],
    )
    assert shaped.config.resolved_fill_shares is not None
    assert [t.to_dict() for t in shaped.config.fill_shares] == [
        {
            "scope": {"cost_center": [label]},
            "shares": {"external": 1.0, "conversion": 0.0, "transfer": 0.0},
        }
    ]
    assert capped.config.ceiling.caps[0] == 0.0
    assert dict(run.config.capacity_caps) == {}


def test_an_unguided_headcount_study_reads_no_guidance(inputs: tuple[Path, Path]) -> None:
    run = run_study(profile(mode="unguided", scope="hcf"), inputs[0])
    assert run.passed
    assert run.config.guidance is None
    assert run.config.awards is None
    assert "accounting_identity_max_residual" not in {m.name for m in run.metrics}
    assert run.provenance["guidance_fingerprint"] == ""
    assert run.config.run_date == RUN  # the month after the world's last snapshot


def test_a_profile_can_start_the_run_earlier(inputs: tuple[Path, Path]) -> None:
    run = run_study(profile(mode="unguided", run_date="2025-07-01", months=2), inputs[0])
    assert run.passed
    assert run.seed.snapshot_date == dt.date(2025, 6, 30)
    assert run.fitted.window[1] == dt.date(2025, 6, 30)


def test_a_study_is_reproducible(inputs: tuple[Path, Path], guided: StudyRun) -> None:
    again = run_study(profile(), *inputs)
    assert np.array_equal(again.simulated.headcount, guided.simulated.headcount)
    for a, b in zip(again.simulated.trajectories, guided.simulated.trajectories, strict=True):
        assert a.sbc is not None
        assert b.sbc is not None
        pd.testing.assert_frame_equal(a.sbc, b.sbc)


def test_inputs_must_fit_the_mode(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    world, guidance = inputs
    with pytest.raises(StudyError, match="needs a guidance artifact"):
        run_study(profile(), world)
    with pytest.raises(StudyError, match="reads no guidance"):
        run_study(profile(mode="unguided"), world, guidance)
    with pytest.raises(StudyError, match="no world"):
        run_study(profile(mode="unguided"), tmp_path / "missing.duckdb")


# --- the evaluator ----------------------------------------------------------------------------


def _with_trajectory(run: StudyRun, **changes: object) -> list[str]:
    """The names of the failing metrics after changing trajectory 0."""
    first = dataclasses.replace(run.simulated.trajectories[0], **changes)  # type: ignore[arg-type]
    result = dataclasses.replace(
        run.simulated, trajectories=(first, *run.simulated.trajectories[1:])
    )
    return [m.name for m in invariants(result) if not m.passed]


def test_invariants_catch_a_broken_quota(guided: StudyRun) -> None:
    row = guided.simulated.trajectories[0].quotas[0]
    record = row.record
    broken = QuotaRecord(record.quota, record.eligible, record.realized + 1)
    quotas = (dataclasses.replace(row, record=broken), *guided.simulated.trajectories[0].quotas[1:])
    assert _with_trajectory(guided, quotas=quotas) == ["quota_violations"]


def test_invariants_catch_a_broken_stock_identity(guided: StudyRun) -> None:
    headcount = guided.simulated.trajectories[0].headcount.copy()
    headcount[2, 0] += 1
    assert _with_trajectory(guided, headcount=headcount) == ["headcount_identity_max_gap"]
    openings = guided.simulated.trajectories[0].open_openings.copy()
    openings[-1, 3] -= 2
    assert _with_trajectory(guided, open_openings=openings) == ["opening_identity_max_gap"]


def test_invariants_catch_an_untied_tranche(guided: StudyRun) -> None:
    expense = ExpenseResult(pd.DataFrame(), np.array([0.0, 0.5]))
    assert _with_trajectory(guided, expense=expense) == ["accounting_identity_max_residual"]
