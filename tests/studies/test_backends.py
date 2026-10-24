"""Tests for the backends: replay and guided-actuals studies in simulation, and offline studies."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from hcsim.engine.config import RunMode
from hcsim.engine.seeding import schema_from_world, seed_state
from hcsim.financial.prices import AnnualGrowthPrice, CustomPrice, StaticPrice
from hcsim.studies import harness
from hcsim.studies.backends import _realized_price
from hcsim.studies.harness import (
    ITSELF,
    ArmRun,
    Closure,
    StudyError,
    StudyRun,
    preflight,
    run_study,
)
from hcsim.studies.profile import ProfileError, StudyProfile
from hcsim.studies.scorecard import run_record

#: A window the tiny world holds the actuals of: it ends with 2025.
PAST = dt.date(2025, 7, 1)
MONTHS = 4


def profile(**fields: object) -> StudyProfile:
    return StudyProfile.from_mapping(
        {
            "name": "backends",
            "seed": 7,
            "trajectories": 2,
            "months": MONTHS,
            "run_date": PAST.isoformat(),
        }
        | fields
    )


def month_end_headcount(world: Path, months: int) -> np.ndarray:
    """(months + 1, cost centers): the world's headcount at each month-end from the one before
    the run, as a run starting there would seed it."""
    start = pd.Period(PAST, freq="M")
    with duckdb.connect(str(world), read_only=True) as con:
        schema = schema_from_world(con)
        rows = [
            seed_state(con, schema, (start + k).start_time.date())[0].headcount
            for k in range(months + 1)
        ]
    return np.stack(rows).astype(np.int64)


@pytest.fixture(scope="module")
def replayed(inputs: tuple[Path, Path]) -> StudyRun:
    return run_study(profile(mode="replay"), inputs[0])


# --- the in-simulation backend: replay --------------------------------------------------------


def test_a_replay_study_reproduces_history(replayed: StudyRun, inputs: tuple[Path, Path]) -> None:
    assert replayed.passed
    assert replayed.closure is Closure.CLOSED
    assert replayed.config.run_mode is RunMode.REPLAY
    assert replayed.config.guidance is None
    truth = month_end_headcount(inputs[0], MONTHS)
    assert replayed.result is not None
    for t in replayed.result.trajectories:
        assert np.array_equal(t.headcount, truth)
        assert t.known_fired["termination"] > 0


def test_a_replay_study_replays_the_recorded_awards(replayed: StudyRun) -> None:
    awards = replayed.config.awards
    assert awards is not None
    assert awards.replayed is not None
    metrics = {m.name: m for m in replayed.metrics}
    assert metrics["accounting_identity_max_residual"].passed
    p = replayed.provenance
    assert (p["mode"], p["backend"], p["guidance_fingerprint"]) == ("replay", "simulation", "")
    assert p["isolates"] == "orchestration, data fidelity, and the Financial Layer"


# --- the in-simulation backend: guided actuals ------------------------------------------------


def test_a_guided_actuals_study_reads_the_windows_realized_totals(
    inputs: tuple[Path, Path],
) -> None:
    run = run_study(profile(mode="guided_actuals", scope="hcf"), inputs[0])
    assert run.passed
    assert run.config.run_mode is RunMode.GUIDED_ACTUALS
    guidance = run.config.guidance
    assert guidance is not None
    assert guidance.snapshot_date == (PAST - dt.timedelta(days=1)).isoformat()
    assert guidance.horizon == MONTHS
    assert run.provenance["guidance_fingerprint"] == guidance.fingerprint()
    assert {m.name: m.value for m in run.metrics}["quota_violations"] == 0
    with pytest.raises(StudyError, match="reads no guidance; drop the artifact"):
        run_study(profile(mode="guided_actuals", scope="hcf"), *inputs)


def test_a_reference_arm_reads_its_own_modes_inputs(inputs: tuple[Path, Path]) -> None:
    arms = [
        {"name": "base", "null": True},
        {"name": "faster", "overrides": {"promotion.gate_probability": 0.5}},
        {"name": "history", "reference": True, "mode": "replay"},
        {"name": "actuals", "reference": True, "mode": "guided_actuals"},
    ]
    checked = preflight(profile(mode="unguided", scope="hcf", arms=arms), inputs[0])
    assert checked.ok, checked.problems
    modes = {name: c.run_mode for name, c in checked.configs.items()}
    assert modes == {
        "base": RunMode.UNGUIDED,
        "faster": RunMode.UNGUIDED,
        "history": RunMode.REPLAY,
        "actuals": RunMode.GUIDED_ACTUALS,
    }
    assert checked.inputs is not None
    assert set(checked.inputs.modes) == {"unguided", "replay", "guided_actuals"}
    assert checked.inputs.modes["unguided"].known is None  # a forecast reads no known events
    assert checked.inputs.modes["replay"].replay is not None
    assert checked.configs["actuals"].guidance is not None
    assert checked.configs["base"].guidance is None


def test_a_replayed_price_continues_from_the_worlds_last_close() -> None:
    with duckdb.connect() as con:
        con.execute("CREATE TABLE stock_price (month_end DATE, close DOUBLE)")
        model = StaticPrice(50.0)
        assert _realized_price(con, PAST, model) is model  # no closes to replay
        con.execute(
            "INSERT INTO stock_price VALUES ('2025-05-31', 100.0), ('2025-06-30', 110.0), "
            "('2025-07-31', 121.0)"
        )
        price = _realized_price(con, PAST, model)
        assert isinstance(price, CustomPrice)
        assert price.path == (110.0, 121.0)
        assert price.extend_with == AnnualGrowthPrice(121.0)
        grown = _realized_price(con, PAST, AnnualGrowthPrice(110.0, growth_rate=1.2))
        assert isinstance(grown, CustomPrice)
        assert grown.extend_with == AnnualGrowthPrice(121.0, growth_rate=1.2)


def test_a_mode_whose_inputs_cant_resolve_is_refused(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object) -> None:
        raise ValueError("no history")

    monkeypatch.setattr(harness, "mode_inputs", broken)
    checked = preflight(profile(mode="replay", scope="hcf"), inputs[0])
    assert checked.problems == ("can't resolve the inputs replay reads: no history",)


# --- deferral ---------------------------------------------------------------------------------


def test_a_reference_arm_on_actuals_waits_for_a_closed_window(inputs: tuple[Path, Path]) -> None:
    arms = [
        {"name": "base", "null": True},
        {"name": "faster", "overrides": {"promotion.gate_probability": 0.5}},
        {"name": "history", "reference": True, "mode": "replay"},
    ]
    run = run_study(
        profile(mode="unguided", scope="hcf", run_date="2025-11-01", arms=arms), inputs[0]
    )
    assert run.closure is Closure.PARTIAL
    assert [a.arm.name for a in run.arms] == ["base", "faster"]
    (waiting,) = run.deferred
    assert waiting.arm == "history"
    assert "partially_closed" in waiting.reason
    assert run.metadata()["deferred"] == [{"arm": "history", "reason": waiting.reason}]


def test_a_study_on_actuals_cant_run_in_an_open_window(inputs: tuple[Path, Path]) -> None:
    with pytest.raises(StudyError) as raised:
        run_study(profile(mode="replay", run_date=None), inputs[0])
    assert raised.value.problems == [
        "nothing can run yet: its mode, replay, reads the window's actuals, which is "
        "wholly_open; it runs once the window is wholly closed"
    ]


def test_an_artifact_is_needed_by_any_mode_that_reads_one(inputs: tuple[Path, Path]) -> None:
    arms = [{"name": "base", "null": True}, {"name": "x"}]
    arms.append({"name": "forecast", "reference": True, "mode": "guided_forecast"})
    with pytest.raises(StudyError, match="mode 'guided_forecast' needs a guidance artifact"):
        run_study(profile(mode="guided_actuals", scope="hcf", arms=arms), inputs[0])


def test_a_study_on_actuals_takes_no_subset() -> None:
    message = "modes ['replay'] read the whole world's actuals, so take no subset"
    with pytest.raises(ProfileError, match=re.escape(message)):
        profile(mode="replay", subset={"cost_center": ["CC-01"]})


# --- the offline backend ----------------------------------------------------------------------


def test_an_offline_study_scores_its_inputs_without_simulating(
    inputs: tuple[Path, Path],
) -> None:
    offline = profile(mode="unguided", scope="hcf", backend="offline", evaluators=["fitted_rates"])
    run = run_study(offline, inputs[0])
    assert run.result is None
    assert all(a.result is None for a in run.arms)
    with pytest.raises(StudyError, match="ran offline, so has no simulated results"):
        _ = run.simulated
    with pytest.raises(StudyError, match="arm study ran offline"):
        _ = ArmRun(ITSELF, run.config, None, run.metrics).simulated
    names = {m.name: m.value for m in run.metrics}
    assert names == {f"{name}_rate": value for name, value in run.fitted.rates.items()}
    assert all(m.threshold is None and m.evaluator == "fitted_rates" for m in run.metrics)
    assert run.passed
    assert [t.phase for t in run.timing] == ["preflight", "evaluate B"]
    assert run.provenance["backend"] == "offline"
    assert run_record(run, 1.0)["results_digest"] == ""


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"backend": "offline"},
            "a study on the offline backend runs offline evaluators, which "
            "['invariants', 'summary'] aren't",
        ),
        (
            {"evaluators": ["invariants", "fitted_rates"]},
            "a study on the simulation backend runs simulation evaluators, which "
            "['fitted_rates'] aren't",
        ),
        (
            {
                "backend": "offline",
                "evaluators": ["fitted_rates"],
                "arms": [{"name": "base", "null": True}, {"name": "x"}],
            },
            "an offline study scores its inputs, which every arm shares, so it has no arms",
        ),
        ({"backend": "batch"}, "backend: Input should be 'simulation' or 'offline'"),
    ],
)
def test_a_backend_runs_its_own_evaluators(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ProfileError, match=re.escape(message)):
        profile(**fields)
