"""Tests for gates: bounds on metrics that mark a run provisional or stop it."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from hcsim.engine.modes import RunMode
from hcsim.studies.evaluators import Metric
from hcsim.studies.gates import (
    BUILTIN,
    BUILTIN_BY_NAME,
    Action,
    Gate,
    GateError,
    GateResult,
    Verdict,
    apply,
    gate_set,
    verdict,
)
from hcsim.studies.harness import run_study
from hcsim.studies.profile import ProfileError, StudyProfile

PAST = dt.date(2025, 7, 1)
ARMS = [
    {"name": "base", "null": True},
    {"name": "x", "overrides": {"promotion.gate_probability": 0.4}},
    {"name": "y", "overrides": {"promotion.gate_probability": 0.6}},
]


def profile(**fields: object) -> StudyProfile:
    return StudyProfile.from_mapping(
        {
            "name": "gates",
            "seed": 7,
            "trajectories": 2,
            "months": 4,
            "run_date": PAST.isoformat(),
            "mode": "unguided",
            "scope": "hcf",
            "evaluators": ["invariants", "fidelity"],
        }
        | fields
    )


def names(gates: tuple[Gate, ...]) -> set[str]:
    return {g.name for g in gates}


# --- a gate -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"metric": "headcount"}, "metric is evaluator.metric"),
        ({"max": 1.0, "min": 0.0}, "exactly one of max and min"),
        ({}, "exactly one of max and min"),
        ({"max": 1.0, "closures": frozenset({"closed"})}, "no window closures ['closed']"),
    ],
)
def test_a_gate_bounds_one_metric_once(fields: dict[str, object], message: str) -> None:
    with pytest.raises(GateError, match=re.escape(message)):
        Gate(**({"name": "g", "metric": "e.m", "action": Action.STOP} | fields))  # type: ignore[arg-type]


def test_a_gate_holds_by_its_bound() -> None:
    at_most = Gate("a", "e.m", Action.STOP, max=0.1)
    assert at_most.holds(0.1)
    assert not at_most.holds(0.11)
    assert at_most.holds(-5)
    assert at_most.bound == "x <= 0.1"
    magnitude = Gate("b", "e.m", Action.STOP, max=0.1, absolute=True)
    assert not magnitude.holds(-0.2)
    assert magnitude.bound == "|x| <= 0.1"
    at_least = Gate("c", "e.m", Action.PROVISIONAL, min=2)
    assert at_least.holds(2)
    assert not at_least.holds(1.9)
    assert at_least.bound == "x >= 2"
    assert at_least.evaluator == "e"


# --- gate sets --------------------------------------------------------------------------------


def test_the_gate_set_follows_the_mode_and_the_window() -> None:
    invariants = {"quota_honored", "headcount_identity", "opening_identity", "accounting_identity"}
    replay = names(gate_set("replay", "wholly_closed"))
    assert invariants | {"replay_headcount", "replay_attrition", "replay_sbc"} <= replay
    assert "forecast_headcount" not in replay
    assert names(gate_set("replay", "partially_closed")) == invariants
    assert names(gate_set(RunMode.UNGUIDED, "wholly_closed")) == invariants | {"forecast_headcount"}
    assert names(gate_set("guided_actuals", "wholly_closed")) == invariants | {"actuals_headcount"}
    assert names(gate_set("guided_forecast", "wholly_open")) == invariants
    assert all(g.action is Action.STOP for g in BUILTIN if g.metric.startswith("invariants."))


def test_a_study_adds_its_own_gates_and_can_drop_the_builtin_ones() -> None:
    own = Gate("tight", "fidelity.headcount_wape", Action.PROVISIONAL, max=0.01)
    assert names(gate_set("unguided", "wholly_closed", [own], builtin=False)) == {"tight"}
    late = Gate("late", "e.m", Action.STOP, max=0, closures=frozenset({"wholly_closed"}))
    assert names(gate_set("unguided", "wholly_open", [own, late], builtin=False)) == {"tight"}
    with pytest.raises(GateError, match=re.escape("gate names repeat: ['quota_honored']")):
        gate_set("unguided", "wholly_open", [BUILTIN_BY_NAME["quota_honored"]])
    with pytest.raises(GateError, match="no window closure"):
        gate_set("unguided", "closed")


# --- applying gates, and the verdict ----------------------------------------------------------


def test_gates_apply_to_the_metrics_an_arm_reports() -> None:
    gates = [
        Gate("q", "invariants.quota_violations", Action.STOP, max=0),
        Gate("f", "fidelity.headcount_wape", Action.PROVISIONAL, max=0.05),
    ]
    results = apply(gates, "base", [Metric("invariants", "quota_violations", 2.0, 0)])
    q, f = results
    assert (q.value, q.passed, q.failed) == (2.0, False, True)
    assert not f.applied
    assert not f.failed
    assert f.reason == "arm base reports no fidelity.headcount_wape"


def _result(action: Action, passed: bool) -> GateResult:
    return GateResult("g", "a", "e.m", action, "x <= 0", 1.0, passed)


def test_the_verdict_is_the_worst_failure() -> None:
    assert verdict([]) is Verdict.PASSED
    assert verdict([], provisional_inputs=True) is Verdict.PROVISIONAL
    assert verdict([_result(Action.PROVISIONAL, False)]) is Verdict.PROVISIONAL
    both = [_result(Action.PROVISIONAL, False), _result(Action.STOP, False)]
    assert verdict(both) is Verdict.STOPPED
    assert verdict([_result(Action.STOP, True)]) is Verdict.PASSED


# --- gates in a study -------------------------------------------------------------------------


def test_a_replay_passes_its_gates(inputs: tuple[Path, Path]) -> None:
    run = run_study(profile(mode="replay"), inputs[0])
    assert run.verdict is Verdict.PASSED
    applied = {g.gate: g for g in run.gates if g.applied}
    assert applied["replay_headcount"].value == 0
    assert applied["replay_promotions"].passed
    sbc = next(g for g in run.gates if g.gate == "replay_sbc")
    assert not sbc.applied  # an hcf study scores no SBC
    assert sbc.reason == "arm study reports no sbc.monthly_wape"
    meta = run.metadata()
    assert meta["verdict"] == "passed"
    assert {g["gate"] for g in meta["gates"]} == {g.gate for g in run.gates}


def test_a_failed_stop_gate_stops_the_study(inputs: tuple[Path, Path]) -> None:
    tight = {"name": "tight", "metric": "fidelity.headcount_wape", "max": 0.0, "action": "stop"}
    run = run_study(profile(arms=ARMS, gates=[tight]), inputs[0])
    assert run.verdict is Verdict.STOPPED
    assert [a.arm.name for a in run.arms] == ["base"]
    why = "the study stopped: gate tight failed on arm base"
    assert [(s.name, s.reason) for s in run.skips if s.what == "arm"] == [("x", why), ("y", why)]


def test_a_failed_provisional_gate_flags_the_study(inputs: tuple[Path, Path]) -> None:
    tight = {"name": "tight", "metric": "fidelity.headcount_wape", "max": 0.0}
    arms = [*ARMS, {"name": "history", "reference": True, "mode": "replay"}]
    run = run_study(profile(arms=arms, gates=[tight], builtin_gates=False), inputs[0])
    assert run.verdict is Verdict.PROVISIONAL
    assert [a.arm.name for a in run.arms] == ["base", "x", "y", "history"]
    assert {g.arm for g in run.gates} == {"base", "x", "y"}  # a reference arm is never gated
    assert {g.gate for g in run.gates} == {"tight"}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {
                "gates": [
                    {"name": "quota_honored", "metric": "invariants.quota_violations", "max": 0}
                ]
            },
            "gate names repeat: ['quota_honored']",
        ),
        (
            {"gates": [{"name": "s", "metric": "sbc.monthly_wape", "max": 0.1}]},
            "gates ['s'] bound metrics of evaluators the study doesn't run",
        ),
        (
            {"gates": [{"name": "s", "metric": "fidelity.headcount_wape"}]},
            "exactly one of max and min",
        ),
        (
            {"gates": [{"name": "s", "metric": "fidelity", "max": 1}]},
            "gates.0.metric: String should match pattern",
        ),
    ],
)
def test_a_profiles_gates_are_checked(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ProfileError, match=re.escape(message)):
        profile(**fields)
    own = {"name": "quota_honored", "metric": "invariants.quota_violations", "max": 0}
    assert profile(gates=[own], builtin_gates=False).own_gates()[0].name == "quota_honored"
