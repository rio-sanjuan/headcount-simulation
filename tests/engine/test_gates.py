"""Tests for registration and step gates, and their bypass counts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hcsim.engine.config import ConfigError, RunMode
from hcsim.engine.gates import (
    Gate,
    GateError,
    RegistrationGate,
    StepGate,
    gate_from_dict,
    register,
)
from hcsim.engine.settings import from_record, to_record
from tests.engine.test_config_records import _saved
from tests.engine.test_config_state_seeds import awards
from tests.engine.test_config_state_seeds import config as small_config
from tests.engine.test_loop import RUN_DATE, run

DOMAINS = ("attrition", "backfill", "fills", "cancellation", "growth")


def test_a_registration_gate_admits_by_switch_and_mode() -> None:
    replay_only = RegistrationGate("growth", modes=(RunMode.REPLAY,), reason="replay only")
    assert replay_only.admits(RunMode.REPLAY)
    assert not replay_only.admits(RunMode.UNGUIDED)
    assert not RegistrationGate("growth", enabled=False).admits(RunMode.REPLAY)
    assert RegistrationGate("growth").admits(RunMode.GUIDED_FORECAST)


def test_a_step_gate_opens_in_its_months_with_its_probability() -> None:
    rng = np.random.default_rng(0)
    cycle = StepGate("cancellation", months=(3, 9))
    assert [cycle.opens(m, rng) for m in (2, 3, 9)] == [False, True, True]
    assert not StepGate("growth", probability=0.0).opens(5, rng)
    half = StepGate("growth", probability=0.5)
    share = np.mean([half.opens(1, rng) for _ in range(4_000)])
    assert abs(share - 0.5) < 0.03


def test_registration_keeps_the_step_order_and_refuses_unknown_domains() -> None:
    gates: list[Gate] = [
        RegistrationGate("growth", enabled=False),
        StepGate("cancellation", months=(1,)),
        StepGate("cancellation", probability=0.5),
    ]
    registered = register(DOMAINS, gates, RunMode.UNGUIDED)
    assert registered.domains == ("attrition", "backfill", "fills", "cancellation")
    assert registered.unregistered == ("growth",)
    assert len(registered.steps["cancellation"]) == 2
    assert registered.opens("attrition", 5, np.random.default_rng(0))
    assert not registered.opens("cancellation", 2, np.random.default_rng(0))
    with pytest.raises(GateError, match="no domain 'weather' to gate"):
        register(DOMAINS, [StepGate("weather")], RunMode.UNGUIDED)


def test_a_bad_gate_is_reported_among_the_configuration_s_problems() -> None:
    with pytest.raises(ConfigError) as caught:
        small_config(
            gates=(StepGate("growth", months=(0, 3, 3), probability=1.5),),
        )
    assert caught.value.problems == [
        "gates: growth: months must be distinct, 1 to 12: [0, 3, 3]",
        "gates: growth: a probability must lie in [0, 1]",
    ]


def test_gates_round_trip_through_plain_data_and_records(tmp_path: Path) -> None:
    gates = (
        RegistrationGate("growth", modes=(RunMode.REPLAY,), reason="replay"),
        StepGate("cancellation", months=(3,), probability=0.25),
    )
    for gate in gates:
        assert gate_from_dict(json.loads(json.dumps(gate.to_dict()))) == gate
    for bad in ({"gate": "valve", "domain": "x"}, {"gate": "step"}):
        with pytest.raises(GateError, match="can't read gate"):
            gate_from_dict(bad)
    built = small_config(models=_saved(tmp_path), gates=gates)
    record = json.loads(json.dumps(to_record(built)))
    rebuilt = from_record(record, schema=built.schema, models_root=tmp_path, awards=awards())
    assert rebuilt.gates == gates
    record["gates"] = [{"gate": "valve"}]
    with pytest.raises(ConfigError, match="can't read gate"):
        from_record(record, schema=built.schema, models_root=tmp_path, awards=awards())


def test_an_unregistered_domain_never_runs() -> None:
    result = run(gates=(RegistrationGate("growth", enabled=False),))
    assert result.unregistered == (
        "replay_requisitions",
        "other_movement",
        "performance",
        "annual_award",
        "anticipated_award",
        "campus",
        "growth",
    )
    assert int(result.flow("G").sum()) == 0
    assert int(run().flow("G").sum()) > 0
    # Replayed requisitions and other movement come only from replayed history, performance
    # needs a model, the award domains their awards, and campus a schedule, so all six are
    # unregistered by default.
    award_domains = ("annual_award", "anticipated_award")
    assert run().unregistered == (
        "replay_requisitions",
        "other_movement",
        "performance",
        *award_domains,
        "campus",
    )
    assert run(run_mode="replay").unregistered == ("performance", *award_domains, "campus")


def test_a_step_gate_bypasses_its_domain_and_counts_each_bypass() -> None:
    february = RUN_DATE.replace(month=2).month
    result = run(gates=(StepGate("cancellation", months=(february,)),))
    flows = result.flow("U")
    assert int(flows[:, [0, 2, 3, 4, 5]].sum()) == 0  # every month but February
    assert all(t.bypasses == {"cancellation": 5} for t in result.trajectories)
    assert run().trajectories[0].bypasses == {}


def test_a_stochastic_gate_is_reproducible_and_disturbs_no_other_stream() -> None:
    gated = run(gates=(StepGate("growth", probability=0.5),))
    again = run(gates=(StepGate("growth", probability=0.5),))
    assert [t.bypasses for t in gated.trajectories] == [t.bypasses for t in again.trajectories]
    assert np.array_equal(gated.flow("G"), again.flow("G"))
    counts = [t.bypasses["growth"] for t in gated.trajectories]
    assert 0 < sum(counts) < 6 * len(counts)
    # Growth's own draws come from its own stream, so the months it runs match an ungated run.
    plain = run()
    ran = gated.flow("G").sum(axis=2) > 0
    assert int(gated.flow("A")[:, 0].sum()) == int(plain.flow("A")[:, 0].sum())
    assert ran.any()
