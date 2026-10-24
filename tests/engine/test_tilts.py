"""Tests for probability tilts: a planner's multiplier on one domain's event probabilities."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.domains import StepContext, attrition
from hcsim.engine.fills import channel_hazards
from hcsim.engine.loop import simulate
from hcsim.engine.planner import PlannerError, PlannerScope, ProbabilityTilt, resolve_tilts
from hcsim.engine.seeds import SeedManager
from hcsim.engine.settings import from_record, to_record
from hcsim.model.core.layouts import EmployeeColumn
from tests.engine.test_loop import RUN_DATE, SCHEMA, config, guidance, initial_state, models
from tests.support.stat_check import stat_check

FIRST = PlannerScope.of(cost_center=["CC-01"])
FIRST_SENIOR = PlannerScope.of(cost_center=["CC-01"], job_level=[2])
DOUBLE = ProbabilityTilt("attrition", FIRST, 2.0)


def test_a_tilt_is_checked_with_every_problem_named() -> None:
    with pytest.raises(PlannerError) as caught:
        ProbabilityTilt("weather", FIRST, -1.0, "2026-05", "2026-02", cap=1.5)
    assert caught.value.problems == [
        f"tilt on weather for {FIRST}: no domain 'weather' to tilt; known: attrition, promotion, "
        "backfill, external_fill, conversion_fill, transfer_fill, cancellation",
        f"tilt on weather for {FIRST}: a multiplier is finite and non-negative, not -1.0",
        f"tilt on weather for {FIRST}: a cap lies in (0, 1], not 1.5",
        f"tilt on weather for {FIRST}: the window starts in 2026-05, after it ends in 2026-02",
    ]
    with pytest.raises(PlannerError, match="the window's months are YYYY-MM"):
        ProbabilityTilt("attrition", FIRST, 2.0, "not a month")
    tilt = ProbabilityTilt("attrition", FIRST, 2.0, "2026-03", "2026-04", cap=0.5, reason="a plan")
    assert ProbabilityTilt.from_dict(tilt.to_dict()) == tilt
    with pytest.raises(PlannerError, match="can't read tilt"):
        ProbabilityTilt.from_dict({"domain": "attrition"})


def test_a_window_resolves_to_the_runs_steps() -> None:
    # The run starts in January 2026 and lasts six months.
    def steps(start: str | None, end: str | None) -> list[int]:
        return list(ProbabilityTilt("attrition", FIRST, 2.0, start, end).steps(RUN_DATE, 6))

    assert steps("2026-03", "2026-04") == [2, 3]
    assert steps(None, "2026-02") == [0, 1]
    assert steps("2026-05", None) == [4, 5]
    assert steps(None, None) == [0, 1, 2, 3, 4, 5]
    assert steps("2026-09", None) == []  # after the run
    assert steps("2025-01", "2025-06") == []  # before it


def test_the_most_specific_tilt_in_force_multiplies_each_row_up_to_its_cap() -> None:
    senior = ProbabilityTilt("attrition", FIRST_SENIOR, 30.0, cap=0.5)
    later = ProbabilityTilt("attrition", PlannerScope.of(cost_center=["CC-02"]), 3.0, "2026-04")
    tilts = resolve_tilts([DOUBLE, senior, later], SCHEMA, RUN_DATE, 6)
    assert tilts is not None
    people = initial_state().live_employees
    cc = people[:, EmployeeColumn.COST_CENTER].astype(int)
    level = people[:, EmployeeColumn.LEVEL].astype(int)
    out = tilts.apply("attrition", people, 0, np.full(len(people), 0.05))
    assert np.allclose(out.p[(cc == 0) & (level == 0)], 0.1)
    assert set(out.p[(cc == 0) & (level == 1)]) == {0.5}  # 1.5, capped
    assert set(out.p[cc > 0]) == {0.05}  # CC-02's tilt waits for April
    assert out.capped.sum() == ((cc == 0) & (level == 1)).sum()
    assert out.tilted.sum() == (cc == 0).sum()
    april = tilts.apply("attrition", people, 3, np.full(len(people), 0.05))
    assert np.allclose(april.p[cc == 1], 0.15)
    untouched = tilts.apply("promotion", people, 0, np.full(len(people), 0.05))
    assert not untouched.tilted.any()
    assert resolve_tilts([], SCHEMA, RUN_DATE, 6) is None


def test_a_tilt_reaches_each_domain_it_names_and_is_recorded() -> None:
    fills = ProbabilityTilt("external_fill", FIRST, 0.5)
    run = config(tilts=(DOUBLE, fills))
    state = initial_state()
    ctx = StepContext(run, 0, 0, SeedManager(1))
    plain = StepContext(config(), 0, 0, SeedManager(1))
    reqs = state.live_openings
    assert channel_hazards(reqs, ctx)[:, 0].tolist() == pytest.approx(
        [0.15, 0.15, 0.3, 0.3, 0.3, 0.3]
    )
    assert channel_hazards(reqs, plain)[:, 0].tolist() == pytest.approx([0.3] * 6)
    attrition(state, ctx)
    assert ctx.tilt_records["tilted_attrition"].tolist() == [30, 0, 0]
    assert ctx.tilt_records["tilt_capped_attrition"].tolist() == [0, 0, 0]
    assert ctx.tilt_records["tilted_external_fill"].tolist() == [2, 0, 0]
    (t,) = simulate(dataclasses.replace(run, num_trajectories=1), state).trajectories
    tilted = t.diagnostics["tilted_attrition"]
    assert tilted.shape == (run.simulation_months, 3)
    assert (tilted[:, 0] > 0).all()
    assert (tilted[:, 1:] == 0).all()


def test_under_guidance_a_tilt_moves_who_leaves_not_how_many() -> None:
    # Guidance fixes CC-01 at 3 departures a month; a tilt on its level-2 people makes them
    # almost all of the 3.
    senior = ProbabilityTilt("attrition", FIRST_SENIOR, 40.0)
    guided = guidance({"attrition": {"CC-01": 3.0}})
    for tilts, seniors in (((), None), ((senior,), True)):
        run = config(guidance=guided, tilts=tilts)
        counts, picked = 0, 0
        for j in range(20):
            ctx = StepContext(run, 0, j, SeedManager(j))
            gone = attrition(initial_state(), ctx).departures
            mine = gone[gone[:, EmployeeColumn.COST_CENTER] == 0]
            counts += len(mine)
            picked += int((mine[:, EmployeeColumn.LEVEL] == 1).sum())
        assert counts == 60  # the quota, tilted or not
        if seniors:
            assert picked >= 55


def test_tilts_are_resolved_and_recorded(tmp_path: Path) -> None:
    from tests.engine.test_config_records import _saved
    from tests.engine.test_config_state_seeds import config as small_config

    junior = ProbabilityTilt("attrition", PlannerScope.of(job_level=[1]), 0.5)
    with pytest.raises(ConfigError, match=r"tilts: tilts on attrition: rules 1 .* are ambiguous"):
        config(tilts=(DOUBLE, junior))
    run = small_config(
        models=_saved(tmp_path), tilts=(ProbabilityTilt("cancellation", FIRST, 0.5, "2026-02"),)
    )
    record = json.loads(json.dumps(to_record(run)))
    assert record["tilts"] == [t.to_dict() for t in run.tilts]
    rebuilt = from_record(record, schema=run.schema, models_root=tmp_path, awards=run.awards)
    assert rebuilt.tilts == run.tilts
    record["tilts"] = [{"domain": "weather", "scope": {}, "multiplier": 1.0}]
    with pytest.raises(ConfigError, match="no domain 'weather' to tilt"):
        from_record(record, schema=run.schema, models_root=tmp_path, awards=run.awards)


@pytest.mark.slow
def test_an_unguided_tilt_moves_the_count_by_its_multiplier() -> None:
    # Each of CC-01's 30 people leaves in the first month with 0.05, doubled to 0.1: 3 expected.
    run = config(models=models(attrition=0.05), tilts=(DOUBLE,), num_trajectories=2_000)
    first = simulate(run, initial_state()).flow("A")[:, 0, :].astype(np.float64)
    n = len(first)
    for c, expected in ((0, 3.0), (1, 1.5)):
        stat_check(
            first[:, c].mean(), expected, first[:, c].std(ddof=1) / np.sqrt(n), label=f"CC-0{c + 1}"
        )
