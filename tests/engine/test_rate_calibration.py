"""Tests for rate calibration: the logit shifts that make the expected path meet the plan."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.engine.calibration import MAX_SHIFT, Calibration, calibrate
from hcsim.engine.domains import StepContext, attrition
from hcsim.engine.fills import channel_hazards
from hcsim.engine.logit import logit_shift
from hcsim.engine.loop import simulate
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.alignment import AlignmentTarget, Interpolation, marginal
from hcsim.model.core.layouts import EmployeeColumn
from tests.engine.test_loop import config, initial_state, models
from tests.support.stat_check import stat_check

#: A 5% attrition hazard and a 30% fill hazard per opening, with 5% of requisitions exiting
#: unfilled a month: uncalibrated, June ends near 33.4, 30.6, and 29.1 heads.
MODELS = models(attrition=0.05, fill=0.3, cancel=0.05)


def june(values: dict[str, float]) -> AlignmentTarget:
    m = marginal("cost_center", list(values), ["2026-06"], [[v] for v in values.values()])
    return AlignmentTarget((m,), pd.Period("2026-01", "M"), 6, 6, Interpolation.HOLD_LAST)


PLAN = june({"CC-01": 36.0, "CC-02": 26.0})


def calibrated(plan: AlignmentTarget = PLAN, **overrides: object) -> Calibration | None:
    run = config(plan=plan, alignment_method="rate_calibration", models=MODELS, **overrides)
    return calibrate(initial_state(), StepContext(run, 0, 0, SeedManager(1)))


def test_a_logit_shift_moves_a_probability_and_leaves_certainty_alone() -> None:
    p = np.array([0.0, 0.25, 0.5, 1.0])
    assert logit_shift(p, np.zeros(4)).tolist() == pytest.approx(p.tolist())
    assert logit_shift(p, np.full(4, np.log(3.0))).tolist() == pytest.approx([0.0, 0.5, 0.75, 1.0])
    assert logit_shift(p, np.full(4, -np.log(3.0)))[2] == pytest.approx(0.25)


def test_the_shifts_meet_each_target_in_expectation() -> None:
    found = calibrated()
    assert found is not None
    assert found.step == 5
    # CC-01 needs more heads: fewer leave and more fill. CC-02 needs fewer: the reverse.
    assert found.attrition[0] < 0 < found.fill[0]
    assert found.fill[1] < 0 < found.attrition[1]
    assert (found.attrition[2], found.fill[2]) == (0.0, 0.0)  # no target, no shift
    assert found.converged.all()
    assert np.abs(found.residual).max() < 1e-6
    assert set(found.to_dict()) == {
        "step",
        "iterations",
        "attrition",
        "fill",
        "converged",
        "residual",
    }


def test_a_target_out_of_reach_is_reported_unconverged() -> None:
    found = calibrated(june({"CC-01": 200.0}))
    assert found is not None
    assert not found.converged[0]
    assert found.fill[0] == pytest.approx(MAX_SHIFT)
    assert found.residual[0] < -100
    assert calibrated(june({"CC-01": 36.0}), simulation_months=3) is None  # t* past the run
    assert calibrate(initial_state(), StepContext(config(), 0, 0, SeedManager(1))) is None


def test_the_handlers_take_the_shifts() -> None:
    run = config(plan=PLAN, alignment_method="rate_calibration", models=MODELS)
    state = initial_state()
    found = calibrate(state, StepContext(run, 0, 0, SeedManager(1)))
    shifted = StepContext(run, 0, 0, SeedManager(1), calibration=found)
    plain = StepContext(run, 0, 0, SeedManager(1))
    openings = state.live_openings
    assert (
        channel_hazards(openings, shifted)[:2, 0].tolist()
        != channel_hazards(openings, plain)[:2, 0].tolist()
    )
    assert (
        channel_hazards(openings, shifted)[4:, 0].tolist()
        == channel_hazards(openings, plain)[4:, 0].tolist()
    )

    # Over the same draws, CC-01's people, shifted down, leave less often, and CC-02's more.
    def leaving(calibration: Calibration | None) -> np.ndarray:
        counts = np.zeros(3, dtype=np.int64)
        for j in range(40):
            ctx = StepContext(run, 0, j, SeedManager(j), calibration=calibration)
            gone = attrition(state, ctx).departures[:, EmployeeColumn.COST_CENTER].astype(int)
            counts += np.bincount(gone, minlength=3)
        return counts

    up, flat = leaving(found), leaving(None)
    assert up[0] < flat[0]
    assert up[1] > flat[1]
    assert up[2] == flat[2]


@pytest.mark.slow
def test_a_calibrated_run_meets_the_plan_over_trajectories() -> None:
    run = config(
        plan=PLAN, alignment_method="rate_calibration", models=MODELS, num_trajectories=1_000
    )
    result = simulate(run, initial_state())
    assert result.calibration is not None
    june_end = result.headcount[:, 6, :2].astype(np.float64)
    for c, target in enumerate((36.0, 26.0)):
        heads = june_end[:, c]
        stat_check(
            heads.mean(), target, heads.std(ddof=1) / np.sqrt(len(heads)), label=f"CC-0{c + 1}"
        )
    plain = simulate(config(plan=PLAN, models=MODELS, num_trajectories=2), initial_state())
    assert plain.calibration is None
