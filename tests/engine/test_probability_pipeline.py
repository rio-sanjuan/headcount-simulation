"""Tests for the fixed probability pipeline every domain draws through: the model, then the
planner's tilts, then the rate-calibration shift, then the eligibility rules and known-event
exclusions, then the sampler."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hcsim.engine.calibration import Calibration
from hcsim.engine.domains import ATTRITION, Departures, StepContext, backfill, cancellation
from hcsim.engine.eligibility import EligibilityRule
from hcsim.engine.known_events import EventKind, KnownEvent, KnownEvents, Source
from hcsim.engine.known_plan import sanitize
from hcsim.engine.known_resolution import resolve_plan
from hcsim.engine.planner import PlannerScope, ProbabilityTilt
from hcsim.engine.promotion import promotion
from hcsim.engine.seeds import SeedManager
from hcsim.model.core.layouts import EmployeeColumn
from hcsim.model.ml.entity import ConstantModel
from tests.engine.test_ceiling import ALWAYS
from tests.engine.test_loop import SCHEMA, config, initial_state, models

FIRST = PlannerScope.of(cost_center=["CC-01"])
#: CC-01's attrition is doubled by a tilt, and shifted by log 3 by rate calibration.
SHIFT = Calibration(
    np.array([np.log(3.0), 0.0, 0.0]), np.zeros(3), np.ones(3, dtype=bool), np.zeros(3), 5, 1
)


def ctx(step: int = 0, calibration: Calibration | None = SHIFT, **overrides: object) -> StepContext:
    run = config(
        models=models(attrition=0.2),
        tilts=(ProbabilityTilt("attrition", FIRST, 2.0),),
        **overrides,
    )
    return StepContext(run, step, 0, SeedManager(1), calibration=calibration)


def scored(context: StepContext, **kwargs: bool) -> np.ndarray:
    people = initial_state().live_employees
    p = context.config.models.attrition.compute_probability(people)
    return context.probabilities(ATTRITION, people, p, **kwargs)


def test_the_tilt_comes_before_the_shift() -> None:
    # Tilted first, 0.2 doubles to 0.4, whose odds of 2/3 the shift triples to 2: 2/3. Shifted
    # first, it would be 3/7 doubled, 6/7.
    p = scored(ctx())
    assert p[:30].tolist() == pytest.approx([2 / 3] * 30)
    assert p[30:].tolist() == pytest.approx([0.2] * 60)
    assert scored(ctx(), shift=False)[:30].tolist() == pytest.approx([0.4] * 30)
    assert scored(ctx(calibration=None))[:30].tolist() == pytest.approx([0.4] * 30)


def test_eligibility_and_known_events_always_win() -> None:
    senior = EligibilityRule.of("attrition", cost_center=["CC-01"], job_level=[2])
    p = scored(ctx(eligibility=(senior,)))
    people = initial_state().live_employees
    level = people[:, EmployeeColumn.LEVEL].astype(int)
    assert (p[:30][level[:30] == 1] == 0).all()
    assert p[:30][level[:30] == 0].tolist() == pytest.approx([2 / 3] * 15)
    # Employee 4 is known to leave in step 3, so no sampled draw takes them before it.
    event = KnownEvent(EventKind.TERMINATION, 3, Source.DECLARED, 4)
    known = resolve_plan(sanitize(KnownEvents((event,)), range(90), []), SCHEMA)
    waiting = dataclasses.replace(ctx(), known=known)
    assert scored(waiting)[4] == 0.0
    assert scored(waiting)[5] == pytest.approx(2 / 3)


def test_recording_can_be_left_off() -> None:
    context = ctx()
    scored(context, record=False)
    assert context.tilt_records == {}
    scored(context)
    assert context.tilt_records["tilted_attrition"].tolist() == [30, 0, 0]


def test_every_domain_draws_through_the_pipeline() -> None:
    nothing = PlannerScope()
    state = initial_state()
    # Cancellation, tilted to nothing, closes nothing.
    closed = cancellation(
        state,
        StepContext(
            config(
                models=models(cancel=0.9),
                tilts=(ProbabilityTilt("cancellation", nothing, 0.0),),
            ),
            0,
            0,
            SeedManager(1),
        ),
    )
    assert closed.openings.is_empty
    # Promotion, tilted to nothing, promotes no one in its cycle month.
    promoting = dataclasses.replace(models(attrition=0.0), promotion=ConstantModel(1.0))
    run = config(models=promoting, tilts=(ProbabilityTilt("promotion", nothing, 0.0),))
    cycle = run.promotion_cycle_months[0] - run.run_date.month
    assert promotion(state, StepContext(run, cycle, 0, SeedManager(1))).employees.is_empty
    # Backfill, tilted to nothing, opens nothing for the month's departures.
    run = config(
        models=dataclasses.replace(models(), backfill=ALWAYS),
        tilts=(ProbabilityTilt("backfill", nothing, 0.0),),
    )
    context = StepContext(run, 0, 0, SeedManager(1))
    context.departures.append(
        Departures(ATTRITION, state.live_employees[:3].copy(), np.zeros(3, dtype=bool))
    )
    assert backfill(state, context).openings.is_empty
