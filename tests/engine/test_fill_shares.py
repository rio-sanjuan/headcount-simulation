"""Tests for fill-share targets: the channel mix a planner sets for a scope's fills."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.domains import StepContext
from hcsim.engine.fills import channel_hazards, rescale
from hcsim.engine.loop import simulate
from hcsim.engine.planner import (
    FillShareTarget,
    PlannerError,
    PlannerScope,
    Unattainable,
    solve_shares,
)
from hcsim.engine.seeds import SeedManager
from hcsim.engine.settings import from_record, to_record
from hcsim.model.core.layouts import OpeningColumn
from hcsim.model.ml.entity import ConstantModel
from tests.engine.test_loop import config, initial_state, models
from tests.support.stat_check import stat_check

FIRST = PlannerScope.of(cost_center=["CC-01"])
#: External fills at 0.3 a month, conversion and transfer at 0.1 each.
CHANNELS = dataclasses.replace(
    models(), conversion_fill=ConstantModel(0.1), transfer_fill=ConstantModel(0.1)
)


def test_a_target_names_channels_and_shares_that_sum_to_one() -> None:
    target = FillShareTarget(FIRST, {"external": 0.5, "conversion": 0.5})
    assert dict(target.shares) == {"external": 0.5, "conversion": 0.5, "transfer": 0.0}
    assert FillShareTarget.from_dict(target.to_dict()) == target
    with pytest.raises(PlannerError) as caught:
        FillShareTarget(FIRST, {"external": 0.7, "referral": 0.3})
    assert caught.value.problems == [
        "fill shares for cost_center in ['CC-01']: no fill channels ['referral']; the channels "
        "are ['external', 'conversion', 'transfer']",
        "fill shares for cost_center in ['CC-01']: shares sum to one, not 0.7",
    ]
    with pytest.raises(PlannerError, match="finite and non-negative"):
        FillShareTarget(FIRST, {"external": 1.5, "conversion": -0.5})
    with pytest.raises(PlannerError, match="can't read fill shares"):
        FillShareTarget.from_dict({"scope": {}})


def test_the_solver_moves_the_mix_and_keeps_the_count() -> None:
    hazards = np.array([[0.2, 0.1, 0.1], [0.3, 0.05, 0.1], [0.6, 0.3, 0.3]])
    goal = np.array([0.5, 0.25, 0.25])
    solved = solve_shares(hazards, goal)
    assert solved.reason is None
    assert solved.achieved.tolist() == pytest.approx(goal.tolist(), abs=1e-6)
    before, after = solved.fills
    assert after == pytest.approx(before, rel=1e-6)
    shaped = rescale(hazards * solved.multipliers).sum(axis=0)
    assert (shaped / shaped.sum()).tolist() == pytest.approx(goal.tolist(), abs=1e-6)


def test_a_target_out_of_reach_says_why() -> None:
    # No conversion hazard anywhere: the target's conversion share has nothing to open.
    closed = solve_shares(np.array([[0.2, 0.0, 0.1]]), np.array([0.6, 0.2, 0.2]))
    assert closed.reason is Unattainable.CLOSED
    assert closed.achieved.tolist() == pytest.approx([0.75, 0.0, 0.25], abs=1e-6)
    # Only the first opening can fill by transfer, at most once a month, so transfers can't
    # take 90% of the 1.9 fills the two openings expect.
    saturated = solve_shares(np.array([[0.9, 0.0, 0.1], [0.9, 0.0, 0.0]]), np.array([0.1, 0, 0.9]))
    assert saturated.reason is Unattainable.SATURATED
    nothing = solve_shares(np.zeros((2, 3)), np.array([1.0, 0.0, 0.0]))
    assert (nothing.fills, nothing.reason) == ((0.0, 0.0), None)


def test_the_fills_domain_shapes_each_scopes_hazards() -> None:
    target = FillShareTarget(FIRST, {"external": 0.4, "conversion": 0.4, "transfer": 0.2})
    run = config(models=CHANNELS, fill_shares=(target,))
    state = initial_state()
    reqs = state.live_openings
    shaped = channel_hazards(reqs, StepContext(run, 0, 0, SeedManager(1)))
    plain = channel_hazards(reqs, StepContext(config(models=CHANNELS), 0, 0, SeedManager(1)))
    first = reqs[:, OpeningColumn.COST_CENTER] == 0
    weights = reqs[first, OpeningColumn.NUM_OPENINGS][:, None]
    fills = (shaped[first] * weights).sum(axis=0)
    assert (fills / fills.sum()).tolist() == pytest.approx([0.4, 0.4, 0.2], abs=1e-6)
    assert fills.sum() == pytest.approx((plain[first] * weights).sum(), rel=1e-6)
    assert shaped[~first].tolist() == plain[~first].tolist()  # outside the scope, untouched
    assert run.resolved_fill_shares is not None
    solved = run.resolved_fill_shares.solve(plain, reqs)
    assert solved[0] is not None
    assert solved[0].reason is None


def test_targets_are_resolved_and_recorded(tmp_path: Path) -> None:
    from tests.engine.test_config_records import _saved
    from tests.engine.test_config_state_seeds import config as small_config

    junior = FillShareTarget(PlannerScope.of(job_level=[1]), {"external": 1.0})
    with pytest.raises(ConfigError, match=r"fill_shares: rules 1 .* are ambiguous"):
        config(fill_shares=(FillShareTarget(FIRST, {"external": 1.0}), junior))
    with pytest.raises(ConfigError, match="fill_shares: rule 1: scope"):
        config(
            fill_shares=(
                FillShareTarget(PlannerScope.of(cost_center=["CC-09"]), {"external": 1.0}),
            )
        )
    target = FillShareTarget(FIRST, {"external": 0.6, "conversion": 0.4})
    run = small_config(models=_saved(tmp_path), fill_shares=(target,))
    record = json.loads(json.dumps(to_record(run)))
    assert record["fill_shares"] == [target.to_dict()]
    rebuilt = from_record(record, schema=run.schema, models_root=tmp_path, awards=run.awards)
    assert rebuilt.fill_shares == (target,)
    record["fill_shares"] = [{"scope": {}, "shares": {"referral": 1.0}}]
    with pytest.raises(ConfigError, match="no fill channels"):
        from_record(record, schema=run.schema, models_root=tmp_path, awards=run.awards)


@pytest.mark.slow
def test_a_runs_fills_land_on_the_target_mix() -> None:
    target = FillShareTarget(FIRST, {"external": 0.4, "conversion": 0.6})
    no_transfers = dataclasses.replace(CHANNELS, transfer_fill=None)
    run = config(models=no_transfers, fill_shares=(target,), num_trajectories=600)
    result = simulate(run, initial_state())
    external = float(result.flow("F_ext")[:, :, 0].sum())
    conversion = float(result.flow("F_conv")[:, :, 0].sum())
    share = conversion / (external + conversion)
    se = np.sqrt(share * (1 - share) / (external + conversion))
    stat_check(share, 0.6, se, label="conversion share in CC-01")
    plain = simulate(dataclasses.replace(run, fill_shares=()), initial_state())
    untouched = float(plain.flow("F_conv")[:, :, 0].sum()) / float(
        plain.flow("F_ext")[:, :, 0].sum() + plain.flow("F_conv")[:, :, 0].sum()
    )
    assert untouched == pytest.approx(0.25, abs=0.03)  # the models' own 0.1 against 0.3
