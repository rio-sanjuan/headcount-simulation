"""Tests for the harness's preflight: every problem at once, the subset lowered, and inputs
resolved once for every arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hcsim.studies import harness
from hcsim.studies.harness import StudyError, preflight, run_study
from hcsim.studies.profile import StudyProfile


def profile(**fields: Any) -> StudyProfile:
    base = {
        "name": "pre",
        "seed": 3,
        "trajectories": 1,
        "months": 3,
        "mode": "unguided",
        "scope": "hcf",
    }
    return StudyProfile.from_mapping(base | fields)


GAINS = {
    "arms": [{"name": "rate", "null": True}],
    "arm_groups": [{"name": "gain", "vary": {"requisitions.supply_gain": [0.5, 1.0]}}],
}


def test_preflight_reports_every_problem_at_once(tmp_path: Path, inputs: tuple[Path, Path]) -> None:
    # A guided study with no guidance and no world: both are named.
    found = preflight(profile(mode="guided_forecast"), tmp_path / "nowhere.duckdb")
    assert found.problems == (
        "mode 'guided_forecast' needs a guidance artifact",
        f"no world at {tmp_path / 'nowhere.duckdb'}",
    )
    world, _ = inputs
    # Every arm that doesn't validate is named, each with its problem.
    bad = preflight(
        profile(
            arm_groups=[{"name": "gain", "vary": {"requisitions.supply_gain": [5.0, 7.0]}}],
            arms=[{"name": "rate", "null": True}],
        ),
        world,
    )
    assert bad.problems == (
        "arm gain[requisitions.supply_gain=5.0]: requisitions_supply_gain must lie in [0, 1]",
        "arm gain[requisitions.supply_gain=7.0]: requisitions_supply_gain must lie in [0, 1]",
    )
    with pytest.raises(StudyError) as caught:
        run_study(profile(mode="guided_forecast"), tmp_path / "nowhere.duckdb")
    assert len(caught.value.problems) == 2
    unreadable = preflight(profile(mode="guided_forecast"), world, tmp_path)
    assert unreadable.problems[0].startswith(f"can't load the guidance artifact at {tmp_path}")
    assert (
        preflight(profile(subset={"cost_center": ["CC-NOPE"]}), world)
        .problems[0]
        .startswith("subset: ")
    )


def test_every_arm_reads_one_set_of_inputs(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    world, _ = inputs
    calls = {"fit": 0, "seed": 0}
    fit, seed = harness.fit_models, harness.seed_for

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        calls["fit"] += 1
        return fit(*args, **kwargs)

    def counted_seed(*args: Any, **kwargs: Any) -> Any:
        calls["seed"] += 1
        return seed(*args, **kwargs)

    monkeypatch.setattr(harness, "fit_models", counted_fit)
    monkeypatch.setattr(harness, "seed_for", counted_seed)
    study = run_study(profile(**GAINS), world)
    assert calls == {"fit": 1, "seed": 1}
    assert [a.arm.name for a in study.arms] == [
        "rate",
        "gain[requisitions.supply_gain=0.5]",
        "gain[requisitions.supply_gain=1.0]",
    ]
    gains = [a.config.requisitions_supply_gain for a in study.arms]
    assert gains == [1.0, 0.5, 1.0]  # the null arm keeps the default
    assert study.config is study.arms[0].config  # the study's own run is its null arm
    assert all(a.config.models is study.arms[0].config.models for a in study.arms)
    assert study.provenance["input_kinds"] == "seed_state,fitted_models"
    assert run_study(profile(), world).arms == ()  # no arms: the profile runs as itself


def test_a_subset_is_lowered_and_seeds_only_its_part(inputs: tuple[Path, Path]) -> None:
    world, _ = inputs
    whole = preflight(profile(), world)
    assert whole.inputs is not None
    labels = whole.inputs.schema.vocabulary("cost_center").labels
    staffed = [labels[c] for c, n in enumerate(whole.inputs.state.headcount) if n > 0]
    part = preflight(profile(subset={"cost_center": staffed[:1]}), world)
    assert part.inputs is not None
    assert part.inputs.subset is not None
    seeded = part.inputs.state.headcount
    assert seeded[labels.index(staffed[0])] > 0
    assert int(np.count_nonzero(seeded)) == 1


def test_a_reference_arm_runs_under_its_own_mode(inputs: tuple[Path, Path]) -> None:
    world, guidance = inputs
    study = run_study(
        profile(
            mode="guided_forecast",
            arms=[
                {"name": "guided", "null": True},
                {"name": "free", "reference": True, "mode": "unguided"},
            ],
        ),
        world,
        guidance,
    )
    modes = {a.arm.name: (a.config.run_mode, a.config.guidance is None) for a in study.arms}
    assert modes == {"guided": ("guided_forecast", False), "free": ("unguided", True)}


def test_a_bad_subset_or_a_missing_plan_is_named(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    from hcsim.studies.profile import ProfileError

    with pytest.raises(ProfileError, match="a subset can't constrain 'weather'"):
        profile(subset={"weather": ["rain"]})
    world, _ = inputs
    missing = preflight(profile(plan={"path": str(tmp_path / "no-plan")}), world)
    assert missing.problems[0].startswith("can't build the plan: no alignment target at")
