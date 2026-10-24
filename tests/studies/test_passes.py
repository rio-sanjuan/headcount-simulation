"""Tests for the harness's passes: window closure, deferred variants, recorded skips, pass A and
pass B, metadata, persistence, and the timing ledger."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hcsim.studies.harness import Closure, Deferred, deferral, run_study, save, window_closure
from hcsim.studies.profile import Arm, ProfileError, StudyProfile

D = dt.date


def profile(**fields: Any) -> StudyProfile:
    base = {
        "name": "passes",
        "seed": 3,
        "trajectories": 2,
        "months": 3,
        "mode": "unguided",
        "scope": "hcf",
    }
    return StudyProfile.from_mapping(base | fields)


def test_a_window_is_open_partly_closed_or_closed_by_its_actuals() -> None:
    through = D(2025, 12, 31)
    assert window_closure(D(2026, 1, 1), 6, through) is Closure.OPEN
    assert window_closure(D(2025, 10, 1), 6, through) is Closure.PARTIAL
    assert window_closure(D(2025, 7, 1), 6, through) is Closure.CLOSED
    assert window_closure(D(2025, 12, 1), 1, through) is Closure.CLOSED


def test_an_arm_that_reads_actuals_waits_for_a_closed_window() -> None:
    arm = Arm(name="history", reference=True)
    waiting = deferral(arm, "replay", Closure.PARTIAL)
    assert waiting == Deferred(
        "history",
        "its mode, replay, reads the window's actuals, which is partially_closed; it runs once "
        "the window is wholly closed",
    )
    assert deferral(arm, "replay", Closure.CLOSED) is None
    assert deferral(arm, "guided_actuals", Closure.OPEN) is not None
    assert deferral(arm, "unguided", Closure.OPEN) is None
    assert deferral(arm, "guided_forecast", Closure.OPEN) is None


SCREENED: dict[str, Any] = {
    "evaluators": ["invariants", "summary", "plan_alignment"],
    "arms": [{"name": "rate", "null": True}],
    "arm_groups": [{"name": "gain", "vary": {"requisitions.supply_gain": [0.0, 0.5, 1.0]}}],
    "screen": {"trajectories": 1, "keep": 1, "rank_by": "summary.headcount_end_mean"},
}


def test_pass_a_screens_the_compared_arms_and_pass_b_runs_the_best(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    world, _ = inputs
    study = run_study(profile(**SCREENED), world)
    assert study.closure is Closure.OPEN  # the run starts the month after the world's last
    # Pass A ran every compared arm once, best first; pass B, the null arm and the best.
    ranked = [r.metrics for r in study.screened]
    values = [next(m.value for m in ms if m.name == "headcount_end_mean") for ms in ranked]
    assert values == sorted(values)
    assert [r.pass_name for r in study.screened] == ["A", "A", "A"]
    assert all(r.config.num_trajectories == 1 for r in study.screened)
    kept = study.screened[0].arm.name
    assert [(a.arm.name, a.pass_name) for a in study.arms] == [("rate", "B"), (kept, "B")]
    assert all(a.config.num_trajectories == 2 for a in study.arms)
    # The two it didn't keep are skipped with their ranks, and so is plan alignment: no plan.
    reasons = {(s.what, s.name): s.reason for s in study.skips}
    assert reasons[("evaluator", "plan_alignment")] == "the study has no plan to score against"
    dropped = [s for s in study.skips if s.what == "arm"]
    assert [s.reason for s in dropped] == [
        "screened out in pass A: ranked 2 of 3 by summary.headcount_end_mean",
        "screened out in pass A: ranked 3 of 3 by summary.headcount_end_mean",
    ]
    assert not any(m.evaluator == "plan_alignment" for m in study.metrics)
    # The timing ledger holds preflight, and each arm's simulation and evaluation per pass.
    phases = [(t.phase, t.arm) for t in study.timing]
    assert phases[0] == ("preflight", "")
    assert ("simulate A", kept) in phases
    assert ("evaluate B", "rate") in phases
    assert all(t.seconds >= 0 for t in study.timing)
    # Metadata and persistence.
    meta = study.metadata()
    assert meta["spec"] == study.profile.spec().digest()
    assert meta["closure"] == "wholly_open"
    assert [a["pass"] for a in meta["arms"]] == ["A", "A", "A", "B", "B"]
    assert len(meta["skips"]) == 3
    written = save(study, tmp_path / "out")
    assert json.loads((written / "study.json").read_text())["spec"] == meta["spec"]
    rows = pd.read_csv(written / "metrics.csv")
    assert set(rows["pass"]) == {"A", "B"}
    assert len(rows) == sum(len(r.metrics) for r in (*study.screened, *study.arms))


def test_a_study_with_no_arms_saves_its_own_run(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    world, _ = inputs
    study = run_study(profile(), world)
    rows = pd.read_csv(save(study, tmp_path / "plain") / "metrics.csv")
    assert set(rows["arm"]) == {"study"}
    assert set(rows["pass"]) == {"B"}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (SCREENED | {"evaluators": ["invariants"]}, "the screen ranks by summary, which the study"),
        (SCREENED | {"arm_groups": []}, "a screen needs compared arms to rank"),
        (
            SCREENED | {"screen": {"trajectories": 2, "keep": 1, "rank_by": "summary.x"}},
            "fewer trajectories",
        ),
        (SCREENED | {"screen": {"trajectories": 1, "keep": 1, "rank_by": "nonsense"}}, "rank_by"),
    ],
)
def test_a_bad_screen_is_refused(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ProfileError) as caught:
        profile(**fields)
    assert message in str(caught.value)
