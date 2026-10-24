"""Tests for the evaluators: the registry, what each needs, and each one's metrics."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import re
from pathlib import Path

import duckdb
import numpy as np
import pytest

from hcsim.engine.seeding import schema_from_world, seed_state
from hcsim.studies.actuals import PARTS, Actuals, ActualsError, month_end_headcount, resolve
from hcsim.studies.evaluators import (
    EVALUATORS,
    OFFLINE_EVALUATORS,
    SKIP_REASONS,
    WHOLE_WORLD,
    Need,
    Scoring,
    run_rates,
)
from hcsim.studies.harness import StudyRun, run_study
from hcsim.studies.inputs import InputError, fit_models
from hcsim.studies.kinds import KINDS
from hcsim.studies.profile import ProfileError, StudyProfile
from tests.support.stat_check import stat_check

PAST = dt.date(2025, 7, 1)
MONTHS = 4
#: Every simulation evaluator but the ones that need a plan.
SCORED = ["invariants", "fidelity", "domain", "dispersion", "sbc", "award_grants", "award_sizing"]
#: The world's awards follow their events by up to two months.
LAG = {"new_hire": 2, "promotion": 2, "transfer": 2}
#: The head events: history counts each exactly as a replay does.
HEAD_EVENTS = (
    "attrition",
    "hires",
    "internal_starts",
    "conversions_out",
    "promotions",
    "requisitions_closed_unfilled",
)


def profile(**fields: object) -> StudyProfile:
    return StudyProfile.from_mapping(
        {
            "name": "evaluators",
            "seed": 7,
            "trajectories": 3,
            "months": MONTHS,
            "run_date": PAST.isoformat(),
            "mode": "unguided",
            "evaluators": SCORED,
        }
        | fields
    )


def metrics(run: StudyRun, evaluator: str) -> dict[str, float]:
    return {m.name: m.value for m in run.metrics if m.evaluator == evaluator}


@pytest.fixture(scope="module")
def replayed(inputs: tuple[Path, Path]) -> StudyRun:
    return run_study(profile(mode="replay", replay_lag=LAG), inputs[0])


@pytest.fixture(scope="module")
def unguided(inputs: tuple[Path, Path]) -> StudyRun:
    return run_study(profile(), inputs[0])


# --- the registry -----------------------------------------------------------------------------


def test_the_registry_holds_every_evaluator_with_what_it_reads() -> None:
    nine = {
        "invariants",
        "fidelity",
        "domain",
        "offline_domain",
        "dispersion",
        "plan_alignment",
        "sbc",
        "award_grants",
        "award_sizing",
    }
    assert nine <= set(EVALUATORS) | set(OFFLINE_EVALUATORS)
    assert not set(EVALUATORS) & set(OFFLINE_EVALUATORS)
    for e in (*EVALUATORS.values(), *OFFLINE_EVALUATORS.values()):
        assert set(e.kinds) <= set(KINDS), e.name
        assert set(e.actuals) <= set(PARTS), e.name
        assert bool(e.actuals) == (Need.ACTUALS in e.needs), e.name


def test_an_evaluator_is_skipped_without_what_it_needs() -> None:
    sbc = EVALUATORS["sbc"]
    can = {"plan": False, "sbc": True, "trajectories": 3, "closed": True, "subset": False}
    assert sbc.skipped(**can) is None  # type: ignore[arg-type]
    cases = [
        ({"sbc": False}, SKIP_REASONS[Need.SBC]),
        ({"closed": False}, SKIP_REASONS[Need.ACTUALS]),
        ({"subset": True}, WHOLE_WORLD),
    ]
    for change, reason in cases:
        assert sbc.skipped(**(can | change)) == reason  # type: ignore[arg-type]
    spread = EVALUATORS["dispersion"].skipped(**(can | {"trajectories": 1}))  # type: ignore[arg-type]
    assert spread == SKIP_REASONS[Need.SPREAD]
    assert EVALUATORS["plan_alignment"].skipped(**can) == SKIP_REASONS[Need.PLAN]  # type: ignore[arg-type]


def test_a_study_records_each_evaluator_it_skips(inputs: tuple[Path, Path]) -> None:
    run = run_study(profile(scope="hcf", run_date="2025-11-01", trajectories=1), inputs[0])
    reasons = {s.name: s.reason for s in run.skips}
    assert reasons == {
        "fidelity": SKIP_REASONS[Need.ACTUALS],
        "dispersion": SKIP_REASONS[Need.SPREAD],
        "sbc": SKIP_REASONS[Need.SBC],
        "award_grants": SKIP_REASONS[Need.SBC],
        "award_sizing": SKIP_REASONS[Need.SBC],
    }
    assert run.closure.value == "partially_closed"
    assert {m.evaluator for m in run.metrics} == {"invariants", "domain"}
    subset = profile(scope="hcf", evaluators=["fidelity"], subset={"cost_center": ["CC-01"]})
    assert [s.reason for s in run_study(subset, inputs[0]).skips] == [WHOLE_WORLD]


def test_scoring_refuses_what_it_doesnt_hold(unguided: StudyRun) -> None:
    offline = Scoring(None, unguided.fitted)
    with pytest.raises(ValueError, match="no simulated results"):
        _ = offline.run
    with pytest.raises(ValueError, match="no actuals"):
        offline.actual("facts")
    with pytest.raises(ActualsError, match="don't hold sbc"):
        Actuals(PAST, MONTHS).need("sbc")


# --- the actuals ------------------------------------------------------------------------------


def test_the_actuals_resolve_only_what_is_asked(inputs: tuple[Path, Path]) -> None:
    with duckdb.connect(str(inputs[0]), read_only=True) as con:
        schema = schema_from_world(con)
        actuals = resolve(con, schema, PAST, MONTHS, ["headcount", "refit"])
        assert (actuals.facts, actuals.sbc, actuals.awards) == (None, None, None)
        assert actuals.refit is not None
        assert actuals.refit.window == (dt.date(2025, 6, 30), dt.date(2025, 10, 31))
        heads = month_end_headcount(con, schema, PAST, MONTHS)
        for k, first in enumerate((dt.date(2025, 8, 1), dt.date(2025, 9, 1))):
            assert np.array_equal(heads[k], seed_state(con, schema, first)[0].headcount)
        with pytest.raises(ActualsError, match=re.escape("no parts of the actuals ['plans']")):
            resolve(con, schema, PAST, MONTHS, ["plans"])
        with pytest.raises(InputError, match="at least a month"):
            fit_models(con, schema, PAST, window_months=0)


# --- fidelity ---------------------------------------------------------------------------------


def test_a_replay_is_faithful_to_history(replayed: StudyRun) -> None:
    fidelity = metrics(replayed, "fidelity")
    for event in HEAD_EVENTS:
        assert fidelity[f"{event}_wape"] == 0, event
        assert fidelity[f"{event}_bias"] == 0, event
    assert fidelity["headcount_wape"] == 0
    assert fidelity["headcount_end_bias"] == 0
    assert "requisitions_opened_wape" in fidelity  # history's snapshots can see fewer


def test_an_unguided_run_is_scored_against_history(unguided: StudyRun) -> None:
    fidelity = metrics(unguided, "fidelity")
    assert 0 < fidelity["headcount_wape"] < 0.2
    assert fidelity["attrition_wape"] > 0
    assert all(math.isfinite(v) for v in fidelity.values())


# --- domain and offline_domain ----------------------------------------------------------------


def test_an_unguided_run_realizes_its_models_rates(unguided: StudyRun) -> None:
    result = unguided.simulated
    domain = metrics(unguided, "domain")
    fitted = unguided.fitted.rates
    rates = run_rates(result)
    assert domain["attrition_rate"] == rates["attrition"]
    assert domain["attrition_rate_ratio"] == pytest.approx(rates["attrition"] / fitted["attrition"])
    at_risk = float(result.headcount[:, :-1, :].sum())
    p = fitted["attrition"]
    observed = float(result.flow("A").sum())
    stat_check(observed, at_risk * p, math.sqrt(at_risk * p * (1 - p)), label="attrition")
    cells = float(np.prod(result.flow("G").shape))
    growth = fitted["growth"]
    stat_check(float(result.flow("G").sum()), cells * growth, math.sqrt(cells * growth))


def test_an_offline_study_scores_its_rates_out_of_time(inputs: tuple[Path, Path]) -> None:
    study = profile(backend="offline", scope="hcf", evaluators=["offline_domain", "fitted_rates"])
    run = run_study(study, inputs[0])
    offline = metrics(run, "offline_domain")
    with duckdb.connect(str(inputs[0]), read_only=True) as con:
        after = dt.date(2025, 11, 1)
        refit = fit_models(con, schema_from_world(con), after, window_months=MONTHS).rates
    for domain, rate in refit.items():
        assert offline[f"{domain}_rate"] == rate
        assert offline[f"{domain}_rate_ratio"] == pytest.approx(rate / run.fitted.rates[domain])
    assert metrics(run, "fitted_rates")["attrition_rate"] == run.fitted.rates["attrition"]
    with pytest.raises(ProfileError, match=re.escape("which ['offline_domain'] aren't")):
        profile(evaluators=["offline_domain"])


# --- dispersion -------------------------------------------------------------------------------


def test_dispersion_measures_how_trajectories_spread(
    unguided: StudyRun, replayed: StudyRun
) -> None:
    spread = metrics(unguided, "dispersion")
    assert spread["headcount_end_cv"] > 0
    assert spread["headcount_end_interval"] > 0
    assert spread["attrition_cv"] > 0
    end = unguided.simulated.headcount[:, -1, :].sum(axis=1)
    assert spread["headcount_end_cv"] == pytest.approx(end.std(ddof=1) / end.mean())
    # Replay is history, whatever the trajectory.
    assert all(v == 0 for v in metrics(replayed, "dispersion").values())


def test_dispersion_reports_nothing_for_one_trajectory(unguided: StudyRun) -> None:
    one = dataclasses.replace(unguided.simulated, trajectories=unguided.simulated.trajectories[:1])
    assert EVALUATORS["dispersion"].score(Scoring(one, unguided.fitted)) == []


# --- sbc --------------------------------------------------------------------------------------


def test_a_replays_sbc_ties_to_the_reference_ledger(replayed: StudyRun) -> None:
    sbc = metrics(replayed, "sbc")
    assert sbc["monthly_wape"] < 0.03
    assert abs(sbc["total_bias"]) < 0.03
    assert abs(sbc["seed_bias"]) < 0.02  # the seed is history's own tranches
    assert sbc["cost_center_wape"] >= sbc["monthly_wape"]


def test_an_unguided_runs_sbc_is_scored(unguided: StudyRun) -> None:
    sbc = metrics(unguided, "sbc")
    assert {"monthly_wape", "cost_center_wape", "total_bias", "seed_bias", "sim_bias"} <= set(sbc)
    assert sbc["monthly_wape"] > 0


# --- award_grants and award_sizing ------------------------------------------------------------


def test_a_replay_grants_historys_awards(replayed: StudyRun) -> None:
    grants = metrics(replayed, "award_grants")
    assert grants["promotion_grants_ratio"] == 1
    assert grants["new_hire_grants"] > 0
    assert grants["unsized_grants"] == 0


def test_a_replay_without_its_lag_grants_no_new_hire_award(inputs: tuple[Path, Path]) -> None:
    run = run_study(profile(mode="replay", evaluators=["award_grants"]), inputs[0])
    grants = metrics(run, "award_grants")
    assert grants["new_hire_grants"] == 0
    assert grants["ungranted_grants"] > 0
    with pytest.raises(ProfileError, match="only a replay at hcf\\+sbc reads"):
        profile(replay_lag=LAG)
    with pytest.raises(ProfileError, match=re.escape("no award types ['bonus']")):
        profile(mode="replay", replay_lag={"bonus": 1})
    with pytest.raises(ProfileError, match="from 0"):
        profile(mode="replay", replay_lag={"new_hire": -1})


def test_award_sizing_scores_the_sized_grants(unguided: StudyRun, replayed: StudyRun) -> None:
    sizing = metrics(unguided, "award_sizing")
    assert sizing["new_hire_shares_mean"] > 0
    assert sizing["new_hire_shares_ratio"] > 0
    replay_sizing = metrics(replayed, "award_sizing")
    assert "promotion_shares_mean" not in replay_sizing  # every promotion took history's award
