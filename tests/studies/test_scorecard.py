"""Tests for the versioned scorecard: its schema, comparisons with the null arm, rankings, and the
run record's keys."""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcsim.studies import scorecard
from hcsim.studies.harness import StudyRun, run_study
from hcsim.studies.profile import ProfileError, Ranking, StudyProfile
from hcsim.studies.scorecard import (
    COLUMNS,
    PROVENANCE_COLUMNS,
    RUN_RECORD_KEYS,
    SCORECARD_VERSION,
    ScorecardError,
    check,
    check_record,
    rankings,
    run_record,
)

ARMS = [
    {"name": "base", "null": True},
    {"name": "low", "overrides": {"promotion.gate_probability": 0.2}},
    {"name": "high", "overrides": {"promotion.gate_probability": 0.9}},
    {"name": "history", "reference": True, "mode": "guided_forecast"},
]
METRIC = "summary.headcount_end_mean"


def profile(**fields: object) -> StudyProfile:
    return StudyProfile.from_mapping(
        {
            "name": "scorecard",
            "seed": 7,
            "trajectories": 3,
            "months": 3,
            "mode": "unguided",
            "scope": "hcf",
            "arms": ARMS,
            "rankings": [{"metric": METRIC, "better": "higher"}],
        }
        | fields
    )


@pytest.fixture(scope="module")
def study(inputs: tuple[Path, Path]) -> StudyRun:
    return run_study(profile(), *inputs)


@pytest.fixture(scope="module")
def card(study: StudyRun) -> pd.DataFrame:
    return scorecard.scorecard(study)


def column(card: pd.DataFrame, name: str, metric: str = "headcount_end_mean") -> dict[str, float]:
    """`name` of `metric` by arm."""
    mine = card[card["metric"] == metric]
    return {
        str(a): float(v) if pd.notna(v) else math.nan
        for a, v in zip(mine["arm"], mine[name], strict=True)
    }


def test_a_row_per_pass_arm_evaluator_and_metric(study: StudyRun, card: pd.DataFrame) -> None:
    assert list(card.columns) == COLUMNS
    assert check(card) == []
    assert (card["scorecard_version"] == SCORECARD_VERSION).all()
    per_arm = sum(len(a.metrics) for a in study.arms)
    assert len(card) == per_arm
    assert not card.duplicated(["pass", "arm", "evaluator", "metric"]).any()
    roles = card.groupby("arm")["role"].first().to_dict()
    assert roles == {"base": "null", "low": "compared", "high": "compared", "history": "reference"}


def test_every_row_carries_its_provenance(study: StudyRun, card: pd.DataFrame) -> None:
    for column in PROVENANCE_COLUMNS:
        assert (card[column] == study.provenance[column]).all(), column
    assert card["spec"].str.startswith("sp1-").all()
    assert (card["backend"] == "simulation").all()


def test_each_arm_is_compared_with_the_null_arm(card: pd.DataFrame) -> None:
    value, vs_null = column(card, "value"), column(card, "vs_null")
    assert math.isnan(vs_null["base"])
    for arm in ("low", "high", "history"):
        assert vs_null[arm] == pytest.approx(value[arm] - value["base"])


def test_the_null_and_compared_arms_rank_by_each_ranking(
    study: StudyRun, card: pd.DataFrame
) -> None:
    value, rank = column(card, "value"), column(card, "rank")
    order = sorted(["base", "low", "high"], key=lambda a: -value[a])
    assert [rank[a] for a in order] == [1, 2, 3]
    assert math.isnan(rank["history"])  # a reference arm is never ranked
    assert card.loc[card["metric"] != "headcount_end_mean", "rank"].isna().all()
    (ranking,) = rankings(study).values()
    assert [r["arm"] for r in ranking] == order
    assert [r["rank"] for r in ranking] == [1, 2, 3]


def test_a_ranking_orders_by_its_direction() -> None:
    values = [3.0, -1.0, 2.0]
    for ranking, first in (
        (Ranking(metric="e.m", better="lower"), -1.0),
        (Ranking(metric="e.m", better="higher"), 3.0),
        (Ranking(metric="e.m", better="nearer_zero"), -1.0),
    ):
        assert min(values, key=ranking.key) == first
    assert Ranking(metric="e.m", better="nearer_zero").key(-0.5) == 0.5


def test_a_screen_ranks_by_its_metric_and_its_rows_say_pass_a(inputs: tuple[Path, Path]) -> None:
    screened = profile(
        rankings=[],
        screen={"trajectories": 1, "keep": 1, "rank_by": "summary.attrition_mean"},
        arms=ARMS[:3],
    )
    assert [r.metric for r in screened.ranked_by()] == ["summary.attrition_mean"]
    run = run_study(screened, inputs[0])
    card = scorecard.scorecard(run)
    passes = card.groupby("pass")["arm"].unique().to_dict()
    assert sorted(passes["A"]) == ["high", "low"]
    assert card.loc[card["pass"] == "A", "rank"].isna().all()
    assert card.loc[card["pass"] == "A", "vs_null"].isna().all()
    assert check(card) == []


def test_a_study_without_arms_is_one_arm(inputs: tuple[Path, Path]) -> None:
    run = run_study(profile(arms=[], rankings=[]), inputs[0])
    card = scorecard.scorecard(run)
    assert set(card["arm"]) == {"study"}
    assert set(card["role"]) == {"study"}
    assert card["vs_null"].isna().all()
    assert card["rank"].isna().all()


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"rankings": [{"metric": "fidelity.headcount_wape"}]}, "read evaluators the study"),
        ({"arms": [], "rankings": [{"metric": METRIC}]}, "it has none"),
        ({"rankings": [{"metric": METRIC, "better": "best"}]}, "rankings.0.better"),
    ],
)
def test_a_profiles_rankings_are_checked(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ProfileError, match=re.escape(message)):
        profile(**fields)


# --- the schema -------------------------------------------------------------------------------


def test_the_schema_refuses_a_card_that_doesnt_fit(card: pd.DataFrame) -> None:
    assert check(card.drop(columns="rank")) == [
        f"a version {SCORECARD_VERSION} scorecard has the columns {COLUMNS}"
    ]
    assert check(card.assign(scorecard_version=1)) == [
        f"scorecard versions [1]; this is {SCORECARD_VERSION}"
    ]
    assert check(card.assign(role="judge")) == ["no roles ['judge']"]
    assert check(card.assign(spec="")) == ["rows without their provenance's spec"]


def test_a_written_card_loads_back_typed(study: StudyRun, tmp_path: Path) -> None:
    path = scorecard.write(study, tmp_path / "out", 1.5)
    loaded = scorecard.load(path)
    assert str(loaded["rank"].dtype) == "Int64"
    assert str(loaded["passed"].dtype) == "bool"
    original = scorecard.scorecard(study)
    assert loaded["arm"].tolist() == original["arm"].tolist()
    assert loaded["role"].tolist() == original["role"].tolist()  # "null" reads as itself
    assert np.allclose(loaded["value"], original["value"], rtol=1e-9)
    bad = tmp_path / "bad.csv"
    original.drop(columns="role").to_csv(bad, index=False)
    with pytest.raises(ScorecardError, match="has the columns"):
        scorecard.load(bad)


# --- the run record ---------------------------------------------------------------------------


def test_the_run_record_holds_exactly_its_keys(study: StudyRun, tmp_path: Path) -> None:
    record = run_record(study, 2.0)
    assert tuple(record) == RUN_RECORD_KEYS
    assert record["verdict"] == study.verdict.value
    assert {a["role"] for a in record["arms"]} == {"null", "compared", "reference"}
    assert list(record["rankings"]) == [METRIC]
    assert record["spec"] == study.profile.spec().digest()
    scorecard.write(study, tmp_path, 2.0)
    written = json.loads((tmp_path / scorecard.RUN_RECORD_FILE).read_text())
    assert sorted(written) == sorted(RUN_RECORD_KEYS)
    assert check_record(written) == []
    extra = written | {"surprise": 1}
    assert check_record(extra) == ["the run record has no keys ['surprise']"]
    lacking = {k: v for k, v in written.items() if k != "gates"}
    assert check_record(lacking) == ["the run record lacks ['gates']"]
    assert check_record(written | {"scorecard_version": 1}) == [
        f"a run record is version {SCORECARD_VERSION}"
    ]


def test_writing_refuses_a_card_that_doesnt_fit(study: StudyRun, tmp_path: Path) -> None:
    broken = dataclasses.replace(study, provenance=study.provenance | {"spec": ""})
    with pytest.raises(ScorecardError, match="provenance's spec"):
        scorecard.write(broken, tmp_path, 1.0)
