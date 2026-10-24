"""Tests for study profiles: the schema, arms and arm groups, the frozen specification, and the
round trip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from hcsim.studies.profile import Arm, ProfileError, StudyProfile

BASE: dict[str, Any] = {"name": "gains", "seed": 3}


def profile(**fields: Any) -> StudyProfile:
    return StudyProfile.from_mapping(BASE | fields)


ARMS: dict[str, Any] = {
    "arms": [
        {"name": "rate", "null": True, "overrides": {"alignment.method": "none"}},
        {"name": "replayed", "reference": True, "mode": "unguided"},
    ],
    "arm_groups": [
        {
            "name": "gain",
            "vary": {
                "requisitions.supply_gain": [0.5, 1.0],
                "requisitions.supply_rule": ["strength", "soft_bridge"],
            },
            "overrides": {"alignment.method": "supply"},
        }
    ],
}


def test_a_group_expands_into_one_arm_per_combination() -> None:
    arms = profile(**ARMS).expanded_arms()
    assert [a.name for a in arms] == [
        "rate",
        "replayed",
        'gain[requisitions.supply_gain=0.5,requisitions.supply_rule="strength"]',
        'gain[requisitions.supply_gain=0.5,requisitions.supply_rule="soft_bridge"]',
        'gain[requisitions.supply_gain=1.0,requisitions.supply_rule="strength"]',
        'gain[requisitions.supply_gain=1.0,requisitions.supply_rule="soft_bridge"]',
    ]
    assert arms[3].overrides == {
        "alignment.method": "supply",
        "requisitions.supply_gain": 0.5,
        "requisitions.supply_rule": "soft_bridge",
    }


def test_the_spec_is_frozen_and_names_its_null_and_compared_arms() -> None:
    spec = profile(**ARMS).spec()
    assert spec.null_arm is not None
    assert spec.null_arm.name == "rate"
    assert len(spec.compared) == 4  # the group's, not the null or the reference arm
    assert spec.digest().startswith("sp1-")
    assert spec.digest() == profile(**ARMS).spec().digest()
    other = profile(**(ARMS | {"arm_groups": []}))
    assert other.spec().digest() != spec.digest()
    with pytest.raises(AttributeError):
        spec.arms = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="frozen"):
        spec.profile.seed = 4  # type: ignore[misc]


def test_a_profile_round_trips_through_plain_data_and_yaml(tmp_path: Path) -> None:
    original = profile(**ARMS, run_date="2026-01-01", capacity_caps={"CC-01": 30})
    assert StudyProfile.from_mapping(original.to_mapping()) == original
    (tmp_path / "study.yaml").write_text(yaml.safe_dump(original.to_mapping()))
    assert StudyProfile.load(tmp_path / "study.yaml") == original
    assert original.spec().digest() == StudyProfile.load(tmp_path / "study.yaml").spec().digest()


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"colour": "blue"}, "colour: Extra inputs are not permitted"),
        ({"evaluators": ["vibes"]}, "no evaluators ['vibes']"),
        (
            {"arms": [{"name": "a", "null": True, "overrides": {"supply.gain": 1}}]},
            "no settings ['supply.gain']; a setting is a group.key path",
        ),
        ({"arms": [{"name": "a"}]}, "exactly one null arm, not 0"),
        (
            {"arms": [{"name": "a", "null": True}, {"name": "b", "null": True}]},
            "exactly one null arm, not 2",
        ),
        (
            {"arms": [{"name": "a", "null": True}, {"name": "a", "reference": True}]},
            "arm names repeat: ['a']",
        ),
        ({"arms": [{"name": "a", "null": True, "reference": True}]}, "both the null arm"),
        ({"arms": [{"name": "a", "null": True, "mode": "unguided"}]}, "only a reference arm"),
        (
            {"arm_groups": [{"name": "g", "vary": {"requisitions.supply_gain": []}}]},
            "varies ['requisitions.supply_gain'] over no values",
        ),
        (
            {
                "arm_groups": [
                    {
                        "name": "g",
                        "vary": {"requisitions.supply_gain": [1.0]},
                        "overrides": {"requisitions.supply_gain": 0.5},
                    }
                ]
            },
            "both varies and fixes ['requisitions.supply_gain']",
        ),
        ({"arms": [{"name": "a", "null": True, "bogus": 1}]}, "Extra inputs are not permitted"),
    ],
)
def test_a_bad_profile_is_refused_by_name(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ProfileError) as caught:
        profile(**fields)
    assert message in str(caught.value)


def test_a_profile_without_arms_runs_as_itself() -> None:
    plain = profile()
    assert plain.expanded_arms() == ()
    assert plain.spec().null_arm is None
    assert plain.evaluators == ["invariants", "summary"]
    assert Arm(name="x").overrides == {}


def test_a_profile_declares_fill_share_targets() -> None:
    from hcsim.engine.planner import FillShareTarget, PlannerScope

    shares = [{"scope": {"cost_center": ["CC-01"]}, "shares": {"external": 0.4, "conversion": 0.6}}]
    declared = profile(fill_shares=shares)
    (target,) = declared.fill_share_targets()
    assert target == FillShareTarget(
        PlannerScope.of(cost_center=["CC-01"]), {"external": 0.4, "conversion": 0.6}
    )
    assert StudyProfile.from_mapping(declared.to_mapping()) == declared
    with pytest.raises(ProfileError, match=r"shares sum to one, not 0\.5"):
        profile(fill_shares=[{"shares": {"external": 0.5}}])
    with pytest.raises(ProfileError, match="no fill channels"):
        profile(fill_shares=[{"shares": {"referral": 1.0}}])
    assert profile().fill_share_targets() == ()


# --- the plan, the alignment method, tilts, and settings ------------------------------------------


def test_a_profile_declares_a_plan_a_method_tilts_and_settings() -> None:
    import datetime as dt

    from hcsim.engine.alignment import AlignmentMethod
    from hcsim.engine.planner import PlannerScope, ProbabilityTilt

    declared = profile(
        plan={"synthetic": {"shape": "relative", "dates": ["2026-06"], "factors": [1.1]}},
        alignment={"method": "hybrid", "overrides": {"rate": "none"}},
        tilts=[{"domain": "attrition", "scope": {"cost_center": ["CC-01"]}, "multiplier": 2.0}],
        settings={"requisitions.supply_rule": "soft_bridge"},
        arms=[{"name": "rate", "null": True}],
    )
    assert declared.alignment.choice().resolve() is AlignmentMethod.HYBRID
    assert declared.alignment.choice().resolve("rate") is AlignmentMethod.NONE
    assert declared.probability_tilts() == (
        ProbabilityTilt("attrition", PlannerScope.of(cost_center=["CC-01"]), 2.0),
    )
    assert declared.plan is not None
    built = declared.plan.build({"CC-01": 100.0, "CC-02": 50.0}, dt.date(2026, 1, 1), 6)
    assert built.target("cost_center", built.evaluation_month).tolist() == pytest.approx(
        [110.0, 55.0]
    )
    assert StudyProfile.from_mapping(declared.to_mapping()) == declared


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"plan": {}}, "a saved target's path or a synthetic one, exactly one"),
        ({"plan": {"path": "p", "synthetic": {"shape": "constant"}}}, "exactly one"),
        ({"plan": {"synthetic": {"shape": "zigzag"}}}, "a synthetic plan's shape is one of"),
        ({"plan": {"synthetic": {"shape": "constant", "base": 3}}}, "not ['base']"),
        ({"alignment": {"method": "both"}}, "alignment.method"),
        (
            {"alignment": {"overrides": {"ghost": "none"}}, "arms": [{"name": "a", "null": True}]},
            "alignment overrides name no arms ['ghost']",
        ),
        (
            {
                "alignment": {"overrides": {"a": "none"}},
                "arms": [{"name": "a", "null": True, "overrides": {"alignment.method": "supply"}}],
            },
            "arms ['a'] set alignment.method and an alignment override both",
        ),
        ({"settings": {"alignment.method": "demand"}}, "not settings.alignment.method"),
        ({"settings": {"run.seed": 4}}, "set the profile's own fields, ['seed for run.seed']"),
        ({"settings": {"supply.rule": "x"}}, "no settings ['supply.rule']"),
        ({"tilts": [{"domain": "weather", "multiplier": 2.0}]}, "no domain 'weather' to tilt"),
    ],
)
def test_a_bad_plan_method_tilt_or_setting_is_refused(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ProfileError) as caught:
        profile(**fields)
    assert message in str(caught.value)
