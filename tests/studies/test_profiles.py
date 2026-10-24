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
