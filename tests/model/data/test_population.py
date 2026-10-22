"""Tests for population specs, stamps, and guards."""

from __future__ import annotations

import pytest

from hcsim.model.data.population import (
    DEFAULT,
    REGISTRY,
    REGULAR,
    PopulationError,
    PopulationSpec,
    PopulationStamp,
    guard,
    population,
)


def test_the_registry_holds_the_built_in_populations() -> None:
    assert sorted(REGISTRY) == ["every_account", "regular", "regular_and_contingent"]
    assert population("regular") is REGULAR is DEFAULT
    assert REGULAR.rules == {"account_group": frozenset({"regular"})}
    assert REGULAR.description
    with pytest.raises(
        PopulationError, match="no built-in population 'staff'; known: every_account"
    ):
        population("staff")


def test_the_digest_is_the_rules_whatever_their_order() -> None:
    a = PopulationSpec.of(account_group=["regular", "contingent"], org_level1=["ORG-A"])
    b = PopulationSpec.of(org_level1=["ORG-A"], account_group=["contingent", "regular", "regular"])
    assert a.digest == b.digest
    assert a.digest.startswith("pd1-")
    assert a.name == f"pop-{a.digest[:12]}"  # unnamed specs are named after their digest
    assert list(a.rules) == ["account_group", "org_level1"]
    assert a != PopulationSpec.of(account_group=["regular"])
    named = PopulationSpec.of("org-a", account_group=["regular"], org_level1=["ORG-A"])
    assert named.name == "org-a"
    assert named.to_dict() == {
        "name": "org-a",
        "digest": named.digest,
        "rules": {"account_group": ["regular"], "org_level1": ["ORG-A"]},
    }


def test_rules_must_use_population_columns_and_admit_something() -> None:
    with pytest.raises(PopulationError, match="'trans_type' can't define a population") as error:
        PopulationSpec.of(trans_type=["HIRE"], salary=["high"], cost_center=[])
    message = str(error.value)
    assert "'salary' isn't in the column catalog" in message
    assert "the rule on 'cost_center' admits nothing" in message


def test_a_row_is_in_when_every_rule_admits_it() -> None:
    spec = PopulationSpec.of(account_group=["regular"], org_level1=["ORG-A", "ORG-B"])
    assert spec.admits({"account_group": "regular", "org_level1": "ORG-B", "extra": 1})
    assert not spec.admits({"account_group": "contingent", "org_level1": "ORG-A"})
    with pytest.raises(PopulationError, match=r"the row has no \['org_level1'\]"):
        spec.admits({"account_group": "regular"})


def test_stamps_round_trip_and_guards_refuse_another_population() -> None:
    stamp = REGULAR.stamp
    assert PopulationStamp.parse(str(stamp)) == stamp
    for bad in ("regular", "@pd1-x", "regular@"):
        with pytest.raises(PopulationError, match="name@digest"):
            PopulationStamp.parse(bad)
    guard(REGULAR, stamp, "guidance forecast")
    guard(REGULAR, str(stamp), "guidance forecast")
    guard(REGULAR, None, "guidance forecast")  # unstamped artifacts are the default's
    other = population("regular_and_contingent")
    with pytest.raises(
        PopulationError,
        match="the attrition model was built for population regular_and_contingent@pd1-",
    ):
        guard(REGULAR, other.stamp, "attrition model")
    with pytest.raises(PopulationError, match="this run is for regular_and_contingent"):
        guard(other, None, "guidance forecast")
