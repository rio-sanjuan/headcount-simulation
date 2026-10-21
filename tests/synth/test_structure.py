"""Tests for the world's reference structures."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import FAMILIES, FAMILY_CODES, WorldConfig, preset
from hcsim.synth.population import draw_roles
from hcsim.synth.structure import (
    account_dimension,
    company_dimension,
    job_dimension,
    level_one_names,
    location_dimension,
    org_dimension,
    org_levels,
)


def org(name: str = "small", seed: int = 1, **overrides: object) -> pd.DataFrame:
    config = preset(name) if not overrides else WorldConfig(**overrides)  # type: ignore[arg-type]
    return org_dimension(config, np.random.default_rng(seed))


def depth(frame: pd.DataFrame) -> pd.Series:
    """Levels above the leaf, plus the leaf itself."""
    levels = [c for c in frame.columns if c.startswith("org_level")]
    return frame[levels].ne(frame["cost_center"], axis=0).sum(axis=1) + 1


def test_every_cost_center_has_a_node_at_every_level() -> None:
    frame = org()
    assert list(frame.columns) == [
        "cost_center",
        "org_level1",
        "org_level2",
        "org_level3",
        "org_level4",
        "cost_type",
        "field_type",
    ]
    assert len(frame) == 250
    assert frame["cost_center"].is_unique
    assert frame[org_levels(preset("small"))].notna().all().all()


def test_depth_is_uneven_and_the_leaf_pads_the_levels_below_it() -> None:
    frame = org()
    d = depth(frame)
    assert d.min() >= 2
    assert d.max() == 4
    assert d.nunique() > 1
    levels = org_levels(preset("small"))
    for row, k in zip(frame.itertuples(index=False), d, strict=True):
        values = [getattr(row, level) for level in levels]
        assert values[k - 1 :] == [row.cost_center] * (len(levels) - k + 1)
        assert row.cost_center not in values[: k - 1]


def test_the_hierarchy_is_a_tree_with_extending_labels() -> None:
    frame = org()
    levels = org_levels(preset("small"))
    for upper, lower in itertools.pairwise(levels):
        internal = frame[frame[lower] != frame["cost_center"]]
        parents = internal.groupby(lower)[upper].nunique()
        assert (parents == 1).all()  # every node has one parent
        assert all(
            child.startswith(f"{parent}-")
            for child, parent in zip(internal[lower], internal[upper], strict=True)
        )


def test_level_one_is_dealt_round_robin() -> None:
    frame = org()
    tops = level_one_names(5)
    assert frame["org_level1"].tolist() == [tops[i % 5] for i in range(250)]


def test_the_tiny_world_keeps_two_levels() -> None:
    frame = org("tiny")
    assert sorted(frame["org_level1"].unique()) == ["ORG-A", "ORG-B"]
    assert (frame["org_level2"] == frame["cost_center"]).all()


def test_cost_and_field_types_follow_their_shares() -> None:
    frame = org("medium", seed=2)
    assert set(frame["cost_type"]) == {"fixed", "variable"}
    assert (frame["cost_type"] == "variable").mean() == pytest.approx(0.20, abs=0.03)
    # A field org is field all the way down.
    assert (frame.groupby("org_level1")["field_type"].nunique() == 1).all()


def test_level_one_names_continue_past_z() -> None:
    names = level_one_names(28)
    assert names[:2] == ["ORG-A", "ORG-B"]
    assert names[25:] == ["ORG-Z", "ORG-AA", "ORG-AB"]


def test_the_same_seed_gives_the_same_hierarchy() -> None:
    pd.testing.assert_frame_equal(org(seed=4), org(seed=4))
    assert not org(seed=4).equals(org(seed=5))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("org_levels", 1, "org_levels must be at least 2"),
        ("org_branching", 1, "org_branching must be at least 2"),
        ("org_stop_probability", 1.5, "org_stop_probability must lie in [0, 1]"),
    ],
)
def test_bad_hierarchy_settings_are_refused(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        WorldConfig(**{field: value})  # type: ignore[arg-type]


# --- the job, location, company, and account catalogs ------------------------------------------


def test_the_job_catalog_marks_manager_titles_and_executive_levels() -> None:
    config = preset("medium")
    jobs = job_dimension(config)
    assert jobs["job_code"].is_unique
    assert set(jobs["job_family"]) == set(config.families)
    assert sorted(jobs["job_level"].unique()) == list(config.levels)
    managers = jobs[jobs["manager_title"]]
    assert (managers["job_level"] >= config.manager_min_level).all()
    assert managers["job_title"].str.contains(" Manager ").all()
    assert managers["job_code"].str.endswith("M").all()
    assert not jobs.loc[~jobs["manager_title"], "job_title"].str.contains("Manager").any()
    assert (jobs["is_executive"] == (jobs["job_level"] >= 7)).all()
    assert (jobs["level_name"] == "L" + jobs["job_level"].astype(str)).all()
    eng = jobs[(jobs["job_family"] == "Engineering") & (jobs["job_level"] == 4)]
    assert eng[["job_code", "job_title"]].values.tolist() == [
        ["ENG-4", "Engineering IV"],
        ["ENG-4M", "Engineering Manager IV"],
    ]


def test_family_codes_are_unique_and_every_family_has_a_tech_indicator() -> None:
    assert len(set(FAMILY_CODES.values())) == len(FAMILY_CODES)
    assert set(FAMILY_CODES) == set(FAMILIES)
    assert set(FAMILIES.values()) == {"software_engineering", "other_technical", "non_technical"}


def test_locations_form_a_region_tree() -> None:
    frame = location_dimension(preset("large"))
    assert len(frame) == 20
    assert (frame.groupby("location")["country"].nunique() == 1).all()
    assert (frame.groupby("country")["region"].nunique() == 1).all()
    assert frame.groupby("country").size().max() == 3
    assert frame["country"].nunique() == 7
    assert frame["region"].nunique() == 3


def test_companies_roll_up_with_offline_subsidiaries_apart() -> None:
    frame = company_dimension(preset("large"))
    assert frame["company"].tolist() == [f"CO-{i}" for i in range(1, 7)]
    assert frame["is_primary"].tolist() == [True] + [False] * 5
    assert frame["is_offline"].tolist() == [False] * 4 + [True] * 2
    assert set(frame.loc[~frame["is_offline"], "planning_company"]) == {"PCO-1"}
    assert set(frame.loc[frame["is_offline"], "planning_company"]) == {"PCO-2"}


def test_accounts_cover_every_group() -> None:
    frame = account_dimension(WorldConfig())
    assert dict(zip(frame["account"], frame["account_group"], strict=True)) == {
        "ACCT-100": "regular",
        "ACCT-200": "contingent",
        "ACCT-300": "temporary",
    }


def test_roles_book_to_the_primary_company_and_a_regular_account() -> None:
    config = preset("medium")
    roles = draw_roles(
        config, np.zeros(20_000, int), np.ones(20_000, int), np.random.default_rng(6)
    )
    shares = np.bincount(roles["company"], minlength=4) / 20_000
    assert shares[0] == pytest.approx(config.primary_company_share, abs=0.02)
    assert shares[1:] == pytest.approx([(1 - config.primary_company_share) / 3] * 3, abs=0.02)
    assert (roles["account"] == 0).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"offline_subsidiaries": 1}, "offline_subsidiaries must leave the primary company online"),
        ({"accounts": (("A", "contingent"),)}, "accounts need a regular account"),
        ({"accounts": (("A", "regular"), ("A", "temporary"))}, "account names must be unique"),
        ({"locations_per_country": 0}, "locations_per_country and countries_per_region"),
    ],
)
def test_bad_catalog_settings_are_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorldConfig(**overrides)  # type: ignore[arg-type]
