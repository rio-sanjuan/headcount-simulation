"""Tests for the world's reference structures."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from hcsim.synth.config import WorldConfig, preset
from hcsim.synth.structure import level_one_names, org_dimension, org_levels


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
