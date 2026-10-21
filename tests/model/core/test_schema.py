"""Tests for the simulation schema."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hcsim.model.core.schema import (
    COST_CENTER,
    JOB_LEVEL,
    UNATTRIBUTED,
    SchemaError,
    SimulationSchema,
    Vocabulary,
    check_codes,
)


def toy_schema() -> SimulationSchema:
    return SimulationSchema.build(
        {COST_CENTER: ["CC-03", "CC-01", "CC-02"], JOB_LEVEL: [10, 2, 1, 9]},
        org_levels={"org_level1": {"CC-01": "ORG-A", "CC-02": "ORG-B", "CC-03": "ORG-A"}},
    )


def test_labels_sort_lexically_and_codes_are_positions() -> None:
    vocab = Vocabulary.build("cost_center", ["CC-03", "CC-01", "CC-02", "CC-01"])
    assert vocab.labels == ("CC-01", "CC-02", "CC-03")
    assert [vocab.code(x) for x in vocab.labels] == [0, 1, 2]


def test_levels_sort_numerically_so_10_follows_9() -> None:
    assert toy_schema().vocabulary(JOB_LEVEL).labels == ("1", "2", "9", "10")


def test_a_numeric_vocabulary_refuses_a_non_numeric_label() -> None:
    with pytest.raises(SchemaError, match="non-numeric"):
        Vocabulary.build(JOB_LEVEL, ["3", "x"])


def test_duplicate_labels_are_refused_when_given_explicitly() -> None:
    with pytest.raises(SchemaError, match="duplicate"):
        Vocabulary("v", ("a", "a"))


def test_unknown_labels_raise_or_become_unattributed() -> None:
    schema = toy_schema()
    with pytest.raises(SchemaError, match="no label"):
        schema.encode(COST_CENTER, ["CC-09"])
    codes = schema.encode(COST_CENTER, ["CC-02", "CC-09"], unknown="unattributed")
    assert codes.tolist() == [1, UNATTRIBUTED]


def test_decode_refuses_unattributed_and_out_of_range_codes() -> None:
    schema = toy_schema()
    assert schema.decode(COST_CENTER, [2, 0]).tolist() == ["CC-03", "CC-01"]
    for bad in ([UNATTRIBUTED], [3]):
        with pytest.raises(SchemaError, match="can't decode"):
            schema.decode(COST_CENTER, bad)


def test_rollups_map_leaves_to_org_nodes_and_keep_unattributed() -> None:
    schema = toy_schema()
    assert schema.vocabulary("org_level1").labels == ("ORG-A", "ORG-B")
    assert schema.rollup("org_level1", [0, 1, 2, UNATTRIBUTED]).tolist() == [0, 1, 0, -1]


def test_a_rollup_must_place_every_cost_center() -> None:
    with pytest.raises(SchemaError, match="doesn't place"):
        SimulationSchema.build(
            {COST_CENTER: ["CC-01", "CC-02"]}, org_levels={"org_level1": {"CC-01": "ORG-A"}}
        )


def test_json_round_trip_preserves_codes_and_fingerprint() -> None:
    schema = toy_schema()
    restored = SimulationSchema.from_json(schema.to_json())
    assert restored == schema
    assert restored.fingerprint() == schema.fingerprint()
    assert schema.fingerprint().startswith("sc1-")


def test_a_different_vocabulary_changes_the_fingerprint() -> None:
    other = SimulationSchema.build({COST_CENTER: ["CC-01", "CC-02", "CC-04"]})
    assert other.fingerprint() != toy_schema().fingerprint()


def test_unknown_vocabulary_names_list_the_known_ones() -> None:
    with pytest.raises(SchemaError, match="known: cost_center"):
        toy_schema().vocabulary("nope")


def test_rollup_arrays_are_read_only() -> None:
    with pytest.raises(ValueError, match="read-only"):
        toy_schema().rollups["org_level1"][0] = 1


def test_check_codes_accepts_unattributed_and_refuses_out_of_range() -> None:
    check_codes("x", np.array([-1, 0, 2]), 3)
    with pytest.raises(SchemaError):
        check_codes("x", [3], 3)
    with pytest.raises(SchemaError):
        check_codes("x", [-2], 3)


@given(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=30))
def test_encode_then_decode_is_the_identity(labels: list[str]) -> None:
    vocab = Vocabulary.build("v", labels, numeric=False)
    assert vocab.decode(vocab.encode(labels)).tolist() == labels
