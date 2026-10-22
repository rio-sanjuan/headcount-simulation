"""Tests for the simulation schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    canonical_numeric,
    check_codes,
    org_levels_from_frame,
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
    assert schema.fingerprint().startswith("sc2-")


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


def test_a_number_has_one_spelling_in_a_numeric_vocabulary() -> None:
    vocab = Vocabulary.build(JOB_LEVEL, [3, 1.0, "2", np.int64(3), 10.0])
    assert vocab.labels == ("1", "2", "3", "10")
    assert vocab.encode([3.0, "3", np.float64(1), 10]).tolist() == [2, 2, 0, 3]
    assert 2.0 in vocab
    assert vocab.encode(["x"], unknown="unattributed").tolist() == [UNATTRIBUTED]
    assert canonical_numeric(2.5) == "2.5"
    for bad in ("x", float("nan"), float("inf")):
        with pytest.raises(SchemaError, match="non-"):
            canonical_numeric(bad)
    with pytest.raises(SchemaError, match="non-canonical labels"):
        Vocabulary(JOB_LEVEL, ("1.0",), numeric=True)


def _org() -> pd.DataFrame:
    """Three levels, with the branch under ORG-B ending early and repeating its leaf."""
    return pd.DataFrame(
        {
            "cost_center": ["CC-01", "CC-02", "CC-03", "CC-04"],
            "org_level1": ["ORG-A", "ORG-A", "ORG-A", "ORG-B"],
            "org_level3": ["CC-01", "CC-02", "CC-03", "CC-04"],
            "org_level2": ["ORG-A-1", "ORG-A-1", "ORG-A-2", "CC-04"],
            "cost_type": ["fixed"] * 4,
        }
    )


def test_every_org_level_becomes_a_rollup_top_level_first() -> None:
    levels = org_levels_from_frame(_org())
    assert list(levels) == ["org_level1", "org_level2", "org_level3"]
    schema = SimulationSchema.build({COST_CENTER: _org()["cost_center"]}, org_levels=levels)
    assert schema.org_levels == ["org_level1", "org_level2", "org_level3"]
    assert schema.decode("org_level2", schema.rollup("org_level2", [0, 2, 3])).tolist() == [
        "ORG-A-1",
        "ORG-A-2",
        "CC-04",
    ]
    with pytest.raises(SchemaError, match="no org_level columns"):
        org_levels_from_frame(_org()[["cost_center"]])


def test_org_levels_must_nest() -> None:
    org = _org().assign(org_level1=["ORG-A", "ORG-B", "ORG-A", "ORG-B"])  # ORG-A-1 has two parents
    with pytest.raises(SchemaError, match="'ORG-A-1' has parents \\['ORG-A', 'ORG-B'\\]"):
        SimulationSchema.build(
            {COST_CENTER: org["cost_center"]}, org_levels=org_levels_from_frame(org)
        )


def test_a_saved_schema_loads_back_identical(tmp_path: Path) -> None:
    schema = toy_schema()
    path = tmp_path / "run" / "schema.json"
    schema.save(path)
    document = json.loads(path.read_text())
    assert document["format_version"] == 2
    assert document["numeric"] == [JOB_LEVEL]
    assert document["fingerprint"] == schema.fingerprint()
    loaded = SimulationSchema.load(path)
    assert loaded == schema
    assert loaded.vocabulary(JOB_LEVEL).numeric


def test_loading_refuses_a_bad_schema_file(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    with pytest.raises(SchemaError, match="no schema at"):
        SimulationSchema.load(path)
    toy_schema().save(path)
    document = json.loads(path.read_text())
    for change, message in (
        ({"format_version": 1}, "format 1 isn't 2"),
        ({"fingerprint": "sc2-0"}, "doesn't match its recorded fingerprint"),
        (
            {
                "vocabularies": {
                    **document["vocabularies"],
                    COST_CENTER: ["CC-02", "CC-01", "CC-03"],
                }
            },
            "not in canonical order: \\['cost_center'\\]",
        ),
    ):
        path.write_text(json.dumps(document | change))
        with pytest.raises(SchemaError, match=message):
            SimulationSchema.load(path)


def test_a_document_without_numeric_flags_reads_levels_as_numeric() -> None:
    data = toy_schema().to_dict()
    del data["numeric"]
    assert SimulationSchema.from_dict(data).vocabulary(JOB_LEVEL).numeric
