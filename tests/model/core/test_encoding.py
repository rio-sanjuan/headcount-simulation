"""Tests for frame and movement encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.model.core.encoding import (
    CoverageWarning,
    encode_frame,
    encode_movements,
    vocabulary_for,
)
from hcsim.model.core.schema import (
    COST_CENTER,
    JOB_FAMILY,
    JOB_LEVEL,
    UNATTRIBUTED,
    SchemaError,
    SimulationSchema,
)


def schema() -> SimulationSchema:
    return SimulationSchema.build(
        {
            COST_CENTER: ["CC-01", "CC-02", "CC-03"],
            JOB_FAMILY: ["Data", "Sales"],
            JOB_LEVEL: [1, 2, 3],
        }
    )


def test_columns_find_their_vocabulary_by_name() -> None:
    s = schema()
    for column in ("cost_center", "cost_center_previous", "to_cost_center", "from_cost_center"):
        assert vocabulary_for(s, column) == COST_CENTER
    with pytest.raises(SchemaError, match="no vocabulary for column 'team'"):
        vocabulary_for(s, "team")


def test_a_fully_covered_frame_encodes_without_a_warning() -> None:
    frame = pd.DataFrame({"cost_center": ["CC-02", "CC-01"], "job_level": [3.0, 1.0]})
    encoded = encode_frame(schema(), frame, ["cost_center", "job_level"])
    assert encoded.array().tolist() == [[1, 2], [0, 0]]
    assert encoded.complete.tolist() == [True, True]
    assert encoded.gaps == ()
    assert [c.share for c in encoded.coverage] == [1.0, 1.0]


def test_gaps_become_unattributed_and_raise_one_warning() -> None:
    frame = pd.DataFrame(
        {
            "cost_center": ["CC-01", "CC-09", "CC-08", "CC-09"],
            "job_family": ["Data", "Data", "Legal", "Sales"],
        }
    )
    with pytest.warns(
        CoverageWarning, match=r"cost_center: 3 of 4 rows, such as \['CC-08', 'CC-09'\]"
    ) as caught:
        encoded = encode_frame(schema(), frame, ["cost_center", "job_family"])
    assert len(caught) == 1
    assert encoded.codes["cost_center"].tolist() == [0, UNATTRIBUTED, UNATTRIBUTED, UNATTRIBUTED]
    assert encoded.complete.tolist() == [True, False, False, False]
    (cc, family) = encoded.coverage
    assert (cc.unattributed, cc.share, cc.examples) == (3, 0.25, ("CC-08", "CC-09"))
    assert (family.unattributed, family.examples) == (1, ("Legal",))
    assert encoded.array(["job_family"]).shape == (4, 1)


def test_warnings_can_be_left_to_the_caller() -> None:
    frame = pd.DataFrame({"cost_center": ["CC-09"]})
    encoded = encode_frame(schema(), frame, ["cost_center"], warn=False)
    assert encoded.gaps[0].column == "cost_center"


def test_bad_column_requests_are_refused() -> None:
    frame = pd.DataFrame({"cost_center": ["CC-01"]})
    with pytest.raises(SchemaError, match="no columns \\['job_family'\\]"):
        encode_frame(schema(), frame, ["job_family"])
    with pytest.raises(SchemaError, match="columns repeat"):
        encode_frame(schema(), frame, ["cost_center", "cost_center"])
    empty = encode_frame(schema(), frame, [])
    assert empty.array().shape == (0, 0)
    assert empty.complete.size == 0


def test_movement_destinations_encode_through_the_origin_s_vocabulary() -> None:
    moves = pd.DataFrame(
        {
            "cost_center_previous": ["CC-01", "CC-02", "CC-03"],
            "cost_center": ["CC-02", "CC-02", "CC-07"],
            "job_level_previous": [1, 2, 2],
            "job_level": [2, 2, 3],
        }
    )
    with pytest.warns(CoverageWarning, match="CC-07"):
        encoded = encode_movements(schema(), moves, [COST_CENTER, JOB_LEVEL])
    assert encoded.origin[COST_CENTER].tolist() == [0, 1, 2]
    assert encoded.destination[COST_CENTER].tolist() == [1, 1, UNATTRIBUTED]
    assert encoded.placeable.tolist() == [True, True, False]
    assert encoded.changes(COST_CENTER).tolist() == [True, False, False]
    assert encoded.changes(JOB_LEVEL).tolist() == [True, False, True]


def test_a_known_movement_frame_names_both_ends() -> None:
    moves = pd.DataFrame({"from_job_family": ["Data"], "to_job_family": ["Sales"]})
    encoded = encode_movements(schema(), moves, [JOB_FAMILY], origin="from_{}", destination="to_{}")
    assert encoded.changes(JOB_FAMILY).tolist() == [True]
    assert np.array_equal(encoded.placeable, [True])
    assert encode_movements(schema(), moves.iloc[0:0], [], origin="from_{}").placeable.size == 0


def test_a_movement_attribute_must_match_its_columns() -> None:
    moves = pd.DataFrame({"job_family_previous": ["Data"], "cost_center": ["CC-01"]})
    with pytest.raises(SchemaError, match="doesn't encode through 'cost_center'"):
        encode_movements(schema(), moves, [COST_CENTER], origin="job_family_previous")
