"""Tests for the offline stream and manual adjustments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcsim.financial.expense import SBC_KEYS, SBC_VALUES
from hcsim.financial.routing import RECORDED_COLUMNS, RecordedAwards
from hcsim.financial.streams import ManualAdjustments, OfflineStream, StreamError
from hcsim.model.core.schema import (
    AWARD_TYPE,
    COST_CENTER,
    JOB_LEVEL,
    SimulationSchema,
)

SCHEMA = SimulationSchema.build(
    {COST_CENTER: ["CC-01", "CC-02"], JOB_LEVEL: [1, 2], AWARD_TYPE: ["new_hire", "annual"]}
)


def offline(**overrides: list[object]) -> pd.DataFrame:
    rows: dict[str, list[object]] = {
        "award_id": [1],
        "holder": [7],
        "award_type": [1],
        "award_month": [-2],
        "vest_month": [3],
        "shares": [12],
        "value_per_share": [5.0],
        "level": [0],
        "cost_center": [1],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_the_offline_stream_runs_through_the_calculator_from_the_runs_first_month() -> None:
    # 60 of value over the six months -2 to 3: 9 a month net, 18 of it before the run, then 9
    # in each of months 0 to 3 and a vesting true-up of 6 at month 3.
    stream = OfflineStream(offline())
    result = stream.expense()
    assert result.opening is not None
    assert result.opening.tolist() == pytest.approx([18.0])
    sbc = stream.sbc()
    assert sbc["month"].tolist() == [0, 1, 2, 3]
    assert sbc["recognized"].tolist() == pytest.approx([9.0, 9.0, 9.0, 15.0])
    assert result.max_identity_residual == pytest.approx(0.0)
    assert (stream.tranches().forfeit_month == -1).all()  # no attrition


def test_a_replays_offline_tranche_forfeits_in_the_month_history_records() -> None:
    # The same tranche, its holder leaving in month 1: the 27 recognized before it, 18 of them
    # before the run, reverse.
    stream = OfflineStream(offline(forfeit_month=[1]))
    assert stream.tranches().forfeit_month.tolist() == [1]
    sbc = stream.sbc()
    assert sbc["month"].tolist() == [0, 1]
    assert sbc["recognized"].tolist() == pytest.approx([9.0, -27.0])


def test_an_empty_offline_stream_has_no_sbc() -> None:
    empty = OfflineStream(offline().iloc[:0])
    assert len(empty) == 0
    assert list(empty.sbc().columns) == [*SBC_KEYS, *SBC_VALUES]


def test_offline_tranches_are_checked() -> None:
    with pytest.raises(StreamError, match=r"need the columns \['level'\]"):
        OfflineStream(offline().drop(columns="level"))
    with pytest.raises(StreamError) as error:
        OfflineStream(offline(vest_month=[-1], award_month=[0], value_per_share=[np.nan]))
    message = str(error.value)
    for part in ("still vests", "can't vest before its award month", "appraisal"):
        assert part in message
    awarded = RecordedAwards.from_frame(
        pd.DataFrame(
            [(7, "annual", 0, 3, 12), (8, "annual", 0, 3, 12)],
            columns=list(RECORDED_COLUMNS),
        )
    )
    with pytest.raises(StreamError) as error:
        OfflineStream(offline(award_month=[0], forfeit_month=[-2]), awarded)
    message = str(error.value)
    assert "forfeits during the run, or never" in message
    assert "1 recorded awards have no tranche awarded to their holder in their month" in message


def manual(**overrides: list[object]) -> pd.DataFrame:
    rows: dict[str, list[object]] = {
        "month": [0, 0, 5],
        "cost_center": ["CC-01", "CC-01", "CC-02"],
        "award_type": ["annual", "annual", "new_hire"],
        "level": [1, 1, 2],
        "amount": [100.0, 50.0, -20.0],
        "note": ["a known grant", "its second part", "a correction"],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_manual_adjustments_add_their_amounts_to_recognized_expense() -> None:
    adjustments = ManualAdjustments.from_labels(manual(), SCHEMA, months=6)
    assert len(adjustments) == 3
    sbc = adjustments.sbc()
    assert list(sbc.columns) == [*SBC_KEYS, *SBC_VALUES]
    code = {name: SCHEMA.vocabulary(name).code for name in (COST_CENTER, AWARD_TYPE, JOB_LEVEL)}
    assert sbc[[*SBC_KEYS, "recognized"]].values.tolist() == [
        [0, code[COST_CENTER]("CC-01"), code[AWARD_TYPE]("annual"), code[JOB_LEVEL]("1"), 150.0],
        [5, code[COST_CENTER]("CC-02"), code[AWARD_TYPE]("new_hire"), code[JOB_LEVEL]("2"), -20.0],
    ]
    assert (sbc[["gross", "net", "trueup"]] == 0).all().all()


def test_manual_adjustments_are_checked_all_at_once() -> None:
    bad = manual(
        month=[0, 6, -1], cost_center=["CC-01", "CC-09", "CC-02"], amount=[1.0, np.inf, 2.0]
    )
    with pytest.raises(StreamError) as error:
        ManualAdjustments.from_labels(bad, SCHEMA, months=6)
    message = str(error.value)
    assert "months [-1, 6] lie outside the run's 0..5" in message
    assert "cost_center labels the schema doesn't know: ['CC-09']" in message
    assert "1 amounts aren't finite" in message
    with pytest.raises(StreamError, match=r"need the columns \['note'\]"):
        ManualAdjustments.from_labels(manual().drop(columns="note"), SCHEMA, months=6)
