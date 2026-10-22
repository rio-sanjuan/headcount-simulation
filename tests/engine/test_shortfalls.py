"""Tests for shortfall reporting."""

from __future__ import annotations

import dataclasses

import numpy as np

from hcsim.engine.loop import SHORTFALL_COLUMNS, simulate
from hcsim.engine.results import QuotaRecord
from hcsim.model.ml.entity import ConstantModel
from tests.engine.test_loop import config, guidance, initial_state, models


def test_a_record_s_shortfall_is_what_the_quota_didn_t_get() -> None:
    record = QuotaRecord(np.array([5, 2, 0]), np.array([3, 4, 0]), np.array([3, 2, 0]))
    assert record.shortfall.tolist() == [2, 0, 0]


def test_a_quota_past_the_openings_shows_up_as_a_shortfall() -> None:
    # CC-01 holds four openings and fills them all, but guidance asks for nine fills a month.
    built = dataclasses.replace(models(fill=1.0, attrition=0.0, cancel=0.0, backfill=0.0))
    result = simulate(
        config(
            models=built,
            guidance=guidance({"external_hires": {"CC-01": 9.0}}),
            simulation_months=2,
            num_trajectories=2,
        ),
        initial_state(),
    )
    frame = result.shortfalls()
    assert tuple(frame.columns) == SHORTFALL_COLUMNS
    assert set(frame["domain"]) == {"fills"}
    assert set(frame["category"]) == {"external_hires"}
    assert set(frame["group"]) == {"CC-01"}
    first = frame[frame["step"] == 0]
    assert (first["quota"] == 9).all()
    assert (first["shortfall"] == first["quota"] - first["realized"]).all()
    assert (first["realized"] <= first["eligible"]).all()
    assert (first["month"] == 1).all()
    totals = result.shortfall_totals()
    assert totals["shortfall"].tolist() == [int(frame["shortfall"].sum())]
    assert totals["per_trajectory"].tolist() == [frame["shortfall"].sum() / 2]


def test_a_run_that_meets_every_quota_reports_no_shortfall() -> None:
    built = dataclasses.replace(models(), conversion_fill=ConstantModel(0.0))
    result = simulate(config(models=built), initial_state())
    assert result.shortfalls().empty
    assert result.shortfall_totals().empty
