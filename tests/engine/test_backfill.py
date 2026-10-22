"""Tests for backfill's draw: a rate per cost center or a classifier, in two registrations."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.domains import (
    ATTRITION,
    TRANSFERS,
    Departures,
    StepContext,
    backfill,
    backfill_transfer,
)
from hcsim.engine.seeds import SeedManager
from hcsim.engine.state import SimulationState
from hcsim.model.core.layouts import Buffer, EmployeeColumn, OpeningColumn
from hcsim.model.ml.group import BackfillRateModel
from tests.engine.test_loop import config, initial_state, models

#: CC-01 always backfills, CC-02 never does, and CC-03 reads the pooled half.
BY_COST_CENTER = BackfillRateModel(3, min_departures=1).fit(
    np.array([0, 0, 1, 1]), np.array([True, True, False, False])
)


class ByLevel:
    """A classifier that backfills only level-code 1."""

    ALGORITHM_NAME: ClassVar[str] = "level_only"

    def compute_probability(self, features: Buffer) -> np.ndarray:
        return (features[:, EmployeeColumn.LEVEL] == 1).astype(np.float64)


def departing(state: SimulationState, rows: list[int], domain: str = ATTRITION) -> Departures:
    taken = state.live_employees[rows].copy()
    return Departures(domain, taken, np.zeros(len(rows), dtype=bool))


def context(**overrides: object) -> StepContext:
    return StepContext(config(**overrides), 0, 0, SeedManager(5))


def test_each_departure_backfills_at_its_cost_center_s_rate() -> None:
    state = initial_state()
    ctx = context(models=dataclasses.replace(models(), backfill=BY_COST_CENTER))
    ctx.departures.append(departing(state, [0, 1, 2, 30, 31, 32]))
    opened = backfill(state, ctx).openings.append
    assert opened[:, OpeningColumn.COST_CENTER].tolist() == [0, 0, 0]


def test_the_draw_is_binomial_over_the_departures() -> None:
    state = initial_state()
    rate = BackfillRateModel.constant(0.3, 3)
    ctx = context(models=dataclasses.replace(models(), backfill=rate))
    ctx.departures.append(departing(state, list(range(90))))
    counts = [len(backfill(state, ctx).openings.append) for _ in range(200)]
    mean = float(np.mean(counts))
    assert abs(mean - 27) < 3 * np.sqrt(90 * 0.3 * 0.7 / 200)


def test_a_classifier_scores_each_departure() -> None:
    state = initial_state()
    ctx = context(
        models=dataclasses.replace(models(), backfill=ByLevel()), backfill_source="classifier"
    )
    ctx.departures.append(departing(state, [0, 1, 2, 3]))  # levels alternate 0 and 1
    opened = backfill(state, ctx).openings.append
    assert opened[:, OpeningColumn.LEVEL].tolist() == [1, 1]


def test_the_backfill_model_must_serve_the_configured_source() -> None:
    with pytest.raises(ConfigError, match="needs a GroupRateModel as the backfill model"):
        config(models=dataclasses.replace(models(), backfill=ByLevel()))
    with pytest.raises(ConfigError, match="needs a EntityModel as the backfill_transfer model"):
        config(
            models=dataclasses.replace(
                models(), backfill=ByLevel(), backfill_transfer=models().growth
            ),
            backfill_source="classifier",
        )


def test_each_registration_reads_its_own_departures() -> None:
    state = initial_state()
    never = BackfillRateModel.constant(0.0, 3)
    always = BackfillRateModel.constant(1.0, 3)
    ctx = context(models=dataclasses.replace(models(), backfill=always))
    ctx.departures.append(departing(state, [0, 1]))
    ctx.departures.append(departing(state, [30, 31, 32], TRANSFERS))
    assert len(backfill(state, ctx).openings.append) == 2
    # Transfers out backfill at the departure rate, without a model of their own ...
    assert len(backfill_transfer(state, ctx).openings.append) == 3
    # ... and at their own rate with one.
    own = context(models=dataclasses.replace(models(), backfill=always, backfill_transfer=never))
    own.departures.extend(ctx.departures)
    assert backfill_transfer(state, own).openings.is_empty
    rows, flags = ctx.departed(ATTRITION, TRANSFERS)
    assert len(rows) == len(flags) == 5
    assert ctx.departed("promotion")[0].shape == (0, rows.shape[1])
