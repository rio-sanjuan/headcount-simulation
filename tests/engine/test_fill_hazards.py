"""Tests for the fill channels' hazards, their rescaling, and the planner-mix hook."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.domains import StepContext
from hcsim.engine.eligibility import EligibilityRule
from hcsim.engine.fills import (
    CHANNELS,
    CONVERSION,
    EXTERNAL,
    TRANSFER,
    channel_hazards,
    fills,
    rescale,
)
from hcsim.engine.seeds import SeedManager
from hcsim.engine.settings import to_record
from hcsim.model.core.layouts import OpeningColumn
from hcsim.model.ml.entity import ConstantModel
from tests.engine.test_config_records import _saved
from tests.engine.test_config_state_seeds import config as small_config
from tests.engine.test_loop import config, initial_state, models


def hazards(**overrides: Any) -> np.ndarray:
    ctx = StepContext(config(**overrides), 0, 0, SeedManager(1))
    return channel_hazards(initial_state().live_openings, ctx)


def with_channels(external: float, conversion: float, transfer: float) -> Any:
    return dataclasses.replace(
        models(fill=external),
        conversion_fill=ConstantModel(conversion),
        transfer_fill=ConstantModel(transfer),
    )


def test_rows_past_one_are_rescaled_and_the_rest_kept() -> None:
    h = np.array([[0.2, 0.3, 0.1], [0.8, 0.8, 0.4], [0.0, 0.0, 0.0]])
    out = rescale(h)
    np.testing.assert_allclose(out[0], h[0])
    np.testing.assert_allclose(out[1], [0.4, 0.4, 0.2])
    np.testing.assert_allclose(out[2], 0.0)


def test_each_channel_reads_its_own_model() -> None:
    assert CHANNELS == ("external", "conversion", "transfer")
    only_external = hazards()
    assert (only_external[:, EXTERNAL] == 0.3).all()
    assert (only_external[:, [CONVERSION, TRANSFER]] == 0).all()
    three = hazards(models=with_channels(0.2, 0.1, 0.05))
    np.testing.assert_allclose(three[0], [0.2, 0.1, 0.05])
    crowded = hazards(models=with_channels(0.9, 0.6, 0.0))
    np.testing.assert_allclose(crowded[0], [0.6, 0.4, 0.0])


def test_each_channel_honours_its_own_eligibility() -> None:
    rule = EligibilityRule.of("transfer_fill", cost_center=["CC-01"])
    h = hazards(models=with_channels(0.2, 0.1, 0.05), eligibility=(rule,))
    openings = initial_state().live_openings
    in_cc01 = openings[:, OpeningColumn.COST_CENTER] == 0
    assert (h[in_cc01, TRANSFER] == 0).all()
    assert (h[~in_cc01, TRANSFER] == 0.05).all()
    assert (h[:, EXTERNAL] == 0.2).all()


def test_a_planner_mix_reshapes_the_hazards_before_rescaling() -> None:
    def favour_conversion(h: np.ndarray, openings: np.ndarray, step: int) -> np.ndarray:
        out = h.copy()
        out[:, CONVERSION] *= 10
        return out

    h = hazards(models=with_channels(0.2, 0.1, 0.0), fill_mix=favour_conversion)
    np.testing.assert_allclose(h[0], [0.2 / 1.2, 1.0 / 1.2, 0.0])


def test_a_bad_planner_mix_is_refused() -> None:
    def narrow(h: np.ndarray, openings: np.ndarray, step: int) -> np.ndarray:
        return h[:, :2]

    def negative(h: np.ndarray, openings: np.ndarray, step: int) -> np.ndarray:
        return h - 1

    with pytest.raises(ValueError, match="must return"):
        hazards(fill_mix=narrow)
    with pytest.raises(ValueError, match="finite, non-negative"):
        hazards(fill_mix=negative)


def test_a_planner_mix_has_no_record_yet(tmp_path: Path) -> None:
    built = small_config(models=_saved(tmp_path), fill_mix=lambda h, o, s: h)
    with pytest.raises(ConfigError, match="planner-mix hook is code"):
        to_record(built)


def test_a_crowded_opening_fills_by_one_channel_in_proportion() -> None:
    state = initial_state()
    ctx = StepContext(config(models=with_channels(1.0, 1.0, 0.0)), 0, 0, SeedManager(3))
    result = fills(state, ctx)
    external, conversion = int(result.flows["F_ext"].sum()), int(result.flows["F_conv"].sum())
    # Each of the 12 openings fills by exactly one channel, about half by each.
    assert external + conversion == 12 == len(result.employees.append)
    assert 0 < external < 12
