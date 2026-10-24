"""Tests for the run-mode ladder: each rung's pins, what follows from them, and the checks a
ladder must pass to import."""

from __future__ import annotations

import pytest

from hcsim.engine.config import ConfigError
from hcsim.engine.loop import REPLAYING
from hcsim.engine.modes import LADDER, Pin, Rung, RunMode, check_ladder, releases, rung
from tests.engine.test_loop import config, guidance

M, P = RunMode, Pin


def test_each_rung_pins_one_input_fewer_than_the_one_above() -> None:
    assert [(r.mode, set(r.pins)) for r in LADDER] == [
        (M.REPLAY, {P.MOVEMENTS, P.VOLUMES, P.QUOTA}),
        (M.GUIDED_ACTUALS, {P.VOLUMES, P.QUOTA}),
        (M.GUIDED_FORECAST, {P.QUOTA}),
        (M.UNGUIDED, set()),
    ]
    assert [releases(m) for m in M] == [None, P.MOVEMENTS, P.VOLUMES, P.QUOTA]
    assert rung("guided_forecast").isolates == "the guidance forecast"


def test_each_rungs_other_properties_follow_from_its_pins() -> None:
    def derived(mode: RunMode) -> tuple[bool, bool, bool, bool]:
        r = rung(mode)
        return r.reads_guidance, r.realized_guidance, r.replays_history, r.needs_actuals

    assert derived(M.REPLAY) == (False, False, True, True)
    assert derived(M.GUIDED_ACTUALS) == (True, True, False, True)
    assert derived(M.GUIDED_FORECAST) == (True, False, False, False)
    assert derived(M.UNGUIDED) == (False, False, False, False)
    assert REPLAYING == (M.REPLAY,)


def test_a_ladder_that_doesnt_hold_together_is_refused() -> None:
    assert check_ladder(LADDER) == []
    everything = frozenset(Pin)
    broken = (
        Rung(M.REPLAY, frozenset({P.MOVEMENTS}), ""),
        Rung(M.GUIDED_ACTUALS, everything, ""),
        Rung(M.GUIDED_ACTUALS, frozenset({P.QUOTA}), ""),
    )
    assert check_ladder(broken) == [
        "the ladder is missing ['guided_forecast', 'unguided']",
        "the ladder repeats ['guided_actuals']",
        "the top rung, replay, must pin everything",
        "the bottom rung, guided_actuals, must pin nothing",
        "guided_actuals must pin replay's pins less exactly one",
        "guided_actuals must pin guided_actuals's pins less exactly one",
    ]
    skipping = (
        Rung(M.REPLAY, everything, ""),
        Rung(M.GUIDED_ACTUALS, frozenset({P.QUOTA}), ""),  # releases two at once
        Rung(M.GUIDED_FORECAST, frozenset({P.QUOTA}), ""),
        Rung(M.UNGUIDED, frozenset(), ""),
    )
    assert check_ladder(skipping) == [
        "guided_actuals must pin replay's pins less exactly one",
        "guided_forecast must pin guided_actuals's pins less exactly one",
    ]


def test_the_configuration_reads_the_ladder() -> None:
    # A rung that reads guidance needs it, and one that doesn't refuses it.
    with pytest.raises(ConfigError, match="run_mode guided_actuals needs guidance"):
        config(run_mode="guided_actuals")
    with pytest.raises(ConfigError, match="run_mode unguided reads no guidance"):
        config(run_mode="unguided", guidance=guidance({"attrition": {"CC-01": 1.0}}))
    assert config(guidance=guidance({"attrition": {"CC-01": 1.0}})).run_mode is M.GUIDED_FORECAST
    assert config().run_mode is M.UNGUIDED
