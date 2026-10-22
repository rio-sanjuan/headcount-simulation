"""Tests for the stock price models."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from hcsim.financial.prices import (
    AnnualGrowthPrice,
    CustomPrice,
    GBMPrice,
    PriceModelError,
    StaticPrice,
    StockPriceModel,
    from_declaration,
)


def test_a_static_price_never_moves() -> None:
    model = StaticPrice(50.0)
    assert isinstance(model, StockPriceModel)
    for month in range(1, 13):
        model.step(month)
    assert model.price == model.projection(3) == 50.0


def test_annual_growth_applies_once_a_year_in_its_month_and_not_on_the_first_step() -> None:
    model = AnnualGrowthPrice(100.0, growth_rate=1.10, growth_month=1)
    model.step(1)  # the run's first step is January: no growth yet
    assert model.price == 100.0
    for month in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        model.step(month)
    assert model.price == 100.0
    model.step(1)  # the next January
    assert model.price == pytest.approx(110.0)


def test_the_projection_compounds_from_the_current_price() -> None:
    model = AnnualGrowthPrice(100.0, growth_rate=1.10)
    assert model.projection(0) == 100.0
    assert model.projection(2) == pytest.approx(121.0)
    with pytest.raises(PriceModelError, match="non-negative"):
        model.projection(-1)


def test_each_trajectory_can_hold_its_own_copy() -> None:
    model = AnnualGrowthPrice(100.0, growth_rate=1.10, growth_month=3)
    clone = copy.deepcopy(model)
    model.step(1)
    model.step(3)
    assert model.price == pytest.approx(110.0)
    assert clone.price == 100.0


def test_a_growth_rate_written_as_a_percentage_is_refused_with_the_convention() -> None:
    with pytest.raises(PriceModelError, match=r"write 1\.08") as error:
        AnnualGrowthPrice(100.0, growth_rate=0.08, growth_month=13)
    assert "calendar month" in str(error.value)  # every problem at once


def test_prices_must_be_positive() -> None:
    with pytest.raises(PriceModelError, match="positive"):
        StaticPrice(0.0)
    with pytest.raises(PriceModelError, match="positive"):
        AnnualGrowthPrice(-1.0)


def test_declarations_round_trip_and_unknown_types_are_refused() -> None:
    for model in (StaticPrice(42.0), AnnualGrowthPrice(80.0, growth_rate=1.2, growth_month=7)):
        assert from_declaration(model.declaration()).declaration() == model.declaration()
    with pytest.raises(PriceModelError, match="unknown price model 'RandomWalk'"):
        from_declaration({"type": "RandomWalk", "params": {}})


# --- custom paths --------------------------------------------------------------------------------


def stepped(model: StockPriceModel, months: list[int]) -> list[float]:
    out = []
    for month in months:
        model.step(month)
        out.append(model.price)
    return out


def test_a_custom_path_gives_each_steps_price_and_projects_along_it() -> None:
    model = CustomPrice([100.0, 101.0] + [102.0] * 22 + [130.0])
    assert model.price == 100.0  # before the first step
    assert model.projection(1) == 102.0  # twelve months on
    assert model.projection(2) == 130.0
    assert stepped(model, [1, 2, 3]) == [100.0, 101.0, 102.0]
    with pytest.raises(PriceModelError, match="no extension"):
        model.projection(3)


def test_an_extension_continues_from_the_paths_last_price() -> None:
    growth = AnnualGrowthPrice(102.0, growth_rate=1.10, growth_month=4)
    model = CustomPrice([100.0, 101.0, 102.0], extend_with=growth)
    # January, February, March on the path; growth's first move is in April, the month after.
    assert stepped(model, [1, 2, 3, 4, 5]) == pytest.approx([100.0, 101.0, 102.0, 112.2, 112.2])
    assert model.projection(1) == pytest.approx(112.2 * 1.10)
    fresh = CustomPrice([100.0, 101.0, 102.0], extend_with=growth)
    fresh.step(1)
    assert fresh.projection(1) == pytest.approx(102.0 * 1.10)  # 12 months on: past the path
    assert growth.price == 102.0  # the path holds its own copy


def test_a_custom_path_is_checked() -> None:
    with pytest.raises(PriceModelError, match="at least one price"):
        CustomPrice([])
    with pytest.raises(PriceModelError, match="finite and positive"):
        CustomPrice([100.0, -1.0])
    with pytest.raises(PriceModelError, match=r"continues from the path's last price, 102\.0"):
        CustomPrice([100.0, 102.0], extend_with=StaticPrice(90.0))
    short = CustomPrice([100.0])
    short.step(1)
    with pytest.raises(PriceModelError, match="covers 1 months and has no extension"):
        short.step(2)


# --- geometric Brownian motion -------------------------------------------------------------------


def test_gbm_opens_at_its_price_then_moves_reproducibly() -> None:
    model = GBMPrice(100.0, drift=0.08, volatility=0.3, seed=7)
    path = stepped(model, list(range(1, 13)))
    assert path[0] == 100.0
    assert len(set(path[1:])) == 11
    assert stepped(GBMPrice(100.0, 0.08, 0.3, seed=7), list(range(1, 13))) == path
    assert stepped(model.reseeded(8), list(range(1, 13)))[1:] != path[1:]
    assert model.projection(2) == pytest.approx(model.price * math.exp(0.16))


def test_gbm_moves_at_its_drift_on_average() -> None:
    months = 12 * 400
    model = GBMPrice(100.0, drift=0.06, volatility=0.2, seed=1)
    path = np.log(stepped(model, [1] * (months + 1)))
    monthly = np.diff(path)
    assert monthly.mean() == pytest.approx((0.06 - 0.02) / 12, abs=0.002)
    assert monthly.std() == pytest.approx(0.2 / math.sqrt(12), rel=0.05)
    flat = GBMPrice(100.0, drift=0.12, volatility=0.0)
    assert stepped(flat, [1, 2])[1] == pytest.approx(100.0 * math.exp(0.01))


def test_gbm_is_checked() -> None:
    with pytest.raises(PriceModelError) as error:
        GBMPrice(-1.0, drift=math.inf, volatility=-0.1)
    message = str(error.value)
    assert all(word in message for word in ("opening_price", "drift", "volatility"))


def test_every_model_round_trips_through_its_declaration() -> None:
    models: list[StockPriceModel] = [
        StaticPrice(50.0),
        AnnualGrowthPrice(100.0, 1.05, 4),
        CustomPrice([100.0, 104.0], extend_with=AnnualGrowthPrice(104.0, 1.07)),
        GBMPrice(100.0, 0.07, 0.25, seed=3),
    ]
    for model in models:
        built = from_declaration(model.declaration())
        assert built.declaration() == model.declaration()
        assert type(built) is type(model)
