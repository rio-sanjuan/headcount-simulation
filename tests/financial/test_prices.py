"""Tests for the stock price models."""

from __future__ import annotations

import copy

import pytest

from hcsim.financial.prices import (
    AnnualGrowthPrice,
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
    with pytest.raises(PriceModelError, match="unknown price model 'GBMPrice'"):
        from_declaration({"type": "GBMPrice", "params": {}})
